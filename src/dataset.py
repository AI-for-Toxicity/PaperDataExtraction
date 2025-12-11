import json
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional


SYSTEM_PROMPT = (
    "You are an assistant specialized in extracting mechanistic toxicology "
    "events (MIE, KE, AO) from scientific text."
)


def csv_quote(value: str) -> str:
    """
    Wrap a string in double quotes and escape internal quotes for CSV-like output.
    """
    if value is None:
        value = ""
    value = str(value)
    value = value.replace('"', '""')
    return f'"{value}"'


def normalize_whitespace(text: str) -> str:
    """
    Basic whitespace normalization to avoid duplicati per differenze di spazi.
    """
    return " ".join((text or "").split())


def extract_chunk_examples_from_file(
    path: Path,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Legge un file *_events.json e restituisce:

      - lista di esempi POSITIVI:
        [
          {
            "chunk_text": <testo del chunk>,
            "events": [
              {"chemical": ..., "event_type": ..., "description": ...},
              ...
            ]
          },
          ...
        ]

      - lista di chunk VUOTI (NEGATIVI):
        [ <chunk_text_senza_eventi>, ... ]

    - Usa SOLO events dentro ai chunks (ignora unmatched_events).
    - Tiene solo event_type in {MIE, KE, AO}.
    - Deduplica per tripla (chemical, event_type, description) all'interno del chunk.
    """
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    chunks: List[Dict[str, Any]] = data.get("chunks", [])
    positive_examples: List[Dict[str, Any]] = []
    empty_chunks: List[str] = []

    for ch in chunks:
        text = (ch.get("text") or "").strip()
        if not text:
            continue

        raw_events = ch.get("events") or []
        seen_triples: set[Tuple[str, str, str]] = set()
        collected_events: List[Dict[str, str]] = []

        for ev in raw_events:
            event_type = (ev.get("event_type") or "").strip().upper()
            if event_type not in {"MIE", "KE", "AO"}:
                continue

            chemical = normalize_whitespace(ev.get("chemical", ""))
            short_desc = normalize_whitespace(ev.get("event_description_short", ""))
            long_desc = normalize_whitespace(ev.get("event_description_long", ""))

            # scegliamo la descrizione: short se c'è, altrimenti long
            description = short_desc or long_desc

            # se manca chemical o descrizione, per il tuo formato non ha senso tenerlo
            if not chemical or not description:
                continue

            triple_key = (chemical.lower(), event_type, description.lower())
            if triple_key in seen_triples:
                continue
            seen_triples.add(triple_key)

            collected_events.append(
                {
                    "chemical": chemical,
                    "event_type": event_type,
                    "description": description,
                }
            )

        if collected_events:
            positive_examples.append(
                {
                    "chunk_text": text,
                    "events": collected_events,
                }
            )
        else:
            # chunk con testo ma zero eventi validi → candidato NEGATIVO
            empty_chunks.append(text)

    return positive_examples, empty_chunks


def build_messages_for_chunk(
    chunk_text: str,
    events: List[Dict[str, str]],
) -> Dict[str, Any]:
    """
    Esempio POSITIVO: chunk con eventi.
    - user: prompt + testo del chunk
    - assistant: lista di righe "chemical","event_type","description"
    """
    user_prompt = (
        "Given the following text from a toxicology article, extract all MIE, KE and AO "
        "events with the associated chemical and a concise description.\n\n"
        "Return one event per line in the exact format:\n"
        "\"chemical\",\"event_type\",\"description\"\n\n"
        "If the text does not contain any MIE, KE or AO events, return an empty output.\n\n"
        f"Text:\n{chunk_text}"
    )

    lines: List[str] = []
    for ev in events:
        chem = csv_quote(ev["chemical"])
        etype = csv_quote(ev["event_type"])
        desc = csv_quote(ev["description"])
        lines.append(f"{chem},{etype},{desc}")

    assistant_output = "\n".join(lines)

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_output},
        ]
    }


def build_messages_for_empty_chunk(chunk_text: str) -> Dict[str, Any]:
    """
    Esempio NEGATIVO: chunk senza eventi.
    - user: stesso prompt
    - assistant: output vuoto (stringa "")
    """
    user_prompt = (
        "Given the following text from a toxicology article, extract all MIE, KE and AO "
        "events with the associated chemical and a concise description.\n\n"
        "Return one event per line in the exact format:\n"
        "\"chemical\",\"event_type\",\"description\"\n\n"
        "If the text does not contain any MIE, KE or AO events, return an empty output.\n\n"
        f"Text:\n{chunk_text}"
    )

    assistant_output = ""  # nessun evento → output vuoto

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_output},
        ]
    }


def build_biomistral_chunk_dataset(
    input_dir: str,
    train_path: str,
    test_path: str,
    test_ratio: float = 0.05,
    empty_ratio: float = 1.0,
    seed: int = 42,
) -> None:
    """
    - Legge tutti i file .json in input_dir (formato *_events.json).
    - Crea esempi a livello di chunk:
        * POSITIVI: chunk con ≥1 evento MIE/KE/AO
        * NEGATIVI: chunk con testo ma 0 eventi MIE/KE/AO
    - Aggiunge un numero di esempi NEGATIVI ≈ empty_ratio * (#positivi),
      campionati dai chunk vuoti disponibili.
    - Fa shuffle e split train/test e salva due JSONL.

    test_ratio piccolo (es. 0.05) perché hai pochi esempi.
    """
    input_dir_path = Path(input_dir)
    files = sorted(input_dir_path.glob("*.json"))

    positive_examples: List[Dict[str, Any]] = []
    empty_candidates: List[str] = []

    for fp in files:
        pos, empties = extract_chunk_examples_from_file(fp)
        positive_examples.extend(pos)
        empty_candidates.extend(empties)

    if not positive_examples:
        raise RuntimeError(
            "No positive (labeled) chunk examples found. "
            "Check input directory / file formats / event filtering."
        )

    # costruisci esempi POSITIVI
    all_examples: List[Dict[str, Any]] = []
    for ex in positive_examples:
        msg_example = build_messages_for_chunk(ex["chunk_text"], ex["events"])
        all_examples.append(msg_example)

    # aggiungi esempi NEGATIVI (vuoti), se disponibili
    if empty_candidates and empty_ratio > 0:
        rng = random.Random(seed)
        rng.shuffle(empty_candidates)

        n_pos = len(positive_examples)
        n_neg_desired = int(n_pos * empty_ratio)
        n_neg = min(n_neg_desired, len(empty_candidates))

        selected_empties = empty_candidates[:n_neg]

        for chunk_text in selected_empties:
            msg_example = build_messages_for_empty_chunk(chunk_text)
            all_examples.append(msg_example)

    # shuffle globale e split train/test
    rng = random.Random(seed + 1)
    rng.shuffle(all_examples)

    n_total = len(all_examples)
    if n_total == 1:
        train_examples = all_examples
        test_examples: List[Dict[str, Any]] = []
    else:
        n_test = max(1, int(n_total * test_ratio))
        n_test = min(n_test, n_total - 1)
        test_examples = all_examples[:n_test]
        train_examples = all_examples[n_test:]

    train_p = Path(train_path)
    test_p = Path(test_path)

    with train_p.open("w", encoding="utf-8") as f_train:
        for ex in train_examples:
            f_train.write(json.dumps(ex, ensure_ascii=False) + "\n")

    with test_p.open("w", encoding="utf-8") as f_test:
        for ex in test_examples:
            f_test.write(json.dumps(ex, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    # Esempio d'uso:
    # metti tutti i tuoi *_events.json in "events_folder" e lancia:
    #
    #   python build_dataset_chunks.py
    #
    build_biomistral_chunk_dataset(
        input_dir="test_data/labels/scored",
        train_path="train.jsonl",
        test_path="test.jsonl",
        test_ratio=0.05,   # test piccolo
        empty_ratio=1.0,   # ~stesso numero di esempi vuoti dei positivi
        seed=42,
    )
