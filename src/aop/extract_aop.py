"""
AOP extraction pipeline (markdown -> entities -> relations -> AOP chains)

Assumptions:
- You already have cleaned Markdown files organized per-paper.
- This script skips PDF parsing / OCR (you said so).

What it does:
1. Load markdown files and split into sections (abstract, results, etc.).
2. Run biomedical NER (Hugging Face token-classification pipeline).
3. Create candidate entity pairs from sentences.
4. Use an LLM (OpenAI by default) to classify/score causal relations for pairs.
5. Build a directed graph of events, collapse into ordered AOP chains.

Usage:
    export OPENAI_API_KEY=...
    pip install -r requirements.txt
    python aop_extraction_pipeline.py --md_dir ./clean_md --out results.jsonl

Requirements (suggested):
transformers>=4.30
torch
sentencepiece
openai
tqdm
networkx
spacy

Notes:
- Replace the HF NER model with one you prefer (PubMedBERT-based token classification or scispacy).
- Relation extraction here uses LLM prompting. If you have a fine-tuned RE model, plug it in.

"""

import os
import re
import json
import argparse
from pathlib import Path
from typing import List, Dict, OrderedDict, Tuple, Any
from tqdm import tqdm

# Hugging Face for NER
from transformers import pipeline

# For sentence splitting
import spacy

# Graph building
import networkx as nx

# LLM (OpenAI) for relation classification / chaining
try:
    import openai
except Exception:
    openai = None


# -------------------------- Utilities --------------------------
def load_markdown_sections(md_text: str) -> Dict[str, str]:
    sections = OrderedDict()
    current_title = 'NOTITLE'
    for line in md_text.splitlines():
        line = line.strip()
        if line.startswith('#'):
            # In case of empty section, keep title as content
            if current_title is not None and current_title != "NOTITLE" and sections.get(current_title) == '':
                sections[current_title] = current_title

            current_title = line.lstrip('#').strip()
            sections[current_title] = ''
        elif current_title and line != '':
            if current_title == 'NOTITLE' and current_title not in sections:
                sections[current_title] = ''
            sections[current_title] += line + ' '
    
    return sections

def sentence_splitter(nlp, text: str) -> List[str]:
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]

# -------------------------- NER --------------------------
class HFNER:
    def __init__(self, model_name: str = 'en_ner_bionlp13cg_md'):
        # token-classification pipeline: returns list of entities with 'word','score','entity','start','end'
        self.ner = pipeline('token-classification', model=model_name, aggregation_strategy='simple')

    def extract(self, text: str) -> List[Dict[str, Any]]:
        # returns entities with text and span
        ents = self.ner(text)
        out = []
        for e in ents:
            out.append({'text': e['word'], 'label': e.get('entity_group', e.get('entity')), 'score': e['score'], 'start': e['start'], 'end': e['end']})
        return out

# -------------------------- Relation / Causality Scoring --------------------------
class LLMCausalScorer:
    """Uses an LLM (OpenAI) to score if entity A causes B in a sentence/context.
    The method returns a score between 0 and 1 and a short rationale.
    """
    def __init__(self, provider: str = 'openai', model: str = 'gpt-4o-mini'):
        self.provider = provider
        self.model = model
        if provider == 'openai' and openai is None:
            raise RuntimeError('openai package not installed. pip install openai')
        if provider == 'openai':
            # openai.api_key should be set in env
            pass

    def score_pair(self, context_sentence: str, ent_a: str, ent_b: str) -> Tuple[float, str]:
        prompt = self._build_prompt(context_sentence, ent_a, ent_b)
        if self.provider == 'openai':
            resp = openai.ChatCompletion.create(
                model=self.model,
                messages=[{'role':'user','content': prompt}],
                max_tokens=150,
                temperature=0.0,
            )
            txt = resp['choices'][0]['message']['content'].strip()
        else:
            # Fallback: very naive heuristic
            txt = self._heuristic(context_sentence, ent_a, ent_b)
        # Parse the LLM answer which should be in JSON like: {"score":0.85, "rationale":"..."}
        try:
            parsed = json.loads(txt)
            score = float(parsed.get('score', 0.0))
            rationale = parsed.get('rationale', '')
        except Exception:
            # attempt to extract a float
            m = re.search(r"([0-9]*\.?[0-9]+)", txt)
            score = float(m.group(1)) if m else 0.0
            rationale = txt
        return max(0.0, min(1.0, score)), rationale

    def _build_prompt(self, sentence: str, a: str, b: str) -> str:
        return (
            "You are an expert toxicologist and curator.\n"
            "Given the following sentence from a scientific paper, tell me how likely it is that the first entity causally contributes to the second entity.\n"
            "Answer in strict JSON with two fields: score (float 0-1) and rationale (short).\n\n"
            f"Sentence: \"{sentence}\"\n"
            f"Entity A: \"{a}\"\n"
            f"Entity B: \"{b}\"\n\n"
            "Guidelines:\n"
            "- If sentence states a causal or mechanistic link (e.g., 'A inhibits B, causing C'), give score >= 0.8.\n"
            "- If evidence is correlational or ambiguous, give score between 0.3 and 0.7.\n"
            "- If sentence denies a causal link or says no effect, score <= 0.1.\n"
        )

    def _heuristic(self, sentence, a, b):
        # quick keyword-based fallback
        kw = ['cause', 'lead to', 'result in', 'induces', 'inhibits', 'inhibition', 'activation', 'mediates', 'promote', 'reduces', 'increases']
        s = sentence.lower()
        score = 0.0
        for k in kw:
            if k in s:
                score += 0.25
        score = min(1.0, score)
        return json.dumps({'score': score, 'rationale': 'heuristic keyword match'})

# -------------------------- Graph / AOP Assembly --------------------------
def build_aop_graph(relations: List[Dict[str, Any]], score_threshold: float = 0.6) -> nx.DiGraph:
    """relations: list of {a, b, score, rationale, context, paper_id}
    Build a directed graph keeping edges with score >= threshold. Collapse identical nodes by text.
    """
    G = nx.DiGraph()
    for r in relations:
        a = r['a'].strip()
        b = r['b'].strip()
        s = r['score']
        if s >= score_threshold and a and b and a.lower() != b.lower():
            G.add_node(a)
            G.add_node(b)
            # keep best score if multiple edges
            if G.has_edge(a, b):
                if G[a][b]['score'] < s:
                    G[a][b]['score'] = s
                    G[a][b]['evidences'] = [r]
                else:
                    G[a][b]['evidences'].append(r)
            else:
                G.add_edge(a, b, score=s, evidences=[r])
    return G

def extract_chains_from_graph(G: nx.DiGraph, min_len: int = 2) -> List[List[str]]:
    # naive approach: find all simple paths up to length 6 and return sorted by sum edge scores
    chains = []
    nodes = list(G.nodes)
    for src in nodes:
        for dst in nodes:
            if src == dst:
                continue
            for path in nx.all_simple_paths(G, source=src, target=dst, cutoff=6):
                if len(path) >= min_len:
                    # compute path score as min(edge scores) or product
                    scores = [G[path[i]][path[i+1]]['score'] for i in range(len(path)-1)]
                    path_score = min(scores)
                    chains.append({'path': path, 'score': path_score})
    # deduplicate by exact path
    unique = {}
    for c in chains:
        key = ' -> '.join(c['path'])
        if key not in unique or unique[key]['score'] < c['score']:
            unique[key] = c
    out = sorted(unique.values(), key=lambda x: x['score'], reverse=True)
    return [o['path'] for o in out]

# -------------------------- Main pipeline --------------------------
def process_paper(md_path: Path, ner_model: HFNER, nlp, scorer: None) -> Dict[str, Any]: #LLMCausalScorer) -> Dict[str, Any]:
    text = md_path.read_text(encoding='utf-8')
    sections = load_markdown_sections(text)
    paper_id = md_path.stem
    extracted_entities = []
    relations = []

    for sec_title, sec_text in sections.items():
        sentences = sentence_splitter(nlp, sec_text)
        for sent in sentences:
            ents = ner_model.extract(sent)
            # filter / normalize
            ents = [e for e in ents if len(e['text']) > 1]
            # deduplicate entities by text span
            seen = set()
            clean_ents = []
            for e in ents:
                t = e['text'].strip()
                if t.lower() in seen:
                    continue
                seen.add(t.lower())
                clean_ents.append(e)
            # store
            for e in clean_ents:
                extracted_entities.append({'paper': paper_id, 'section': sec_title, 'sentence': sent, 'text': e['text'], 'label': e['label'], 'score': e['score']})
            # candidate pairs: all pairs in the sentence
            for i in range(len(clean_ents)):
                for j in range(len(clean_ents)):
                    if i == j:
                        continue
                    a = clean_ents[i]['text']
                    b = clean_ents[j]['text']
                    # score via LLM
                    try:
                        sc, rationale = scorer.score_pair(sent, a, b)
                    except Exception as ex:
                        sc, rationale = 0.0, f'error: {ex}'
                    relations.append({'paper': paper_id, 'section': sec_title, 'sentence': sent, 'a': a, 'b': b, 'score': sc, 'rationale': rationale})
    # assemble graph
    G = build_aop_graph(relations, score_threshold=0.6)
    chains = extract_chains_from_graph(G, min_len=2)
    return {'paper': paper_id, 'entities': extracted_entities, 'relations': relations, 'chains': chains}


def main(args):
    nlp = spacy.load('en_core_web_sm')
    print('Loading NER model...')
    ner_model = HFNER(model_name=args.ner_model)
    print('Initializing LLM scorer...')
    scorer = LLMCausalScorer(provider=args.llm_provider, model=args.llm_model)

    md_dir = Path(args.md_dir)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    results = []
    for p in list(md_dir.glob('*.md')):
        try:
            res = process_paper(p, ner_model, nlp, scorer)
            results.append(res)
            with open(out / f'{p.stem}.jsonl', 'w', encoding='utf-8') as fh:
                json_dump = json.dumps(res, ensure_ascii=False, default=lambda o: o.item() if hasattr(o, "item") else (o.tolist() if hasattr(o, "tolist") else str(o)))
                fh.write(json_dump + '\n')
        except Exception as e:
            print(f'Error processing {p}: {e}')

    # aggregated index
    with open(out / 'index.json', 'w', encoding='utf-8') as fh:
        json.dump({'papers': [r['paper'] for r in results]}, fh, indent=2)

    print('Done. Results in', out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--md_dir', type=str, default='data/processed/gold', help='Directory with cleaned markdown files')
    parser.add_argument('--out', type=str, default='data/processed/gold/out', help='Output directory for results')
    parser.add_argument('--ner_model', type=str, default='d4data/biomedical-ner-all', help='HF token-classification model')
    parser.add_argument('--llm_provider', type=str, default='heuristic', choices=['openai','heuristic'], help='LLM provider for causality scoring')
    parser.add_argument('--llm_model', type=str, default='gpt-4o-mini', help='LLM model name (OpenAI)')
    args = parser.parse_args()
    # quick env check for OpenAI
    if args.llm_provider == 'openai':
        if openai is None:
            raise RuntimeError('openai not available, install openai package')
        if os.getenv('OPENAI_API_KEY') is None:
            raise RuntimeError('Please set OPENAI_API_KEY in your environment')
        openai.api_key = os.getenv('OPENAI_API_KEY')
    main(args)
