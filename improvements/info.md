considering all of the gold is LLM generated, I have more than 100 papers of data, and I have no domain expertise, how would you suggest tackling other problems? should I ask different LLMs to review extractions

# User
add a full_eval_analysis_folder_path agrument and write a new PredEvaluator function that generates for each chunk a md file with chunk text, then gold events, then pred events. order events alphabetically by chemical name, then event_type (MIE then KE then AO), then alphabetically on the description.
execute it at the end of analyze_eval_jsonl

# Assistant
Checked chunks 1 to 6.

Now I have enough to give a thorough diagnosis. Here's what the data shows:

---

## Diagnosis

### Problem 1 — Gold labels are selectively incomplete (not a model failure)

**Chunk 1:** Gold has 4 events. Pred has 11, all text-grounded (score 52–100), chemicals found. But only 1 matches. Looking at what the model actually extracted: COX inhibition, ROS generation, OATP2B1 uptake, gastrointestinal injury — these are **valid toxicology events mentioned in the text**. The annotator selected only the 4 events that map to the AOP Wiki entries they were tracking, not all events mentioned.

**Chunk 4:** The text literally contains Table 2 listing all known biological effects of 1,4-benzoquinone. The model reads the table and outputs them all (14 events, all score ~100, all chemically grounded). Gold only has 2 of them. Again, the model is right, the annotation is just selective.

**Consequence:** The precision/recall of 0.16/0.20 and F1=0.177 is largely an artifact of the annotation strategy, not of the model outputting wrong things. The real grounded-precision (pred events that are textually supported) is **324/325 = ~99.7%**, which tells a very different story.

---

### Problem 2 — Model generates garbage on negative examples (real training failure)

Records 2, 3, 8, 9... have loss **20–27** with gold=0 and pred=0. Chunk 3's text is a QC/methods section — no events. The model outputs something unparseable (0 events extracted) but the raw string is long garbage that doesn't match \"### END\", hence the extreme loss. This means the model **has not reliably learned to output nothing for negative chunks**. This is a real failure: on ~50% of records (79/158 are negative), the model is hallucinating or failing to terminate.

---

### Problem 3 — Abstraction level mismatch

**Chunk 5:** Gold has `Roundup | MIE | Cell entry of glyphosate` (AOP Wiki abstraction). The model outputs `Roundup | KE | Disrupted CYP1A2 activity restored by Dig2` — a specific mechanism mentioned in the abstract. Both are textually grounded, but they're at different abstraction levels. The model is learning to extract what's **said in the text**, but gold labels are what the **annotator decided is the AOP-relevant framing**, which often doesn't match the phrasing used in the paper.

**Chunk 6:** Gold has `DAPT | MIE | Notch pathway inhibition`. The word \"Notch pathway inhibition\" does appear in the text, but the model only outputs KE-level effects (reduced neuronal differentiation, neurite length). MIEs — the molecular trigger — are systematically underrepresented in model output, likely because they require more reasoning rather than surface extraction.

---

### Problem 4 — Chemical name variant matching gaps

`Roundup` vs `Roundup (glyphosate and adjuvants)` fails both exact and fuzzy matching because the extra parenthetical inflates the string distance. `1,4-Benzoquinone` vs `1,4-benzoquinone` is caught by `norm()`. `DAPT (Notch inhibitor)` in gold vs nothing in pred — the model simply doesn't extract DAPT events at all.

---

## Summary table

| Issue | Root cause | Impact on metrics |
|---|---|---|
| Model extracts all textually-valid events, gold is selective | Annotation strategy | Inflates FP, suppresses precision |
| Garbage output on negative chunks | Training: insufficient negative signal | Inflates loss, ~50% of records |
| AOP Wiki abstraction vs text-level extraction | Data labeling philosophy | Suppresses recall |
| Missing MIE extractions | Model limitation / lack of MIE training signal | Suppresses recall on MIEs |

---

## Proposed solutions

**1. Fix the evaluation philosophy first.**
The F1=0.177 is measuring \"did the model pick the same 4 events the annotator chose\" not \"did the model extract correct toxicology events.\" Before changing the model, verify manually whether the 10 \"FP\" events from chunk 1 are actually valid. If they are, the metric is broken, not the model. Consider using a human-reviewed precision on a sample.

**2. Fix the negative example generation failure.**
Inspect the raw `pred` strings for records with loss >10 to confirm they're garbage (not truncated CSV). Then either:
- Increase the proportion of negative examples in training
- Add a stopping criterion: force generation to stop at `### END` (constrained decoding)
- Or check if `max_new_tokens` is too small, causing the model to be cut off mid-output on longer positive examples and producing truncated garbage for negatives too

**3. Bridge the abstraction gap in the training data.**
The training JSONL was built from AOP Wiki-style annotations. Options:
- Augment the gold with text-extracted paraphrases of the same events (the model is already doing this correctly — reward it)
- Or conversely, during inference post-process pred descriptions by matching them to AOP Wiki event names (you have a RAG index for this)

**4. Add MIE-specific training signal.**
MIEs are underrepresented in predictions. Check the ratio of MIE/KE/AO in `train.jsonl` — if MIEs are rare, the model underweights them. Oversample MIE-containing examples or add a note in the prompt that MIEs (molecular initiating events, the primary interaction) are especially important.

**5. Normalize chemical names before matching.**
In `compare_gold_pred`, strip parenthetical variants before computing the norm key, so `Roundup (glyphosate and adjuvants)` and `Roundup` resolve to the same base name.