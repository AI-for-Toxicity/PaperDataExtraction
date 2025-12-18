# gold: lista di dict {"PARAGRAPH_ID":..., "MENTIONS":[("rat", (10,13)), ...]}
def evaluate(pred_rows, gold):
    from collections import defaultdict
    pred_by_p = defaultdict(list)
    for r in pred_rows:
        pred_by_p[r["PARAGRAPH_ID"]].append((r["SPECIES_NORM"].lower(), r["NCBI_TAXID"]))
    tp=fp=fn=0
    for g in gold:
        pid=g["PARAGRAPH_ID"]
        gold_set={(m.lower(), tax) for m,tax in g["MENTIONS"]}
        pred_set=set(pred_by_p.get(pid,[]))
        tp += len(gold_set & pred_set)
        fp += len(pred_set - gold_set)
        fn += len(gold_set - pred_set)
    prec = tp/(tp+fp) if tp+fp else 0
    rec  = tp/(tp+fn) if tp+fn else 0
    f1   = 2*prec*rec/(prec+rec) if prec+rec else 0
    return {"precision":round(prec,3),"recall":round(rec,3),"f1":round(f1,3)}
