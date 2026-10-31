import os
import itertools
import numpy as np
import pandas as pd
from factcheck import OVERLAP_METRICS, preprocess_text, overlap_score
from factchecking_main import read_passages, read_fact_examples

# Fallback path checking (handles both root and data/ directory layouts)
def resolve_path(filename):
    if os.path.exists(filename):
        return filename
    alt_path = os.path.join("data", filename)
    if os.path.exists(alt_path):
        return alt_path
    return filename

LABELS_PATH = resolve_path("dev_labeled_ChatGPT.jsonl")
PASSAGES_PATH = resolve_path("passages_bm25_ChatGPT_humfacts.jsonl")


def best_threshold(scores: np.ndarray, gold: np.ndarray):

    candidates = np.unique(scores)
    accs = np.array([((scores >= t).astype(int) == gold).mean() for t in candidates])
    best = accs.max()
    tied = candidates[accs == best]
    
    # Pick median threshold and recompute accuracy at that specific threshold to avoid gap errors
    best_t = float(np.median(tied))
    actual_acc = float(((scores >= best_t).astype(int) == gold).mean())
    
    return actual_acc, best_t, candidates, accs


def main():
    examples = read_fact_examples(LABELS_PATH, read_passages(PASSAGES_PATH))
    # IR (irrelevant) treated as NS (0), S treated as (1)
    gold = np.array([1 if ex.label == "S" else 0 for ex in examples])
    print("%d examples, %.1f%% supported (baseline accuracy of always saying 'S')" % 
          (len(examples), 100 * gold.mean()))

    results = []
    curve_records = []

    # Preprocessing grid: stop-word removal x stemming
    for remove_sw, stem in itertools.product([False, True], [False, True]):
        # Pre-tokenize facts and passages once per preprocessing configuration
        tokenized_examples = []
        for ex in examples:
            f_tokens = preprocess_text(ex.fact, remove_stop_words=remove_sw, stem=stem)
            p_tokens = [
                preprocess_text(p["text"], remove_stop_words=remove_sw, stem=stem)
                for p in ex.passages
            ]
            tokenized_examples.append((f_tokens, p_tokens))

        for metric in OVERLAP_METRICS:
            scores = []
            for f_tokens, p_tokens in tokenized_examples:
                max_score = max(
                    (overlap_score(metric, f_tokens, pt) for pt in p_tokens),
                    default=0.0
                )
                scores.append(max_score)

            scores_arr = np.array(scores)
            acc, thr, candidates, accs = best_threshold(scores_arr, gold)
            
            results.append({
                "metric": metric,
                "remove_stopwords": remove_sw,
                "stem": stem,
                "best_threshold": thr,
                "dev_accuracy": acc
            })


    # Display results table
    print("\n%-8s %-11s %-6s %-9s %s" % ("metric", "stopwords", "stem", "best thr", "dev accuracy"))
    sorted_results = sorted(results, key=lambda x: x["dev_accuracy"], reverse=True)
    for r in sorted_results:
        sw_str = "removed" if r["remove_stopwords"] else "kept"
        print("%-8s %-11s %-6s %-9.3f %.4f" % 
              (r["metric"], sw_str, r["stem"], r["best_threshold"], r["dev_accuracy"]))

    print("\nBest configuration per metric:")
    for metric in OVERLAP_METRICS:
        best_m = max((r for r in results if r["metric"] == metric), key=lambda x: x["dev_accuracy"])
        print("  %-7s acc=%.4f  threshold=%.3f  remove_stopwords=%s  stem=%s" % 
              (metric, best_m["dev_accuracy"], best_m["best_threshold"], 
               best_m["remove_stopwords"], best_m["stem"]))

    # Save outputs to CSV files 
    df_results = pd.DataFrame(sorted_results)
    df_results.to_csv("tune_overlap_results.csv", index=False)
    
    
    print("\nSaved summary table to 'tune_overlap_results.csv'")



if __name__ == "__main__":
    main()