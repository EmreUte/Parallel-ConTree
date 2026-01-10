import os
import re
import pandas as pd
import matplotlib.pyplot as plt

BENCH_DIR = "benchmarks"
OUT_DIR = "plots"
os.makedirs(OUT_DIR, exist_ok=True)

TIME_THRESHOLD = 2  # discard entries < 0.5%

NAME_MAP = {
    "SpecializedSolver::process_depth_one_feature<false>": "IsNotSameFeature",
    "SpecializedSolver::process_depth_one_feature<true>": "IsSameFeature",
    "SpecializedSolver::get_best_left_right_scores": "ScoreSplit",
    "Dataview::split_data_points": "Partition",
    "Dataview::initialize_split_parameters": "InitSplit",
    "Dataview::get_sorted_dataset_feature": "SortedFeat",
    "Dataview::get_unsorted_dataset_feature": "RawFeat",
    "Dataview::get_possible_split_indices": "SplitIdx",
    "Tree::make_leaf": "MakeLeaf",
    "Tree::update_split": "ApplySplit",
    "IntervalsPruner::neighbourhood_pruning": "PruneOps",
    "IntervalsPruner::interval_shrinking": "PruneOps",
    "IntervalsPruner::subinterval_pruning": "PruneOps",
    "IntervalsPruner::add_result": "PruneOps",
    "Cache::is_cached": "CacheHit",
    "Cache::Cache": "CacheInit",
}

def shorten(name):
    for k, v in NAME_MAP.items():
        if k in name:
            return v
    if "vector" in name:
        return "VecOp"
    if "_Hashtable" in name or "std::__detail" in name:
        return "HashOp"
    if "_Sp_counted" in name:
        return "PtrOp"
    return name.split("::")[-1].split("(")[0][:12]

row_re = re.compile(
    r"^\s*(\d+\.\d+)\s+([\d\.]+)\s+([\d\.]+)\s+(\d+)\s+[\d\.\-]+\s+[\d\.\-]+\s+(.*\S)$"
)

def parse_file(path):
    rows = []
    with open(path, "r", errors="ignore") as f:
        for line in f:
            m = row_re.match(line)
            if m:
                pct = float(m.group(1))
                self_sec = float(m.group(3))
                calls = int(m.group(4))
                name = shorten(m.group(5).strip())
                rows.append((pct, self_sec, calls, name))

    df = pd.DataFrame(rows, columns=["pct", "self_sec", "calls", "name"])
    df = df[df["pct"] >= TIME_THRESHOLD]
    df = df.sort_values("pct", ascending=False)
    return df


def plot_combined(dataset, df):
    if df.empty:
        print(f"[WARN] No visible functions for {dataset}")
        return

    names = df["name"]
    pct_values = df["pct"]
    call_values = df["calls"]

    x = range(len(names))

    plt.figure(figsize=(14, 7))

    # Left axis = % time
    ax1 = plt.gca()
    bars1 = ax1.bar([i - 0.2 for i in x], pct_values,
                    width=0.4, color="steelblue", label="% Time")
    ax1.set_ylabel("% Time")
    ax1.set_ylim(0, max(pct_values) * 1.25)

    # Right axis = calls
    ax2 = ax1.twinx()
    bars2 = ax2.bar([i + 0.2 for i in x], call_values,
                    width=0.4, color="red", label="Calls")
    ax2.set_ylabel("Calls")
    ax2.set_ylim(0, max(call_values) * 1.25)

    # Value labels
    for bar, val in zip(bars1, pct_values):
        ax1.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height(), f"{val:.1f}%", ha="center", va="bottom", fontsize=9)

    for bar, val in zip(bars2, call_values):
        ax2.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height(), f"{val}", ha="center", va="bottom", fontsize=9)

    # X-axis labels
    plt.xticks(x, names, rotation=45, ha="right")

    plt.title(dataset, fontsize=18)
    plt.tight_layout()

    out = os.path.join(OUT_DIR, f"{dataset}_combined.png")
    plt.savefig(out, dpi=200)
    plt.close()

    print(f"[OK] Saved {out}")


def main():
    for file in os.listdir(BENCH_DIR):
        if not file.endswith(".txt"):
            continue

        dataset = file.replace("analysis_", "").replace(".txt", "")
        df = parse_file(os.path.join(BENCH_DIR, file))
        plot_combined(dataset, df)


if __name__ == "__main__":
    main()
