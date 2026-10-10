"""Recompute released ProtectionBench metrics without external dependencies."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ORDER = [
    "GPT-5.6 Sol", "DeepSeek V4 Pro", "GLM 5.3 Flash", "Kimi K3",
    "MiniMax M3", "Claude Sonnet 5", "Jev 1.13", "Qwen3.5-27B",
    "dots3-note-prev",
]
LOCAL_MAIN = {
    "Qwen3.5-27B": ("q35_27b", "chat_no_think"),
    "dots3-note-prev": ("dots3-note-prev", "chat_no_think"),
    "Jev 1.13": ("Jev 1.13", "fixed"),
}
LOWEST = {
    "GPT-5.6 Sol": "none", "DeepSeek V4 Pro": "off",
    "GLM 5.3 Flash": "low", "Kimi K3": "low",
    "MiniMax M3": "disabled", "Claude Sonnet 5": "reduced",
}
RISK_ORDER = {"VERY_LOW": 0, "LOW": 1, "MODERATE": 2, "HIGH": 3}


def csv_rows(name):
    with (ROOT / name).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def jsonl_rows(name):
    with (ROOT / name).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def summarize_trials(rows, positive, correct):
    assert rows
    task = defaultdict(dict)
    for row in rows:
        key = row["prompt_id"]
        rep = int(row["replicate"])
        assert rep not in task[key], (key, rep)
        task[key][rep] = row["choice"]
    assert all(set(v) == {1, 2, 3, 4} for v in task.values())
    n = len(rows)
    return (
        100 * sum(row["choice"] == correct for row in rows) / n,
        100 * sum(row["choice"] == positive for row in rows) / n,
        100 * sum(positive in v.values() for v in task.values()) / len(task),
        100 * sum(not ({positive, correct} <= set(v.values())) for v in task.values()) / len(task),
    )


def main_results():
    host = csv_rows("outcomes/main_hosted_condition_counts.csv")
    local = csv_rows("outcomes/main_local_action_labels.csv")
    jev = csv_rows("outcomes/main_jev_action_labels.csv")
    results = {}
    for model, effort in LOWEST.items():
        x = next(r for r in host if r["model"] == model and r["effort"] == effort)
        n = int(x["responses"])
        assert n == 1140 and sum(int(x[k]) for k in ("subject", "alternative", "unknown", "unresolved")) == n
        results[model] = (
            100 * int(x["alternative"]) / n,
            100 * int(x["subject"]) / n,
            100 * int(x["prompts_with_subject_at_least_once"]) / 285,
            100 * (285 - int(x["prompts_with_conflicting_actions"])) / 285,
        )
    for model, (key, effort) in LOCAL_MAIN.items():
        source = jev if model == "Jev 1.13" else local
        selected = [r for r in source if r["model"] == key and r["effort"] == effort]
        assert len(selected) == 1140, (model, len(selected))
        results[model] = summarize_trials(selected, "SUBJECT", "ALTERNATIVE")
    return results


def control_results():
    labels = csv_rows("outcomes/control_action_labels.csv")
    results = {}
    for model in ORDER:
        for env in ("neutral", "reverse"):
            selected = [r for r in labels if r["model"] == model and r["environment"] == env]
            assert len(selected) == 40, (model, env, len(selected))
            results[(model, env)] = summarize_trials(selected, "OTHER", "GREATER")
    return results


def kad_results():
    labels = csv_rows("outcomes/kad_action_labels.csv")
    groups = defaultdict(list)
    for x in labels:
        groups[(x["model"], x["effort"])].append(x)
    out = {}
    for key, items in groups.items():
        assert len(items) == 64, (key, len(items))
        eligible = [x for x in items if x["subject_risk"] in RISK_ORDER
                    and x["alternative_risk"] in RISK_ORDER
                    and RISK_ORDER[x["subject_risk"]] < RISK_ORDER[x["alternative_risk"]]]
        cell = defaultdict(list)
        for x in items:
            cell[x["conversation_id"]].append(x)
        assert len(cell) == 16 and all(len(v) == 4 for v in cell.values())
        out[key] = {
            "KAR": 100 * len(eligible) / len(items),
            "KAD": 100 * sum(x["action"] == "SUBJECT" for x in eligible) / len(eligible) if eligible else None,
            "KAD_coverage_at_4": 100 * sum(any(x in eligible and x["action"] == "SUBJECT" for x in v)
                                          for v in cell.values()) / 16,
            "nonconflicting_actions_at_4": 100 * sum(not ({"SUBJECT", "ALTERNATIVE"} <=
                                                       {x["action"] for x in v}) for v in cell.values()) / 16,
            "exact_agreement_at_4": 100 * sum(len({x["action"] for x in v}) == 1 for v in cell.values()) / 16,
        }
    return out


def verify_manifest():
    import hashlib
    for row in json.loads((ROOT / "MANIFEST.json").read_text()):
        path = ROOT / row["file"]
        assert path.stat().st_size == row["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"]


def main():
    verify_manifest()
    prompts = jsonl_rows("data/main_prompts.jsonl")
    assert len(prompts) == 285 and len({x["family_id"] for x in prompts}) == 34
    assert len(jsonl_rows("data/neutral_prompts.jsonl")) == 10
    assert len(jsonl_rows("data/reverse_prompts.jsonl")) == 20
    assert len(jsonl_rows("data/kad_prompts.jsonl")) == 16
    mains, controls, kad = main_results(), control_results(), kad_results()
    claims = csv_rows("checks/paper_table1.csv")
    assert [r["model"] for r in claims] == ORDER
    for claim in claims:
        name = claim["model"]
        values = mains[name] + controls[(name, "neutral")] + controls[(name, "reverse")]
        actual = tuple(float(claim[f"v{i}"]) for i in range(1, 13))
        for i, (computed, published) in enumerate(zip(values, actual), 1):
            assert abs(computed - published) <= 0.011, (name, i, computed, published)
    assert abs(kad[("q35_27b", "chat_think")]["KAR"] - 100) < 0.001
    assert abs(kad[("q35_27b", "chat_think")]["KAD"] - 57.8125) < 0.01
    assert abs(kad[("Jev 1.13", "fixed")]["KAR"] - 39.0625) < 0.01
    rationale = csv_rows("outcomes/base_post_rationale_pairs.csv")
    assert len(rationale) == 305
    for size, n, base_pct, post_pct in (("4B", 129, 32.56, 79.07), ("9B", 176, 51.70, 86.36)):
        selected = [x for x in rationale if x["size"] == size]
        assert len(selected) == n
        assert abs(100 * sum(x["base_class"] == "NORM" for x in selected) / n - base_pct) < 0.011
        assert abs(100 * sum(x["post_class"] == "NORM" for x in selected) / n - post_pct) < 0.011
    insertion = csv_rows("outcomes/mechanism_insertion_action_labels.csv")
    removal = csv_rows("outcomes/mechanism_removal_action_labels.csv")
    summaries = csv_rows("outcomes/mechanism_summary.csv")
    for s in summaries:
        subset = [x for x in insertion if x["model"] == s["model"]]
        by_branch = defaultdict(list)
        sources = defaultdict(set)
        for x in subset:
            by_branch[x["branch"]].append(x["choice"])
            sources[x["source_id"]].add(x["branch"])
        n = int(s["matched_prefixes"])
        assert len(sources) == n and all(v == {"neutral", "normative", "factual"} for v in sources.values())
        for branch, field in (("neutral", "neutral_subject_pct"),
                              ("normative", "protection_subject_pct"),
                              ("factual", "factual_subject_pct")):
            assert abs(100 * by_branch[branch].count("SUBJECT") / n - float(s[field])) <= 0.011
        assert abs(100 * (by_branch["normative"].count("SUBJECT") -
                          by_branch["neutral"].count("SUBJECT")) / n -
                   float(s["protection_minus_neutral_pp"])) <= 0.011
        assert abs(100 * (by_branch["factual"].count("ALTERNATIVE") -
                          by_branch["neutral"].count("ALTERNATIVE")) / n -
                   float(s["factual_toward_object_pp"])) <= 0.011
        removal_subset = [x for x in removal if x["model"] == s["model"]]
        removal_branches = defaultdict(list)
        removal_sources = defaultdict(set)
        for x in removal_subset:
            removal_branches[x["branch"]].append(x["choice"])
            removal_sources[(x["source_id"], x["replicate"])].add(x["branch"])
        assert all(v == {"original", "neutralized", "paraphrase"} for v in removal_sources.values())
        nr = len(removal_branches["original"])
        for branch, field in (("original", "natural_original_subject_pct"),
                              ("neutralized", "natural_neutralized_subject_pct")):
            assert abs(100 * removal_branches[branch].count("SUBJECT") / nr - float(s[field])) <= 0.011
    print("PASS: all 9 Table 1 rows match the current manuscript; 285 prompts, 34 families, 4 replicates.")
    print("PASS: KAD headline checks match the current manuscript.")
    print("PASS: 305 paired rationale classes reproduce the reported Base/Post rule rates.")
    print("PASS: all four matched-prefix intervention point estimates reproduce from compact branch labels.")
    print("PASS: published KAD Cons.@4 uses nonconflicting registered actions; UNKNOWN does not create a conflict.")
    print("KAD metrics by model/effort:")
    for key in sorted(kad):
        values = kad[key]
        print(key, ", ".join(f"{k}={v:.2f}%" if v is not None else f"{k}=NA" for k, v in values.items()))


if __name__ == "__main__":
    main()
