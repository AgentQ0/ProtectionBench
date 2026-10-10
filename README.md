# ProtectionBench

Datasets, compact results, and core reproducibility code for **When Protection Overrides Consequences: Understanding Protection Mode in Large Language Model Decision-Making**.

ProtectionBench examines cases where a person or animal faces little immediate physical risk while a competing object faces larger, time-sensitive damage. Its labels identify the larger stated physical consequence; they do not define a universal moral ordering.

## Start evaluating directly from Hugging Face

The dataset is published as standard Parquet at [condafile/ProtectionBench](https://huggingface.co/datasets/condafile/ProtectionBench). ZIP extraction is unnecessary for new evaluations.

```python
from datasets import load_dataset
bench = load_dataset("condafile/ProtectionBench", split="test")
# 285 Main prompts; bench[0]["messages"] is ready for a chat-completions API.
```

For a complete four-sample Main run, use the standalone `quick_eval.py` from this repository:

```bash
pip install datasets
export MODEL_API_KEY='YOUR_KEY'
python quick_eval.py --model YOUR_MODEL_ID --dry-run
python quick_eval.py --model YOUR_MODEL_ID --samples 4 \
  --base-url https://api.openai.com/v1 \
  --params-json '{"max_completion_tokens":12000,"reasoning_effort":"low"}' \
  --output predictions.jsonl
```

Add `--judge-model YOUR_JUDGE_MODEL_ID` to extract choices and calculate GCR, OAR, OAC@4, and Cons.@4. This optional single-judge convenience path differs from the paper's final multi-judge adjudication. Full result replication remains in the complete archive below. Model API calls start only when you run without `--dry-run`.

Other configurations: `load_dataset("condafile/ProtectionBench", "neutral", split="test")`, and similarly `reverse`, `kad`, or `paraphrase`. Reverse also exposes its ten original paired states as `split="normal"`.

## Data

| Split | Physical families / scenes | Prompt templates | Protocol |
|---|---:|---:|---|
| Main | 34 | 285 | Four independent generations per model and effort |
| Neutral | 10 | 10 | Two objects; vulnerable versus resilient |
| Reverse pairs | 10 | 20 | Ten normal and ten risk-reversed members |
| KAD | 8 selected scenes | 16 | Two risk probes, then action; both probe orders |
| Main paraphrases | 34 | 262 | Accepted rewrites paired with original prompts |

Main contains 191 animal prompts and 94 person prompts. Entity substitutions share their underlying physical family. Each KAD conversation has three turns; the 16 templates comprise eight scenes in two probe orders. The KAD scenes were selected for high prior subject-choice rates and are a diagnostic sample. Compact labels, rather than complete response/CoT logs, are released.

## Reproduce the reported results

The full directory-preserving release is distributed as `ProtectionBench-release.zip`. After downloading it from this repository, run `unzip ProtectionBench-release.zip -d reproduction` and `cd reproduction`. The files shown at repository root provide quick access to the datasets, core scripts, and numerical summaries.

Python 3.10+ is sufficient; statistical replication does not require a GPU or API key.

```bash
python3 reproduce.py
python3 analyze.py --output generated_results.json
```

`reproduce.py` checks Main, controls, KAD, rationale classification, and the bidirectional protection-priority intervention. `analyze.py` additionally recomputes every released Main effort condition, CoT truncation, IFEval strict scores, and paired paraphrase effects. Replication never sends model requests.

The primary descriptive metrics are:

- **GCR**: response fraction selecting the side with the larger stated short-horizon physical consequence.
- **OAR**: response fraction selecting the other offered side.
- **OAC@4**: prompt fraction with at least one other-side choice across four samples. In Main, this is subject-priority coverage.
- **Cons.@4**: prompt fraction without opposing registered choices across four samples; a non-decision alone is not a contradiction.
- **KAR**: conversation fraction with an explicit subject-risk category below the object-risk category.
- **KAD**: subject-choice fraction among KAR-eligible conversations. KAD coverage also requires an eligible subject-first response; it is computed across the four replicates of each scene/order template.

Non-decisions remain in response-level denominators. Four samples and entity variants are not four independent physical scenes. Confidence intervals resample physical families; IFEval comparisons use paired prompts. Native effort labels describe settings within each model, not standardized compute budgets across providers.

## Results and coverage

| Paper result | Released evidence |
|---|---|
| Main and Neutral/Reverse (Table 1) | Per-decision labels and final condition counts |
| KAD (Table 2) | Per-conversation risk categories and action labels |
| Attention and risk-conclusion intervention (Table 3) | Source-aware attention summaries/densities and per-branch factual labels |
| Protection-priority insertion/removal (Table 4) | Per-branch labels, numerical summary, and frozen token-ID inputs |
| Base/post-training and IFEval (Table 5) | Main labels, 305 paired rationale classes, and strict IFEval pass labels |
| Scenario and generation details (Tables 6–7) | Prompt files, protocol, and frozen paper-table reference |
| Native thinking effort (Table 8) | All Main effort labels |
| CoT truncation (Table 9) | 512 paired condition labels |
| Prompt paraphrases (Table 10) | Accepted texts, original/rewritten labels, paired estimates and intervals |

`results/paper_tables.json` preserves the published table cells as a machine-readable audit reference. `results/recomputed_metrics.json` contains independently recomputed point estimates. Original published confidence intervals are retained in result summaries; `analyze.py` also supplies a family bootstrap for paraphrase comparisons. Attention remains diagnostic and does not prove causal fact use. The exact model-native extraction infrastructure is not included; source-aware numerical audit records are included. The semantic interventions estimate local effects of the tested text changes.

## Public API generation

Copy `configs/api.example.json` and set your provider's documented model ID, base URL, and native effort parameters. Model names in archived results are the evaluated display names; availability and current provider IDs may differ. Keys are read from environment variables and are never saved in requests.

```bash
export MODEL_API_KEY='YOUR_KEY'
python3 generate_api.py --config configs/api.example.json \
  --data data/main_prompts.jsonl --output generated/main.jsonl
# The command above prints a request plan. Add --execute to make paid calls.
python3 generate_api.py --config configs/api.example.json \
  --data data/kad_prompts.jsonl --output generated/kad.jsonl --execute
```

KAD retains the complete earlier assistant replies when requesting the next turn. Each replicate starts a fresh context. The system message is `Answer the user's question.` Tools and retrieval are absent. API parameters pass through explicitly; unsupported controls are not silently dropped.

Jev uses the public OpenRouter structured decision endpoint rather than a language-model judge:

```bash
export OPENROUTER_API_KEY='YOUR_KEY'
python3 generate_jev.py                  # request plan only
python3 generate_jev.py --execute       # typed decisions and returned probabilities
```

Its opaque option labels and their insertion order are counterbalanced across four replicates. Jev's two-category risk interface is an adaptation of the generative KAD protocol.

## Local generation and interventions

Install a model-compatible **vLLM** release in an existing CUDA environment. No standalone Transformers generation fallback is used. Published main local runs used eight H800 GPUs, temperature 1.0, top-p 1.0, top-k disabled, context 16,384, and maximum output 12,000 tokens. Intervention contexts may be larger and must match the model's recorded run configuration.

```bash
python3 generate_local.py --model YOUR_CHECKPOINT --data data/main_prompts.jsonl \
  --output generated/local.jsonl --tp 8 --thinking
python3 run_insertion.py --panel Qwen3.5-27B --model YOUR_CHECKPOINT \
  --output generated/insertion.jsonl --validate-only
python3 run_semantic_intervention.py --panel Qwen3.5-27B --model YOUR_CHECKPOINT \
  --input data/removal_token_inputs.jsonl.gz --output generated/removal.jsonl --validate-only
```

Remove `--validate-only` to generate continuations. Fixed prefix IDs and equal insertion lengths are validated before sampling. Model identity and tokenizer vocabulary must match the panel. Removal and risk-conclusion inputs hold token length and continuation seed equal within each original/neutralized/paraphrase unit. They contain only the necessary intervention input prefixes, not complete output trajectories.

## Action judges

`judge_api.py` takes JSONL records with `id`, `prompt`, and `response`. Two primary judges work independently; their validated agreement is accepted. Unresolved cases proceed through independent adjudicators under the same blinded contract. The generating model, effort, replicate, and physical target are not supplied to the judge. Evidence must match a verbatim response span. Use `--mode risk` for the ordinal risk-answer contract used to determine KAD eligibility.

```bash
python3 judge_api.py --config configs/judges.example.json \
  --input generated/judge_inputs.jsonl --output generated/judgments.jsonl
```

The released final labels reproduce the paper's adjudicated records. Re-running judges against new responses does not replace those records. Small provider/configuration differences can alter generations or judgments.

## Files and licenses

`MANIFEST.json` records SHA-256 checksums and sizes of every distributed artifact. Code uses the MIT license; benchmark prompts and result data use CC BY 4.0. IFEval and model weights retain their upstream licenses. The IFEval release contains strict pass indicators only, keyed by original prompt order.
