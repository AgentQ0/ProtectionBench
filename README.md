# ProtectionBench

This work was completed at Xiaohongshu.

Datasets, compact results, and core reproducibility code for **When Protection Overrides Consequences: Understanding Protection Mode in Large Language Model Decision-Making**.

ProtectionBench examines cases where a person or animal faces little immediate physical risk while a competing object faces larger, time-sensitive damage. Its labels identify the larger stated physical consequence; they do not define a universal moral ordering.


![Figure 1: Protection Mode examples and model-level results](figure1.png)

*Figure 1. Protection Mode examples and results for subject-priority coverage and knowledge–action divergence.* [Original PDF](figure1.pdf) · [PowerPoint source](figure1.pptx).

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

## Evaluation results

These are the results reported in the [paper](https://openreview.net/forum?id=I35nG4HR52). Rates are percentages; changes and confidence intervals are in percentage points. The reference side is defined by the stated short-horizon physical consequences, not by a universal moral ordering.

### Main benchmark and matched controls

Main contains **285 prompts from 34 physical families**, with four independent model responses per prompt and setting. Neutral and Reverse each evaluate ten prespecified control prompts with four responses. Main pairs a low-risk living subject with a damage-prone object; Neutral compares a resilient object with a vulnerable object; Reverse changes the state so the living subject faces the larger consequence.

- **GCR:** response fraction choosing the side with the larger stated physical consequence: the object in Main, vulnerable object in Neutral, and subject in Reverse.
- **OAR:** response fraction choosing the other offered side. In Main, this is the subject-choice rate.
- **OAC@4:** prompt fraction with at least one other-side choice in four responses. In Main, this is **Subject-Priority Coverage@4**.
- **Cons.@4:** prompt fraction without opposing registered choices across four responses; non-decisions alone do not create a contradiction.

Non-decisions remain in response-level denominators, so GCR and OAR need not sum to 100%. For Human Raters, @4 aggregates four different people, each rating each prompt once. Native effort labels are interpreted within each model, not as equal computation budgets across models.

| Model | Main effort | Main GCR | Main OAR | Main OAC@4 | Main Cons.@4 | Neutral GCR | Reverse GCR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GPT-5.6 Sol | none | 68.77 | 31.05 | 34.74 | 92.28 | 100.00 | 100.00 |
| DeepSeek V4 Pro | off | 85.53 | 14.12 | 35.79 | 64.91 | 95.00 | 100.00 |
| GLM 5.3 Flash | low | 82.46 | 17.54 | 41.75 | 62.46 | 100.00 | 95.00 |
| Kimi K3 | low | 82.98 | 16.93 | 38.25 | 64.91 | 100.00 | 100.00 |
| MiniMax M3 | disabled | 55.18 | 44.21 | 84.56 | 24.91 | 90.00 | 97.50 |
| Claude Sonnet 5 | reduced | 92.54 | 7.11 | 14.04 | 88.42 | 100.00 | 100.00 |
| Jev 1.13 | fixed | 20.88 | 79.12 | 84.56 | 90.18 | 100.00 | 100.00 |
| Qwen3.5-27B | no-think | 52.89 | 46.84 | 83.16 | 31.23 | 100.00 | 97.50 |
| dots3-note-prev | no-think | 66.49 | 32.02 | 74.04 | 29.82 | 85.00 | 85.00 |
| Human Raters | four raters | 100.00 | 0.00 | 0.00 | 100.00 | 100.00 | 100.00 |

Subject-priority coverage is broad but varies substantially across models. Sol reaches 34.74% Main OAC@4; MiniMax and Qwen3.5-27B reach 84.56% and 83.16%. Human Main OAC@4 is 0%. Separately, human GCR is 100% in all three environments. The control results show that subject-first behavior is not a fixed response to every physical situation.

<details>
<summary>All four metrics for Neutral and Reverse</summary>

#### Neutral

| Model | GCR | OAR | OAC@4 | Cons.@4 |
| --- | ---: | ---: | ---: | ---: |
| GPT-5.6 Sol | 100.00 | 0.00 | 0.00 | 100.00 |
| DeepSeek V4 Pro | 95.00 | 2.50 | 10.00 | 90.00 |
| GLM 5.3 Flash | 100.00 | 0.00 | 0.00 | 100.00 |
| Kimi K3 | 100.00 | 0.00 | 0.00 | 100.00 |
| MiniMax M3 | 90.00 | 7.50 | 30.00 | 70.00 |
| Claude Sonnet 5 | 100.00 | 0.00 | 0.00 | 100.00 |
| Jev 1.13 | 100.00 | 0.00 | 0.00 | 100.00 |
| Qwen3.5-27B | 100.00 | 0.00 | 0.00 | 100.00 |
| dots3-note-prev | 85.00 | 10.00 | 40.00 | 60.00 |
| Human Raters | 100.00 | 0.00 | 0.00 | 100.00 |

#### Reverse

| Model | GCR | OAR | OAC@4 | Cons.@4 |
| --- | ---: | ---: | ---: | ---: |
| GPT-5.6 Sol | 100.00 | 0.00 | 0.00 | 100.00 |
| DeepSeek V4 Pro | 100.00 | 0.00 | 0.00 | 100.00 |
| GLM 5.3 Flash | 95.00 | 5.00 | 10.00 | 90.00 |
| Kimi K3 | 100.00 | 0.00 | 0.00 | 100.00 |
| MiniMax M3 | 97.50 | 0.00 | 0.00 | 100.00 |
| Claude Sonnet 5 | 100.00 | 0.00 | 0.00 | 100.00 |
| Jev 1.13 | 100.00 | 0.00 | 0.00 | 100.00 |
| Qwen3.5-27B | 97.50 | 2.50 | 10.00 | 90.00 |
| dots3-note-prev | 85.00 | 10.00 | 30.00 | 70.00 |
| Human Raters | 100.00 | 0.00 | 0.00 | 100.00 |

</details>

### Knowledge–action divergence (KAD)

The three-turn protocol asks about each side's risk separately and then presents the joint action question. Eight selected scenes with frequent prior subject-first responses, two probe orders, and four replicates yield **64 conversations per model configuration**. This selected diagnostic sample is not an estimate of natural prevalence.

**KAR** is the conversation fraction explicitly assigning lower risk to the subject than the object. **Conditional KAD** is the subject-choice fraction among those eligible conversations. **KAD Coverage@4** is the scene–probe-order fraction with at least one eligible subject-first response among four replicates.

| Model | Effort | KAR | Conditional KAD | KAD Coverage@4 | Cons.@4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| GPT-5.6 Sol | higher | 82.81 | 3.77 | 12.50 | 87.50 |
| GPT-5.6 Sol | lower | 79.69 | 1.96 | 6.25 | 87.50 |
| DeepSeek V4 Pro | higher | 92.19 | 6.78 | 18.75 | 81.25 |
| DeepSeek V4 Pro | lower | 93.75 | 5.00 | 12.50 | 87.50 |
| GLM 5.3 Flash | higher | 67.19 | 6.98 | 12.50 | 68.75 |
| GLM 5.3 Flash | lower | 79.69 | 15.69 | 25.00 | 56.25 |
| Kimi K3 | higher | 81.25 | 3.85 | 12.50 | 75.00 |
| Kimi K3 | lower | 95.31 | 26.23 | 56.25 | 37.50 |
| MiniMax M3 | higher | 87.50 | 10.71 | 31.25 | 62.50 |
| MiniMax M3 | lower | 90.63 | 39.66 | 68.75 | 31.25 |
| Claude Sonnet 5 | default | 100.00 | 18.75 | 31.25 | 68.75 |
| Claude Sonnet 5 | reduced | 84.38 | 38.89 | 62.50 | 50.00 |
| Jev 1.13 | fixed | 39.06 | 8.00 | 12.50 | 93.75 |
| Qwen3.5-27B | on | 100.00 | 57.81 | 75.00 | 62.50 |
| Qwen3.5-27B | off | 96.88 | 22.58 | 37.50 | 68.75 |
| dots3-note-prev | on | 78.13 | 12.00 | 31.25 | 37.50 |
| dots3-note-prev | off | 67.19 | 9.30 | 25.00 | 50.00 |

The same subject-first behavior can accompany different risk assessments. Jev has low KAR (39.06%) but comparatively low conditional KAD (8.00%); many chat models state the risk ordering and still choose the subject. Qwen3.5-27B with thinking reaches 100% KAR and 57.81% conditional KAD. These are observable judgments and actions, not a direct measurement of all internal knowledge use.

### What influences the final action?

We separately intervene on explicit risk conclusions and protection-priority reasoning. In the risk-conclusion experiment, **Original** retains the model's elicited judgment, **Neutralized** removes its risk direction while retaining the scene, and **Paraphrase** preserves its meaning. The table reports object-choice rates; the effect is Original minus Neutralized, with family-clustered 95% confidence intervals.

| Model | Original | Neutralized | Paraphrase | Effect [95% CI] |
| --- | ---: | ---: | ---: | ---: |
| Qwen3.5-27B | 39.84 | 45.83 | 38.80 | −5.99 [−21.13, +4.05] |
| dots3-note-prev | 97.14 | 81.51 | 91.41 | +15.63 [+4.29, +19.10] |

The explicit judgment has a positive incremental action effect in dots3-note-prev, while the Qwen3.5-27B interval includes zero. Attention audits find no stable reduction in attention to prior risk judgments or key physical facts in KAD cases. Attention does not establish causal use, and neutralizing one stated judgment does not measure the total influence of physical knowledge elsewhere in a model.

**Matched insertion.** We hold prefix token IDs fixed, match insertion lengths, and continue generation after a neutral statement, protection-priority statement, or factual reminder. The three condition columns report subject-choice rates. Protection effect is the change in subject choice; factual effect is the change in object choice.

| Model | Matched units | Neutral | Protection | Factual | Protection effect [95% CI] | Factual object-choice effect |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5-4B | 586 | 31.74 | 60.24 | 32.76 | +28.50 [23.48,33.98] | +0.68 |
| Qwen3.5-9B | 748 | 39.04 | 58.16 | 40.11 | +19.12 [15.62,26.79] | +4.41 |
| Qwen3.5-27B | 883 | 57.30 | 85.73 | 52.21 | +28.43 [22.81,41.41] | +5.21 |
| dots3-note-prev | 50 | 18.00 | 56.00 | 14.00 | +38.00 [18.64,60.47] | +4.00 |

**Natural-statement neutralization.** We also neutralize protection-priority statements from naturally generated reasoning, with meaning-preserving paraphrases as controls. These columns report subject-choice rates and the reduction after neutralization.

| Model | Original | Neutralized | Reduction [95% CI] |
| --- | ---: | ---: | ---: |
| Qwen3.5-4B | 62.76 | 51.56 | 11.20 [8.12,16.16] |
| Qwen3.5-9B | 62.24 | 52.34 | 9.90 [3.06,16.76] |
| Qwen3.5-27B | 82.55 | 67.71 | 14.84 [6.25,21.30] |
| dots3-note-prev | 52.27 | 43.18 | 9.09 [-2.08,34.09] |

Protection-priority insertion raises subject choice by **19.12–38.00 points** across the four models. Natural-statement neutralization reduces it by **9.90–14.84 points** across the three Qwen models, with all three intervals above zero; the dots reduction is less certain because its interval crosses zero. The bidirectional results identify a local causal contribution from protection-priority semantic content to action selection.

### Pretraining and post-training

We compare pretrained-only Qwen3.5 Base raw completions with post-trained no-thinking responses. Subject-choice and Coverage@4 changes have confidence intervals crossing zero at both sizes; this does not establish equivalence. Strict IFEval performance improves substantially. On **305 matched subject-first pairs** (129 at 4B and 176 at 9B), the visible rationale shifts toward general protection principles.

| Qwen3.5 size | Measure | Base | Post-trained | Change 95% CI |
| --- | ---: | ---: | ---: | ---: |
| 4B | Main OAR | 28.25 | 31.58 | [-3.60,+10.22] |
| 4B | Main OAC@4 | 71.23 | 69.82 | [-12.59,+9.30] |
| 4B | Strict IFEval | 38.26 | 83.18 | [+40.30,+49.54] |
| 4B | General protection principle | 32.56 | 79.07 | [+37.21,+55.15] |
| 4B | Additional subject-risk argument | 38.76 | 14.73 | [-34.36,-11.93] |
| 9B | Main OAR | 33.33 | 37.02 | [-2.36,+9.24] |
| 9B | Main OAC@4 | 75.44 | 73.68 | [-11.59,+6.71] |
| 9B | Strict IFEval | 44.18 | 84.47 | [+35.67,+44.92] |
| 9B | General protection principle | 51.70 | 86.36 | [+25.74,+41.12] |
| 9B | Additional subject-risk argument | 33.52 | 9.66 | [-30.00,-16.11] |

Protection Mode is observable in pretrained-only outputs. Post-trained models give more decisive responses and more standardized protection-priority explanations, without a reliable reduction in aggregate subject choice in this comparison.

<details>
<summary>Complete matched rationale classification</summary>

The percentages below describe the dominant rationale among paired decisions in which both versions choose the subject; they are not frequencies among all responses.

| Size | Dominant rationale | Base | Post-trained | Change 95% CI |
| --- | ---: | ---: | ---: | ---: |
| 4B | General protection principle | 32.56 | 79.07 | [+37.21,+55.15] |
| 4B | Additional subject-risk argument | 38.76 | 14.73 | [-34.36,-11.93] |
| 4B | Procedural/order argument | 7.75 | 3.10 | [-10.29,+0.87] |
| 4B | Local scene-fact argument | 5.43 | 1.55 | [-8.39,0.00] |
| 4B | Other/unclear argument | 15.50 | 1.55 | [-21.74,-7.75] |
| 9B | General protection principle | 51.70 | 86.36 | [+25.74,+41.12] |
| 9B | Additional subject-risk argument | 33.52 | 9.66 | [-30.00,-16.11] |
| 9B | Procedural/order argument | 3.98 | 0.57 | [-5.80,-1.12] |
| 9B | Local scene-fact argument | 4.55 | 2.84 | [-6.25,+2.35] |
| 9B | Other/unclear argument | 6.25 | 0.57 | [-12.16,-1.40] |

</details>

### Additional reported evaluations

<details>
<summary>All evaluated Main thinking-effort settings</summary>

The same four Main metrics are used at each native effort setting. These response distributions vary by model and need not change monotonically with the setting label.

| Model | Effort | GCR | OAR | OAC@4 | Cons.@4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| GPT-5.6 Sol | none | 68.77 | 31.05 | 34.74 | 92.28 |
| GPT-5.6 Sol | low | 93.51 | 6.40 | 13.33 | 87.37 |
| GPT-5.6 Sol | medium | 96.32 | 3.68 | 10.18 | 89.82 |
| GPT-5.6 Sol | high | 97.37 | 2.46 | 7.37 | 92.63 |
| GPT-5.6 Sol | xhigh | 96.84 | 3.16 | 10.88 | 89.12 |
| GPT-5.6 Sol | max | 97.46 | 2.54 | 7.72 | 92.28 |
| DeepSeek V4 Pro | off | 85.53 | 14.12 | 35.79 | 64.91 |
| DeepSeek V4 Pro | low | 91.40 | 8.60 | 14.74 | 88.77 |
| DeepSeek V4 Pro | high | 93.33 | 6.58 | 12.28 | 89.82 |
| DeepSeek V4 Pro | max | 92.63 | 7.19 | 12.63 | 89.82 |
| GLM 5.3 Flash | low | 82.46 | 17.54 | 41.75 | 62.46 |
| GLM 5.3 Flash | high | 87.89 | 12.02 | 34.39 | 65.96 |
| GLM 5.3 Flash | max | 93.68 | 5.96 | 18.25 | 81.75 |
| Kimi K3 | low | 82.98 | 16.93 | 38.25 | 64.91 |
| Kimi K3 | high | 90.96 | 9.04 | 20.70 | 80.70 |
| Kimi K3 | max | 94.56 | 5.26 | 15.79 | 84.56 |
| MiniMax M3 | disabled | 55.18 | 44.21 | 84.56 | 24.91 |
| MiniMax M3 | low | 52.89 | 46.32 | 85.26 | 27.37 |
| MiniMax M3 | adaptive | 94.74 | 5.09 | 16.14 | 83.86 |
| Claude Sonnet 5 | reduced | 92.54 | 7.11 | 14.04 | 88.42 |
| Claude Sonnet 5 | default | 91.58 | 7.54 | 16.84 | 85.26 |
| Qwen3.5-27B | no-think | 52.89 | 46.84 | 83.16 | 31.23 |
| Qwen3.5-27B | think | 72.37 | 27.54 | 53.33 | 54.04 |
| dots3-note-prev | no-think | 66.49 | 32.02 | 74.04 | 29.82 |
| dots3-note-prev | think | 81.75 | 18.07 | 41.05 | 61.40 |

</details>

<details>
<summary>Reasoning-prefix truncation</summary>

Natural reasoning is compared with continuations after 256-, 128-, and 64-token prefixes. The same four choice metrics are reported. These selected diagnostic scenes differ from the full Main benchmark. Effects are model-dependent and not uniformly monotonic.

| Model | Prefix cutoff | GCR | OAR | OAC@4 | Cons.@4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5-4B | Natural | 56.25 | 43.75 | 75.00 | 37.50 |
| Qwen3.5-4B | 256 | 56.25 | 37.50 | 62.50 | 37.50 |
| Qwen3.5-4B | 128 | 50.00 | 37.50 | 75.00 | 25.00 |
| Qwen3.5-4B | 64 | 53.12 | 34.38 | 87.50 | 25.00 |
| Qwen3.5-9B | Natural | 59.38 | 40.62 | 62.50 | 62.50 |
| Qwen3.5-9B | 256 | 68.75 | 25.00 | 37.50 | 75.00 |
| Qwen3.5-9B | 128 | 59.38 | 28.12 | 50.00 | 75.00 |
| Qwen3.5-9B | 64 | 56.25 | 40.62 | 62.50 | 62.50 |
| Qwen3.5-27B | Natural | 34.38 | 65.62 | 87.50 | 50.00 |
| Qwen3.5-27B | 256 | 71.88 | 28.12 | 37.50 | 62.50 |
| Qwen3.5-27B | 128 | 62.50 | 37.50 | 37.50 | 100.00 |
| Qwen3.5-27B | 64 | 68.75 | 28.12 | 50.00 | 62.50 |
| dots3-note-prev | Natural | 65.62 | 34.38 | 62.50 | 37.50 |
| dots3-note-prev | 256 | 71.88 | 18.75 | 62.50 | 37.50 |
| dots3-note-prev | 128 | 65.62 | 34.38 | 75.00 | 25.00 |
| dots3-note-prev | 64 | 75.00 | 25.00 | 62.50 | 37.50 |

</details>

<details>
<summary>Prompt-paraphrase robustness</summary>

The comparison uses the **262 accepted Main paraphrases and their matched originals**, with four samples each. Every reported change interval includes zero; no statistically detectable wording effect is observed in these comparisons, which does not prove exact equivalence.

| Model | Metric | Original | Paraphrase | Change [95% CI] |
| --- | ---: | ---: | ---: | ---: |
| DeepSeek V4 Pro | OAR | 14.69 | 11.74 | -2.96 [-6.30,+0.28] |
| DeepSeek V4 Pro | OAC@4 | 36.64 | 30.92 | -5.73 [-11.89,+0.84] |
| GLM 5.3 Flash | OAR | 17.37 | 19.47 | +2.10 [-4.82,+9.06] |
| GLM 5.3 Flash | OAC@4 | 40.08 | 40.08 | 0.00 [-8.55,+8.56] |
| Kimi K3 | OAR | 16.70 | 15.65 | -1.05 [-5.64,+3.78] |
| Kimi K3 | OAC@4 | 38.55 | 33.21 | -5.34 [-12.90,+2.53] |
| Jev 1.13 | OAR | 78.63 | 73.00 | -5.63 [-22.27,+9.78] |
| Jev 1.13 | OAC@4 | 84.35 | 81.30 | -3.05 [-19.92,+12.89] |

</details>

The [published table cells](https://github.com/AgentQ0/ProtectionBench/blob/main/paper_tables.json) and [recomputed metrics](https://github.com/AgentQ0/ProtectionBench/blob/main/recomputed_metrics.json) accompany the full numerical summaries and reproduction code in this repository.

### Released evidence

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

Unless explicitly stated otherwise, this repository does not include or distribute any third-party model weights. Users who independently download or enable third-party models or services must also comply with the applicable upstream project, model-weight, and data license terms. Use of third-party API services is subject to the applicable service providers' terms and conditions.
