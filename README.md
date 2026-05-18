# ShikshaGemma: A Hindi-Native Science & Math Tutor for India

*A fine-tuned Gemma 4 E4B model that runs fully offline on-device via Google AI Edge and LiteRT.*

ShikshaGemma is a single Hindi-native tutor for science and mathematics, built to support students from Class 6 through MSc. It gives step-by-step explanations in Hindi across school science, JEE and NEET preparation, undergraduate science, and postgraduate-level physics, chemistry, mathematics, and biology.


## 1. Why Gemma 4 E4B

1. **On-device deployment envelope.** The single hardest constraint of the project is that the model must run on a low-end Android device with no internet. Gemma 4 E4B, 4-bit quantised, fits this envelope. Larger open models do not.
2. **Hindi prior.** Gemma 4 E4B handles Hindi prompts and Hindi reasoning out-of-the-box better than other open ~4B-class models we screened. Fine-tuning sharpens an existing capability instead of teaching a new language.
3. **First-party edge runtime.** Gemma 4 has first-party support in the Google AI Edge / LiteRT stack, which is the deployment target. There is no extra port effort for inference.
4. **Vision and tool-use ready.** The vision tower is preserved through fine-tuning. The chat template natively supports function calling. Diagram support (JEE physics, NEET biology figures) and a symbolic-calculator tool are direct extensions, not rewrites.

## 2. Training

The fine-tuning is staged into two phases, with explicit goals per phase, using Unsloth for memory-efficient 4-bit LoRA SFT.

### 2.1 Stage 1: English reasoning foundation

**Goal:** build deep scientific and mathematical reasoning capability in the base model, in the language where high-quality CoT material is most abundant: English.

**Data:** 120,000 samples total.
- **100,000 samples** from established high-quality public English chain-of-thought / reasoning datasets in math and the sciences.
- **20,000 synthetic samples** generated from classical school-and-undergraduate physics, chemistry, mathematics, and biology textbooks, with explicit step-by-step CoT solutions.

**Compute:** ~60 GPU-hours, 4-bit LoRA SFT via Unsloth.

**Rationale:** transferring a reasoning capability across languages is a substantially easier learning problem than learning reasoning from scratch in a low-resource language. Stage 1 builds the reasoning prior; Stage 2 specialises the language.

### 2.2 Stage 2: Hindi specialisation

**Goal:** teach the Stage-1-reasoning-strong checkpoint to speak and think in Hindi across school, entrance, undergrad, and postgrad science.

**Data:** 40,000 samples total.
- **36,000 synthetic Hindi samples** generated from:
  - **NCERT** Class 6–12 science and math (Hindi medium).
  - **IGNOU** undergraduate and postgraduate Hindi science material.
  - **VMOU** (Vardhman Mahaveer Open University) Hindi BSc / MSc material.
  Each sample is a Hindi problem with a Hindi step-by-step CoT solution and a strict-format final answer.
- **4,000 samples** from **JEE Main and NEET previous-year papers**, translated and verified into Hindi, with step-by-step Hindi solutions.

**Compute:** ~5 GPU-hours, 4-bit LoRA SFT via Unsloth on top of the Stage-1 checkpoint.

**Total fine-tuning compute:** ~65 GPU-hours end-to-end on a single 24 GB consumer-class GPU (L4).

---

## 3. Deployment

Deployment uses **Google AI Edge** with the model converted to **LiteRT** format.

- **Conversion:** the merged Stage-2 checkpoint is converted to LiteRT, quantised to 4-bit, and packaged for on-device execution.
- **Runtime:** Google AI Edge runtime on Android. The model executes locally on the device CPU / NPU. No network roundtrip.
- **Optional path:** the same merged checkpoint can also be served via `transformers` + 4-bit BNB or via vLLM for classroom / lab deployments with a shared GPU; this is what we use for benchmarking.

---

## 4. Evaluation

We evaluate on two tracks: stock peer-comparable benchmarks (Track A) and custom Hindi JEE benchmarks we built ourselves (Track B). Each source dataset is broader than the slice we evaluated on; the tables below state explicitly what each source contains and which subset we used.

### 5.1 Track A: stock benchmarks (`lm-evaluation-harness`)

| Benchmark | What the source dataset contains | Subset we evaluated on | N |
|---|---|---|---|
| **GPQA Diamond** | GPQA is 448 expert-written graduate-level MCQs in biology, physics, chemistry. Diamond is the hardest 198 (the subset where domain-expert non-specialist validators score under ~30%). | Full Diamond split, zero-shot, English. | 198 |
| **MILU Hindi STEM** | MILU (AI4Bharat) is a 70k+ Hindi MCQ benchmark across 11 domains: humanities, social sciences, sciences, mathematics, engineering, medicine, law, business, religion, etc. | STEM-relevant subjects only: physics, chemistry, mathematics, biology, computer science. | 2,624 |
| **IndicMMLU-Pro Hindi STEM** | IndicMMLU-Pro is the Hindi translation of MMLU-Pro, a hardened MMLU rebuild with 14 broad categories and mixed 4-option / 10-option items. | STEM categories only: math, physics, chemistry, biology, engineering, computer science. | 4,499 |

### 5.2 Track B: custom Hindi JEE benchmarks (`inspect-ai`)

| Benchmark | What the source paper contains | What we used | N |
|---|---|---|---|
| **JEE Main 2026 (Hindi)** | January 24, 2026 morning-shift Hindi paper. 75 questions, 25 each in physics, chemistry, mathematics. Question types: single_correct + numerical. | Full text-only set, after hand-cleaning Symbol-font PUA encoding artefacts, collapsed match-list options, and mis-parsed numerical answers. 1 unrecoverable item dropped. | 74 |
| **JEE Advanced 2025 (Hindi, Papers 1 + 2)** | Official Hindi medium papers, ~108 questions across physics, chemistry, mathematics. Question types: single_correct, multi_correct (multiple options correct simultaneously), numerical. | Text-only items. 38 diagram-dependent questions dropped for this first-pass text-only run. | 70 |

Each question carries `question_type ∈ {single_correct, multi_correct, numerical}`. The scorer dispatches per type: exact letter match for single, sorted-letter-set equality for multi, 1% relative tolerance for numerical with `a/b` fraction support and LaTeX normalisation (`\frac{}{}`, `\boxed{}` preprocessing). Scores break down per subject and per question type.

---

## 6. Results

Both tracks were run twice with the same scorer: once on stock Gemma 4 E4B (baseline), once on the Stage 2 ShikshaGemma checkpoint.

### 7.1 Track A: stock benchmarks

| Benchmark | N | Base Gemma 4 E4B | ShikshaGemma (Stage 2) | Δ (pp) |
|---|---|---|---|---|
| GPQA Diamond (zero-shot, English) | 198 | 24.24% | 26.26% | +2.02 |
| MILU Hindi STEM | 2,624 | 41.01% | 36.97% | −4.04 |
| IndicMMLU-Pro Hindi STEM | 4,499 | 14.80% | 13.34% | −1.46 |

### 7.2 Track B: custom Hindi JEE benchmarks

| Benchmark | N | Base Gemma 4 E4B | ShikshaGemma (Stage 2) | Δ (pp) |
|---|---|---|---|---|
| JEE Main 2026 (Hindi) | 74 | 44.59% | 25.68% | −18.92 |
| JEE Advanced 2025 (Hindi, text-only) | 70 | 40.00% | 11.43% | −28.57 |

Per-subject and per-question-type breakdowns (Track B):

**JEE Main 2026 (Hindi):**

| Cut | Base | Stage 2 |
|---|---|---|
| physics | 51.85% | 25.93% |
| chemistry | 43.48% | 26.09% |
| mathematics | 38.10% | 28.57% |
| single_correct | 47.46% | 30.51% |
| numerical | 33.33% | 6.67% |

**JEE Advanced 2025 (Hindi, text-only):**

| Cut | Base | Stage 2 |
|---|---|---|
| physics | 28.57% | 14.29% |
| chemistry | 41.67% | 8.33% |
| mathematics | 43.75% | 12.50% |
| single_correct | 45.45% | 22.73% |
| multi_correct | 15.38% | 7.69% |
| numerical | 45.71% | 5.71% |

### 7.3 Reading of the numbers

The Stage 2 ShikshaGemma checkpoint **regresses against base Gemma 4 E4B on every Hindi benchmark we report.** The regression is small on the multilingual MCQ benchmarks (MILU, IndicMMLU-Pro Hindi STEM) and large on the JEE-grade benchmarks. GPQA in English shifts slightly upward, consistent with Stage 1 having had a positive effect on English reasoning before Stage 2.

Stage 2 is not production-ready yet, needs more debugging and fixing.

## 8. Repository layout

```
sikshagemma/
├── src/                           #training scripts, eval scripts
└── custom_benchmark/
    ├── jee_main_2026_hindi_benchmark.jsonl       (74 rows)
    └── jee_advanced_2025_hindi_benchmark.jsonl   (70 rows)
└── custom_tasks/                  # lm eval tasks
└── inspect_tasks/                 # inspect-ai tasks
└── utils/                         # util scripts
```
