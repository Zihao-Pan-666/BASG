# BASG: Behavior-Augmented Semantic Generalization for Zero-Shot Cross-Domain Sequential Recommendation

Official code repository for the paper:

> **A Behavior-Augmented Semantic Generalization Framework for Zero-Shot Cross-Domain Sequential Recommendation** *(submitted to Big Data and Cognitive Computing)*

BASG is a backbone-compatible framework for zero-shot cross-domain sequential recommendation (ZCDSR). It augments a frozen, source-trained semantic backbone with a parameter-decoupled **Behavior Expert** and combines their scores with source-calibrated **Static Fusion**:

```
q_BASG(H, i) = q̃_sem(H, i) + β⋆ · q̃_beh(H, i)
```

- **Semantic backbone** — a frozen source-selected sequential recommender exposed only through a candidate-score interface. Three backbones are supported: SAGERec, BERT4Rec-Llama (SEM), and UniSRec.
- **Behavior Expert** — a separately trained dual-channel scorer:
  - *Temporal channel*: query-local log-gap and relative-position features computed from the timestamps of the current prefix only.
  - *Interaction-property channel*: a popularity-related scalar proxy and a co-occurrence-informed vector proxy, predicted from frozen item-text embeddings by two source-fitted mappers (shared across backbones and seeds).
- **Static Fusion** — query-wise standardization of the two score vectors followed by a fixed scalar combination; β⋆ is selected exclusively on a user-disjoint source validation partition and never touches target data.

All trainable BASG components receive interaction supervision only from the source domain (Amazon Movies & TV). Target domains (Amazon CDs & Vinyl, Amazon Industrial & Scientific) are used strictly for final zero-shot evaluation.

---

## Repository structure

```
BASG/
├── basg/                      # Core library
│   ├── models/                #   Behavior Expert, BERT4Rec-family backbones, checkpoint loading
│   ├── features/              #   Temporal features, popularity/behavior mappers, embedding loading
│   ├── data/                  #   Interaction loading, splits, prefix datasets
│   ├── training/              #   Behavior Expert and backbone trainers
│   ├── evaluation/            #   Sampled-ranking evaluation, static-fusion tuning, metrics
│   ├── utils/                 #   Runtime logging, seeding, device, source partitioning
│   └── baselines/             #   UniSRec baseline
├── scripts/
│   ├── basg/                  # BASG pipeline: mappers → expert → fusion → zero-shot eval
│   ├── baselines/             # Baseline training/evaluation (SEM, RecG, SAGE, UniSRec)
│   └── data/                  # Protocol manifest writer
├── configs/
│   ├── basg/                  # expert.yaml (SAGE), expert_sem.yaml, expert_unisrec.yaml
│   └── baselines/             # sem.yaml, recg.yaml, sage.yaml, unisrec.yaml
└── docs/
    └── DATA.md                # Data and artifact acquisition guide
```

---

## Quick start

### 0. Environment and data

```bash
pip install -r requirements.txt
```

PyTorch ≥ 2.0 with CUDA is recommended (the reported runs use PyTorch 2.6.0 + CUDA 12.6 on an NVIDIA RTX 5880 Ada). Place the processed interactions, the LLM2Vec item embeddings, and (optionally) pre-trained checkpoints as described in [docs/DATA.md](docs/DATA.md).

### 1. Train the semantic backbone (source domain only)

```bash
# SAGE backbone (used by configs/basg/expert.yaml)
python scripts/baselines/train_family.py --config configs/baselines/sage.yaml --seed 2026
# SEM backbone (BERT4Rec-Llama; used by configs/basg/expert_sem.yaml)
python scripts/baselines/train_family.py --config configs/baselines/sem.yaml --seed 2026
# UniSRec backbone (used by configs/basg/expert_unisrec.yaml)
python scripts/baselines/train_unisrec.py --config configs/baselines/unisrec.yaml --seed 2026
```

Each run writes a checkpoint under `artifacts/checkpoints/` and a source-validation summary under `results/mainline/`.

### 2. Build the source-fitted property mappers (run once, shared by all backbones and seeds)

```bash
# Popularity proxy: per-week percentile trajectory + LLM→pop predictor
python scripts/basg/build_popularity.py --source amazon_movies_and_tv

# Co-occurrence proxy: source co-occurrence neighborhood + semantic filter + PCA + LLM→z mapper
python scripts/basg/build_behavior.py --source amazon_movies_and_tv
```

### 3. Train the Behavior Expert (per seed)

```bash
python scripts/basg/train_expert.py --config configs/basg/expert.yaml --seed 2026
```

### 4. Select the static fusion coefficient β⋆ on the source fusion partition (per seed, per backbone)

```bash
python scripts/basg/tune_fusion.py --config configs/basg/expert.yaml --seed 2026
```

### 5. Zero-shot evaluation on both target domains

```bash
python scripts/basg/eval_zero_shot.py --config configs/basg/expert.yaml --seed 2026 \
  --targets "amazon_cds_and_vinyl,amazon_industrial_and_scientific"
```

For the SEM- and UniSRec-based BASG variants, repeat steps 4–5 with `configs/basg/expert_sem.yaml` and `configs/basg/expert_unisrec.yaml` (the Behavior Expert checkpoints from step 3 are reused; only β⋆ is re-selected per backbone), or use the convenience wrappers:

```bash
python scripts/basg/eval_sem_basg.py          # tune + eval for SEM-BASG, seeds 2026/2027/2028
python scripts/basg/eval_unisrec_basg.py      # tune + eval for UniSRec-BASG, seeds 2026/2027/2028
```

Paper results aggregate seeds **2026, 2027, 2028** — run steps 3–5 once per seed (and step 1 per seed for each backbone).

---

## Reproducing the paper results

The main table (3 seeds, mean ± sample std) is reproduced by the commands above:

| Model | ACV NDCG@10 | AIS NDCG@10 |
|---|---|---|
| BERT4Rec-Llama (SEM) | 0.2361 | 0.1318 |
| LLM-RecG | 0.2622 | 0.1035 |
| SAGERec | 0.2656 | 0.1136 |
| UniSRec | 0.2521 | 0.1420 |
| **SAGERec-BASG** | **0.2763** | 0.1231 |
| **SEM-BASG** | 0.2522 | 0.1361 |
| **UniSRec-BASG** | 0.2689 | **0.1507** |

Static Fusion yields positive NDCG@10 gains in all 18 backbone × domain × seed evaluations (three backbones, two target domains, three seeds).

The `eval_zero_shot.py` CSV reports `Backbone`, `BehaviorExpert`, `StaticFusion`, and `OracleDiagnostic` (upper-bound diagnostic) rows per target domain; Recall@10, NDCG@10, and MRR@10 are computed for each.

### RQ2 ablations

Inference-time ablations of the two Behavior Expert channels are built into the evaluator:

```bash
# w/o temporal channel (gap + recency zeroed; popularity/co-occurrence retained)
python scripts/basg/eval_zero_shot.py --config configs/basg/expert.yaml --seed 2026 \
  --ablation "w/o temporal-features"

# w/o interaction-property channel (popularity/co-occurrence zeroed; temporal retained)
python scripts/basg/eval_zero_shot.py --config configs/basg/expert.yaml --seed 2026 \
  --ablation "w/o item-properties"
```

---

## Reusing artifacts from the original experimental project

If you already produced artifacts with the internal (pre-release) project layout, you can reuse them by copying the files to the names expected by this repository:

| Original artifact | Copy to |
|---|---|
| `artifacts/checkpoints/bert4rec_family/bert4rec_{sem,recg_a01,sage}_matched_amazon_movies_and_tv_seedYYYY.pt` | `artifacts/checkpoints/bert4rec_family/bert4rec_{sem,recg_a01,sage}_amazon_movies_and_tv_seedYYYY.pt` |
| `artifacts/checkpoints/temporal_expert_alphafree/temporal_expert_alphafree_amazon_movies_and_tv_seedYYYY.pt` | `artifacts/checkpoints/behavior_expert/behavior_expert_amazon_movies_and_tv_seedYYYY.pt` |
| `artifacts/popularity/amazon_movies_and_tv_pop_predictor.pt` | `artifacts/popularity/amazon_movies_and_tv_pop_predictor.pt` (unchanged) |
| `artifacts/behavior/source_behavior_mapper.pt` | `artifacts/behavior/source_behavior_mapper.pt` (unchanged) |
| `artifacts/checkpoints/unisrec/unisrec_sem_amazon_movies_and_tv_seedYYYY.pt` | unchanged |

With these in place, you can skip directly to step 4 (fusion tuning + evaluation).

---

## Scope notes

Kept intentionally small for readability and quick reproduction. The following items from the full experimental project are **not** included here and are available from the authors upon request:

- **RecFormer baseline** — text-centric Longformer pipeline with its own 768-dim item encoder.
- **RQ4 diagnostic router** (RouterFusion, 18-feature gate MLP) — a source-trained fusion baseline reported as a negative result in the paper; it is not part of the deployed BASG pipeline.
- **RQ5 complementarity/query-conditioning analysis** scripts.
- The optional `dynamic_role` distillation mode inside the backbone trainer (dormant; disabled by default in all provided configs).

---

## Citation

```bibtex
@article{basg2026,
  title   = {A Behavior-Augmented Semantic Generalization Framework for Zero-Shot Cross-Domain Sequential Recommendation},
  journal = {Big Data and Cognitive Computing},
  year    = {2026},
  note    = {under review}
}
```

## License

This repository is released under the [MIT License](LICENSE). The experiments use the public Amazon Reviews 2023 dataset; processed files and pre-computed embeddings are distributed separately (see [docs/DATA.md](docs/DATA.md)).
