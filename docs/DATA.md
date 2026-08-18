# Data and Artifacts

This repository ships code and configuration only. Processed interaction files, pre-computed item embeddings, and trained checkpoints must be obtained or regenerated as described below. For the exact files used in the paper, please contact the corresponding author.

## Expected layout

```
BASG/
├── data/
│   ├── processed/
│   │   ├── amazon_movies_and_tv.csv               # source domain interactions + item text
│   │   ├── amazon_cds_and_vinyl.csv               # target domain (ACV)
│   │   └── amazon_industrial_and_scientific.csv   # target domain (AIS)
│   └── semantic_embeddings/
│       ├── amazon_movies_and_tv_embedding_llama.parquet
│       ├── amazon_cds_and_vinyl_embedding_llama.parquet
│       └── amazon_industrial_and_scientific_embedding_llama.parquet
└── artifacts/
    ├── checkpoints/
    │   ├── bert4rec_family/    # SEM / RecG / SAGE backbone checkpoints (train_family.py)
    │   ├── unisrec/            # UniSRec backbone checkpoints (train_unisrec.py)
    │   ├── behavior_expert/    # Behavior Expert checkpoints (train_expert.py)
    │   └── static_fusion/      # β⋆ selection payloads (tune_fusion.py)
    ├── behavior/               # source_behavior_mapper.pt, source_behavior_z64.pt
    └── popularity/             # {source}_pop_table.pt, {source}_pop_predictor.pt
```

## Processed interactions

Each processed CSV contains the chronological review interactions of one domain:

```
UserId,ItemId,Timestamp,title,description,features
```

- `UserId` / `ItemId` — raw identifiers (the loader builds a stable item map on first appearance).
- `Timestamp` — millisecond Unix timestamp.
- `title` / `description` / `features` — item text metadata used for semantic encoding.

### Obtaining the files

The raw corpus is the public **Amazon Reviews 2023** dataset:

- <https://amazon-reviews-2023.github.io/>
- Hugging Face mirror: `McAuley-Lab/Amazon-Reviews-2023`

The Amazon Movies & TV and Amazon CDs & Vinyl files follow the processed releases used in the SAGERec experimental pipeline, and Amazon Industrial & Scientific follows the processed release accompanying LLM-RecG. To rebuild them yourself:

1. Download the review and metadata files of each domain and place them under `data/{domain}/` as `{domain}.csv` and `meta_{domain}.jsonl`.
2. Run the preprocessing script from the SAGERec repository
   (<https://github.com/Zihao-Pan-666/SAGERec>, `scripts/preprocess_amazon.py`),
   which produces `data/{domain}/processed_data.csv`; copy it to
   `data/processed/{domain}.csv`.

Downstream, the BASG loader sorts interactions chronologically per user and discards sequences with fewer than 3 interactions (`min_len: 3` in the configs).

## Semantic item embeddings

One Parquet file per domain, keyed by raw item id (`RawItemId`) with a 4096-dimensional vector column (`item_text_embedding`). The embeddings are produced offline with **LLM2Vec** (`McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp`) and frozen throughout all experiments; the language model is never invoked online.

To regenerate them, run the encoding script from the SAGERec repository (`scripts/encode_items.py`) with the domain names above, or contact the authors for the exact files.

## Trained artifacts

All artifacts are produced by the scripts in this repository:

| Artifact | Produced by | Size (approx.) |
|---|---|---|
| SEM / RecG / SAGE backbone checkpoints (`bert4rec_{sem,recg_a01,sage}_amazon_movies_and_tv_seedYYYY.pt`) | `scripts/baselines/train_family.py` | ~6 MB each |
| UniSRec backbone checkpoints (`unisrec_sem_amazon_movies_and_tv_seedYYYY.pt`) | `scripts/baselines/train_unisrec.py` | ~17 MB each |
| Behavior Expert checkpoints (`behavior_expert_amazon_movies_and_tv_seedYYYY.pt`) | `scripts/basg/train_expert.py` | ~22 MB each |
| Fusion payloads (`{backbone}_basg_static_fusion_amazon_movies_and_tv_seedYYYY.pt`) | `scripts/basg/tune_fusion.py` | <1 MB each |
| Popularity table + predictor (`{source}_pop_table.pt`, `{source}_pop_predictor.pt`) | `scripts/basg/build_popularity.py` | ~4 MB predictor |
| Co-occurrence mapper (`source_behavior_mapper.pt`, `source_behavior_z64.pt`) | `scripts/basg/build_behavior.py` | ~10 MB mapper |

The property mappers are fitted once on source data and reused across all backbones and seeds. If you have the artifacts from the original experimental project, see the copy-and-rename table in the main [README](../README.md).

## Compute requirements

- Behavior Expert training: 1.4–2.6 hours per run on an NVIDIA RTX 5880 Ada GPU (reported in the paper).
- Backbone training and mapper construction run on a single GPU of comparable class; CPU-only execution is possible but slow.
- The large consumers of RAM are the co-occurrence construction in `build_behavior.py` (sparse item–item matrix over ~85k source items) and loading the 4096-dim embedding tables.

## Zero-shot protocol compliance

All pipeline scripts load the source domain for training/selection and touch target domains only at final evaluation. The three-way user-disjoint source validation split (34% checkpoint selection / 33% static-β selection / 33% reserved) is deterministic per seed via `basg/utils/source_partition.py`. A protocol manifest can be written with:

```bash
python scripts/data/protocol_manifest.py --config configs/basg/expert.yaml
```
