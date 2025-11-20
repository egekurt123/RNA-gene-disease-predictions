## RNA / Gene Disease Predictions

This repository evaluates a variety of published and DNA/RNA embeddings on cancer and rare disease classification tasks. The code compares embeddings (including Omics, STRING, PoPS, Orthrus, ESM2, Orthrus RNA embeddings, and others) using tree models.
---

### Requirements

- Python 3.9+

---

### Data inputs

- **Embedding files** live in the shared path referenced inside `Cancer Predictions/common_helper_functions.py` (`path = .../funcrvp_embeddings/`).  Update this path if you relocate embeddings locally.
- **Preprocessed gene sets** (`preprocessed_data/*.csv` or `.tsv`) contain curated cohorts for mitochondria, neuromuscular, epilepsy, ophthalmology, neurology, and inborn immunity.

- Verify that every embedding DataFrame has numeric-only feature columns; helper functions drop non-numeric columns automatically.
- Many helper functions expect embeddings to share a common `gene_id` column; use `filter_to_common_genes` before training/evaluating.
- Ensure every dataset you load includes a `gene_id` column and, when used as labels, a `label` column with binary targets.

---

### Core workflows

1. **Cancer (Emogi) benchmarks**
	- `Cancer Predictions/compare_embeddings_all_columns/compare_embeddings_PR.py` or `compare_embeddings_AuRPC.py` to produce PR curves or AuPRC summaries.

2. **Other Disease benchmarks**
	- Scripts in `Disease Predictions/` (e.g., `compare_combination&normal_pca.py`, `datasets_combinations_compare.py`) reuse the same helper module to evaluate diseases individually.

---

### Typical usage pattern

1. **Load embeddings + labels** into DataFrames (e.g., via `pd.read_csv`).
2. **Align genes** using `filter_to_common_genes(embedding_dict, target_df)`.
4. **Train + evaluate** with `compare_embeddings_pr_curves`, `compare_embeddings_auprc_barplots`, or the XGBoost-only scatter/lollipop alternatives.
5. **Inspect plots** under `plots/PR_curves` and `plots/AuPRC` or inside notebook outputs.

All helper functions reside in `Cancer Predictions/common_helper_functions.py` (mirrored version under `Disease Predictions/` for disease scripts).
