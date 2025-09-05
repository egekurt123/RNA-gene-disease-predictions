from common_helper_functions import *
from sklearn.model_selection import StratifiedKFold
import os

# Define embedding datasets
embedding_datasets = {
    'Omics': omics,
    'STRING': string,
    'STRING_EXP': string_exp,
    'Orthrus': orthrus,
    'Emogi_Predictions': emogi_predictions
}

# Compare embeddings using emogi labels
print("Comparing embeddings using Emogi labels as ground truth (5-Fold Cross-Validation)")
print("="*80)

# PR Curves comparison
print("\nGenerating PR curves...")
results = compare_embeddings_pr_curves(embedding_datasets, emogi)
print("\nPR Curves Results summary (Mean ± Std):")
print(results.pivot(index='Embedding', columns='Model', values='auPRC'))

# auPRC bar plots comparison
print("\n" + "="*80)
print("Generating auPRC bar plots...")
auprc_results = compare_embeddings_auprc_barplots(embedding_datasets, emogi)
print("\nauPRC Bar Plot Results summary (Mean ± Std):")
print(auprc_results.pivot(index='Embedding', columns='Model', values='auPRC'))