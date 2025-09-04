from common_helper_functions import *
from sklearn.model_selection import StratifiedKFold
import os

path = "../../../../../../../../../../../s/project/gene_embedding/funcrvp_embeddings/"

omics = pd.read_csv(path + 'omics_d256.tsv', sep='\t')

string = pd.read_csv(path + 'STRING_d128.tsv', sep='\t')

string_exp = pd.read_csv(path + 'STRING_EXP_d128.tsv', sep='\t')

orthrus_embeddings = pd.read_csv('../preprocessed_data/orthrus_processed.csv')

emogi = pd.read_csv("../preprocessed_data/emogi_preprocessed.tsv", sep='\t')

# Create emogi predictions as a feature dataset (just the prediction column)
emogi_predictions = emogi[['gene_id', 'pred']].copy()

# Define embedding datasets
embedding_datasets = {
    'Omics': omics,
    'STRING': string,
    'STRING_EXP': string_exp,
    'Orthrus': orthrus_embeddings,
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