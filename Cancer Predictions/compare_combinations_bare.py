from common_helper_functions import *

# Define combinated datasets
orthrus_omics = orthrus.merge(omics, on='gene_id', how='inner')
orthrus_string = orthrus.merge(string, on='gene_id', how='inner')
orthrus_string_exp = orthrus.merge(string_exp, on='gene_id', how='inner')
orthrus_emogi = orthrus.merge(emogi, on='gene_id', how='inner').drop(columns=['label'])

# normalize combined datasets using z-score normalization
orthrus_omics_normalized = orthrus_omics.copy()
numeric_cols = orthrus_omics.select_dtypes(include=[np.number]).columns
orthrus_omics_normalized[numeric_cols] = (orthrus_omics[numeric_cols] - orthrus_omics[numeric_cols].mean()) / orthrus_omics[numeric_cols].std()

orthrus_string_normalized = orthrus_string.copy()
numeric_cols = orthrus_string.select_dtypes(include=[np.number]).columns
orthrus_string_normalized[numeric_cols] = (orthrus_string[numeric_cols] - orthrus_string[numeric_cols].mean()) / orthrus_string[numeric_cols].std()

orthrus_string_exp_normalized = orthrus_string_exp.copy()
numeric_cols = orthrus_string_exp.select_dtypes(include=[np.number]).columns
orthrus_string_exp_normalized[numeric_cols] = (orthrus_string_exp[numeric_cols] - orthrus_string_exp[numeric_cols].mean()) / orthrus_string_exp[numeric_cols].std()

orthrus_emogi_normalized = orthrus_emogi.copy()
numeric_cols = orthrus_emogi.select_dtypes(include=[np.number]).columns
orthrus_emogi_normalized[numeric_cols] = (orthrus_emogi[numeric_cols] - orthrus_emogi[numeric_cols].mean()) / orthrus_emogi[numeric_cols].std()

# set containing everything

embedding_datasets = {
    'Omics': omics,
    'STRING': string,
    'STRING_EXP': string_exp,
    'Orthrus': orthrus,
    'Emogi_Predictions': emogi_predictions,
    'Omics + Orthrus': orthrus_omics_normalized,
    'STRING + Orthrus': orthrus_string_normalized,
    'STRING_EXP + Orthrus': orthrus_string_exp_normalized,
    'Emogi + Orthrus ': orthrus_emogi_normalized,
}

def add_random_guesser(embedding_datasets, reference_df, seed=42):
    genes = reference_df['gene_id'].drop_duplicates()
    rng = np.random.default_rng(seed)
    rand_df = pd.DataFrame({
        'gene_id': genes.values,
        'rand_feature': rng.random(len(genes))  # uniform [0,1]
    })
    embedding_datasets['Random'] = rand_df

add_random_guesser(embedding_datasets, omics)

#results = compare_embeddings_pr_curves(embedding_datasets, emogi, save_plots=True, title_suffix="All_Combinations")
#print("\nPR Curves Results summary (Mean ± Std):")
#print(results.pivot(index='Embedding', columns='Model', values='auPRC'))

auprc_results = compare_embeddings_auprc_barplots(embedding_datasets, emogi, save_plots=True, title_suffix="All_Combinations_No_Reduction")
print("\nauPRC Bar Plot Results summary (Mean ± Std):")
print(auprc_results.pivot(index='Embedding', columns='Model', values='auPRC'))


