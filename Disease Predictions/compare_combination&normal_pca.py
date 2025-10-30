from common_helper_functions import *

# Define combinated datasets
orthrus_omics = orthrus.merge(omics, on='gene_id', how='inner')
orthrus_string = orthrus.merge(string, on='gene_id', how='inner')
orthrus_string_exp = orthrus.merge(string_exp, on='gene_id', how='inner')
orthrus_pops = orthrus.merge(pops, on='gene_id', how='inner')
orthrus_pops_exp = orthrus.merge(pops_exp, on='gene_id', how='inner')

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

orthrus_pops_normalized = orthrus_pops.copy()
numeric_cols = orthrus_pops.select_dtypes(include=[np.number]).columns
orthrus_pops_normalized[numeric_cols] = (orthrus_pops[numeric_cols] - orthrus_pops[numeric_cols].mean()) / orthrus_pops[numeric_cols].std()

orthrus_pops_exp_normalized = orthrus_pops_exp.copy()
numeric_cols = orthrus_pops_exp.select_dtypes(include=[np.number]).columns
orthrus_pops_exp_normalized[numeric_cols] = (orthrus_pops_exp[numeric_cols] - orthrus_pops_exp[numeric_cols].mean()) / orthrus_pops_exp[numeric_cols].std()

# apply pca
orthrus_omics_pca = pca_reduce(orthrus_omics_normalized, n_components=256)
orthrus_string_pca = pca_reduce(orthrus_string_normalized, n_components=256)
orthrus_string_exp_pca = pca_reduce(orthrus_string_exp_normalized, n_components=256)
orthrus_pops_pca = pca_reduce(orthrus_pops_normalized, n_components=256)
orthrus_pops_exp_pca = pca_reduce(orthrus_pops_exp_normalized, n_components=256)

embedding_datasets = {
    "STRING": string,
    'STRING + Orthrus': orthrus_string_pca,
    "STRING Exp": string_exp,
    'STRING_EXP + Orthrus': orthrus_string_exp_pca,
    "PoPS": pops,
    'PoPS + Orthrus': orthrus_pops_pca,
    "PoPS Exp": pops_exp,
    'PoPS_EXP + Orthrus': orthrus_pops_exp_pca,
    "Omics": omics,
    'Omics + Orthrus': orthrus_omics_pca,
    "Orthrus": orthrus,
}

disease_datasets = {
    "Mito": mito,
    "Ophthalmology": opthamology,
    "Inborn errors of immunity": inborn_errors_immunity,
    "Neurology": neurology,
    "Neuromuscular": neuromuscular,
    "Epilepsy": epilepsy
}

plot_disease_embeddings_auprc_paper(embedding_datasets, disease_datasets, save_dir="plots/AuPRC", original=False, title_suffix="_combinations_compare")

