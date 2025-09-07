from common_helper_functions import *

embedding_datasets = {
    'Omics': omics,
    'STRING': string,
    'STRING_Exp': string_exp,
    'PoPS': pops,
    'PoPS_Exp': pops_exp,
    'Orthrus': orthrus
    }


compare_embeddings_auprc_barplots(embedding_datasets, mito, save_plots=True, use_random_forest=False, disease_name="Mitochondrial Disease")
compare_embeddings_auprc_barplots(embedding_datasets, opthamology, save_plots=True, use_random_forest=False, disease_name="Ophthalmology Disease")
compare_embeddings_auprc_barplots(embedding_datasets, inborn_errors_immunity, save_plots=True, use_random_forest=False, disease_name="Inborn Errors of Immunity")
compare_embeddings_auprc_barplots(embedding_datasets, neurology, save_plots=True, use_random_forest=False, disease_name="Neurology Disease")
compare_embeddings_auprc_barplots(embedding_datasets, neuromuscular, save_plots=True, use_random_forest=False, disease_name="Neuromuscular Disease")
compare_embeddings_auprc_barplots(embedding_datasets, epilepsy, save_plots=True, use_random_forest=False, disease_name="Epilepsy")