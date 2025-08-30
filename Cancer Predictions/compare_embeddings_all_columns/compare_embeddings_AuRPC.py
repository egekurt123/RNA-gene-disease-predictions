import sys
sys.path.append('..')
from common_helper_functions import *

path = "../../../../../../../../../../../s/project/gene_embedding/funcrvp_embeddings/"

#Paper Embeddings
esm2 = pd.read_csv(path + 'ESM2_PCA_d512.tsv', sep='\t')

omics = pd.read_csv(path + 'omics_d256.tsv', sep='\t')

pops = pd.read_csv(path + 'pops_mat_d256.tsv', sep='\t')

pops_exp = pd.read_csv(path + 'pops_mat_exp_d256.tsv', sep='\t')

string = pd.read_csv(path + 'STRING_d128.tsv', sep='\t')

string_exp = pd.read_csv(path + 'STRING_EXP_d128.tsv', sep='\t')

#Cancer columns
emogi_cancer_predictions = pd.read_csv('../../../../../../../../../s/project/gene_embedding/input_data/cancer_eval/emogi_predictions.tsv', sep='\t')
emogi_relevant_columns = emogi_cancer_predictions[['ID','Name', "NCG_Known_Cancer_Gene", "NCG_Candidate_Cancer_Gene", "OncoKB_Cancer_Gene", "Bailey_et_al_Cancer_Gene"]]


#RNA embeddings
orthrus_embeddings = pd.read_csv('../../preprocessed_data/orthrus_processed.csv')


def compare_embeddings_pr_curves(embedding_datasets, cancer_data, target_column):
    """
    Compute auPRC for multiple embeddings using RF and XGBoost and create separate bar plots per model.

    Parameters:
    - embedding_datasets: dict[name] -> DataFrame
    - cancer_data: DataFrame with cancer gene annotations
    - target_column: label column to predict

    Returns:
    - DataFrame with rows per (Embedding, Model)
    """
    results = []
    # Added consistent color palette (same as compare_embeddings_PR.py)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f'][:len(embedding_datasets)]
    color_map = dict(zip(embedding_datasets.keys(), colors))

    for embedding_name, embedding_df in embedding_datasets.items():
        try:
            # Merge embedding with cancer data
            if embedding_name == 'Orthrus':
                merged = embedding_df.merge(cancer_data, left_on='gene_id', right_on='ID', how='inner')
                merged.drop(columns=['gene_id'], inplace=True)
                merged.rename(columns={'gene_id': 'ID'}, inplace=True)
                merged.set_index('ID', inplace=True)
            else:
                if 'gene_id' in embedding_df.columns:
                    merged = embedding_df.merge(cancer_data, left_on='gene_id', right_on='ID', how='inner')
                    merged.set_index('gene_id', inplace=True)
                else:
                    merged = embedding_df.merge(cancer_data, left_index=True, right_on='ID', how='inner')
                    merged.set_index('ID', inplace=True)

            cancer_columns = ["NCG_Known_Cancer_Gene", "NCG_Candidate_Cancer_Gene",
                              "OncoKB_Cancer_Gene", "Bailey_et_al_Cancer_Gene", "ID"]
            cols_to_drop = [c for c in cancer_columns if c in merged.columns and c != target_column]

            X = merged.drop(cols_to_drop + [target_column], axis=1, errors='ignore')
            X = X.select_dtypes(include=[np.number]).reset_index(drop=True)
            y = merged[target_column].reset_index(drop=True)

            if len(X) < 10 or y.sum() < 5:
                print(f"Skipping {embedding_name}: insufficient samples")
                continue

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            # Random Forest
            clf_rf = RandomForestClassifier(n_estimators=100, random_state=42)
            clf_rf.fit(X_train, y_train)
            y_prob_rf = clf_rf.predict_proba(X_test)[:, 1]
            ap_rf = average_precision_score(y_test, y_prob_rf)

            # XGBoost
            clf_xgb = XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
            clf_xgb.fit(X_train, y_train)
            y_prob_xgb = clf_xgb.predict_proba(X_test)[:, 1]
            ap_xgb = average_precision_score(y_test, y_prob_xgb)

            results.append({
                'Embedding': embedding_name,
                'Model': 'Random Forest',
                'auPRC': ap_rf,
                'Samples': len(X),
                'Positive_Rate': y.mean()
            })
            results.append({
                'Embedding': embedding_name,
                'Model': 'XGBoost',
                'auPRC': ap_xgb,
                'Samples': len(X),
                'Positive_Rate': y.mean()
            })
            print(f"{embedding_name}: RF auPRC={ap_rf:.3f}, XGB auPRC={ap_xgb:.3f}")
        except Exception as e:
            print(f"Error processing {embedding_name}: {e}")
            continue

    if not results:
        print("No results to plot.")
        return pd.DataFrame(columns=['Embedding','Model','auPRC','Samples','Positive_Rate'])

    results_df = pd.DataFrame(results)

    # Ensure all embeddings appear (even if one model failed)
    embeddings_order = list(embedding_datasets.keys())

    # Plot separately per model
    for model in results_df['Model'].unique():
        sub = results_df[results_df.Model == model].set_index('Embedding').reindex(embeddings_order)
        fig, ax = plt.subplots(figsize=(10, 5))
        # Use per-embedding colors instead of single model color
        bar_colors = [color_map.get(e, '#333333') for e in sub.index]
        bars = ax.bar(sub.index, sub['auPRC'], color=bar_colors)
        for bar in bars:
            h = bar.get_height()
            if not np.isnan(h):
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.005, f"{h:.3f}", ha='center', va='bottom', fontsize=8)

        # Removed positive rate (baseline) line as requested
        # baseline_map and ax.plot call removed

        ax.set_ylabel('auPRC')
        ax.set_title(f'{model} auPRC Across Embeddings: {target_column}')
        ax.set_xticklabels(sub.index, rotation=45, ha='right')
        ymax = sub['auPRC'].max() if sub['auPRC'].notna().any() else 0
        ax.set_ylim(0, min(1.0, ymax + 0.05))
        ax.grid(axis='y', alpha=0.3)
        # Legend not strictly necessary (colors map to x-ticks), so omitted

        fig.tight_layout()
        out_file = f"../plots/AuPRC/AuPRC_{target_column}_{model}_barplot.png"
        fig.savefig(out_file, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Bar plot saved as: {out_file}")

    return results_df


embedding_datasets = {
    'ESM2': esm2,
    'Omics': omics,
    'POPS': pops,
    'POPS_EXP': pops_exp,
    'STRING': string,
    'STRING_EXP': string_exp,
    'Orthrus': orthrus_embeddings
}

# Compare embeddings for each target
for target in ["NCG_Known_Cancer_Gene", "NCG_Candidate_Cancer_Gene", "OncoKB_Cancer_Gene", "Bailey_et_al_Cancer_Gene"]:
    print(f"\n{'='*50}")
    print(f"Comparing embeddings for: {target}")
    print(f"{'='*50}")
    
    results = compare_embeddings_pr_curves(embedding_datasets, emogi_relevant_columns, target)
    print(f"\nResults summary for {target}:")
    print(results.pivot(index='Embedding', columns='Model', values='auPRC'))


