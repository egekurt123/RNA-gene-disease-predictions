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
    Compare multiple embeddings by training RF and XGBoost models and plotting PR curves.
    Creates separate image files for Random Forest and XGBoost predictions.
    
    Parameters:
    - embedding_datasets: dict with embedding names as keys and DataFrames as values
    - cancer_data: DataFrame with cancer gene annotations
    - target_column: column name to predict
    """
    results = []
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f'][:len(embedding_datasets)]    
    # Create separate plots for RF and XGBoost
    fig_rf, ax_rf = plt.subplots(figsize=(12, 8))
    fig_xgb, ax_xgb = plt.subplots(figsize=(12, 8))
    
    for i, (embedding_name, embedding_df) in enumerate(embedding_datasets.items()):
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
            
            # Remove non-feature columns
            cancer_columns = ["NCG_Known_Cancer_Gene", "NCG_Candidate_Cancer_Gene", 
                            "OncoKB_Cancer_Gene", "Bailey_et_al_Cancer_Gene", "ID"]
            columns_to_drop = [col for col in cancer_columns if col in merged.columns and col != target_column]
            
            X = merged.drop(columns_to_drop + [target_column], axis=1, errors='ignore')
            X = X.select_dtypes(include=[np.number]).reset_index(drop=True)
            y = merged[target_column].reset_index(drop=True)
            
            # Skip if not enough samples
            if len(X) < 10 or y.sum() < 5:
                print(f"Skipping {embedding_name}: insufficient samples")
                continue
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Train Random Forest
            clf_rf = RandomForestClassifier(n_estimators=100, random_state=42)
            clf_rf.fit(X_train, y_train)
            y_prob_rf = clf_rf.predict_proba(X_test)[:, 1]
            
            # Train XGBoost
            clf_xgb = XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
            clf_xgb.fit(X_train, y_train)
            y_prob_xgb = clf_xgb.predict_proba(X_test)[:, 1]
            
            # Calculate PR curves
            precision_rf, recall_rf, _ = precision_recall_curve(y_test, y_prob_rf)
            ap_score_rf = average_precision_score(y_test, y_prob_rf)
            
            precision_xgb, recall_xgb, _ = precision_recall_curve(y_test, y_prob_xgb)
            ap_score_xgb = average_precision_score(y_test, y_prob_xgb)
            
            # Plot curves on separate axes
            color = colors[i]
            ax_rf.plot(recall_rf, precision_rf, color=color, linestyle='-', 
                      label=f'{embedding_name} (AP={ap_score_rf:.3f})', linewidth=2)
            ax_xgb.plot(recall_xgb, precision_xgb, color=color, linestyle='-', 
                       label=f'{embedding_name} (AP={ap_score_xgb:.3f})', linewidth=2)
            
            # Store results
            results.append({
                'Embedding': embedding_name,
                'Model': 'Random Forest',
                'auPRC': ap_score_rf,
                'Samples': len(X),
                'Positive_Rate': y.mean()
            })
            results.append({
                'Embedding': embedding_name,
                'Model': 'XGBoost',
                'auPRC': ap_score_xgb,
                'Samples': len(X),
                'Positive_Rate': y.mean()
            })
            
            print(f"{embedding_name} - RF auPRC: {ap_score_rf:.3f}, XGB auPRC: {ap_score_xgb:.3f}")
            
        except Exception as e:
            print(f"Error processing {embedding_name}: {str(e)}")
            continue
    
    # Add random baseline to both plots
    if results:
        baseline_rate = results[0]['Positive_Rate']
        ax_rf.axhline(y=baseline_rate, color='red', linestyle=':', 
                     label=f'Random baseline ({baseline_rate:.3f})', linewidth=2)
        ax_xgb.axhline(y=baseline_rate, color='red', linestyle=':', 
                      label=f'Random baseline ({baseline_rate:.3f})', linewidth=2)
    
    # Configure Random Forest plot
    ax_rf.set_xlabel('Recall', fontsize=12)
    ax_rf.set_ylabel('Precision', fontsize=12)
    ax_rf.set_title(f'Random Forest - Precision-Recall Curves: {target_column}', fontsize=14)
    ax_rf.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax_rf.grid(True, alpha=0.3)
    fig_rf.tight_layout()
    
    # Configure XGBoost plot
    ax_xgb.set_xlabel('Recall', fontsize=12)
    ax_xgb.set_ylabel('Precision', fontsize=12)
    ax_xgb.set_title(f'XGBoost - Precision-Recall Curves: {target_column}', fontsize=14)
    ax_xgb.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax_xgb.grid(True, alpha=0.3)
    fig_xgb.tight_layout()
    
    # Save the plots
    rf_filename = f'{target_column}_RandomForest_comparison.png'
    xgb_filename = f'{target_column}_XGBoost_comparison.png'

    fig_rf.savefig("../plots/PR_curves/" + rf_filename, dpi=300, bbox_inches='tight')
    fig_xgb.savefig("../plots/PR_curves/" + xgb_filename, dpi=300, bbox_inches='tight')

    plt.show()
    
    print(f"Random Forest plot saved as: {rf_filename}")
    print(f"XGBoost plot saved as: {xgb_filename}")
    
    # Return results DataFrame
    return pd.DataFrame(results)


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


