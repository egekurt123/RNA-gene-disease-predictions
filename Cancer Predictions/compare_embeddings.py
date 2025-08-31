from common_helper_functions import *

path = "../../../../../../../../../../../s/project/gene_embedding/funcrvp_embeddings/"

omics = pd.read_csv(path + 'omics_d256.tsv', sep='\t')

string = pd.read_csv(path + 'STRING_d128.tsv', sep='\t')

string_exp = pd.read_csv(path + 'STRING_EXP_d128.tsv', sep='\t')

orthrus_embeddings = pd.read_csv('../preprocessed_data/orthrus_processed.csv')

emogi = pd.read_csv("../preprocessed_data/emogi_preprocessed.tsv", sep='\t')

# Create emogi predictions as a feature dataset (just the prediction column)
emogi_predictions = emogi[['gene_id', 'pred']].copy()

def compare_embeddings_pr_curves(embedding_datasets, emogi_data):
    """
    Compare multiple embeddings by training RF and XGBoost models and plotting PR curves.
    Uses emogi datasource label column as ground truth.
    Creates separate image files for Random Forest and XGBoost predictions.
    
    Parameters:
    - embedding_datasets: dict with embedding names as keys and DataFrames as values
    - emogi_data: DataFrame with gene_id, label (ground truth), and prediction columns
    """
    results = []
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f'][:len(embedding_datasets)]    
    
    # Create separate plots for RF and XGBoost
    fig_rf, ax_rf = plt.subplots(figsize=(12, 8))
    fig_xgb, ax_xgb = plt.subplots(figsize=(12, 8))
    
    for i, (embedding_name, embedding_df) in enumerate(embedding_datasets.items()):
        try:
            # Merge embedding with emogi data
            if embedding_name == 'Orthrus':
                merged = embedding_df.merge(emogi_data, on='gene_id', how='inner')
                merged.set_index('gene_id', inplace=True)
            elif embedding_name == 'Emogi_Predictions':
                # For emogi predictions, we already have gene_id
                merged = embedding_df.merge(emogi_data, on='gene_id', how='inner')
                merged.set_index('gene_id', inplace=True)
            else: 
                if 'gene_id' in embedding_df.columns:
                    merged = embedding_df.merge(emogi_data, on='gene_id', how='inner')
                    merged.set_index('gene_id', inplace=True)
                else:
                    merged = embedding_df.merge(emogi_data, left_index=True, right_on='gene_id', how='inner')
                    merged.set_index('gene_id', inplace=True)
            
            # Prepare features and target
            # Remove non-feature columns (keep only embedding features)
            columns_to_drop = ['label']
            if 'pred' in merged.columns and embedding_name != 'Emogi_Predictions':
                columns_to_drop.append('pred')
            
            X = merged.drop(columns_to_drop, axis=1, errors='ignore')
            X = X.select_dtypes(include=[np.number]).reset_index(drop=True)
            y = merged['label'].reset_index(drop=True)
            
            # Skip if not enough samples
            if len(X) < 10 or y.sum() < 5:
                print(f"Skipping {embedding_name}: insufficient samples (total: {len(X)}, positive: {y.sum()})")
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
    ax_rf.set_title('Random Forest - Precision-Recall Curves (Emogi Labels)', fontsize=14)
    ax_rf.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax_rf.grid(True, alpha=0.3)
    fig_rf.tight_layout()
    
    # Configure XGBoost plot
    ax_xgb.set_xlabel('Recall', fontsize=12)
    ax_xgb.set_ylabel('Precision', fontsize=12)
    ax_xgb.set_title('XGBoost - Precision-Recall Curves (Emogi Labels)', fontsize=14)
    ax_xgb.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax_xgb.grid(True, alpha=0.3)
    fig_xgb.tight_layout()
    
    # Save the plots
    rf_filename = 'Emogi_Labels_RandomForest_comparison.png'
    xgb_filename = 'Emogi_Labels_XGBoost_comparison.png'

    # Create directory if it doesn't exist
    import os
    os.makedirs("plots/PR_curves", exist_ok=True)
    
    fig_rf.savefig(f"plots/PR_curves/{rf_filename}", dpi=300, bbox_inches='tight')
    fig_xgb.savefig(f"plots/PR_curves/{xgb_filename}", dpi=300, bbox_inches='tight')

    plt.show()
    
    print(f"Random Forest plot saved as: {rf_filename}")
    print(f"XGBoost plot saved as: {xgb_filename}")
    
    # Return results DataFrame
    return pd.DataFrame(results)

# Define embedding datasets
embedding_datasets = {
    'Omics': omics,
    'STRING': string,
    'STRING_EXP': string_exp,
    'Orthrus': orthrus_embeddings,
    'Emogi_Predictions': emogi_predictions
}

# Compare embeddings using emogi labels
print("Comparing embeddings using Emogi labels as ground truth")
print("="*60)

results = compare_embeddings_pr_curves(embedding_datasets, emogi)
print("\nResults summary:")
print(results.pivot(index='Embedding', columns='Model', values='auPRC'))



