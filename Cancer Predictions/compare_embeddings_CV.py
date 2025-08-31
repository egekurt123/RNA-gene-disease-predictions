from common_helper_functions import *
from sklearn.model_selection import StratifiedKFold

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
    Uses emogi datasource label column as ground truth with 5-fold cross-validation.
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
    
    # 5-fold cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
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
            X = X.select_dtypes(include=[np.number])
            y = merged['label']
            
            # Skip if not enough samples
            if len(X) < 10 or y.sum() < 5:
                print(f"Skipping {embedding_name}: insufficient samples (total: {len(X)}, positive: {y.sum()})")
                continue
            
            # Storage for CV results
            cv_precision_rf = []
            cv_recall_rf = []
            cv_ap_scores_rf = []
            cv_precision_xgb = []
            cv_recall_xgb = []
            cv_ap_scores_xgb = []
            
            # Perform 5-fold cross-validation
            for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                # Train Random Forest
                clf_rf = RandomForestClassifier(n_estimators=100, random_state=42)
                clf_rf.fit(X_train, y_train)
                y_prob_rf = clf_rf.predict_proba(X_test)[:, 1]
                
                # Train XGBoost
                clf_xgb = XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
                clf_xgb.fit(X_train, y_train)
                y_prob_xgb = clf_xgb.predict_proba(X_test)[:, 1]
                
                # Calculate PR curves for this fold
                precision_rf, recall_rf, _ = precision_recall_curve(y_test, y_prob_rf)
                ap_score_rf = average_precision_score(y_test, y_prob_rf)
                
                precision_xgb, recall_xgb, _ = precision_recall_curve(y_test, y_prob_xgb)
                ap_score_xgb = average_precision_score(y_test, y_prob_xgb)
                
                # Store fold results
                cv_precision_rf.append(precision_rf)
                cv_recall_rf.append(recall_rf)
                cv_ap_scores_rf.append(ap_score_rf)
                cv_precision_xgb.append(precision_xgb)
                cv_recall_xgb.append(recall_xgb)
                cv_ap_scores_xgb.append(ap_score_xgb)
            
            # Calculate mean AP scores across folds
            mean_ap_rf = np.mean(cv_ap_scores_rf)
            std_ap_rf = np.std(cv_ap_scores_rf)
            mean_ap_xgb = np.mean(cv_ap_scores_xgb)
            std_ap_xgb = np.std(cv_ap_scores_xgb)
            
            # For plotting, use the fold with AP closest to mean
            closest_fold_rf = np.argmin(np.abs(cv_ap_scores_rf - mean_ap_rf))
            closest_fold_xgb = np.argmin(np.abs(cv_ap_scores_xgb - mean_ap_xgb))
            
            # Plot curves on separate axes using representative fold
            color = colors[i]
            ax_rf.plot(cv_recall_rf[closest_fold_rf], cv_precision_rf[closest_fold_rf], 
                      color=color, linestyle='-', 
                      label=f'{embedding_name} (AP={mean_ap_rf:.3f}±{std_ap_rf:.3f})', linewidth=2)
            ax_xgb.plot(cv_recall_xgb[closest_fold_xgb], cv_precision_xgb[closest_fold_xgb], 
                       color=color, linestyle='-', 
                       label=f'{embedding_name} (AP={mean_ap_xgb:.3f}±{std_ap_xgb:.3f})', linewidth=2)
            
            # Store results
            results.append({
                'Embedding': embedding_name,
                'Model': 'Random Forest',
                'auPRC': mean_ap_rf,
                'auPRC_std': std_ap_rf,
                'Samples': len(X),
                'Positive_Rate': y.mean()
            })
            results.append({
                'Embedding': embedding_name,
                'Model': 'XGBoost',
                'auPRC': mean_ap_xgb,
                'auPRC_std': std_ap_xgb,
                'Samples': len(X),
                'Positive_Rate': y.mean()
            })
            
            print(f"{embedding_name} - RF auPRC: {mean_ap_rf:.3f}±{std_ap_rf:.3f}, XGB auPRC: {mean_ap_xgb:.3f}±{std_ap_xgb:.3f}")
            
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
    ax_rf.set_title('Random Forest - Precision-Recall Curves (5-Fold CV, Emogi Labels)', fontsize=14)
    ax_rf.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax_rf.grid(True, alpha=0.3)
    fig_rf.tight_layout()
    
    # Configure XGBoost plot
    ax_xgb.set_xlabel('Recall', fontsize=12)
    ax_xgb.set_ylabel('Precision', fontsize=12)
    ax_xgb.set_title('XGBoost - Precision-Recall Curves (5-Fold CV, Emogi Labels)', fontsize=14)
    ax_xgb.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax_xgb.grid(True, alpha=0.3)
    fig_xgb.tight_layout()
    
    # Save the plots
    rf_filename = 'Emogi_Labels_RandomForest_5FoldCV_comparison.png'
    xgb_filename = 'Emogi_Labels_XGBoost_5FoldCV_comparison.png'
    
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

def compare_embeddings_auprc_barplots(embedding_datasets, emogi_data):
    """
    Compute auPRC for multiple embeddings using RF and XGBoost with 5-fold CV and create separate bar plots per model.
    Uses emogi datasource label column as ground truth.

    Parameters:
    - embedding_datasets: dict[name] -> DataFrame
    - emogi_data: DataFrame with gene_id, label (ground truth), and prediction columns

    Returns:
    - DataFrame with rows per (Embedding, Model)
    """
    results = []
    # Consistent color palette
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f'][:len(embedding_datasets)]
    color_map = dict(zip(embedding_datasets.keys(), colors))
    
    # 5-fold cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for embedding_name, embedding_df in embedding_datasets.items():
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
            columns_to_drop = ['label']
            if 'pred' in merged.columns and embedding_name != 'Emogi_Predictions':
                columns_to_drop.append('pred')

            X = merged.drop(columns_to_drop, axis=1, errors='ignore')
            X = X.select_dtypes(include=[np.number])
            y = merged['label']

            if len(X) < 10 or y.sum() < 5:
                print(f"Skipping {embedding_name}: insufficient samples")
                continue

            # Storage for CV results
            cv_ap_scores_rf = []
            cv_ap_scores_xgb = []
            
            # Perform 5-fold cross-validation
            for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

                # Random Forest
                clf_rf = RandomForestClassifier(n_estimators=100, random_state=42)
                clf_rf.fit(X_train, y_train)
                y_prob_rf = clf_rf.predict_proba(X_test)[:, 1]
                ap_rf = average_precision_score(y_test, y_prob_rf)
                cv_ap_scores_rf.append(ap_rf)

                # XGBoost
                clf_xgb = XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
                clf_xgb.fit(X_train, y_train)
                y_prob_xgb = clf_xgb.predict_proba(X_test)[:, 1]
                ap_xgb = average_precision_score(y_test, y_prob_xgb)
                cv_ap_scores_xgb.append(ap_xgb)

            # Calculate mean and std across folds
            mean_ap_rf = np.mean(cv_ap_scores_rf)
            std_ap_rf = np.std(cv_ap_scores_rf)
            mean_ap_xgb = np.mean(cv_ap_scores_xgb)
            std_ap_xgb = np.std(cv_ap_scores_xgb)

            results.append({
                'Embedding': embedding_name,
                'Model': 'Random Forest',
                'auPRC': mean_ap_rf,
                'auPRC_std': std_ap_rf,
                'Samples': len(X),
                'Positive_Rate': y.mean()
            })
            results.append({
                'Embedding': embedding_name,
                'Model': 'XGBoost',
                'auPRC': mean_ap_xgb,
                'auPRC_std': std_ap_xgb,
                'Samples': len(X),
                'Positive_Rate': y.mean()
            })
            print(f"{embedding_name}: RF auPRC={mean_ap_rf:.3f}±{std_ap_rf:.3f}, XGB auPRC={mean_ap_xgb:.3f}±{std_ap_xgb:.3f}")
        except Exception as e:
            print(f"Error processing {embedding_name}: {e}")
            continue

    if not results:
        print("No results to plot.")
        return pd.DataFrame(columns=['Embedding','Model','auPRC','auPRC_std','Samples','Positive_Rate'])

    results_df = pd.DataFrame(results)

    # Ensure all embeddings appear (even if one model failed)
    embeddings_order = list(embedding_datasets.keys())

    # Plot separately per model
    for model in results_df['Model'].unique():
        sub = results_df[results_df.Model == model].set_index('Embedding').reindex(embeddings_order)
        fig, ax = plt.subplots(figsize=(10, 6))
        # Use per-embedding colors
        bar_colors = [color_map.get(e, '#333333') for e in sub.index]
        bars = ax.bar(sub.index, sub['auPRC'], 
                     yerr=sub['auPRC_std'], 
                     color=bar_colors, 
                     capsize=5, 
                     alpha=0.8)
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            h = bar.get_height()
            std_val = sub['auPRC_std'].iloc[i]
            if not np.isnan(h):
                ax.text(bar.get_x() + bar.get_width()/2, h + std_val + 0.005, 
                       f"{h:.3f}", ha='center', va='bottom', fontsize=9)

        ax.set_ylabel('auPRC (5-Fold CV)')
        ax.set_title(f'{model} auPRC Across Embeddings (Emogi Labels, 5-Fold CV)')
        ax.set_xticklabels(sub.index, rotation=45, ha='right')
        ymax = (sub['auPRC'] + sub['auPRC_std']).max() if (sub['auPRC'] + sub['auPRC_std']).notna().any() else 0
        ax.set_ylim(0, min(1.0, ymax + 0.05))
        ax.grid(axis='y', alpha=0.3)

        fig.tight_layout()
        
        # Create directory if it doesn't exist
        import os
        os.makedirs("plots/AuPRC", exist_ok=True)
        
        out_file = f"plots/AuPRC/AuPRC_Emogi_Labels_{model}_5FoldCV_barplot.png"
        fig.savefig(out_file, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Bar plot saved as: {out_file}")

    return results_df

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