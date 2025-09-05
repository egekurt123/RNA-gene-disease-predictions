import pandas as pd
import numpy as np
import pickle 
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os
import matplotlib.pyplot as plt
import pandas as pd
from xgboost import XGBClassifier

# datasets
path = "../../../../../../../../../../../s/project/gene_embedding/funcrvp_embeddings/"

omics = pd.read_csv(path + 'omics_d256.tsv', sep='\t')

string = pd.read_csv(path + 'STRING_d128.tsv', sep='\t')

string_exp = pd.read_csv(path + 'STRING_EXP_d128.tsv', sep='\t')

orthrus = pd.read_csv('../preprocessed_data/orthrus_processed.csv')

emogi = pd.read_csv("../preprocessed_data/emogi_preprocessed.tsv", sep='\t')

emogi_predictions = emogi[['gene_id', 'pred']].copy()

# functions
def investigate_dataset(dataset):
    """
    Print basic information about the dataset.
    """
    print(dataset.files)
    print("\nArray shapes:")
    print(f"X shape: {dataset['X'].shape}")
    print(f"y shape: {dataset['y'].shape}")
    print(f"genes shape: {dataset['genes'].shape}")
    print(f"\nFirst few y values: {dataset['y'][:5]}")
    print(f"\nX dimensions: {dataset['X'].ndim}")

    if dataset['X'].ndim == 3:
       print(f"X first sample shape: {dataset['X'][0].shape}")
       print(f"X first sample, first feature vector: {dataset['X'][0, 0, :5]}")
    elif dataset['X'].ndim == 2:
       print(f"X first sample (first 10 features): {dataset['X'][0, :10]}")

def create_embedding_dataframe(embeddings, genes, pca=True, n_components=2048):

    if pca:
        embeddings = embeddings.reshape(embeddings.shape[0], -1)
        embeddings = gene_embeddings_pca(embeddings, n_components=n_components)
        n_dims = embeddings.shape[1]
        column_names = [f'PCA_{i}' for i in range(n_dims)]
    else:
        column_names = [f'Feature_{i}' for i in range(embeddings.shape[1])]
    
    # Create DataFrame
    df = pd.DataFrame(embeddings, columns=column_names)
    df['gene_id'] = genes
    df = df.set_index('gene_id')

    # Keep only the first occurrence of each gene
    df = df[~df.index.duplicated(keep='first')]
    
    return df
    
def gene_embeddings_pca(X, n_components=2048):
  
    pca = PCA(n_components=n_components)
    gene_embeddings = pca.fit_transform(X)

    print(f"PCA embeddings shape: {gene_embeddings.shape}")

    total_explained_variance = np.sum(pca.explained_variance_ratio_)
    print(f"Total explained variance by PCA {total_explained_variance:.2%}")
    
    return gene_embeddings

def predict_and_plot_pr_curve(merged, target_column):
    """
    Train Random Forest and XGBoost to predict target_column from merged DataFrame,
    and plot precision-recall curves.
    """
    # Prepare features and target
    X = merged.drop([target_column], axis=1)
    y = merged[target_column]

    # Train-test split (80-20, stratified)
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

    # Precision-recall curves
    precision_rf, recall_rf, _ = precision_recall_curve(y_test, y_prob_rf)
    ap_score_rf = average_precision_score(y_test, y_prob_rf)

    precision_xgb, recall_xgb, _ = precision_recall_curve(y_test, y_prob_xgb)
    ap_score_xgb = average_precision_score(y_test, y_prob_xgb)

    # Plot Precision-Recall curves
    plt.figure(figsize=(7, 5))
    plt.plot(recall_rf, precision_rf, label=f'Random Forest (AP={ap_score_rf:.3f})', linewidth=2)
    plt.plot(recall_xgb, precision_xgb, label=f'XGBoost (AP={ap_score_xgb:.3f})', linewidth=2)
    plt.axhline(y=sum(y_test)/len(y_test), color='red', linestyle='--', label='Random baseline')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve: {target_column}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # Print auPRC scores
    print(f"Random Forest - auPRC: {ap_score_rf:.3f}")
    print(f"XGBoost - auPRC: {ap_score_xgb:.3f}")
    print()

def filter_to_common_genes(embedding_datasets, emogi_data):
    """Return copies of embedding_datasets and emogi_data filtered to the intersection of gene_ids."""
    def _get_gene_set(df):
        if 'gene_id' in df.columns:
            return set(df['gene_id'].astype(str))
        else:
            return set(df.index.astype(str))

    all_sets = [_get_gene_set(emogi_data)]
    for df in embedding_datasets.values():
        all_sets.append(_get_gene_set(df))
    common_genes = set.intersection(*all_sets)
    if len(common_genes) == 0:
        raise ValueError("No common genes found across all embedding datasets and emogi_data.")

    filtered_embeddings = {}
    for name, df in embedding_datasets.items():
        if 'gene_id' in df.columns:
            filtered_embeddings[name] = df[df['gene_id'].astype(str).isin(common_genes)].copy()
        else:
            filtered_embeddings[name] = df.loc[df.index.astype(str).isin(common_genes)].copy()

    emogi_filtered = emogi_data[emogi_data['gene_id'].astype(str).isin(common_genes)].copy()
    print(f"Filtered to {len(common_genes)} common genes across all datasets.")
    return filtered_embeddings, emogi_filtered, common_genes

def compare_embeddings_pr_curves(embedding_datasets, emogi_data, save_plots=True):
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

    #filter datasets to common genes
    embedding_datasets, emogi_data, _ = filter_to_common_genes(embedding_datasets, emogi_data)
    
    for i, (embedding_name, embedding_df) in enumerate(embedding_datasets.items()):
        try:
            # Merge embedding with emogi data
            if embedding_name == 'Orthrus' or embedding_name == 'Emogi_Predictions':
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
            
            # Print gene count and class distribution
            n_positives = y.sum()
            n_negatives = len(y) - n_positives
            print(f"{embedding_name}: Using {len(merged)} genes (Pos: {n_positives}, Neg: {n_negatives})")
            
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

    if save_plots:
        # Save the plots
        rf_filename = 'Emogi_Labels_RandomForest_5FoldCV_comparison.png'
        xgb_filename = 'Emogi_Labels_XGBoost_5FoldCV_comparison.png'

        # Create directory if it doesn't exist
        import os
        os.makedirs("plots/PR_curves", exist_ok=True)
    
        fig_rf.savefig(f"plots/PR_curves/{rf_filename}", dpi=300, bbox_inches='tight')
        fig_xgb.savefig(f"plots/PR_curves/{xgb_filename}", dpi=300, bbox_inches='tight')

        print(f"Random Forest plot saved as: {rf_filename}")
        print(f"XGBoost plot saved as: {xgb_filename}")

        plt.show()

    
    # Return results DataFrame
    return pd.DataFrame(results)

def compare_embeddings_auprc_barplots(embedding_datasets, emogi_data, save_plots=True):
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

    #filter datasets to common genes
    embedding_datasets, emogi_data, _ = filter_to_common_genes(embedding_datasets, emogi_data)

    for embedding_name, embedding_df in embedding_datasets.items():
        try:
            # Merge embedding with emogi data
            if embedding_name == 'Orthrus' or embedding_name == 'Emogi_Predictions':
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

            # Print gene count and class distribution
            n_positives = y.sum()
            n_negatives = len(y) - n_positives
            print(f"{embedding_name}: Using {len(merged)} genes (Pos: {n_positives}, Neg: {n_negatives})")

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
        
        if save_plots:
            # Create directory if it doesn't exist
            os.makedirs("plots/AuPRC", exist_ok=True)
        
            out_file = f"plots/AuPRC/AuPRC_Emogi_Labels_{model}_5FoldCV_barplot.png"
            fig.savefig(out_file, dpi=300, bbox_inches='tight')
            plt.show()
            print(f"Bar plot saved as: {out_file}")

    return results_df

def pca_reduce(df, n_components=256, id_cols=['gene_id']):
    df_copy = df.copy()
    # keep id columns if present
    present_id_cols = [c for c in id_cols if c in df_copy.columns]
    numeric_cols = df_copy.select_dtypes(include=[np.number]).columns.tolist()
    # exclude numeric id columns if any (e.g., gene index)
    numeric_cols = [c for c in numeric_cols if c not in present_id_cols]
    if len(numeric_cols) <= n_components:
        # nothing to reduce; return original frame (keep order)
        return df_copy
    X = df_copy[numeric_cols].values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    pca = PCA(n_components=n_components, random_state=42)
    Xp = pca.fit_transform(Xs)
    pc_cols = [f"PC_{i+1}" for i in range(Xp.shape[1])]
    df_pca = pd.DataFrame(Xp, columns=pc_cols, index=df_copy.index)
    non_numeric = df_copy.drop(columns=numeric_cols)
    return pd.concat([non_numeric.reset_index(drop=True), df_pca.reset_index(drop=True)], axis=1)

