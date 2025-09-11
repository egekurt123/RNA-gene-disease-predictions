import pandas as pd
import numpy as np
import os
import pickle 
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from xgboost import XGBClassifier


# embedding datasets
path = "../../../../../../../../../../../s/project/gene_embedding/funcrvp_embeddings/"

omics = pd.read_csv(path + 'omics_d256.tsv', sep='\t')

string = pd.read_csv(path + 'STRING_d128.tsv', sep='\t')

string_exp = pd.read_csv(path + 'STRING_EXP_d128.tsv', sep='\t')

pops = pd.read_csv(path + 'pops_mat_d256.tsv', sep='\t')

pops_exp = pd.read_csv(path + 'pops_mat_exp_d256.tsv', sep='\t')

orthrus = pd.read_csv('../preprocessed_data/orthrus_processed.csv')

# disease gene datasets
mito = pd.read_csv('../preprocessed_data/mito_genes.csv')
epilepsy = pd.read_csv('../preprocessed_data/epilepsy_genes.csv')
neurology = pd.read_csv('../preprocessed_data/neurology_genes.csv')
inborn_errors_immunity = pd.read_csv('../preprocessed_data/inborn_errors_immunity_genes.csv')
neuromuscular = pd.read_csv('../preprocessed_data/neuromuscular_genes.csv')
opthamology = pd.read_csv('../preprocessed_data/opthamology_genes.csv')


def filter_to_common_genes(embedding_datasets, target_data):
    """Return copies of embedding_datasets and target_data filtered to the intersection of gene_ids."""
    def _get_gene_set(df):
        if 'gene_id' in df.columns:
            return set(df['gene_id'].astype(str))
        else:
            return set(df.index.astype(str))

    all_sets = [_get_gene_set(target_data)]
    for df in embedding_datasets.values():
        all_sets.append(_get_gene_set(df))
    common_genes = set.intersection(*all_sets)
    if len(common_genes) == 0:
        raise ValueError("No common genes found across all embedding datasets and target dataset.")

    filtered_embeddings = {}
    for name, df in embedding_datasets.items():
        if 'gene_id' in df.columns:
            filtered_embeddings[name] = df[df['gene_id'].astype(str).isin(common_genes)].copy()
        else:
            filtered_embeddings[name] = df.loc[df.index.astype(str).isin(common_genes)].copy()

    target_filtered = target_data[target_data['gene_id'].astype(str).isin(common_genes)].copy()
    print(f"Filtered to {len(common_genes)} common genes across all datasets.")
    return filtered_embeddings, target_filtered, common_genes

def compare_embeddings_auprc_barplots(embedding_datasets, target_data, save_plots=True, use_random_forest=True, disease_name="Disease"):
    """
    Compute auPRC for multiple embeddings using RF and XGBoost with 5-fold CV and create separate bar plots per model.
    Uses target dataset label column as ground truth.

    Parameters:
    - embedding_datasets: dict[name] -> DataFrame
    - target_data: DataFrame with gene_id, label (ground truth), and prediction columns

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
    embedding_datasets, target_data, _ = filter_to_common_genes(embedding_datasets, target_data)

    for embedding_name, embedding_df in embedding_datasets.items():
        try:
            # Merge embedding with target data
            if embedding_name == 'Orthrus':
                merged = embedding_df.merge(target_data, on='gene_id', how='inner')
                merged.set_index('gene_id', inplace=True)
            else: 
                if 'gene_id' in embedding_df.columns:
                    merged = embedding_df.merge(target_data, on='gene_id', how='inner')
                    merged.set_index('gene_id', inplace=True)
                else:
                    merged = embedding_df.merge(target_data, left_index=True, right_on='gene_id', how='inner')
                    merged.set_index('gene_id', inplace=True)

            X = merged.drop(['target'], axis=1, errors='ignore')
            X = X.select_dtypes(include=[np.number])
            y = merged['target']

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

                if use_random_forest:
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

            if use_random_forest:
                mean_ap_rf = np.mean(cv_ap_scores_rf)
                std_ap_rf = np.std(cv_ap_scores_rf)
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
            if use_random_forest:
                print(f"{embedding_name}: RF auPRC={mean_ap_rf:.3f}±{std_ap_rf:.3f}, XGB auPRC={mean_ap_xgb:.3f}±{std_ap_xgb:.3f}")
            else:
                print(f"{embedding_name}: XGB auPRC={mean_ap_xgb:.3f}±{std_ap_xgb:.3f}")
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
        ax.set_title(f'{model} auPRC Across Embeddings ({disease_name}, 5-Fold CV)')
        ax.set_xticklabels(sub.index, rotation=45, ha='right')
        ymax = (sub['auPRC'] + sub['auPRC_std']).max() if (sub['auPRC'] + sub['auPRC_std']).notna().any() else 0
        ax.set_ylim(0, min(1.0, ymax + 0.05))
        ax.grid(axis='y', alpha=0.3)

        fig.tight_layout()
        
        if save_plots:
            # Create directory if it doesn't exist
            os.makedirs("plots/AuPRC", exist_ok=True)

            out_file = f"plots/AuPRC/AuPRC_{disease_name}_{model}_5FoldCV_barplot.png"
            fig.savefig(out_file, dpi=300, bbox_inches='tight')
            plt.show()
            print(f"Bar plot saved as: {out_file}")

    return results_df

def plot_disease_embeddings_auprc_paper(embedding_datasets, disease_datasets, save_dir="plots/AuPRC", original=False, title_suffix=""):
    """
    Compute auPRC across embeddings with XGBoost (5-fold CV),
    plot grouped barplot with black dots for folds, styled like the paper figure.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Define consistent color palette (added Orthrus as purple)
    if original:
        color_map = {
            "STRING": "#1f77b4",       # dark blue
            "STRING Exp": "#aec7e8",   # light blue
            "PoPS": "#2ca02c",         # green
            "PoPS Exp": "#98df8a",     # light green
            "Omics": "#ff7f0e",        # orange
            "Orthrus": "#9467bd"       # purple
        }
    else:
        color_map = {
            "Omics + Orthrus": "#ff7f0e",        # orange
            "STRING + Orthrus": "#1f77b4",       # dark blue
            "STRING_EXP + Orthrus": "#aec7e8",   # light blue
            "PoPS + Orthrus": "#2ca02c",         # green
            "PoPS_EXP + Orthrus": "#98df8a"      # light green
        }

    results = []
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for disease_name, target_df in disease_datasets.items():
        for emb_name, emb_df in embedding_datasets.items():
            try:
                merged = emb_df.merge(target_df, on="gene_id", how="inner")
                merged.set_index("gene_id", inplace=True)
                X = merged.drop("target", axis=1, errors="ignore").select_dtypes(include=[np.number])
                y = merged["target"]

                fold_scores = []
                for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
                    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

                    clf = XGBClassifier(n_estimators=100, random_state=42, eval_metric="logloss")
                    clf.fit(X_train, y_train)
                    y_prob = clf.predict_proba(X_test)[:, 1]
                    auprc = average_precision_score(y_test, y_prob)
                    fold_scores.append(auprc)

                    results.append({
                        "Disease": disease_name,
                        "Embedding": emb_name,
                        "Fold": fold,
                        "auPRC": auprc
                    })

                print(f"{disease_name} - {emb_name}: mean auPRC = {np.mean(fold_scores):.3f}")

            except Exception as e:
                print(f"Error with {disease_name} / {emb_name}: {e}")
                continue

    results_df = pd.DataFrame(results)

    # === Plotting ===
    plt.figure(figsize=(12, 6))
    sns.set(style="whitegrid")

    # Bars = mean auPRC
    sns.barplot(
        data=results_df,
        x="Disease", y="auPRC", hue="Embedding",
        ci=None, palette=color_map, edgecolor="black", alpha=0.9
    )

    # Black dots = fold results
    sns.stripplot(
        data=results_df,
        x="Disease", y="auPRC", hue="Embedding",
        dodge=True, jitter=False, marker="o", alpha=0.8, color="black"
    )

    # Clean legend (remove duplicates)
    handles, labels = plt.gca().get_legend_handles_labels()
    n_embeds = len(embedding_datasets)
    plt.legend(handles[:n_embeds], labels[:n_embeds],
               bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)

    # Style adjustments
    plt.ylabel("auPRC", fontsize=14, weight="bold")
    plt.xlabel("")
    plt.title("XGBoost auPRC across diseases and embeddings (5-fold CV)", fontsize=15, weight="bold")
    plt.xticks(rotation=25, ha="right", fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylim(0, 0.7)
    plt.tight_layout()

    # Save
    out_file = os.path.join(save_dir, f"Disease_Embedding_auPRC_with_orthrus{title_suffix}.png")
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Saved plot: {out_file}")

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