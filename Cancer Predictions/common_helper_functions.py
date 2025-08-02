import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import pandas as pd
from xgboost import XGBClassifier

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

def create_embedding_dataframe(embeddings, genes, pca=True):

    embeddings_flat = embeddings.reshape(embeddings.shape[0], -1)

    if pca:
        embeddings = gene_embeddings_pca(embeddings_flat)
        n_dims = embeddings.shape[1]
        column_names = [f'PCA_{i}' for i in range(n_dims)]
    else:
        column_names = [f'Feature_{i}' for i in range(embeddings_flat.shape[1])]
    
    # Create DataFrame
    df = pd.DataFrame(embeddings_flat, columns=column_names)
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