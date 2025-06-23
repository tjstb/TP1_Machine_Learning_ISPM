import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def perform_pca_analysis():
    """
    Effectue l'analyse en composantes principales (PCA)
    """
    # Chargement des données
    df = pd.read_csv('data/engineered_data.csv')
    
    print("=== ANALYSE EN COMPOSANTES PRINCIPALES (PCA) ===")
    
    # Sélection des variables pour la PCA
    features_for_pca = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
    X = df[features_for_pca]
    
    # Standardisation des données
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Application de la PCA
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)
    
    # Analyse des composantes principales
    print(f"Variance expliquée par chaque composante:")
    for i, var_exp in enumerate(pca.explained_variance_ratio_):
        print(f"PC{i+1}: {var_exp:.3f} ({var_exp*100:.1f}%)")
    
    print(f"\nVariance cumulée:")
    cumsum_var = np.cumsum(pca.explained_variance_ratio_)
    for i, cum_var in enumerate(cumsum_var):
        print(f"PC1 à PC{i+1}: {cum_var:.3f} ({cum_var*100:.1f}%)")
    
    # Visualisation de la variance expliquée
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Graphique en barres
    axes[0].bar(range(1, len(pca.explained_variance_ratio_) + 1), 
                pca.explained_variance_ratio_)
    axes[0].set_xlabel('Composante Principale')
    axes[0].set_ylabel('Variance Expliquée')
    axes[0].set_title('Variance Expliquée par Composante')
    axes[0].set_xticks(range(1, len(pca.explained_variance_ratio_) + 1))
    
    # Graphique cumulé
    axes[1].plot(range(1, len(cumsum_var) + 1), cumsum_var, 'bo-')
    axes[1].axhline(y=0.8, color='r', linestyle='--', label='80% de variance')
    axes[1].axhline(y=0.95, color='g', linestyle='--', label='95% de variance')
    axes[1].set_xlabel('Nombre de Composantes')
    axes[1].set_ylabel('Variance Cumulée')
    axes[1].set_title('Variance Cumulée')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/pca_variance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Analyse des loadings (contributions des variables)
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    loadings_df = pd.DataFrame(loadings, 
                              columns=[f'PC{i+1}' for i in range(len(features_for_pca))],
                              index=features_for_pca)
    
    print(f"\nLoadings (contributions des variables):")
    print(loadings_df)
    
    # Visualisation des loadings
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Heatmap des loadings
    sns.heatmap(loadings_df.T, annot=True, cmap='RdBu_r', center=0, 
                ax=axes[0], fmt='.3f')
    axes[0].set_title('Loadings des Variables sur les Composantes Principales')
    
    # Biplot (PC1 vs PC2)
    axes[1].scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6, c=df['Gender_encoded'], 
                   cmap='viridis')
    
    # Ajout des vecteurs des variables
    for i, (var, loading) in enumerate(zip(features_for_pca, loadings)):
        axes[1].arrow(0, 0, loading[0]*3, loading[1]*3, 
                     head_width=0.1, head_length=0.1, fc='red', ec='red')
        axes[1].text(loading[0]*3.2, loading[1]*3.2, var, 
                    fontsize=10, ha='center', va='center')
    
    axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    axes[1].set_title('Biplot PCA (coloré par Genre)')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/pca_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Création du dataset avec les composantes principales
    pca_df = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(X_pca.shape[1])])
    pca_df['Gender'] = df['Gender']
    pca_df['Gender_encoded'] = df['Gender_encoded']
    
    # Ajout des variables originales pour référence
    for col in features_for_pca:
        pca_df[col] = df[col]
    
    # Sauvegarde
    pca_df.to_csv('data/pca_data.csv', index=False)
    
    print(f"\nDonnées PCA sauvegardées. Forme: {pca_df.shape}")
    
    return pca, X_pca, pca_df, scaler

if __name__ == "__main__":
    pca_model, X_pca, pca_df, scaler = perform_pca_analysis()
    print("\nAnalyse PCA terminée!")
