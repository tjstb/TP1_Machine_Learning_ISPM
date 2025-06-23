import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, silhouette_samples
import warnings
warnings.filterwarnings('ignore')

def perform_clustering_analysis():
    """
    Effectue l'analyse de clustering avec K-means
    """
    # Chargement des données
    df = pd.read_csv('data/engineered_data.csv')
    pca_df = pd.read_csv('data/pca_data.csv')
    
    print("=== ANALYSE DE CLUSTERING ===")
    
    # Préparation des données pour le clustering
    features_for_clustering = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
    X = df[features_for_clustering]
    
    # Standardisation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 1. MÉTHODE DU COUDE (ELBOW METHOD)
    print("1. Méthode du coude...")
    
    k_range = range(2, 7)
    inertias = []
    silhouette_scores = []
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))
    
    # Visualisation de la méthode du coude et des scores de silhouette
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Méthode du coude
    axes[0].plot(k_range, inertias, 'bo-')
    axes[0].set_xlabel('Nombre de Clusters (k)')
    axes[0].set_ylabel('Inertie (WCSS)')
    axes[0].set_title('Méthode du Coude')
    axes[0].grid(True, alpha=0.3)
    
    # Scores de silhouette
    axes[1].plot(k_range, silhouette_scores, 'ro-')
    axes[1].set_xlabel('Nombre de Clusters (k)')
    axes[1].set_ylabel('Score de Silhouette')
    axes[1].set_title('Scores de Silhouette')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/clustering_evaluation.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Scores de silhouette par k:")
    for k, score in zip(k_range, silhouette_scores):
        print(f"k={k}: {score:.3f}")
    
    # 2. SÉLECTION DU NOMBRE OPTIMAL DE CLUSTERS
    # Basé sur le score de silhouette le plus élevé
    optimal_k = k_range[np.argmax(silhouette_scores)]
    print(f"\nNombre optimal de clusters (silhouette): {optimal_k}")
    
    # 3. CLUSTERING FINAL
    print(f"\n2. Clustering final avec k={optimal_k}...")
    
    kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    cluster_labels = kmeans_final.fit_predict(X_scaled)
    
    # Ajout des labels aux données
    df['Cluster'] = cluster_labels
    pca_df['Cluster'] = cluster_labels
    
    # Calcul des métriques finales
    final_silhouette = silhouette_score(X_scaled, cluster_labels)
    print(f"Score de silhouette final: {final_silhouette:.3f}")
    
    # 4. ANALYSE DES CLUSTERS
    print(f"\n3. Analyse des clusters...")
    
    # Statistiques par cluster
    cluster_stats = df.groupby('Cluster')[features_for_clustering].agg(['mean', 'std', 'count'])
    print(f"\nStatistiques par cluster:")
    print(cluster_stats)
    
    # Répartition par genre et cluster
    gender_cluster = pd.crosstab(df['Gender'], df['Cluster'], normalize='columns')
    print(f"\nRépartition par genre (en pourcentage par cluster):")
    print(gender_cluster * 100)
    
    # 5. VISUALISATIONS
    
    # Visualisation 2D avec PCA
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Visualisation des Clusters (k={optimal_k})', fontsize=16)
    
    # PCA colorée par cluster
    scatter = axes[0,0].scatter(pca_df['PC1'], pca_df['PC2'], 
                               c=pca_df['Cluster'], cmap='viridis', alpha=0.7)
    axes[0,0].set_xlabel('PC1')
    axes[0,0].set_ylabel('PC2')
    axes[0,0].set_title('Clusters dans l\'espace PCA')
    plt.colorbar(scatter, ax=axes[0,0])
    
    # Age vs Income
    scatter = axes[0,1].scatter(df['Annual Income (k$)'], df['Age'], 
                               c=df['Cluster'], cmap='viridis', alpha=0.7)
    axes[0,1].set_xlabel('Revenu Annuel (k$)')
    axes[0,1].set_ylabel('Âge')
    axes[0,1].set_title('Clusters: Âge vs Revenu')
    plt.colorbar(scatter, ax=axes[0,1])
    
    # Income vs Spending Score
    scatter = axes[1,0].scatter(df['Annual Income (k$)'], df['Spending Score (1-100)'], 
                               c=df['Cluster'], cmap='viridis', alpha=0.7)
    axes[1,0].set_xlabel('Revenu Annuel (k$)')
    axes[1,0].set_ylabel('Score de Dépense')
    axes[1,0].set_title('Clusters: Revenu vs Score de Dépense')
    plt.colorbar(scatter, ax=axes[1,0])
    
    # Age vs Spending Score
    scatter = axes[1,1].scatter(df['Age'], df['Spending Score (1-100)'], 
                               c=df['Cluster'], cmap='viridis', alpha=0.7)
    axes[1,1].set_xlabel('Âge')
    axes[1,1].set_ylabel('Score de Dépense')
    axes[1,1].set_title('Clusters: Âge vs Score de Dépense')
    plt.colorbar(scatter, ax=axes[1,1])
    
    plt.tight_layout()
    plt.savefig('plots/cluster_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Analyse de silhouette détaillée
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    silhouette_vals = silhouette_samples(X_scaled, cluster_labels)
    y_lower = 10
    
    for i in range(optimal_k):
        cluster_silhouette_vals = silhouette_vals[cluster_labels == i]
        cluster_silhouette_vals.sort()
        
        size_cluster_i = cluster_silhouette_vals.shape[0]
        y_upper = y_lower + size_cluster_i
        
        color = plt.cm.viridis(float(i) / optimal_k)
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_silhouette_vals,
                        facecolor=color, edgecolor=color, alpha=0.7)
        
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10
    
    ax.axvline(x=final_silhouette, color="red", linestyle="--", 
               label=f'Score moyen: {final_silhouette:.3f}')
    ax.set_xlabel('Valeurs de Silhouette')
    ax.set_ylabel('Clusters')
    ax.set_title('Analyse de Silhouette par Cluster')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('plots/silhouette_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Sauvegarde des résultats
    df.to_csv('data/clustered_data.csv', index=False)
    
    return kmeans_final, cluster_labels, optimal_k, final_silhouette

if __name__ == "__main__":
    model, labels, k_opt, silhouette_final = perform_clustering_analysis()
    print(f"\nClustering terminé! k optimal: {k_opt}, Silhouette: {silhouette_final:.3f}")
