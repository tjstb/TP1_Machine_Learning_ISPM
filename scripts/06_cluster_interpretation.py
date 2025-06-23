import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def interpret_clusters():
    """
    Interprète et analyse les clusters obtenus
    """
    # Chargement des données avec clusters
    df = pd.read_csv('data/clustered_data.csv')
    
    print("=== INTERPRÉTATION DES CLUSTERS ===")
    
    # 1. PROFILS DES CLUSTERS
    print("1. Profils des clusters...")
    
    features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
    
    # Statistiques détaillées par cluster
    cluster_profiles = df.groupby('Cluster')[features].agg(['mean', 'median', 'std'])
    print("\nProfils détaillés des clusters:")
    print(cluster_profiles.round(2))
    
    # Tailles des clusters
    cluster_sizes = df['Cluster'].value_counts().sort_index()
    print(f"\nTailles des clusters:")
    for cluster, size in cluster_sizes.items():
        percentage = (size / len(df)) * 100
        print(f"Cluster {cluster}: {size} clients ({percentage:.1f}%)")
    
    # 2. CARACTÉRISATION DES CLUSTERS
    print(f"\n2. Caractérisation des clusters...")
    
    # Moyennes par cluster pour caractérisation
    cluster_means = df.groupby('Cluster')[features].mean()
    
    # Normalisation pour comparaison (z-score par rapport à la moyenne globale)
    global_means = df[features].mean()
    global_stds = df[features].std()
    
    cluster_profiles_normalized = pd.DataFrame()
    for cluster in cluster_means.index:
        normalized_profile = (cluster_means.loc[cluster] - global_means) / global_stds
        cluster_profiles_normalized[f'Cluster_{cluster}'] = normalized_profile
    
    print("\nProfils normalisés (z-scores):")
    print(cluster_profiles_normalized.round(2))
    
    # 3. VISUALISATIONS AVANCÉES
    
    # Radar chart des profils
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Profils des Clusters', fontsize=16)
    
    # Moyennes par cluster
    for i, cluster in enumerate(sorted(df['Cluster'].unique())):
        row = i // 3
        col = i % 3
        
        cluster_data = df[df['Cluster'] == cluster]
        
        # Histogrammes des variables principales
        for j, feature in enumerate(features):
            if j == 0:
                axes[row, col].hist(cluster_data[feature], alpha=0.7, 
                                  label=f'Cluster {cluster}', bins=15)
                axes[row, col].axvline(cluster_data[feature].mean(), 
                                     color='red', linestyle='--', 
                                     label=f'Moyenne: {cluster_data[feature].mean():.1f}')
                axes[row, col].set_title(f'Cluster {cluster} - {feature}')
                axes[row, col].legend()
    
    plt.tight_layout()
    plt.savefig('plots/cluster_profiles_detailed.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Heatmap des profils moyens
    plt.figure(figsize=(10, 6))
    sns.heatmap(cluster_means.T, annot=True, cmap='RdYlBu_r', 
                center=cluster_means.values.mean(), fmt='.1f')
    plt.title('Heatmap des Profils Moyens par Cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Variables')
    plt.tight_layout()
    plt.savefig('plots/cluster_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Box plots par cluster
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, feature in enumerate(features):
        df.boxplot(column=feature, by='Cluster', ax=axes[i])
        axes[i].set_title(f'{feature} par Cluster')
        axes[i].set_xlabel('Cluster')
        axes[i].set_ylabel(feature)
    
    plt.suptitle('Distribution des Variables par Cluster')
    plt.tight_layout()
    plt.savefig('plots/cluster_boxplots.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 4. ANALYSE PAR GENRE
    print(f"\n3. Analyse par genre...")
    
    gender_analysis = pd.crosstab([df['Cluster'], df['Gender']], 
                                 columns='count', normalize='index')
    print("\nRépartition par genre dans chaque cluster:")
    print(gender_analysis.round(3))
    
    # Visualisation genre par cluster
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Répartition absolue
    pd.crosstab(df['Cluster'], df['Gender']).plot(kind='bar', ax=axes[0])
    axes[0].set_title('Répartition par Genre (Nombres absolus)')
    axes[0].set_xlabel('Cluster')
    axes[0].set_ylabel('Nombre de Clients')
    axes[0].legend(title='Genre')
    axes[0].tick_params(axis='x', rotation=0)
    
    # Répartition relative
    pd.crosstab(df['Cluster'], df['Gender'], normalize='index').plot(kind='bar', ax=axes[1])
    axes[1].set_title('Répartition par Genre (Proportions)')
    axes[1].set_xlabel('Cluster')
    axes[1].set_ylabel('Proportion')
    axes[1].legend(title='Genre')
    axes[1].tick_params(axis='x', rotation=0)
    
    plt.tight_layout()
    plt.savefig('plots/gender_by_cluster.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 5. NOMMAGE ET INTERPRÉTATION DES CLUSTERS
    print(f"\n4. Interprétation et nommage des clusters...")
    
    cluster_interpretations = {}
    
    for cluster in sorted(df['Cluster'].unique()):
        cluster_data = df[df['Cluster'] == cluster]
        
        avg_age = cluster_data['Age'].mean()
        avg_income = cluster_data['Annual Income (k$)'].mean()
        avg_spending = cluster_data['Spending Score (1-100)'].mean()
        size = len(cluster_data)
        
        # Logique de nommage basée sur les caractéristiques
        if avg_income < 40 and avg_spending < 40:
            name = "Économes à Faible Revenu"
        elif avg_income < 40 and avg_spending > 60:
            name = "Dépensiers à Faible Revenu"
        elif avg_income > 70 and avg_spending < 40:
            name = "Riches Économes"
        elif avg_income > 70 and avg_spending > 60:
            name = "Riches Dépensiers"
        elif 40 <= avg_income <= 70 and 40 <= avg_spending <= 60:
            name = "Classe Moyenne Équilibrée"
        else:
            name = f"Cluster {cluster}"
        
        cluster_interpretations[cluster] = {
            'name': name,
            'size': size,
            'avg_age': avg_age,
            'avg_income': avg_income,
            'avg_spending': avg_spending,
            'description': f"Âge moyen: {avg_age:.1f}, Revenu moyen: {avg_income:.1f}k$, Score dépense moyen: {avg_spending:.1f}"
        }
        
        print(f"\nCluster {cluster} - '{name}':")
        print(f"  Taille: {size} clients ({size/len(df)*100:.1f}%)")
        print(f"  Âge moyen: {avg_age:.1f} ans")
        print(f"  Revenu moyen: {avg_income:.1f}k$")
        print(f"  Score de dépense moyen: {avg_spending:.1f}")
        
        # Caractéristiques distinctives
        if avg_age < 30:
            print(f"  → Population jeune")
        elif avg_age > 50:
            print(f"  → Population mature")
        
        if avg_income < 30:
            print(f"  → Faible revenu")
        elif avg_income > 80:
            print(f"  → Revenu élevé")
        
        if avg_spending < 30:
            print(f"  → Faible propension à dépenser")
        elif avg_spending > 70:
            print(f"  → Forte propension à dépenser")
    
    # Sauvegarde des interprétations
    interpretation_df = pd.DataFrame(cluster_interpretations).T
    interpretation_df.to_csv('data/cluster_interpretations.csv')
    
    return cluster_interpretations

if __name__ == "__main__":
    interpretations = interpret_clusters()
    print("\nInterprétation des clusters terminée!")
