import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def generate_conclusions_and_recommendations():
    """
    Génère les conclusions et recommandations de l'analyse de clustering
    """
    # Chargement des données
    df = pd.read_csv('data/clustered_data.csv')
    interpretations = pd.read_csv('data/cluster_interpretations.csv', index_col=0)
    
    print("=== CONCLUSIONS ET RECOMMANDATIONS ===")
    
    # 1. RÉSUMÉ DE L'ANALYSE
    print("1. RÉSUMÉ DE L'ANALYSE")
    print("=" * 50)
    
    n_clusters = df['Cluster'].nunique()
    total_customers = len(df)
    
    print(f"• Dataset analysé: {total_customers} clients")
    print(f"• Nombre de clusters identifiés: {n_clusters}")
    print(f"• Variables utilisées: Âge, Revenu Annuel, Score de Dépense")
    
    # Qualité du clustering
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler
    
    features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
    X = df[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    silhouette_avg = silhouette_score(X_scaled, df['Cluster'])
    
    print(f"• Score de silhouette: {silhouette_avg:.3f}")
    
    if silhouette_avg > 0.5:
        quality = "Excellente"
    elif silhouette_avg > 0.3:
        quality = "Bonne"
    else:
        quality = "Modérée"
    
    print(f"• Qualité du clustering: {quality}")
    
    # 2. PROFILS DES SEGMENTS IDENTIFIÉS
    print(f"\n2. PROFILS DES SEGMENTS IDENTIFIÉS")
    print("=" * 50)
    
    cluster_summary = df.groupby('Cluster')[features].agg(['mean', 'count'])
    
    for cluster in sorted(df['Cluster'].unique()):
        cluster_data = df[df['Cluster'] == cluster]
        size = len(cluster_data)
        percentage = (size / total_customers) * 100
        
        print(f"\n📊 CLUSTER {cluster} ({size} clients - {percentage:.1f}%)")
        print(f"   Âge moyen: {cluster_data['Age'].mean():.1f} ans")
        print(f"   Revenu moyen: {cluster_data['Annual Income (k$)'].mean():.1f}k$")
        print(f"   Score dépense moyen: {cluster_data['Spending Score (1-100)'].mean():.1f}/100")
        
        # Répartition par genre
        gender_dist = cluster_data['Gender'].value_counts(normalize=True)
        print(f"   Répartition genre: {gender_dist.to_dict()}")
        
        # Caractérisation
        avg_income = cluster_data['Annual Income (k$)'].mean()
        avg_spending = cluster_data['Spending Score (1-100)'].mean()
        avg_age = cluster_data['Age'].mean()
        
        characteristics = []
        if avg_age < 30:
            characteristics.append("Population jeune")
        elif avg_age > 50:
            characteristics.append("Population mature")
        
        if avg_income < 40:
            characteristics.append("Revenus modestes")
        elif avg_income > 70:
            characteristics.append("Revenus élevés")
        
        if avg_spending < 40:
            characteristics.append("Dépenses faibles")
        elif avg_spending > 60:
            characteristics.append("Dépenses élevées")
        
        if characteristics:
            print(f"   Caractéristiques: {', '.join(characteristics)}")
    
    # 3. INSIGHTS BUSINESS
    print(f"\n3. INSIGHTS BUSINESS")
    print("=" * 50)
    
    # Analyse de la relation revenu-dépense
    correlation_income_spending = df['Annual Income (k$)'].corr(df['Spending Score (1-100)'])
    print(f"• Corrélation Revenu-Dépense: {correlation_income_spending:.3f}")
    
    if abs(correlation_income_spending) < 0.3:
        print("  → Faible corrélation: le revenu ne prédit pas directement les dépenses")
    
    # Segments à fort potentiel
    high_value_segments = []
    for cluster in sorted(df['Cluster'].unique()):
        cluster_data = df[df['Cluster'] == cluster]
        avg_income = cluster_data['Annual Income (k$)'].mean()
        avg_spending = cluster_data['Spending Score (1-100)'].mean()
        
        if avg_income > 60 or avg_spending > 60:
            high_value_segments.append(cluster)
    
    print(f"• Segments à fort potentiel: {high_value_segments}")
    
    # 4. RECOMMANDATIONS STRATÉGIQUES
    print(f"\n4. RECOMMANDATIONS STRATÉGIQUES")
    print("=" * 50)
    
    recommendations = []
    
    for cluster in sorted(df['Cluster'].unique()):
        cluster_data = df[df['Cluster'] == cluster]
        avg_income = cluster_data['Annual Income (k$)'].mean()
        avg_spending = cluster_data['Spending Score (1-100)'].mean()
        avg_age = cluster_data['Age'].mean()
        size = len(cluster_data)
        
        print(f"\n🎯 CLUSTER {cluster} - Stratégies recommandées:")
        
        if avg_income > 70 and avg_spending > 60:
            print("   • Segment PREMIUM - Clients VIP")
            print("     - Produits haut de gamme et services exclusifs")
            print("     - Programme de fidélité premium")
            print("     - Marketing personnalisé et expériences sur-mesure")
            recommendations.append(f"Cluster {cluster}: Stratégie Premium")
            
        elif avg_income > 70 and avg_spending < 40:
            print("   • Segment POTENTIEL ÉLEVÉ - Riches économes")
            print("     - Campagnes de stimulation des achats")
            print("     - Offres de valeur et promotions ciblées")
            print("     - Éducation sur les bénéfices produits")
            recommendations.append(f"Cluster {cluster}: Activation du potentiel")
            
        elif avg_income < 40 and avg_spending > 60:
            print("   • Segment IMPULSIF - Dépensiers à faible revenu")
            print("     - Produits abordables et accessibles")
            print("     - Facilités de paiement")
            print("     - Marketing émotionnel")
            recommendations.append(f"Cluster {cluster}: Accessibilité")
            
        elif avg_income < 40 and avg_spending < 40:
            print("   • Segment ÉCONOME - Clients sensibles au prix")
            print("     - Stratégie prix bas et promotions")
            print("     - Produits de base et essentiels")
            print("     - Communication sur le rapport qualité-prix")
            recommendations.append(f"Cluster {cluster}: Prix compétitifs")
            
        else:
            print("   • Segment ÉQUILIBRÉ - Classe moyenne")
            print("     - Offre diversifiée et équilibrée")
            print("     - Promotions saisonnières")
            print("     - Communication sur la qualité")
            recommendations.append(f"Cluster {cluster}: Approche équilibrée")
    
    # 5. MÉTRIQUES DE PERFORMANCE SUGGÉRÉES
    print(f"\n5. MÉTRIQUES DE PERFORMANCE SUGGÉRÉES")
    print("=" * 50)
    
    print("• Métriques par segment:")
    print("  - Taux de conversion par cluster")
    print("  - Panier moyen par cluster")
    print("  - Fréquence d'achat par cluster")
    print("  - Taux de rétention par cluster")
    print("  - Customer Lifetime Value (CLV) par cluster")
    
    print("\n• Métriques globales:")
    print("  - Évolution de la taille des segments")
    print("  - Migration entre segments")
    print("  - ROI des campagnes par segment")
    
    # 6. LIMITES ET AMÉLIORATIONS
    print(f"\n6. LIMITES ET AMÉLIORATIONS POSSIBLES")
    print("=" * 50)
    
    print("• Limites actuelles:")
    print("  - Analyse basée uniquement sur 3 variables")
    print("  - Pas de données comportementales détaillées")
    print("  - Pas de dimension temporelle")
    print("  - Taille d'échantillon relativement petite")
    
    print("\n• Améliorations suggérées:")
    print("  - Intégrer des données transactionnelles")
    print("  - Ajouter des variables comportementales (fréquence, récence)")
    print("  - Inclure des données démographiques supplémentaires")
    print("  - Analyser l'évolution temporelle des segments")
    print("  - Tester d'autres algorithmes de clustering")
    print("  - Validation avec des données externes")
    
    # Visualisation finale - Dashboard de synthèse
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Dashboard de Synthèse - Segmentation Clients', fontsize=16)
    
    # Taille des segments
    cluster_sizes = df['Cluster'].value_counts().sort_index()
    axes[0,0].pie(cluster_sizes.values, labels=[f'Cluster {i}' for i in cluster_sizes.index], 
                  autopct='%1.1f%%')
    axes[0,0].set_title('Répartition des Segments')
    
    # Profil revenu-dépense par cluster
    for cluster in sorted(df['Cluster'].unique()):
        cluster_data = df[df['Cluster'] == cluster]
        axes[0,1].scatter(cluster_data['Annual Income (k$)'], 
                         cluster_data['Spending Score (1-100)'],
                         label=f'Cluster {cluster}', alpha=0.7, s=50)
    axes[0,1].set_xlabel('Revenu Annuel (k$)')
    axes[0,1].set_ylabel('Score de Dépense')
    axes[0,1].set_title('Profil Revenu-Dépense par Segment')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # Âge moyen par cluster
    age_by_cluster = df.groupby('Cluster')['Age'].mean()
    axes[1,0].bar(age_by_cluster.index, age_by_cluster.values)
    axes[1,0].set_xlabel('Cluster')
    axes[1,0].set_ylabel('Âge Moyen')
    axes[1,0].set_title('Âge Moyen par Segment')
    
    # Heatmap des profils
    cluster_profiles = df.groupby('Cluster')[features].mean()
    sns.heatmap(cluster_profiles.T, annot=True, cmap='RdYlBu_r', ax=axes[1,1], fmt='.1f')
    axes[1,1].set_title('Profils Moyens par Segment')
    
    plt.tight_layout()
    plt.savefig('plots/final_dashboard.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Sauvegarde du rapport
    report = {
        'total_customers': total_customers,
        'n_clusters': n_clusters,
        'silhouette_score': silhouette_avg,
        'clustering_quality': quality,
        'recommendations': recommendations
    }
    
    # Sauvegarde en CSV pour référence
    pd.DataFrame([report]).to_csv('data/analysis_summary.csv', index=False)
    
    print(f"\n✅ ANALYSE TERMINÉE")
    print("=" * 50)
    print(f"Tous les résultats ont été sauvegardés dans le dossier 'data/'")
    print(f"Toutes les visualisations ont été sauvegardées dans le dossier 'plots/'")
    
    return report

if __name__ == "__main__":
    final_report = generate_conclusions_and_recommendations()
    print("\nAnalyse complète terminée avec succès!")
