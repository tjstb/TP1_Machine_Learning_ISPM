"""
Script principal pour exécuter l'analyse complète de segmentation des clients
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from scipy import stats
import os
import warnings
warnings.filterwarnings('ignore')

def run_complete_analysis():
    """
    Exécute l'analyse complète de segmentation des clients du centre commercial
    """
    print("🚀 DÉMARRAGE DE L'ANALYSE COMPLÈTE DE SEGMENTATION")
    print("=" * 60)
    
    # Création des dossiers nécessaires
    os.makedirs('data', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    
    # ÉTAPE 1: Chargement et prétraitement des données
    print("\n📊 ÉTAPE 1/7: Chargement et prétraitement des données")
    print("-" * 50)
    
    try:
        # Chargement des données
        df = pd.read_csv('data/Mall_Customers.csv')
        
        print(f"Dataset chargé avec succès!")
        print(f"Forme du dataset: {df.shape}")
        print(f"Colonnes: {list(df.columns)}")
        print("\nPremières lignes:")
        print(df.head())
        
        # Vérification des valeurs manquantes
        print(f"\nValeurs manquantes: {df.isnull().sum().sum()}")
        
        # Préparation des données
        df_analysis = df.drop('CustomerID', axis=1)
        df_analysis['Gender_encoded'] = df_analysis['Gender'].map({'Male': 0, 'Female': 1})
        
        # Variables numériques
        numeric_features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
        
        # Standardisation
        scaler = StandardScaler()
        df_scaled = df_analysis.copy()
        df_scaled[numeric_features] = scaler.fit_transform(df_analysis[numeric_features])
        
        print("✅ Étape 1 terminée avec succès!")
        
    except Exception as e:
        print(f"❌ Erreur étape 1: {e}")
        return
    
    # ÉTAPE 2: Analyse exploratoire
    print("\n📊 ÉTAPE 2/7: Analyse exploratoire des données")
    print("-" * 50)
    
    try:
        # Configuration des graphiques
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Histogrammes et boxplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Analyse Univariée - Variables Numériques', fontsize=16)
        
        for i, var in enumerate(numeric_features):
            # Histogrammes
            axes[0, i].hist(df_analysis[var], bins=20, alpha=0.7, edgecolor='black')
            axes[0, i].set_title(f'Distribution de {var}')
            axes[0, i].set_xlabel(var)
            axes[0, i].set_ylabel('Fréquence')
            
            # Boxplots
            axes[1, i].boxplot(df_analysis[var])
            axes[1, i].set_title(f'Boxplot de {var}')
            axes[1, i].set_ylabel(var)
        
        plt.tight_layout()
        plt.savefig('plots/univariate_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Matrice de corrélation
        correlation_matrix = df_analysis[numeric_features + ['Gender_encoded']].corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                    square=True, fmt='.3f')
        plt.title('Matrice de Corrélation')
        plt.tight_layout()
        plt.savefig('plots/correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Étape 2 terminée avec succès!")
        
    except Exception as e:
        print(f"❌ Erreur étape 2: {e}")
    
    # ÉTAPE 3: Feature Engineering
    print("\n📊 ÉTAPE 3/7: Ingénierie des caractéristiques")
    print("-" * 50)
    
    try:
        # Catégorisation de l'âge
        def categorize_age(age):
            if age < 25:
                return 'Young'
            elif age < 40:
                return 'Adult'
            elif age < 60:
                return 'Middle-aged'
            else:
                return 'Senior'
        
        df_analysis['Age_Category'] = df_analysis['Age'].apply(categorize_age)
        
        # Ratio Spending/Income
        df_analysis['Spending_Income_Ratio'] = df_analysis['Spending Score (1-100)'] / df_analysis['Annual Income (k$)']
        
        print("Nouvelles variables créées:")
        print(f"- Age_Category: {df_analysis['Age_Category'].value_counts().to_dict()}")
        print(f"- Spending_Income_Ratio: moyenne = {df_analysis['Spending_Income_Ratio'].mean():.3f}")
        
        print("✅ Étape 3 terminée avec succès!")
        
    except Exception as e:
        print(f"❌ Erreur étape 3: {e}")
    
    # ÉTAPE 4: PCA
    print("\n📊 ÉTAPE 4/7: Analyse en composantes principales")
    print("-" * 50)
    
    try:
        # Sélection des variables pour la PCA
        features_for_pca = numeric_features
        X = df_analysis[features_for_pca]
        
        # Standardisation
        scaler_pca = StandardScaler()
        X_scaled = scaler_pca.fit_transform(X)
        
        # Application de la PCA
        pca = PCA()
        X_pca = pca.fit_transform(X_scaled)
        
        print(f"Variance expliquée par chaque composante:")
        for i, var_exp in enumerate(pca.explained_variance_ratio_):
            print(f"PC{i+1}: {var_exp:.3f} ({var_exp*100:.1f}%)")
        
        # Visualisation PCA 2D
        pca_2d = PCA(n_components=2)
        X_pca_2d = pca_2d.fit_transform(X_scaled)
        
        plt.figure(figsize=(10, 6))
        plt.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], 
                   c=df_analysis['Gender_encoded'], cmap='viridis', alpha=0.7)
        plt.xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}%)')
        plt.ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}%)')
        plt.title('Projection PCA 2D (colorée par Genre)')
        plt.colorbar(label='Genre (0=Male, 1=Female)')
        plt.grid(True, alpha=0.3)
        plt.savefig('plots/pca_2d.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Étape 4 terminée avec succès!")
        
    except Exception as e:
        print(f"❌ Erreur étape 4: {e}")
    
    # ÉTAPE 5: Clustering
    print("\n📊 ÉTAPE 5/7: Analyse de clustering")
    print("-" * 50)
    
    try:
        # Préparation des données pour le clustering
        features_for_clustering = numeric_features
        X_clustering = df_analysis[features_for_clustering]
        
        # Standardisation
        scaler_clustering = StandardScaler()
        X_scaled_clustering = scaler_clustering.fit_transform(X_clustering)
        
        # Méthode du coude et scores de silhouette
        k_range = range(2, 7)
        inertias = []
        silhouette_scores = []
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X_scaled_clustering)
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(X_scaled_clustering, kmeans.labels_))
        
        # Visualisation
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
        
        # Sélection du k optimal
        optimal_k = k_range[np.argmax(silhouette_scores)]
        print(f"Nombre optimal de clusters: {optimal_k}")
        print(f"Score de silhouette optimal: {max(silhouette_scores):.3f}")
        
        # Clustering final
        kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        cluster_labels = kmeans_final.fit_predict(X_scaled_clustering)
        
        # Ajout des labels
        df_analysis['Cluster'] = cluster_labels
        
        print("✅ Étape 5 terminée avec succès!")
        
    except Exception as e:
        print(f"❌ Erreur étape 5: {e}")
        return
    
    # ÉTAPE 6: Interprétation des clusters
    print("\n📊 ÉTAPE 6/7: Interprétation des clusters")
    print("-" * 50)
    
    try:
        # Profils des clusters
        cluster_profiles = df_analysis.groupby('Cluster')[features_for_clustering].agg(['mean', 'count'])
        print("\nProfils des clusters:")
        print(cluster_profiles.round(2))
        
        # Visualisation des clusters
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Visualisation des Clusters (k={optimal_k})', fontsize=16)
        
        # PCA colorée par cluster
        scatter = axes[0,0].scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], 
                                   c=df_analysis['Cluster'], cmap='viridis', alpha=0.7)
        axes[0,0].set_xlabel('PC1')
        axes[0,0].set_ylabel('PC2')
        axes[0,0].set_title('Clusters dans l\'espace PCA')
        plt.colorbar(scatter, ax=axes[0,0])
        
        # Age vs Income
        scatter = axes[0,1].scatter(df_analysis['Annual Income (k$)'], df_analysis['Age'], 
                                   c=df_analysis['Cluster'], cmap='viridis', alpha=0.7)
        axes[0,1].set_xlabel('Revenu Annuel (k$)')
        axes[0,1].set_ylabel('Âge')
        axes[0,1].set_title('Clusters: Âge vs Revenu')
        plt.colorbar(scatter, ax=axes[0,1])
        
        # Income vs Spending Score
        scatter = axes[1,0].scatter(df_analysis['Annual Income (k$)'], df_analysis['Spending Score (1-100)'], 
                                   c=df_analysis['Cluster'], cmap='viridis', alpha=0.7)
        axes[1,0].set_xlabel('Revenu Annuel (k$)')
        axes[1,0].set_ylabel('Score de Dépense')
        axes[1,0].set_title('Clusters: Revenu vs Score de Dépense')
        plt.colorbar(scatter, ax=axes[1,0])
        
        # Age vs Spending Score
        scatter = axes[1,1].scatter(df_analysis['Age'], df_analysis['Spending Score (1-100)'], 
                                   c=df_analysis['Cluster'], cmap='viridis', alpha=0.7)
        axes[1,1].set_xlabel('Âge')
        axes[1,1].set_ylabel('Score de Dépense')
        axes[1,1].set_title('Clusters: Âge vs Score de Dépense')
        plt.colorbar(scatter, ax=axes[1,1])
        
        plt.tight_layout()
        plt.savefig('plots/cluster_visualization.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Étape 6 terminée avec succès!")
        
    except Exception as e:
        print(f"❌ Erreur étape 6: {e}")
    
    # ÉTAPE 7: Conclusions et recommandations
    print("\n📊 ÉTAPE 7/7: Conclusions et recommandations")
    print("-" * 50)
    
    try:
        print("=== CONCLUSIONS DE L'ANALYSE ===")
        print(f"• Dataset analysé: {len(df_analysis)} clients")
        print(f"• Nombre de clusters identifiés: {optimal_k}")
        print(f"• Score de silhouette: {max(silhouette_scores):.3f}")
        
        # Profils des segments
        print(f"\n=== PROFILS DES SEGMENTS ===")
        for cluster in sorted(df_analysis['Cluster'].unique()):
            cluster_data = df_analysis[df_analysis['Cluster'] == cluster]
            size = len(cluster_data)
            percentage = (size / len(df_analysis)) * 100
            
            print(f"\n📊 CLUSTER {cluster} ({size} clients - {percentage:.1f}%)")
            print(f"   Âge moyen: {cluster_data['Age'].mean():.1f} ans")
            print(f"   Revenu moyen: {cluster_data['Annual Income (k$)'].mean():.1f}k$")
            print(f"   Score dépense moyen: {cluster_data['Spending Score (1-100)'].mean():.1f}/100")
            
            # Recommandations basiques
            avg_income = cluster_data['Annual Income (k$)'].mean()
            avg_spending = cluster_data['Spending Score (1-100)'].mean()
            
            if avg_income > 70 and avg_spending > 60:
                print("   🎯 Stratégie: Segment PREMIUM - Produits haut de gamme")
            elif avg_income > 70 and avg_spending < 40:
                print("   🎯 Stratégie: POTENTIEL ÉLEVÉ - Stimuler les achats")
            elif avg_income < 40 and avg_spending > 60:
                print("   🎯 Stratégie: IMPULSIF - Produits abordables")
            elif avg_income < 40 and avg_spending < 40:
                print("   🎯 Stratégie: ÉCONOME - Prix compétitifs")
            else:
                print("   🎯 Stratégie: ÉQUILIBRÉ - Offre diversifiée")
        
        # Sauvegarde des données
        df_analysis.to_csv('data/clustered_data.csv', index=False)
        
        print("\n✅ Étape 7 terminée avec succès!")
        
    except Exception as e:
        print(f"❌ Erreur étape 7: {e}")
    
    print(f"\n🎉 ANALYSE COMPLÈTE TERMINÉE!")
    print("=" * 60)
    print("📁 Fichiers générés:")
    print("   • data/clustered_data.csv : Données avec clusters")
    print("   • plots/ : Toutes les visualisations")

if __name__ == "__main__":
    run_complete_analysis()
