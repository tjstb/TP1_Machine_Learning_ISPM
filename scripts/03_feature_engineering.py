import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def feature_engineering():
    """
    Effectue l'ingénierie des caractéristiques
    """
    # Chargement des données
    df = pd.read_csv('data/processed_data.csv')
    
    print("=== FEATURE ENGINEERING ===")
    
    # 1. Création de nouvelles variables
    
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
    
    df['Age_Category'] = df['Age'].apply(categorize_age)
    
    # Catégorisation du revenu
    income_quartiles = df['Annual Income (k$)'].quantile([0.25, 0.5, 0.75])
    def categorize_income(income):
        if income <= income_quartiles[0.25]:
            return 'Low'
        elif income <= income_quartiles[0.5]:
            return 'Medium-Low'
        elif income <= income_quartiles[0.75]:
            return 'Medium-High'
        else:
            return 'High'
    
    df['Income_Category'] = df['Annual Income (k$)'].apply(categorize_income)
    
    # Catégorisation du score de dépense
    spending_quartiles = df['Spending Score (1-100)'].quantile([0.25, 0.5, 0.75])
    def categorize_spending(score):
        if score <= spending_quartiles[0.25]:
            return 'Low'
        elif score <= spending_quartiles[0.5]:
            return 'Medium-Low'
        elif score <= spending_quartiles[0.75]:
            return 'Medium-High'
        else:
            return 'High'
    
    df['Spending_Category'] = df['Spending Score (1-100)'].apply(categorize_spending)
    
    # Création d'un ratio Spending/Income
    df['Spending_Income_Ratio'] = df['Spending Score (1-100)'] / df['Annual Income (k$)']
    
    print(f"Nouvelles variables créées:")
    print(f"- Age_Category: {df['Age_Category'].value_counts().to_dict()}")
    print(f"- Income_Category: {df['Income_Category'].value_counts().to_dict()}")
    print(f"- Spending_Category: {df['Spending_Category'].value_counts().to_dict()}")
    print(f"- Spending_Income_Ratio: moyenne = {df['Spending_Income_Ratio'].mean():.3f}")
    
    # 2. Visualisation des nouvelles variables
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Nouvelles Variables Créées', fontsize=16)
    
    # Age categories
    df['Age_Category'].value_counts().plot(kind='bar', ax=axes[0,0])
    axes[0,0].set_title('Distribution des Catégories d\'Âge')
    axes[0,0].set_xlabel('Catégorie d\'Âge')
    axes[0,0].set_ylabel('Nombre de Clients')
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # Income categories
    df['Income_Category'].value_counts().plot(kind='bar', ax=axes[0,1])
    axes[0,1].set_title('Distribution des Catégories de Revenu')
    axes[0,1].set_xlabel('Catégorie de Revenu')
    axes[0,1].set_ylabel('Nombre de Clients')
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # Spending categories
    df['Spending_Category'].value_counts().plot(kind='bar', ax=axes[1,0])
    axes[1,0].set_title('Distribution des Catégories de Dépense')
    axes[1,0].set_xlabel('Catégorie de Dépense')
    axes[1,0].set_ylabel('Nombre de Clients')
    axes[1,0].tick_params(axis='x', rotation=45)
    
    # Spending/Income ratio
    axes[1,1].hist(df['Spending_Income_Ratio'], bins=20, alpha=0.7, edgecolor='black')
    axes[1,1].set_title('Distribution du Ratio Dépense/Revenu')
    axes[1,1].set_xlabel('Ratio Dépense/Revenu')
    axes[1,1].set_ylabel('Fréquence')
    
    plt.tight_layout()
    plt.savefig('plots/feature_engineering.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. Sélection des variables pour le clustering
    # Variables numériques principales
    clustering_features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
    
    # Optionnel: ajouter le ratio et le genre encodé
    clustering_features_extended = clustering_features + ['Spending_Income_Ratio', 'Gender_encoded']
    
    print(f"\nVariables sélectionnées pour le clustering:")
    print(f"Version de base: {clustering_features}")
    print(f"Version étendue: {clustering_features_extended}")
    
    # Analyse de corrélation avec les nouvelles variables
    numeric_cols = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)', 
                   'Spending_Income_Ratio', 'Gender_encoded']
    
    correlation_extended = df[numeric_cols].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_extended, annot=True, cmap='coolwarm', center=0,
                square=True, fmt='.3f')
    plt.title('Matrice de Corrélation - Variables Étendues')
    plt.tight_layout()
    plt.savefig('plots/correlation_extended.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Sauvegarde des données avec nouvelles variables
    df.to_csv('data/engineered_data.csv', index=False)
    
    return df, clustering_features, clustering_features_extended

if __name__ == "__main__":
    df_engineered, features_basic, features_extended = feature_engineering()
    print("\nFeature engineering terminé!")
