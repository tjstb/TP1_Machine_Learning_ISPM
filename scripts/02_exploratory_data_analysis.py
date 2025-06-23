import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def perform_eda():
    """
    Effectue l'analyse exploratoire des données (EDA)
    """
    # Chargement des données
    df = pd.read_csv('data/processed_data.csv')
    
    # Configuration des graphiques
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. ANALYSE UNIVARIÉE
    print("=== ANALYSE UNIVARIÉE ===")
    
    # Variables numériques
    numeric_vars = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
    
    # Histogrammes et boxplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Analyse Univariée - Variables Numériques', fontsize=16)
    
    for i, var in enumerate(numeric_vars):
        # Histogrammes
        axes[0, i].hist(df[var], bins=20, alpha=0.7, edgecolor='black')
        axes[0, i].set_title(f'Distribution de {var}')
        axes[0, i].set_xlabel(var)
        axes[0, i].set_ylabel('Fréquence')
        
        # Boxplots
        axes[1, i].boxplot(df[var])
        axes[1, i].set_title(f'Boxplot de {var}')
        axes[1, i].set_ylabel(var)
    
    plt.tight_layout()
    plt.savefig('plots/univariate_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Analyse de la variable Gender
    plt.figure(figsize=(8, 6))
    gender_counts = df['Gender'].value_counts()
    plt.pie(gender_counts.values, labels=gender_counts.index, autopct='%1.1f%%')
    plt.title('Répartition par Genre')
    plt.savefig('plots/gender_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Statistiques descriptives détaillées
    print("\nStatistiques descriptives:")
    for var in numeric_vars:
        print(f"\n{var}:")
        print(f"  Moyenne: {df[var].mean():.2f}")
        print(f"  Médiane: {df[var].median():.2f}")
        print(f"  Écart-type: {df[var].std():.2f}")
        print(f"  Asymétrie: {stats.skew(df[var]):.2f}")
        print(f"  Aplatissement: {stats.kurtosis(df[var]):.2f}")
    
    # 2. ANALYSE MULTIVARIÉE
    print("\n=== ANALYSE MULTIVARIÉE ===")
    
    # Matrice de corrélation
    correlation_matrix = df[numeric_vars + ['Gender_encoded']].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, fmt='.3f')
    plt.title('Matrice de Corrélation')
    plt.tight_layout()
    plt.savefig('plots/correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Scatterplot matrix
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle('Matrice de Scatterplots', fontsize=16)
    
    for i, var1 in enumerate(numeric_vars):
        for j, var2 in enumerate(numeric_vars):
            if i == j:
                # Diagonale: histogrammes
                axes[i, j].hist(df[var1], bins=20, alpha=0.7)
                axes[i, j].set_title(f'{var1}')
            else:
                # Scatterplots colorés par genre
                for gender in df['Gender'].unique():
                    mask = df['Gender'] == gender
                    axes[i, j].scatter(df.loc[mask, var2], df.loc[mask, var1], 
                                     alpha=0.6, label=gender)
                axes[i, j].set_xlabel(var2)
                axes[i, j].set_ylabel(var1)
                axes[i, j].legend()
    
    plt.tight_layout()
    plt.savefig('plots/scatterplot_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Analyse par genre
    print("\nAnalyse par genre:")
    for var in numeric_vars:
        print(f"\n{var} par genre:")
        gender_stats = df.groupby('Gender')[var].agg(['mean', 'std', 'median'])
        print(gender_stats)
    
    # Tests statistiques
    print("\nTests de différence entre genres (t-test):")
    for var in numeric_vars:
        male_data = df[df['Gender'] == 'Male'][var]
        female_data = df[df['Gender'] == 'Female'][var]
        t_stat, p_value = stats.ttest_ind(male_data, female_data)
        print(f"{var}: t-statistic = {t_stat:.3f}, p-value = {p_value:.3f}")
    
    return correlation_matrix

if __name__ == "__main__":
    # Créer le dossier plots s'il n'existe pas
    import os
    os.makedirs('plots', exist_ok=True)
    
    correlation_matrix = perform_eda()
    print("\nAnalyse exploratoire terminée!")
