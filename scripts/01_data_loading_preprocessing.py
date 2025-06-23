import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_preprocess_data():
    """
    Charge et prétraite les données des clients du centre commercial
    """
    # Chargement des données
    df = pd.read_csv('data/Mall_Customers.csv')
    
    print("=== INFORMATIONS SUR LE DATASET ===")
    print(f"Forme du dataset: {df.shape}")
    print(f"\nTypes de données:")
    print(df.dtypes)
    
    print(f"\nPremières lignes:")
    print(df.head())
    
    print(f"\nStatistiques descriptives:")
    print(df.describe())
    
    # Vérification des valeurs manquantes
    print(f"\nValeurs manquantes:")
    print(df.isnull().sum())
    
    # Vérification des doublons
    print(f"\nNombre de doublons: {df.duplicated().sum()}")
    
    # Préparation des données pour l'analyse
    # Suppression de CustomerID (non informatif pour le clustering)
    df_analysis = df.drop('CustomerID', axis=1)
    
    # Encodage de la variable Gender
    df_analysis['Gender_encoded'] = df_analysis['Gender'].map({'Male': 0, 'Female': 1})
    
    # Sélection des variables numériques pour la standardisation
    numeric_features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
    
    # Standardisation des variables quantitatives
    scaler = StandardScaler()
    df_scaled = df_analysis.copy()
    df_scaled[numeric_features] = scaler.fit_transform(df_analysis[numeric_features])
    
    print(f"\n=== DONNÉES APRÈS PRÉTRAITEMENT ===")
    print(f"Variables standardisées: {numeric_features}")
    print(f"Forme finale: {df_scaled.shape}")
    
    return df, df_analysis, df_scaled, scaler, numeric_features

if __name__ == "__main__":
    df_original, df_analysis, df_scaled, scaler, numeric_features = load_and_preprocess_data()
    
    # Sauvegarde des données prétraitées
    df_analysis.to_csv('data/processed_data.csv', index=False)
    df_scaled.to_csv('data/scaled_data.csv', index=False)
    
    print("\nDonnées prétraitées sauvegardées avec succès!")
