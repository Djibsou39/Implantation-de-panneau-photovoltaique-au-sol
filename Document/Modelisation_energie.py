# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 17:32:47 2024

@author: Projet tutoré groupe cadastre solaire
"""
#----------------------------------------------------------------------------------------------------------------
# Importation des bibliotheques
#----------------------------------------------------------------------------------------------------------------
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

#  Définition du chemin d'accès
path="D:/LPGGAT/PtutGGAT/Collectif/LDHD/Montegut/prediction"

os.chdir(path) # changement du chemin de dossier

# Definition du nom de fichier energie produite Ordan larroque
energie_prod="energie_produite_ordan_larroque.xlsx"

# concatenation des path et du file
path_energie_prod=os.path.join(path,energie_prod)

# Importation de fichiers shp dans des dataframes
df_ml=pd.read_excel(path_energie_prod)

df_ml.info()
#----------------------------------------------------------------------------------------------------------------
# Analyse bivariée entre energie produite et orientation de toitures
#----------------------------------------------------------------------------------------------------------------

# Séparez les caractéristiques et la cible
X = df_ml['orientation'].values.reshape(-1, 1)
y = df_ml['janvier']

# Créez une instance de transformation polynomiale
degree = 2  # Degré du polynôme
poly = PolynomialFeatures(degree)
X_poly = poly.fit_transform(X)

# Créez et entraînez un modèle de régression linéaire
model = LinearRegression()
model.fit(X_poly, y)

# Générez des prédictions sur une plage de valeurs X
X_values = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
X_values_poly = poly.transform(X_values)
y_pred = model.predict(X_values_poly)

# Calculez le coefficient de détermination R²
y_pred_train = model.predict(X_poly)
r2 = r2_score(y, y_pred_train)

# Tracez les données réelles et les prédictions
plt.figure(figsize=(10, 6))
plt.scatter(X, y, label='Production énergie', color='blue')
plt.plot(X_values, y_pred, color='red', label=f'Régression polynomiale (degré {degree}) (R² = {r2:.2f})')
plt.xlabel('Orientation de toits')
plt.ylabel('Energie en janvier')
plt.title('Régression polynomiale')
plt.legend()
plt.show()

df_ml.columns
#----------------------------------------------------------------------------------------------------------------
# Apprentissage automatique du modele
#----------------------------------------------------------------------------------------------------------------
# Selection des mois
colonne_mois =['janvier', 'fevrier', 'mars', 'avril', 'mai', 'juin', 'juillet', 'aout', 'septembre', 'octobre', 'novembre', 'decembre']

# Séparez vos données en variables indépendantes (X) et la variable cible (y)
X = df_ml.drop(colonne_mois, axis=1)

# Créez une boucle pour prédire les valeurs pour chaque mois
for mois in colonne_mois:
    print("Le mois en cours de traitement:", mois)
    
    # Isoler les valeurs pour le mois en cours
    y = df_ml[mois]

    # Diviser les données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    # Créer une instance de RandomForestRegressor
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

    # Entraîner le modèle
    rf_model.fit(X_train, y_train)

    # Prédire les valeurs sur l'ensemble d'entraînement
    y_train_pred_rf = rf_model.predict(X_train)
    
    # Calculer le coefficient de détermination R² pour l'ensemble d'entraînement
    r2_train_rf = r2_score(y_train, y_train_pred_rf)

    # Tracez les données réelles et les prédictions
    plt.figure(figsize=(10, 6))
    plt.scatter(y_train, y_train_pred_rf, color='blue')
    plt.xlabel('train')
    plt.ylabel('predict')
    plt.grid()
    plt.text(25,160,"r²= "+str(round(r2_train_rf,2)))
    plt.title(mois)
    plt.legend(loc='upper left', bbox_to_anchor=(0, 1))
    plt.show()

    print("Coefficient de détermination R² :", round(r2_train_rf,2))

#----------------------------------------------------------------------------------------------------------------
# Prédictions sur les nouvelles données
#----------------------------------------------------------------------------------------------------------------
    # Charger les nouvelles données pour la prédiction de l'énergie produite
    df_pred = pd.read_excel("D:/LPGGAT/PtutGGAT/Collectif/LDHD/Montegut/prediction/data_montegut.xlsx")

    # Prétraiter les nouvelles données (sélectionner les mêmes caractéristiques que X et transformer)
    X_poly_pred = df_pred[X.columns]  # Garder les mêmes caractéristiques que dans X
    
       # Prédire les valeurs sur les nouvelles données
    y_pred = rf_model.predict(X_poly_pred)
    
    # Ajouter les valeurs prédites à une colonne 'volum_predite' dans df_pred
    df_pred[mois + '_pred'] = y_pred
    
    # Enregistrement des prédictions dans un fichier Excel
    output_filename = f"energie_pred_montegut_pred_{mois}.xlsx"
    output_filepath = os.path.join(path, output_filename)
    df_pred.to_excel(output_filepath, index=False)
    
########################################### C'est fantastique la programmation ################################################################