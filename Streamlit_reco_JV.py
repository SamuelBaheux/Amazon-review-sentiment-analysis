import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from nlp_script import *

# Définir un thème personnalisé
custom_theme = """
    [theme]
    primaryColor = "#008080"
    backgroundColor = "#f0f2f6"
    secondaryBackgroundColor = "#edf0f2"
    textColor = "#262730"
    font = "sans serif"
"""

# Appliquer le thème personnalisé
st.markdown(f'<style>{custom_theme}</style>', unsafe_allow_html=True)

# Titre
st.title("Modèle Streamlit Joli")

# Sous-titre
st.markdown("Ceci est un exemple de modèle Streamlit avec un thème personnalisé.")

# Affichage de texte
st.write("Vous pouvez ajouter du texte, des graphiques et d'autres éléments interactifs à cette interface.")

# Ajout de graphiques
st.subheader("Graphique de distribution aléatoire")
data = np.random.randn(1000)
fig, ax = plt.subplots()
sns.histplot(data, ax=ax, kde=True, color='skyblue')
st.pyplot(fig)

# Ajout d'un dataframe
st.subheader("Exemple de dataframe")
data = {'Nom': ['Alice', 'Bob', 'Charlie', 'David'],
        'Âge': [25, 30, 35, 40],
        'Score': [85, 90, 88, 92]}
df = pd.DataFrame(data)
st.write(df)

# Ajout d'un bouton
if st.button("Cliquez-moi"):
    st.write("Le bouton a été cliqué!")
