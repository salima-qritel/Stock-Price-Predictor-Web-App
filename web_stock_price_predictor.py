import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

# -------------------------
# Design amélioré avec fond sombre
# -------------------------
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, rgba(60, 0, 85, 0.8), rgba(90, 45, 130, 0.8));  /* Violet sombre avec un peu de transparence */
        backdrop-filter: blur(8px);  /* Effet flou pour un fond dégradé plus doux */
        color: #FFFFFF;  /* Texte en blanc */
    }
    </style>
    """,
    unsafe_allow_html=True
)




# -------------------------
# Titre de l'application
# -------------------------
st.title("📈 Stock Price Predictor App")

# -------------------------
# Entrée de l'utilisateur
# -------------------------
stock = st.text_input("🔍 Entrez le ticker de l'action", "GOOG")

# Sélection de la période
defaut_end = datetime.now()
defaut_start = datetime(defaut_end.year - 10, defaut_end.month, defaut_end.day)
start = st.date_input("📅 Date de début", defaut_start)
end = st.date_input("📅 Date de fin", defaut_end)

if start >= end:
    st.error("La date de début doit être antérieure à la date de fin.")
    st.stop()

# Téléchargement des données
data = yf.download(stock, start, end)
if data.empty:
    st.error("Aucune donnée disponible pour ce ticker. Vérifiez l'orthographe ou la période.")
    st.stop()

st.subheader("Données téléchargées")
st.write(data)

# Vérification de la colonne 'Close'
if "Close" in data.columns:
    close_col = "Close"
elif "Adj Close" in data.columns:
    close_col = "Adj Close"
else:
    st.error("Les colonnes 'Close' ou 'Adj Close' sont introuvables.")
    st.stop()

# -------------------------
# Sélection de moyennes mobiles
# -------------------------
options_ma = st.multiselect("Sélectionnez les moyennes mobiles à afficher", [100, 200, 250], default=[100, 200])
colors = ['magenta', 'yellow', 'lime']  # Couleurs pour chaque MA

for ma in options_ma:
    data[f'MA_{ma}'] = data[close_col].rolling(ma).mean()

# Affichage des moyennes mobiles
st.subheader("Prix de clôture et moyennes mobiles")
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(data[close_col], label="Prix de clôture", color='cyan')

for i, ma in enumerate(options_ma):
    ax.plot(data[f'MA_{ma}'], label=f'MA {ma} jours', linestyle='dashed', color=colors[i % len(colors)])

ax.legend()
st.pyplot(fig)

# -------------------------
# Normalisation et préparation des données
# -------------------------
splitting_len = int(len(data) * 0.7)
x_test = data[[close_col]][splitting_len:]
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(x_test)

x_data, y_data = [], []
for i in range(100, len(scaled_data)):
    x_data.append(scaled_data[i-100:i])
    y_data.append(scaled_data[i])
x_data, y_data = np.array(x_data), np.array(y_data)

# -------------------------
# Prédiction avec le modèle
# -------------------------
model = load_model("Latest_stock_price_model.keras")
predictions = model.predict(x_data)
inv_pre = scaler.inverse_transform(predictions)
inv_y_test = scaler.inverse_transform(y_data)

# -------------------------
# Affichage des prédictions
# -------------------------
st.subheader("Prédictions vs Valeurs réelles")
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(inv_y_test, label="Valeurs réelles", color='cyan')
ax.plot(inv_pre, label="Prédictions", color='magenta')
ax.legend()
st.pyplot(fig)

st.success("Analyse terminée !")
