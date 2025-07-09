import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# ----- Fonctions utilitaires -----
def generate_bits(n):
    return np.random.randint(0, 2, n)

def qam_modulate(bits):
    # QAM 16 : 4 bits par symbole
    bits = bits[:len(bits) // 4 * 4]  # Multiple de 4
    symbols = []

    for i in range(0, len(bits), 4):
        b = bits[i:i+4]
        real = 2 * (2 * b[0] + b[1]) - 3
        imag = 2 * (2 * b[2] + b[3]) - 3
        symbols.append(complex(real, imag))

    return np.array(symbols)

def add_noise(symbols, sigma):
    noise = (np.random.normal(0, sigma, symbols.shape)
             + 1j * np.random.normal(0, sigma, symbols.shape))
    return symbols + noise

def qam_demodulate(symbols):
    bits = []
    for s in symbols:
        r = np.real(s)
        i = np.imag(s)

        b0 = int(r > 0)
        b1 = int(abs(r) < 2)
        b2 = int(i > 0)
        b3 = int(abs(i) < 2)

        bits.extend([b0, b1, b2, b3])
    return np.array(bits)

def get_sigma(noise_level, interference, rain):
    levels = {"Faible": 0.1, "Moyen": 0.3, "Fort": 0.6}
    sigma_squared = levels[noise_level] ** 2
    sigma_squared += 0.5 * levels[interference] ** 2  # Interférence
    sigma_squared += 0.7 * levels[rain] ** 2          # Effet pluie
    return np.sqrt(sigma_squared)

# ----- Interface Streamlit -----
st.title("🌐 Simulation de transmission QAM 16 avec bruit")

st.sidebar.header("Paramètres environnementaux")
noise_level = st.sidebar.selectbox("Niveau de bruit", ["Faible", "Moyen", "Fort"])
interference_level = st.sidebar.selectbox("Niveau d'interférences", ["Faible", "Moyen", "Fort"])
rain_level = st.sidebar.selectbox("Niveau de pluie", ["Faible", "Moyen", "Fort"])

st.sidebar.markdown("---")
n_bits = st.sidebar.slider("Nombre de bits à transmettre", 1000, 100_000, 10_000, step=1000)

# ----- Simulation -----
bits_tx = generate_bits(n_bits)
symbols_tx = qam_modulate(bits_tx)
sigma = get_sigma(noise_level, interference_level, rain_level)
symbols_rx = add_noise(symbols_tx, sigma)
bits_rx = qam_demodulate(symbols_rx)

# Troncature pour égalité des tailles
n = min(len(bits_tx), len(bits_rx))
ber = np.sum(bits_tx[:n] != bits_rx[:n]) / n

# ----- Affichage constellation -----
fig, ax = plt.subplots()
ax.scatter(np.real(symbols_rx[:500]), np.imag(symbols_rx[:500]), alpha=0.5, s=10)
ax.set_title("Constellation QAM 16 bruitée (500 symboles)")
ax.set_xlabel("I (In-phase)")
ax.set_ylabel("Q (Quadrature)")
ax.grid(True)
st.pyplot(fig)

# ----- Résultats -----
st.subheader("📊 Résultats")
st.write(f"**Taux d'erreur binaire (TEB)** : `{ber:.4f}`")
st.write(f"**Variance du bruit effective (σ²)** : `{sigma**2:.4f}`")
