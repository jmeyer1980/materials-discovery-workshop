import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import random
import time

# Simple VAE class (same as in notebook)
class SimpleVAE(nn.Module):
    def __init__(self, input_dim=6, latent_dim=5):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU())
        self.fc_mu = nn.Linear(16, latent_dim)
        self.fc_var = nn.Linear(16, latent_dim)
        self.decoder = nn.Sequential(nn.Linear(latent_dim, 16), nn.ReLU(), nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, input_dim), nn.Sigmoid())

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_var(h)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var

@st.cache_resource
def load_and_train_model():
    # Quick training for demo
    data = pd.read_csv('materials_dataset.csv')
    binary_data = data[data['alloy_type'] == 'binary']
    features = binary_data[['composition_1', 'composition_2', 'melting_point', 'density', 'electronegativity', 'atomic_radius']].values
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    model = SimpleVAE()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    features_tensor = torch.FloatTensor(features_scaled)

    model.train()
    for _ in range(20):  # Quick training
        reconstructed, mu, log_var = model(features_tensor)
        loss = nn.functional.mse_loss(reconstructed, features_tensor) - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model, scaler

st.title("🧪 Materials Discovery ML Generator")
st.markdown("Generate new alloy compositions using machine learning!")

# Sidebar
st.sidebar.header("Generation Parameters")
num_samples = st.sidebar.slider("Number of materials to generate", 5, 100, 20)
generate_button = st.sidebar.button("Generate New Materials")

if generate_button:
    with st.spinner("Training model and generating materials..."):
        model, scaler = load_and_train_model()

        # Generate
        model.eval()
        with torch.no_grad():
            z = torch.randn(num_samples, 5)
            generated = model.decode(z).numpy()
            generated = scaler.inverse_transform(generated)

        # Create materials
        elements = ['Al', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn']
        materials = []
        for i, feat in enumerate(generated):
            elem1, elem2 = random.sample(elements, 2)
            comp1 = max(0.1, min(0.9, feat[0]))
            comp2 = 1 - comp1
            materials.append({
                'Formula': f'{elem1}{comp1:.2f}{elem2}{comp2:.2f}',
                'Melting Point (K)': round(abs(feat[2]), 1),
                'Density (g/cm³)': round(abs(feat[3]), 3),
                'Electronegativity': round(max(0, feat[4]), 2)
            })

        df = pd.DataFrame(materials)

        # Display results
        st.success(f"Generated {num_samples} new materials!")
        st.dataframe(df)

        # Quick plot
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.scatter(df['Density (g/cm³)'], df['Melting Point (K)'], alpha=0.7, c='red', s=50)
        ax.set_xlabel('Density (g/cm³)')
        ax.set_ylabel('Melting Point (K)')
        ax.set_title('Generated Materials Properties')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

st.markdown("---")
st.markdown("**How it works:** This app uses a Variational Autoencoder (VAE) trained on existing alloy data to generate new material compositions. The generated materials follow learned patterns but explore new regions of the materials space.")
st.markdown("**Note:** Always validate ML-generated materials experimentally before fabrication!")
