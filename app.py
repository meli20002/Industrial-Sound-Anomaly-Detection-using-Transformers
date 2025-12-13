# app_simple.py

import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import io
# Import des utilitaires du Simple Classifier
from model_utils import load_simple_classifier, predict_simple_classifier, SAMPLE_RATE, N_MFCC, N_FFT, HOP_LENGTH, preprocess_mfcc

# --- Configuration et Chargement du Modèle (avec mise en cache) ---

st.set_page_config(layout="wide", page_title="Détection d'Anomalies Sonores (Simple MLP)")

@st.cache_resource 
def load_the_model():
    """Fonction wrapper pour le chargement du modèle Simple Classifier."""
    return load_simple_classifier() 

SIMPLE_MODEL = load_the_model()

# --- Fonctions de Visualisation ---

def plot_waveform(audio_data, sr):
    """Crée et retourne un plot Matplotlib de la forme d'onde."""
    fig, ax = plt.subplots(figsize=(4, 3)) # Taille ajustée pour 3 colonnes
    librosa.display.waveshow(audio_data, sr=sr, ax=ax)
    ax.set_title("1. Forme d'Onde")
    plt.tight_layout()
    return fig

def plot_log_mel_spectrogram(audio_data, sr):
    """Crée et retourne un plot Matplotlib du Spectrogramme Log-Mel."""
    audio_resampled = librosa.resample(audio_data, orig_sr=sr, target_sr=SAMPLE_RATE)
    mel_spectrogram = librosa.feature.melspectrogram(
        y=audio_resampled, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    
    fig, ax = plt.subplots(figsize=(4, 3)) # Taille ajustée
    img = librosa.display.specshow(log_mel_spectrogram, x_axis='time', y_axis='mel', ax=ax, 
                                   sr=SAMPLE_RATE)
    ax.set_title("2. Spectrogramme Log-Mel")
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    plt.tight_layout()
    return fig

def plot_raw_mfcc(mfcc_features, sr):
    """Crée et retourne un plot Matplotlib des MFCC bruts (non padés/tronqués pour l'exploration)."""
    fig, ax = plt.subplots(figsize=(4, 3)) # Taille ajustée
    
    # Affichage des MFCC (y_axis='linear' ou 'none' est habituel pour les coefficients)
    img = librosa.display.specshow(mfcc_features, x_axis='time', y_axis='linear', ax=ax, sr=sr)
    
    ax.set_title("3. Coefficients MFCC (Bruts)")
    fig.colorbar(img, ax=ax, format="%+2.0f")
    plt.tight_layout()
    return fig

# Le plot_features original pour l'input du modèle
def plot_model_input_features(features, title):
    """Crée et retourne un plot Matplotlib des features (MFCC) pour le modèle (padées/tronquées)."""
    fig, ax = plt.subplots(figsize=(8, 3))
    img = librosa.display.specshow(features, x_axis='time', y_axis='linear', ax=ax, 
                                   sr=SAMPLE_RATE)
    ax.set_title(title)
    fig.colorbar(img, ax=ax, format="%+2.0f")
    plt.tight_layout() 
    return fig


# --- Interface Utilisateur ---

st.title("Projet Master: Anomaly Sound Detection on Fan")
st.subheader("Simple MLP classifier with MFCC features extractor")

# 1. Section Téléchargement et Lecture
st.sidebar.header("1. Upload your Audio (.wav)")
uploaded_file = st.sidebar.file_uploader("chose an audio file", type=['wav'])

if uploaded_file is not None:
    audio_bytes = uploaded_file.read()
    st.sidebar.audio(audio_bytes, format='audio/wav')
    
    try:
        audio_data, sr = librosa.load(io.BytesIO(audio_bytes), sr=None) 
    except Exception as e:
        st.error(f"Erreur lors du chargement de l'audio: {e}.")
        st.stop()
        
    # --- PRÉ-CALCUL DES MFCC BRUTS POUR L'EXPLORATION ---
    # Nous utilisons la logique de preprocess_mfcc mais on charge juste les MFCC numpy 
    # avant le padding/truncation pour l'affichage brut.
    mfcc_brut = librosa.feature.mfcc(y=audio_data, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
        
    # --- EXPLORATION AUDIO ---
    st.header("2. Exploration du Signal Audio et des Features")
    col_wave, col_logmel, col_mfcc_raw = st.columns(3) # 3 colonnes pour l'exploration
    
    with col_wave:
        fig_wave = plot_waveform(audio_data, sr)
        st.pyplot(fig_wave)
        
    with col_logmel:
        fig_logmel = plot_log_mel_spectrogram(audio_data, sr)
        st.pyplot(fig_logmel)
        
    with col_mfcc_raw:
        fig_mfcc_raw = plot_raw_mfcc(mfcc_brut, sr)
        st.pyplot(fig_mfcc_raw)


    # --- PRÉDICTION et FEATURES (Input du Modèle) ---
    st.header("3. Prédiction et Features du Modèle")
    
    col_pred, col_viz = st.columns([1, 2])
    
    # --- PRÉDICTION DU MODÈLE ---
    with col_pred:
        st.markdown("### Simple Classifieur MLP (MFCC)")
        
        # Ici, on utilise la fonction qui fait le pré-traitement complet (padding/truncation)
        prob_mlp, mfcc_features_model = predict_simple_classifier(audio_data, sr, SIMPLE_MODEL) 
        
        # Seuil d'anomalie
        SEUIL_MLP = 0.50 
        
        st.metric(label="Probabilité d'Anomalie (Sortie du MLP)", value=f"{prob_mlp*100:.2f}%")
        
        if prob_mlp >= SEUIL_MLP:
             st.error(" **ANOMALIE detected **")
        else:
             st.success("**no anomaly **")

        st.markdown("---")
        st.info(f"Le modèle a classifié ce son sur la base de {N_MFCC} coefficients MFCC, formatés à 40x200.")

    # --- Visualisation des Features Utilisées ---
    with col_viz:
        st.markdown("### Coefficients MFCC (Input du Modèle après Traitement)")
        
        # On utilise la nouvelle fonction de plot pour l'input final
        fig_input_mlp = plot_model_input_features(mfcc_features_model, "MFCC 40x200 Normalisés (Input Modèle)")
        st.pyplot(fig_input_mlp)