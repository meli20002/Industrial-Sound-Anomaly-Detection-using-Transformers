# model_utils.py

import torch
import torch.nn as nn
import numpy as np
import librosa
import os
import io

# --------------------------------------------------------------------------------------
# --- 1. PARAMÈTRES DE PRÉTRAITEMENT EXACTS (MFCC) ---
# --------------------------------------------------------------------------------------
SAMPLE_RATE = 16000
N_MFCC = 40 
MAX_LEN_MFCC = 200 # Longueur temporelle max (frames)
N_FFT = 1024
HOP_LENGTH = 512

# Taille d'entrée pour le MLP : 40 * 200 = 8000
SIMPLE_CLASSIFIER_INPUT_DIM = N_MFCC * MAX_LEN_MFCC 

# --------------------------------------------------------------------------------------
# --- 2. ARCHITECTURE DU MODÈLE (Simple Classifier) ---
# --------------------------------------------------------------------------------------

class SimpleClassifier(nn.Module):
    def __init__(self, input_dim=SIMPLE_CLASSIFIER_INPUT_DIM):
        super(SimpleClassifier, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3), 

            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 64), 
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.classifier = nn.Linear(64, 2) 

    def forward(self, x):
        batch_size = x.size(0)
        
        # SÉCURITÉ : Retire la dimension Channel si elle existe
        if x.dim() > 2:
            x = x.squeeze(1) 
        
        # Aplatissement : [Batch, 40, 200] -> [Batch, 8000]
        x = x.reshape(batch_size, -1) 
        
        features = self.encoder(x)
        logits = self.classifier(features)
        return logits

# --------------------------------------------------------------------------------------
# --- 3. FONCTION DE PRÉTRAITEMENT MFCC ---
# --------------------------------------------------------------------------------------
def preprocess_mfcc(audio_data, sr):
    """Prétraitement MFCC pour le SimpleClassifier."""
    
    # 1. Resampling
    audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=SAMPLE_RATE)
    
    # 2. Compute MFCC
    mfcc = librosa.feature.mfcc(
        y=audio_data, sr=SAMPLE_RATE, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    
    # 3. Normalisation Z-Score
    mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-6)
    
    # 4. Pad / truncate
    if mfcc.shape[1] < MAX_LEN_MFCC:
        pad_len = MAX_LEN_MFCC - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_len)), mode="constant")
    else:
        mfcc = mfcc[:, :MAX_LEN_MFCC]

    # 5. Conversion en Tenseur : Utilisation de torch.tensor() pour garantir la contiguïté
    # Forme : [Batch=1, N_MFCC, Frames]
    mfccs_tensor = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0) 

    return mfcc, mfccs_tensor

# --------------------------------------------------------------------------------------
# --- 4. CHARGEMENT DU MODÈLE ---
# --------------------------------------------------------------------------------------
def load_simple_classifier():
    """Charge le SimpleClassifier entraîné depuis le disque."""
    
    CNN_PATH = "anomaly_detection_model2.pth" # Chemin vers le fichier de poids
    
    simple_model = SimpleClassifier()
    try:
        simple_model.load_state_dict(torch.load(CNN_PATH, map_location=torch.device('cpu')))
        simple_model.eval() # Mode évaluation initial
    except Exception as e:
        print(f"ERREUR FATALE: Le chargement du SimpleClassifier a échoué : {e}")
        
    return simple_model

# --------------------------------------------------------------------------------------
# --- 5. FONCTION DE PRÉDICTION ---
# --------------------------------------------------------------------------------------
@torch.no_grad()
def predict_simple_classifier(audio_data, sr, model):
    
    # 0. CORRECTION FINALE : Assurez-vous que le modèle est en mode évaluation
    model.eval()
    
    # 1. Prétraitement MFCC
    mfcc_np, input_tensor = preprocess_mfcc(audio_data, sr)
        
    # 2. Prédiction (forward pass)
    logits = model(input_tensor)
    
    # 3. Probabilité de la classe 'Anomalie' (indice 1)
    probabilities = torch.softmax(logits, dim=1)
    prob_anomalie = probabilities.squeeze().cpu().numpy()[1].item()
    
    return prob_anomalie, mfcc_np