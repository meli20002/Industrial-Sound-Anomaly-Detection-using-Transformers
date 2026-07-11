# Industrial Sound Anomaly Detection using Transformers

A master's project exploring **audio-based anomaly detection for industrial machinery** (fan sounds), comparing a Transformer-based approach — fine-tuning the **Audio Spectrogram Transformer (AST)** — against a lightweight **MFCC + MLP classifier** baseline, with an interactive Streamlit demo for real-time inference.


## Overview

Detecting mechanical faults from sound is a classic predictive-maintenance problem: a normal fan produces smooth, stable airflow noise, while a faulty one introduces rattling, scraping, or imbalance artifacts that are often audible before a failure becomes critical. Manual listening doesn't scale, so this project frames the problem as a binary audio classification task (normal vs. anomalous) and explores two modeling strategies:

1. **Transformer approach** — fine-tuning [MIT's AST](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593) (pretrained on AudioSet) on log-Mel spectrograms of fan recordings.
2. **Baseline approach** — a simple MLP classifier trained on MFCC features, used as a lightweight point of comparison and as the model currently served in the demo app.

## Repository Structure

```
.
├── app.py                                                  # Streamlit web app (upload a .wav, get a prediction)
├── model_utils.py                                          # Model architecture, preprocessing, load/predict utilities
├── notebook.ipynb                                          # Data loading, EDA, spectrogram/MFCC exploration
├── fine_tuning_ast(on_fan_set).ipynb                       # Fine-tuning the AST transformer on the fan dataset
├── autoencoder_calssifier_supervised_(with_mfcc).ipynb     # MFCC feature extraction + MLP classifier training
├── anomaly_detection_model2.pth                            # Trained MLP classifier weights (used by the app)
├── images/
│   └── waveform_logmel_comparison.png
└── .gitignore
```

## Dataset

The project uses the **fan subset** of the [MIMII dataset](https://zenodo.org/record/3384388) (Malfunctioning Industrial Machine Investigation and Inspection), a public benchmark of normal/abnormal operating sounds recorded from real industrial machines. Recordings are organized by machine ID, each with `normal/` and `abnormal/` subfolders of 10-second `.wav` clips.

The dataset is not included in this repository (see `.gitignore`). Download it from Zenodo and place it locally, or adapt the download cells in `notebook.ipynb`.

## Approach

**Preprocessing.** Audio is resampled to 16 kHz mono. For the AST pipeline, clips are converted to 128-band log-Mel spectrograms; for the MLP baseline, 40 MFCC coefficients are extracted and z-score normalized, then padded/truncated to a fixed 40×200 shape.

**Models.**
- *AST fine-tuning*: the pretrained `MIT/ast-finetuned-audioset-10-10-0.4593` model is fine-tuned end-to-end on the fan dataset using the Hugging Face `Trainer` API.
- *MLP baseline*: a small feed-forward network (`Linear → BatchNorm → ReLU → Dropout`, 3 hidden layers) over flattened MFCC features, trained with standard supervised classification.

**Evaluation** uses accuracy, precision, recall, F1, and confusion matrices (see `fine_tuning_ast(on_fan_set).ipynb` and `autoencoder_calssifier_supervised_(with_mfcc).ipynb`).

## Demo App

An interactive Streamlit app lets you upload a `.wav` file and see:
- the waveform, log-Mel spectrogram, and raw MFCC visualizations of the uploaded audio,
- the MLP classifier's anomaly probability and prediction.

### Running locally

```bash
# 1. Clone the repository
git clone https://github.com/meli20002/Industrial-Sound-Anomaly-Detection-using-Transformers.git
cd Industrial-Sound-Anomaly-Detection-using-Transformers

# 2. Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate   # on Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install streamlit torch torchaudio transformers librosa numpy pandas scikit-learn matplotlib seaborn tqdm evaluate gdown

# 4. Launch the app
streamlit run app.py
```

The app expects `anomaly_detection_model2.pth` to be present in the same directory (already included in this repo).

## Notebooks

| Notebook | Description |
|---|---|
| [`notebook.ipynb`](notebook.ipynb) | Dataset loading, exploratory data analysis, waveform/spectrogram/MFCC visualization, and the overall AST-based approach write-up. |
| [`fine_tuning_ast(on_fan_set).ipynb`](fine_tuning_ast%28on_fan_set%29.ipynb) | Fine-tuning the AST transformer on the fan subset of MIMII. |
| [`autoencoder_calssifier_supervised_(with_mfcc).ipynb`](autoencoder_calssifier_supervised_%28with_mfcc%29.ipynb) | MFCC feature extraction and training of the supervised MLP classifier (the model served by the demo app). |

## Tech Stack

- **Modeling:** PyTorch, torchaudio, Hugging Face `transformers` (AST), scikit-learn
- **Audio processing:** librosa
- **App:** Streamlit
- **Tooling:** pandas, NumPy, Matplotlib, Seaborn, tqdm

## Roadmap / Ideas

- [ ] Serve the fine-tuned AST model (not just the MLP baseline) in the Streamlit app, with a model-selector toggle.
- [ ] Add automated evaluation scripts and a results table comparing AST vs. MLP on held-out test data.
- [ ] Track experiments (e.g., with Weights & Biases or MLflow) instead of ad hoc notebook cells.
- [ ] Add a `requirements.txt` for reproducible installs.
- [ ] Add unit tests for `model_utils.py` preprocessing functions.

## Author

Developed as part of a Master's-level Advanced Deep Learning project (November 2025).


