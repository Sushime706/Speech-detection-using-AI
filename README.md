# Speech-Based Stress Detection

A machine learning project that classifies speech samples into **Calm** and **Stressed** categories using audio features and an LSTM neural network.

## Project overview

This project uses emotion-labelled speech recordings from the **CREMA-D** and **TESS** datasets. Audio is processed with Librosa, and the extracted features are used to train and evaluate an LSTM classifier.

The notebook also compares the LSTM with two baseline models:
- Random Forest
- Support Vector Machine (SVM)

## Tools and libraries

- Python
- TensorFlow / Keras
- Librosa
- Scikit-learn
- NumPy
- Matplotlib
- SoundFile
- tqdm
- Kaggle API (for downloading datasets)

## Audio features

The notebook extracts:
- 13 MFCC coefficients
- 13 delta MFCC coefficients
- 13 delta-delta MFCC coefficients

The feature sequence is prepared at a fixed length for the LSTM model.

## Model

The main model is an LSTM network built with TensorFlow/Keras. The notebook includes:
- Training, validation and test splits
- Feature standardisation
- Class weighting
- Early stopping and learning-rate reduction
- Model evaluation using a classification report, confusion matrix, ROC-AUC and inference-time measurement

## Datasets

The notebook downloads these datasets through Kaggle:
- [CREMA-D](https://www.kaggle.com/datasets/jayeshlandge/cremad)
- [Toronto Emotional Speech Set (TESS)](https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess)

You may need to accept the dataset terms and configure Kaggle API access before running the notebook.

### Kaggle credentials

Never upload your Kaggle API token or `kaggle.json` file to GitHub.

For Google Colab, configure your Kaggle credentials securely in the runtime or upload `kaggle.json` directly to the runtime (not to this repository). The notebook includes comments showing where credentials are expected.

## How to run

1. Open the notebook in Google Colab or Jupyter.
2. Install the required libraries from `requirements.txt`.
3. Configure Kaggle API access securely.
4. Run the notebook cells in order.

The notebook currently uses Google Colab-style paths such as `/content/datasets/`. If you run it locally, update those paths to match your dataset location.

## Repository contents

```text
speech-stress-detection/
├── speech_stress_detection.ipynb
├── README.md
├── requirements.txt
└── .gitignore
```

## Notes

- This is an academic/project implementation, not a clinically validated stress assessment tool.
- The labels are derived from emotion categories in the source datasets; they should not be interpreted as a medical diagnosis.
- Trained model files and datasets are not included in this repository. Run the notebook to generate model artifacts.

## Author

Arushi Tiwari
