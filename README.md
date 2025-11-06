# EEG-Based Lie Detection System

A machine learning approach to detect deception using Electroencephalogram (EEG) signals. This project analyzes brain activity patterns to distinguish between truthful and deceptive responses.

## Project Overview

This research project explores the potential of using EEG data for automated lie detection. By analyzing various frequency bands (delta, theta, alpha, beta, and gamma) and their relationships, we aim to identify neural signatures associated with deceptive behavior.

## Dataset

The dataset consists of EEG recordings from multiple subjects under different experimental conditions:

- Honest responses to true identity
- Deceitful responses to true identity
- Honest responses to fake identity
- Deceitful responses to fake identity

### Data Structure

```plaintext
dataset/
├── EEG_features_with_labels.csv
└── [Subject_ID]/
    ├── EEG_ExperimentBlock.HONEST_RESPONSE_TO_TRUE_IDENTITY_raw.fif
    ├── EEG_ExperimentBlock.DECEITFUL_RESPONSE_TO_TRUE_IDENTITY_raw.fif
    ├── EEG_ExperimentBlock.HONEST_RESPONSE_TO_FAKE_IDENTITY_raw.fif
    └── EEG_ExperimentBlock.DECEITFUL_RESPONSE_TO_FAKE_IDENTITY_raw.fif
```

## Features

- **Raw EEG Analysis**: Processing and visualization of raw EEG signals
- **Spectral Analysis**: Power spectral density analysis across different frequency bands
- **Feature Engineering**:  
  - Band power extraction (delta, theta, alpha, beta, gamma)
  - Power band ratios
  - Statistical features
- **Advanced Analysis**:  
  - Time-frequency analysis
  - Topographical mapping
  - Principal Component Analysis (PCA)
- **Machine Learning Pipeline**:  
  - Data preprocessing
  - Feature selection
  - Model training and validation
  - Performance evaluation

## Key Components

1. **Exploratory Data Analysis** (`eda.ipynb`)
   - Comprehensive data visualization
   - Statistical analysis
   - Feature relationship exploration
   - Demographic analysis

2. **Model Training** (`model_training.ipynb`)
   - Implementation of machine learning models
   - Cross-validation
   - Performance metrics
   - Model evaluation

3. **Data Preparation** (`prepare_data.py`)
   - Data preprocessing
   - Feature extraction
   - Label preparation

## Prerequisites

Required Python packages:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- mne (for EEG signal processing)

## Installation

1. Clone the repository:

```bash
git clone https://github.com/gawlowski-mateusz/Lie-Detector.git
cd Lie-Detector
```

1. Install required packages:

```bash
pip install -r requirements.txt
```

## Usage

1. **Data Exploration**:

```bash
jupyter notebook eda.ipynb
```

1. **Model Training**:

```bash
jupyter notebook model_training.ipynb
```

## Key Findings

- Different EEG frequency bands show distinct patterns between truthful and deceptive responses
- Demographic factors (age and sex) may influence EEG patterns during deception
- Complex relationships exist between different frequency bands
- Statistical tests indicate significant differences in brain activity between truth-telling and lying
- PCA suggests potential separability between truthful and deceptive states
