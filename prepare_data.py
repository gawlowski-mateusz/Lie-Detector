"""
Processes .fif EEG files for multiple participants and blocks, 
extracts bandpower features (delta, theta, alpha, beta, gamma), 
merges with demographic metadata, and labels each epoch according 
to experimental condition (truth vs lie).
"""

import os
import mne
import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
from mne.time_frequency import psd_array_welch

DATASET_DIRECTORY = "./dataset"
METADATA_FILE = "./dataset/Ankiety.xlsx"
OUTPUT_FILE = "./dataset/EEG_features_with_labels.csv"

# EEG frequency bands (Hz)
bands = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 45)
}

# Block mapping for experiment phases
block_mapping = {
    "HONEST_RESPONSE_TO_TRUE_IDENTITY": 1,    # Block 1: Truth about true identity
    "DECEITFUL_RESPONSE_TO_TRUE_IDENTITY": 2, # Block 2: Lie about true identity
    "HONEST_RESPONSE_TO_FAKE_IDENTITY": 3,    # Block 3: Truth about fake identity
    "DECEITFUL_RESPONSE_TO_FAKE_IDENTITY": 4  # Block 4: Lie about fake identity
}

# Helper functions
def is_control_trial(annotations):
    """Check if the trial is a control trial with known/random data."""
    return any('control' in str(annot).lower() for annot in annotations)

def validate_participant_data(uuid, files):
    """Verify that participant has all required experimental blocks."""
    required_blocks = [
        "HONEST_RESPONSE_TO_TRUE_IDENTITY",
        "DECEITFUL_RESPONSE_TO_TRUE_IDENTITY",
        "HONEST_RESPONSE_TO_FAKE_IDENTITY",
        "DECEITFUL_RESPONSE_TO_FAKE_IDENTITY"
    ]
    return all(any(block in f for f in files) for block in required_blocks)

def calculate_signal_quality(raw_data):
    """Calculate basic signal quality metrics."""
    return {
        "signal_mean": np.mean(raw_data),
        "signal_std": np.std(raw_data),
        "signal_noise_ratio": np.mean(np.abs(raw_data)) / np.std(raw_data)
    }

def bandpower_epoch(epoch_data, sfreq, band):
    """Compute relative bandpower for a single EEG epoch (np.array)."""
    low, high = band
    psd, freqs = psd_array_welch(
        epoch_data, sfreq=sfreq,
        fmin=low, fmax=high,
        n_fft=min(256, epoch_data.shape[-1]),  # avoid n_fft > n_times
        n_per_seg=None, verbose=False
    )
    psd = np.mean(psd, axis=0)
    band_power = trapezoid(psd, freqs)
    return band_power

def extract_features_from_epoch(data, sfreq):
    """Extract bandpower features from one EEG epoch (numpy array)."""
    features = {}
    
    # Basic band powers
    band_powers = {}
    for band_name, freq_range in bands.items():
        power = bandpower_epoch(data, sfreq, freq_range).mean()
        band_powers[band_name] = power
        features[f"{band_name}_power"] = power
    
    # Power ratios (useful for detecting cognitive states)
    features["theta_alpha_ratio"] = band_powers["theta"] / (band_powers["alpha"] + 1e-10)
    features["beta_alpha_ratio"] = band_powers["beta"] / (band_powers["alpha"] + 1e-10)
    features["gamma_beta_ratio"] = band_powers["gamma"] / (band_powers["beta"] + 1e-10)
    
    # Signal characteristics
    features["signal_complexity"] = np.std(data) / np.mean(np.abs(data))
    features["peak_frequency"] = np.argmax(np.abs(np.fft.fft(data))) * sfreq / len(data)
    
    return features

# Load Metadata
metadata = pd.read_excel(METADATA_FILE)
metadata.columns = ["UUID", "Plec", "Wiek", "Uwagi"]
metadata = metadata[~metadata["Uwagi"].astype(str).str.contains("zespute", case=False, na=False)]
print(f"Loaded metadata for {len(metadata)} valid participants.\n")

# Main EEG Processing
EXCLUDE_UUIDS = {"asd793jd", "6a517891"}

all_features = []
subject_folders = [f for f in os.listdir(DATASET_DIRECTORY) if os.path.isdir(os.path.join(DATASET_DIRECTORY, f))]
print(f"Found {len(subject_folders)} subject folders.\n")

for uuid in subject_folders:
    uuid_lower = uuid.lower()

    # Skip excluded/broken/expired participants
    if any(excluded in uuid_lower for excluded in EXCLUDE_UUIDS):
        print(f" -> Skipping excluded/broken subject: {uuid}")
        continue

    subject_path = os.path.join(DATASET_DIRECTORY, uuid)
    subject_metadata = metadata[metadata["UUID"].str.lower() == uuid_lower]

    # Load sex and age
    if not subject_metadata.empty:
        sex = subject_metadata.iloc[0]["Plec"]
        age = subject_metadata.iloc[0]["Wiek"]
    else:
        sex, age = None, None
        print(f" -> No metadata found for {uuid}")

    # Validate participant data
    subject_files = [f for f in os.listdir(subject_path) if f.endswith(".fif")]
    if not validate_participant_data(uuid, subject_files):
        print(f" -> Warning: Participant {uuid} missing some experimental blocks")
    
    # Process EEG files
    for fif_file in subject_files:
        fif_path = os.path.join(subject_path, fif_file)
        print(f"Processing: participant={uuid}, file={fif_file}")

        try:
            # Load EEG
            raw = mne.io.read_raw_fif(fif_path, preload=True, verbose=False)
            raw.pick("eeg")
            raw.filter(0.5, 45., fir_design='firwin', verbose=False)
            
            # Calculate signal quality metrics
            signal_quality = calculate_signal_quality(raw.get_data())

            # Extract epochs
            events, event_id = mne.events_from_annotations(raw, verbose=False)
            epochs = mne.Epochs(
                raw, events=events, event_id=None, tmin=-0.2, tmax=0.8,
                baseline=(None, 0), preload=True, verbose=False
            )
            epochs.drop_bad()
            data = epochs.get_data(picks="eeg")
            sfreq = epochs.info["sfreq"]

            # Determine condition & label
            if "HONEST_RESPONSE_TO_TRUE_IDENTITY" in fif_file:
                condition, label = "truth_true", 1
            elif "DECEITFUL_RESPONSE_TO_TRUE_IDENTITY" in fif_file:
                condition, label = "lie_true", 0
            elif "HONEST_RESPONSE_TO_FAKE_IDENTITY" in fif_file:
                condition, label = "lie_fake", 0
            elif "DECEITFUL_RESPONSE_TO_FAKE_IDENTITY" in fif_file:
                condition, label = "truth_fake", 1
            else:
                condition, label = "unknown", None

            # Extract features for all epochs
            for ep_data in data:
                feature = extract_features_from_epoch(ep_data, sfreq)
                # Get block number and check for control trial
                block_num = next((num for name, num in block_mapping.items() if name in fif_file), None)
                
                feature.update({
                    "UUID": uuid,
                    "File": fif_file,
                    "Block": block_num,
                    "Sex": sex,
                    "Age": age,
                    "Condition": condition,
                    "Label": label,
                    "Signal_Quality": signal_quality["signal_mean"],
                    "Signal_Noise": signal_quality["signal_std"],
                    "SNR": signal_quality["signal_noise_ratio"]
                })
                all_features.append(feature)

        except Exception as e:
            print(f"Error processing {fif_file}: {e}")
            continue


# Save all extracted features
df = pd.DataFrame(all_features)
df.to_csv(OUTPUT_FILE, index=False)
print(f"\nSaved {len(df)} feature rows to {OUTPUT_FILE}")
