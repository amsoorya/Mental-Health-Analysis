import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import welch
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import mne
import neurokit2 as nk
from scipy import signal
from PIL import Image
import io
import pyedflib
import base64
from io import StringIO

# Set page config and styling
st.set_page_config(
    page_title="Mental Health Assessment",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for improved aesthetics
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #ffffff;
        border-radius: 4px 4px 0px 0px;
        padding: 10px 16px;
        border: none;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4e9ed4;
        color: white;
    }
    .stButton>button {
        background-color: #4e9ed4;
        color: white;
        border-radius: 4px;
        border: none;
        padding: 8px 16px;
        font-weight: bold;
    }
    h1, h2, h3 {
        color: #2c3e50;
    }
    .highlight {
        background-color: #e3f2fd;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 5px solid #4e9ed4;
    }
    .metric-card {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        text-align: center;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Function to add a background image
def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://www.transparenttextures.com/patterns/cubes.png");
             background-attachment: fixed;
             background-size: cover;
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

add_bg_from_url()

# Sidebar styling
st.sidebar.markdown("<h2 style='text-align: center; color: #eff2f6;'>🧠 Navigation</h2>", unsafe_allow_html=True)
st.sidebar.markdown("<div style='text-align: center; padding: 10px; background-color: #083a5e; border-radius: 5px; margin-bottom: 15px;'>Mental Health Assessment Tool</div>", unsafe_allow_html=True)

# Main page header
st.markdown("<h1 style='text-align: center;'>Mental Health Assessment Platform</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-style: italic; color: #5a6268;'>Analyze EEG signals and health metrics to identify potential mental health conditions</p>", unsafe_allow_html=True)

# Display app description
with st.expander("ℹ️ About this application", expanded=True):
    st.markdown("""
    This application uses advanced machine learning models to analyze either:
    - EEG (electroencephalogram) signals
    - HRV (Heart Rate Variability) data
    - Manual input of important health metrics
    
    The application can help identify potential signs of:
    - Insomnia
    - Anxiety
    - Depression
    - ADHD
    - Sleep Apnea
    - Cognitive Decline
    
    **Disclaimer**: This tool is for informational purposes only and does not replace professional medical advice, diagnosis, or treatment.
    Always consult with a qualified healthcare provider for medical concerns.
    """)

# Load models
@st.cache_resource
def load_models():
    models = {}
    model_paths = {
        "Healthy": "C:\\Users\\JAYA SOORYA\\shhs\\healthy_cnn_lstm_model.joblib",
        "Anxiety": "C:\\Users\\JAYA SOORYA\\shhs\\anxiety_cnn_lstm_model.joblib",
        "Depression": "C:\\Users\\JAYA SOORYA\\shhs\\depression_cnn_lstm_model.joblib",
        "ADHD": "C:\\Users\\JAYA SOORYA\\shhs\\adhd_cnn_lstm_model.joblib",
        "Sleep Apnea": "C:\\Users\\JAYA SOORYA\\shhs\\apnea_cnn_lstm_model.joblib",
        "Cognitive Decline": "C:\\Users\\JAYA SOORYA\\shhs\\cognitive_decline_cnn_lstm_model.joblib"
    }
    
    for condition, path in model_paths.items():
        try:
            models[condition] = joblib.load(path)
            st.sidebar.success(f"✅ {condition} model loaded")
        except Exception as e:
            st.sidebar.error(f"❌ Failed to load {condition} model: {e}")
    
    return models

# Dictionary of features required for each condition
feature_sets = {
    "Insomnia": ['avg_orp_wake', 'avg_orp_nonrem', 'avg_orp_n1', 'sum_deciles_1and2'],
    
    "Anxiety": ['ihr', 'AVNN', 'SDNN', 'rMSSD', 'PNN50', 'LF_HF', 'IHR', 'SDNNIDX', 'pNN50', 
                'baseline_hr', 'delta_hr', 'dhr_intensity', 'change_in_hr_perh_trt'],
    
    "Depression": ['powers_C3A2_alpha', 'powers_C4A1_alpha', 'alpha_intrusion_pct_3sepochs', 
                  'avg_orp_n3', 'avg_orp_rem'],
    
    "ADHD": ['powers_C3A2_theta1', 'powers_C4A1_theta1', 'powers_C3A2_beta2', 
             'powers_C3A2_beta1', 'powers_C4A1_beta1', 'powers_C4A1_beta2'],
    
    "Sleep Apnea": ['pct_epoch_0to_0_25pct','pct_epoch_0_25to0_5pct', 'pct_epoch_0_5to0_75pct', 
                    'pct_epoch_0_75to1pct', 'pct_epoch_1to1_25pct', 'pct_epoch_1_25to1_5pct', 
                    'pct_epoch_1_5to1_75pct', 'pct_epoch_1_75to2pct', 'pct_epoch_2to2_25pct', 
                    'pct_epoch_2_25to2_5pct', 'orpto9'],
    
    "Cognitive Decline": ['powers_C3A2_delta', 'powers_C3A2_theta1', 'powers_C4A1_delta', 
                          'powers_C4A1_theta1', 'spindle_char_N2_density_C3', 'spindle_char_N2_power_C3', 
                          'spindle_char_N2_freq_C3', 'spindle_char_N2_pctfast_C3', 'spindle_char_N2_density_C4', 
                          'spindle_char_N2_power_C4', 'spindle_char_N2_freq_C4', 'spindle_char_N2_pctfast_C4', 
                          'avg_orp_n3', 'diff_orp', 'avg_normalized_EEG_power', 'avg_orp_n3_z', 
                          'avg_orp_n3_inv', 'powers_C3A2_theta1_z', 'powers_C4A1_theta1_z', 
                          'powers_C3A2_theta1_beta_ratio', 'powers_C4A1_theta1_beta_ratio']
}

# All unique features
all_features = list(set([feature for features in feature_sets.values() for feature in features]))

# Feature descriptions for tooltips and explanations
feature_descriptions = {
    # Insomnia features
    'avg_orp_wake': 'Average odds ratio product during wake periods (related to sleep quality)',
    'avg_orp_nonrem': 'Average odds ratio product during non-REM sleep (measures sleep depth)',
    'avg_orp_n1': 'Average odds ratio product during N1 sleep stage (light sleep)',
    'sum_deciles_1and2': 'Sum of EEG power in the lowest two deciles (related to insomnia)',
    
    # Anxiety features
    'ihr': 'Instantaneous heart rate (beats per minute)',
    'AVNN': 'Average of all normal-to-normal intervals (ms)',
    'SDNN': 'Standard deviation of normal-to-normal intervals (ms)',
    'rMSSD': 'Root mean square of successive differences between normal heartbeats (ms)',
    'PNN50': 'Percentage of successive normal-to-normal intervals > 50 ms',
    'LF_HF': 'Ratio of low frequency to high frequency power (autonomic balance)',
    'IHR': 'Instantaneous heart rate (alternative calculation)',
    'SDNNIDX': 'Mean of standard deviations of NN intervals for each 5-minute segment',
    'pNN50': 'Percentage of adjacent NN intervals that differ by more than 50ms',
    'baseline_hr': 'Baseline heart rate prior to interventions or events',
    'delta_hr': 'Change in heart rate from baseline',
    'dhr_intensity': 'Intensity of heart rate changes',
    'change_in_hr_perh_trt': 'Change in heart rate per hour of treatment',
    
    # Depression features
    'powers_C3A2_alpha': 'EEG alpha wave power at C3A2 electrode position',
    'powers_C4A1_alpha': 'EEG alpha wave power at C4A1 electrode position',
    'alpha_intrusion_pct_3sepochs': 'Percentage of alpha wave intrusion across 3 successive epochs',
    'avg_orp_n3': 'Average odds ratio product during N3 sleep stage (deep sleep)',
    'avg_orp_rem': 'Average odds ratio product during REM sleep',
    
    # ADHD features
    'powers_C3A2_theta1': 'EEG theta1 wave power at C3A2 electrode position',
    'powers_C4A1_theta1': 'EEG theta1 wave power at C4A1 electrode position',
    'powers_C3A2_beta2': 'EEG beta2 wave power at C3A2 electrode position',
    'powers_C3A2_beta1': 'EEG beta1 wave power at C3A2 electrode position',
    'powers_C4A1_beta1': 'EEG beta1 wave power at C4A1 electrode position',
    'powers_C4A1_beta2': 'EEG beta2 wave power at C4A1 electrode position',
    
    # Sleep Apnea features
    'pct_epoch_0to_0_25pct': 'Percentage of epochs with 0-0.25% desaturation',
    'pct_epoch_0_25to0_5pct': 'Percentage of epochs with 0.25-0.5% desaturation',
    'pct_epoch_0_5to0_75pct': 'Percentage of epochs with 0.5-0.75% desaturation',
    'pct_epoch_0_75to1pct': 'Percentage of epochs with 0.75-1% desaturation',
    'pct_epoch_1to1_25pct': 'Percentage of epochs with 1-1.25% desaturation',
    'pct_epoch_1_25to1_5pct': 'Percentage of epochs with 1.25-1.5% desaturation',
    'pct_epoch_1_5to1_75pct': 'Percentage of epochs with 1.5-1.75% desaturation',
    'pct_epoch_1_75to2pct': 'Percentage of epochs with 1.75-2% desaturation',
    'pct_epoch_2to2_25pct': 'Percentage of epochs with 2-2.25% desaturation',
    'pct_epoch_2_25to2_5pct': 'Percentage of epochs with 2.25-2.5% desaturation',
    'orpto9': 'Odds ratio product measure up to 9th decile',
    
    # Cognitive Decline features
    'powers_C3A2_delta': 'EEG delta wave power at C3A2 electrode position',
    'powers_C4A1_delta': 'EEG delta wave power at C4A1 electrode position',
    'spindle_char_N2_density_C3': 'Sleep spindle density during N2 sleep at C3 electrode',
    'spindle_char_N2_power_C3': 'Sleep spindle power during N2 sleep at C3 electrode',
    'spindle_char_N2_freq_C3': 'Sleep spindle frequency during N2 sleep at C3 electrode',
    'spindle_char_N2_pctfast_C3': 'Percentage of fast sleep spindles at C3 during N2 sleep',
    'spindle_char_N2_density_C4': 'Sleep spindle density during N2 sleep at C4 electrode',
    'spindle_char_N2_power_C4': 'Sleep spindle power during N2 sleep at C4 electrode',
    'spindle_char_N2_freq_C4': 'Sleep spindle frequency during N2 sleep at C4 electrode',
    'spindle_char_N2_pctfast_C4': 'Percentage of fast sleep spindles at C4 during N2 sleep',
    'diff_orp': 'Difference in odds ratio product between sleep stages',
    'avg_normalized_EEG_power': 'Average normalized EEG power across frequency bands',
    'avg_orp_n3_z': 'Z-score of average odds ratio product during N3 sleep',
    'avg_orp_n3_inv': 'Inverse of average odds ratio product during N3 sleep',
    'powers_C3A2_theta1_z': 'Z-score of EEG theta1 wave power at C3A2',
    'powers_C4A1_theta1_z': 'Z-score of EEG theta1 wave power at C4A1',
    'powers_C3A2_theta1_beta_ratio': 'Ratio of theta1 to beta power at C3A2',
    'powers_C4A1_theta1_beta_ratio': 'Ratio of theta1 to beta power at C4A1'
}

# Function to process EDF files (EEG data)
def process_edf_file(uploaded_file):
    # Save the uploaded file temporarily
    temp_file = "temp_eeg.edf"
    with open(temp_file, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    try:
        # Read EDF file using pyedflib
        f = pyedflib.EdfReader(temp_file)
        n = f.signals_in_file
        signal_labels = f.getSignalLabels()
        
        # Extract needed channels (focusing on C3A2 and C4A1 which are common in sleep EEG)
        data = {}
        c3a2_idx = None
        c4a1_idx = None
        
        # Try to find C3A2 and C4A1 channels or similar
        for i in range(n):
            if 'C3' in signal_labels[i] or 'C3A2' in signal_labels[i]:
                c3a2_idx = i
            if 'C4' in signal_labels[i] or 'C4A1' in signal_labels[i]:
                c4a1_idx = i
        
        # If we found the channels, extract the data
        if c3a2_idx is not None:
            data['C3A2'] = f.readSignal(c3a2_idx)
        if c4a1_idx is not None:
            data['C4A1'] = f.readSignal(c4a1_idx)
        
        # Close the file
        f.close()
        
        # If we have at least one channel, process it
        if data:
            # Extract features
            features = extract_eeg_features(data)
            return features, data
        else:
            st.error("Could not find required EEG channels in the file.")
            return None, None
            
    except Exception as e:
        st.error(f"Error processing EDF file: {e}")
        return None, None
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_file):
            os.remove(temp_file)

# Function to extract features from EEG data
def extract_eeg_features(eeg_data):
    features = {}
    
    # Process C3A2 channel if available
    if 'C3A2' in eeg_data:
        signal = eeg_data['C3A2']
        # Calculate power spectrum
        f, psd = welch(signal, fs=100, nperseg=1024)
        
        # Define frequency bands
        delta_idx = np.logical_and(f >= 0.5, f <= 4)
        theta1_idx = np.logical_and(f >= 4, f <= 6)
        alpha_idx = np.logical_and(f >= 8, f <= 13)
        beta1_idx = np.logical_and(f >= 13, f <= 20)
        beta2_idx = np.logical_and(f >= 20, f <= 30)
        
        # Calculate power in each band
        features['powers_C3A2_delta'] = np.mean(psd[delta_idx])
        features['powers_C3A2_theta1'] = np.mean(psd[theta1_idx])
        features['powers_C3A2_alpha'] = np.mean(psd[alpha_idx])
        features['powers_C3A2_beta1'] = np.mean(psd[beta1_idx])
        features['powers_C3A2_beta2'] = np.mean(psd[beta2_idx])
        
        # Calculate theta/beta ratio
        features['powers_C3A2_theta1_beta_ratio'] = features['powers_C3A2_theta1'] / (features['powers_C3A2_beta1'] + features['powers_C3A2_beta2'] + 1e-10)
        
        # Calculate z-scores (using typical values, would be better with population stats)
        features['powers_C3A2_theta1_z'] = (features['powers_C3A2_theta1'] - 1.0) / 0.5  # Placeholder values
    
    # Process C4A1 channel if available
    if 'C4A1' in eeg_data:
        signal = eeg_data['C4A1']
        # Calculate power spectrum
        f, psd = welch(signal, fs=100, nperseg=1024)
        
        # Define frequency bands
        delta_idx = np.logical_and(f >= 0.5, f <= 4)
        theta1_idx = np.logical_and(f >= 4, f <= 6)
        alpha_idx = np.logical_and(f >= 8, f <= 13)
        beta1_idx = np.logical_and(f >= 13, f <= 20)
        beta2_idx = np.logical_and(f >= 20, f <= 30)
        
        # Calculate power in each band
        features['powers_C4A1_delta'] = np.mean(psd[delta_idx])
        features['powers_C4A1_theta1'] = np.mean(psd[theta1_idx])
        features['powers_C4A1_alpha'] = np.mean(psd[alpha_idx])
        features['powers_C4A1_beta1'] = np.mean(psd[beta1_idx])
        features['powers_C4A1_beta2'] = np.mean(psd[beta2_idx])
        
        # Calculate theta/beta ratio
        features['powers_C4A1_theta1_beta_ratio'] = features['powers_C4A1_theta1'] / (features['powers_C4A1_beta1'] + features['powers_C4A1_beta2'] + 1e-10)
        
        # Calculate z-scores
        features['powers_C4A1_theta1_z'] = (features['powers_C4A1_theta1'] - 1.0) / 0.5  # Placeholder values
    
    # Estimate some sleep parameters based on alpha power
    if 'powers_C3A2_alpha' in features and 'powers_C4A1_alpha' in features:
        # Alpha intrusion (simplified estimate)
        features['alpha_intrusion_pct_3sepochs'] = min(100, max(0, 20 * (features['powers_C3A2_alpha'] + features['powers_C4A1_alpha']) / 2))
        
        # Simplified sleep stage estimates
        features['avg_orp_wake'] = 0.8 + 0.2 * features['powers_C3A2_alpha'] / (features['powers_C3A2_delta'] + 1e-10)
        features['avg_orp_nonrem'] = 0.6 - 0.2 * features['powers_C3A2_alpha'] / (features['powers_C3A2_delta'] + 1e-10)
        features['avg_orp_n1'] = 0.7 + 0.1 * features['powers_C3A2_alpha'] / (features['powers_C3A2_delta'] + 1e-10)
        features['avg_orp_n3'] = 0.4 - 0.3 * features['powers_C3A2_alpha'] / (features['powers_C3A2_delta'] + 1e-10)
        features['avg_orp_rem'] = 0.7 + 0.1 * features['powers_C4A1_alpha'] / (features['powers_C4A1_delta'] + 1e-10)
        
        # Sleep spindle characteristics (very simplified estimates)
        features['spindle_char_N2_density_C3'] = 2.5 + 0.5 * features['powers_C3A2_alpha'] / (features['powers_C3A2_delta'] + 1e-10)
        features['spindle_char_N2_power_C3'] = features['powers_C3A2_alpha'] * 0.7
        features['spindle_char_N2_freq_C3'] = 12.5 + 0.5 * features['powers_C3A2_beta1'] / (features['powers_C3A2_alpha'] + 1e-10)
        features['spindle_char_N2_pctfast_C3'] = 45 + 5 * features['powers_C3A2_beta1'] / (features['powers_C3A2_alpha'] + 1e-10)
        
        features['spindle_char_N2_density_C4'] = 2.3 + 0.5 * features['powers_C4A1_alpha'] / (features['powers_C4A1_delta'] + 1e-10)
        features['spindle_char_N2_power_C4'] = features['powers_C4A1_alpha'] * 0.75
        features['spindle_char_N2_freq_C4'] = 12.7 + 0.3 * features['powers_C4A1_beta1'] / (features['powers_C4A1_alpha'] + 1e-10)
        features['spindle_char_N2_pctfast_C4'] = 48 + 5 * features['powers_C4A1_beta1'] / (features['powers_C4A1_alpha'] + 1e-10)
        
        # Additional cognitive metrics
        features['diff_orp'] = features['avg_orp_wake'] - features['avg_orp_n3']
        features['avg_normalized_EEG_power'] = (features['powers_C3A2_delta'] + features['powers_C4A1_delta']) / 2
        features['avg_orp_n3_z'] = (features['avg_orp_n3'] - 0.4) / 0.15  # Placeholder normalization
        features['avg_orp_n3_inv'] = 1 / (features['avg_orp_n3'] + 0.1)
        
        # For insomnia
        features['sum_deciles_1and2'] = (features['powers_C3A2_delta'] * 0.2 + features['powers_C4A1_delta'] * 0.2) * 100
    
    # For sleep apnea (simplified)
    for i in range(10):
        pct_key = f'pct_epoch_{i//4}to{i//4}_25pct' if i % 4 == 0 else f'pct_epoch_{i//4}_{i%4*25}to{i//4}_{(i%4+1)*25}pct'
        features[pct_key] = max(0, min(100, 10 - i))
    
    features['orpto9'] = 0.85
    
    # Fill in any missing features with reasonable defaults
    for feature in all_features:
        if feature not in features:
            features[feature] = 0.5  # Default value
    
    return features

# Function to process CSV data
def process_csv_file(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
        
        # Check if the CSV has all required features
        missing_features = [feature for feature in all_features if feature not in df.columns]
        
        if missing_features:
            st.warning(f"The uploaded CSV is missing {len(missing_features)} required features. These will be filled with default values.")
            
            # Fill missing features with default values for the first row
            for feature in missing_features:
                df[feature] = 0.5
        
        # Get the first row of data
        features = df.iloc[0].to_dict()

        # Return only the required features
        required_features = {feature: features.get(feature, 0.5) for feature in all_features}
        return required_features, None
        
    except Exception as e:
        st.error(f"Error processing CSV file: {e}")
        return None, None

# Function to extract HRV features from ECG/PPG data
def extract_hrv_features(data, sampling_rate=250):
    try:
        # Process the ECG signal
        ecg_signals, info = nk.ecg_process(data, sampling_rate=sampling_rate)
        
        # Extract heart rate variability metrics
        hrv = nk.hrv_time(info["ECG_R_Peaks"], sampling_rate=sampling_rate, show=False)
        hrv_freq = nk.hrv_frequency(info["ECG_R_Peaks"], sampling_rate=sampling_rate, show=False)
        
        # Combine all HRV features
        features = {}
        features['AVNN'] = hrv['HRV_MeanNN'].values[0]
        features['SDNN'] = hrv['HRV_SDNN'].values[0]
        features['rMSSD'] = hrv['HRV_RMSSD'].values[0]
        features['pNN50'] = hrv['HRV_pNN50'].values[0]
        features['LF_HF'] = hrv_freq['HRV_LF/HF'].values[0]
        
        # Calculate additional metrics
        features['ihr'] = np.mean(ecg_signals['ECG_Rate'])
        features['IHR'] = features['ihr']  # Duplicate for model compatibility
        features['SDNNIDX'] = features['SDNN'] * 0.9  # Simplified approximation
        features['PNN50'] = features['pNN50']  # Duplicate for model compatibility
        
        # Placeholder values for metrics that would require longitudinal data
        features['baseline_hr'] = features['ihr'] * 0.95
        features['delta_hr'] = features['ihr'] - features['baseline_hr']
        features['dhr_intensity'] = abs(features['delta_hr']) / features['baseline_hr']
        features['change_in_hr_perh_trt'] = features['delta_hr'] / 1.0  # Assumes 1 hour
        
        return features
    except Exception as e:
        st.error(f"Error extracting HRV features: {e}")
        return None

# Function to prepare the data for model prediction
def prepare_data_for_models(features):
    model_inputs = {}
    
    # Prepare data for each condition's model
    for condition, feature_list in feature_sets.items():
        # Create input for this specific model
        inputs = np.array([[features[feature] for feature in feature_list]])

        # Ensure the inputs are reshaped to match the expected shape (1, 1, 100)
        if inputs.shape[1] != 100:
            # Pad or truncate the feature list to match 100 features
            inputs = np.pad(inputs, ((0, 0), (0, 100 - inputs.shape[1])), mode='constant', constant_values=0)
        
        # Reshape to match (1, 1, 100) or (1, timesteps, features) as needed
        inputs = np.reshape(inputs, (1, 1, 100))  # Update 100 to match your model's required feature count if necessary
        
        model_inputs[condition] = inputs
    
    return model_inputs

# Function to make predictions with all models
def run_predictions(models, model_inputs):
    predictions = {}
    
    for condition, model in models.items():
        if condition in model_inputs:
            try:
                # Make prediction
                inputs = model_inputs[condition]
                
                # Check if the model has a predict_proba method (for probability)
                if hasattr(model, 'predict_proba'):
                    prob = model.predict_proba(inputs)[0][1]  # Probability of positive class
                else:
                    # If not, use the raw prediction
                    prob = model.predict(inputs)[0]
                
                predictions[condition] = float(prob)
            except Exception as e:
                st.error(f"Error predicting {condition}: {e}")
                predictions[condition] = 0.0
    
    return predictions

# Generate recommendations based on predictions
def generate_recommendations(predictions):
    recommendations = {}
    
    # Sort conditions by probability (excluding "Healthy")
    sorted_conditions = sorted(
        [(cond, prob) for cond, prob in predictions.items() if cond != "Healthy"], 
        key=lambda x: x[1], 
        reverse=True
    )
    
    # Generate recommendations for the top conditions
    for condition, prob in sorted_conditions:
        if prob >= 0.5:  # Only give recommendations for conditions with high probability
            if condition == "Anxiety":
                recommendations[condition] = [
                    "Consider consulting with a mental health professional for a proper diagnosis",
                    "Practice daily mindfulness meditation and deep breathing exercises",
                    "Limit caffeine and alcohol consumption",
                    "Maintain a regular sleep schedule",
                    "Try progressive muscle relaxation techniques before bed",
                    "Consider cognitive behavioral therapy (CBT) which is effective for anxiety",
                    "Regular physical exercise can help reduce anxiety symptoms"
                ]
            elif condition == "Depression":
                recommendations[condition] = [
                    "Consult with a healthcare provider for proper evaluation",
                    "Engage in regular physical activity, even short walks can help",
                    "Maintain social connections and avoid isolation",
                    "Establish a consistent sleep routine",
                    "Consider light therapy, especially during winter months",
                    "Mindfulness practices and meditation can help manage symptoms",
                    "Cognitive behavioral therapy (CBT) has shown effectiveness for depression"
                ]
            elif condition == "ADHD":
                recommendations[condition] = [
                    "Seek evaluation from a specialist who can diagnose ADHD",
                    "Implement organizational strategies like planners and reminders",
                    "Break tasks into smaller, manageable steps",
                    "Consider mindfulness meditation to improve focus",
                    "Regular exercise can help reduce hyperactivity and improve attention",
                    "Minimize distractions in your work environment",
                    "Adequate sleep is crucial for attention and focus"
                ]
            elif condition == "Sleep Apnea":
                recommendations[condition] = [
                    "Consult with a sleep specialist for diagnosis and treatment options",
                    "Consider a sleep study to confirm the diagnosis",
                    "Maintain a healthy weight as excess weight can worsen symptoms",
                    "Sleep on your side rather than your back",
                    "Avoid alcohol and sedatives before bed",
                    "Establish a consistent sleep schedule",
                    "If diagnosed, use CPAP therapy as prescribed"
                ]
            elif condition == "Insomnia":
                recommendations[condition] = [
                    "Establish a regular sleep schedule, even on weekends",
                    "Create a relaxing bedtime routine",
                    "Make your sleep environment comfortable and free from distractions",
                    "Limit screen time before bed",
                    "Avoid caffeine and alcohol close to bedtime",
                    "Consider cognitive behavioral therapy for insomnia (CBT-I)",
                    "Try relaxation techniques like deep breathing or progressive muscle relaxation"
                ]
            elif condition == "Cognitive Decline":
                recommendations[condition] = [
                    "Consult with a neurologist or geriatric specialist",
                    "Engage in regular mental stimulation and brain training exercises",
                    "Maintain physical exercise as it benefits brain health",
                    "Follow a Mediterranean-style diet rich in fruits, vegetables, and healthy fats",
                    "Stay socially active and engaged",
                    "Ensure adequate sleep and manage stress",
                    "Control cardiovascular risk factors like blood pressure and cholesterol"
                ]
    
    if len(recommendations) == 0 or predictions.get("Healthy", 0) > 0.6:
        recommendations["Healthy"] = [
            "Continue maintaining a balanced lifestyle",
            "Regular physical activity benefits both mental and physical health",
            "Practice stress management techniques like meditation or deep breathing",
            "Maintain a healthy sleep schedule of 7-9 hours per night",
            "Stay socially connected and engaged",
            "Regular health check-ups are still recommended"
        ]
    
    return recommendations

# Function to create visualizations based on the features and predictions
def create_visualizations(features, predictions):
    visualizations = {}
    
    # 1. Radar chart of condition probabilities
    if predictions:
        categories = list(predictions.keys())
        values = [predictions[cat] for cat in categories]
        
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='Mental Health Assessment',
            line_color='rgba(78, 158, 212, 0.8)',
            fillcolor='rgba(78, 158, 212, 0.3)'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            title="Mental Health Assessment Results",
            showlegend=False,
            height=500
        )
        
        visualizations["radar_chart"] = fig
    
    # 2. Brain wave distribution visualization
    eeg_features = {}
    wave_types = ['delta', 'theta1', 'alpha', 'beta1', 'beta2']
    
    for wave in wave_types:
        c3a2_key = f'powers_C3A2_{wave}'
        c4a1_key = f'powers_C4A1_{wave}'
        
        if c3a2_key in features and c4a1_key in features:
            eeg_features[wave] = {
                'C3A2': features[c3a2_key],
                'C4A1': features[c4a1_key]
            }
        elif c3a2_key in features:
            eeg_features[wave] = {
                'C3A2': features[c3a2_key],
                'C4A1': 0  # Default value
            }
        elif c4a1_key in features:
            eeg_features[wave] = {
                'C3A2': 0,  # Default value
                'C4A1': features[c4a1_key]
            }
    
    if eeg_features:
        # Create dataframe for visualization
        eeg_df = pd.DataFrame({
            'Wave Type': [k for k in eeg_features.keys() for _ in range(2)],
            'Channel': ['C3A2', 'C4A1'] * len(eeg_features),
            'Power': [v[ch] for k, v in eeg_features.items() for ch in ['C3A2', 'C4A1']]
        })
        
        # Create a grouped bar chart
        fig = px.bar(
            eeg_df, 
            x='Wave Type', 
            y='Power', 
            color='Channel',
            barmode='group',
            title='Brain Wave Distribution',
            color_discrete_map={'C3A2': '#4e9ed4', 'C4A1': '#d44e9e'},
            height=500
        )
        
        fig.update_layout(
            xaxis_title="Wave Type",
            yaxis_title="Power",
            legend_title="EEG Channel"
        )
        
        visualizations["eeg_waves"] = fig
    
    # 3. Sleep stage distribution
    sleep_stages = ['wake', 'n1', 'nonrem', 'n3', 'rem']
    sleep_data = {}
    
    for stage in sleep_stages:
        key = f'avg_orp_{stage}'
        if key in features:
            sleep_data[stage.upper()] = features[key]
    
    if sleep_data:
        fig = px.pie(
            values=list(sleep_data.values()),
            names=list(sleep_data.keys()),
            title='Sleep Stage Distribution (Estimated)',
            color_discrete_sequence=px.colors.sequential.Blues,
            hole=0.4,
            height=500
        )
        
        fig.update_traces(textposition='inside', textinfo='percent+label')
        visualizations["sleep_stages"] = fig
    
    # 4. HRV metrics visualization if available
    hrv_metrics = ['AVNN', 'SDNN', 'rMSSD', 'pNN50', 'LF_HF']
    hrv_data = {}
    
    for metric in hrv_metrics:
        if metric in features:
            hrv_data[metric] = features[metric]
    
    if hrv_data:
        # Normalize the values for better visualization
        max_vals = {'AVNN': 1000, 'SDNN': 100, 'rMSSD': 100, 'pNN50': 100, 'LF_HF': 5}
        norm_hrv_data = {k: min(v / max_vals.get(k, 1), 1) for k, v in hrv_data.items()}
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=list(norm_hrv_data.keys()),
            y=list(norm_hrv_data.values()),
            marker_color='#4e9ed4',
            name='Normalized Value'
        ))
        
        fig.update_layout(
            title="Heart Rate Variability Metrics",
            xaxis_title="Metric",
            yaxis_title="Normalized Value",
            height=500
        )
        
        # Add hover text with actual values
        fig.update_traces(
            text=[f"{hrv_data[k]:.2f}" for k in norm_hrv_data.keys()],
            hovertemplate='%{x}: %{text}<extra></extra>'
        )
        
        visualizations["hrv_metrics"] = fig
    
    # 5. Cognitive function visualization
    if 'powers_C3A2_theta1_beta_ratio' in features and 'powers_C4A1_theta1_beta_ratio' in features:
        cognitive_data = {
            'Metric': ['Theta/Beta Ratio Left', 'Theta/Beta Ratio Right'],
            'Value': [features['powers_C3A2_theta1_beta_ratio'], features['powers_C4A1_theta1_beta_ratio']]
        }
        
        fig = px.bar(
            cognitive_data,
            x='Metric',
            y='Value',
            title='Cognitive Function Metrics',
            color='Metric',
            color_discrete_sequence=['#4e9ed4', '#d44e9e'],
            height=400
        )
        
        fig.update_layout(showlegend=False)
        visualizations["cognitive"] = fig
    
    # 6. Brain map visualization (simplified)
    if 'powers_C3A2_alpha' in features and 'powers_C4A1_alpha' in features:
        # Create a very simple brain map visualization
        brain_data = np.zeros((10, 10))
        # Set values for C3 and C4 locations (approximate)
        c3_loc = (3, 3)
        c4_loc = (3, 6)
        
        # Normalize values for visualization
        alpha_max = max(features['powers_C3A2_alpha'], features['powers_C4A1_alpha'])
        if alpha_max > 0:
            c3_val = features['powers_C3A2_alpha'] / alpha_max
            c4_val = features['powers_C4A1_alpha'] / alpha_max
            
            # Create a simple gaussian distribution around the electrode locations
            x = np.linspace(0, 9, 10)
            y = np.linspace(0, 9, 10)
            xx, yy = np.meshgrid(x, y)
            
            # C3 distribution
            brain_data += c3_val * np.exp(-0.5 * ((xx - c3_loc[0])**2 + (yy - c3_loc[1])**2) / 1.5)
            # C4 distribution
            brain_data += c4_val * np.exp(-0.5 * ((xx - c4_loc[0])**2 + (yy - c4_loc[1])**2) / 1.5)
            
            # Create the heatmap
            fig = px.imshow(
                brain_data,
                color_continuous_scale='Blues',
                title='Brain Activity Map (Alpha Power)',
                height=400
            )
            
            # Add electrode markers
            fig.add_trace(go.Scatter(
                x=[c3_loc[1], c4_loc[1]],
                y=[c3_loc[0], c4_loc[0]],
                mode='markers+text',
                marker=dict(size=10, color='black'),
                text=['C3', 'C4'],
                textposition="top center",
                showlegend=False
            ))
            
            fig.update_layout(
                xaxis=dict(showticklabels=False),
                yaxis=dict(showticklabels=False)
            )
            
            visualizations["brain_map"] = fig
    
    return visualizations

# Main function to run the Streamlit app
def main():
    # Try to load models
    models = load_models()
    
    # Input method tabs
    tab1, tab2, tab3 = st.tabs(["📊 Upload File", "📋 Manual Input", "ℹ️ Information"])
    
    # Global state to store results
    if 'features' not in st.session_state:
        st.session_state.features = None
    if 'predictions' not in st.session_state:
        st.session_state.predictions = None
    if 'recommendations' not in st.session_state:
        st.session_state.recommendations = None
    if 'visualizations' not in st.session_state:
        st.session_state.visualizations = None
    if 'raw_data' not in st.session_state:
        st.session_state.raw_data = None
    
    # Tab 1: File Upload
    with tab1:
        st.markdown("### Upload Your Data")
        st.write("Upload your EEG data (EDF format) or a CSV file with the required features")
        
        uploaded_file = st.file_uploader("Choose a file", type=["edf", "csv"])
        
        if uploaded_file is not None:
            # Process based on file type
            if uploaded_file.name.lower().endswith('.edf'):
                st.success("EDF file detected (EEG data)")
                with st.spinner("Processing EEG data..."):
                    features, raw_data = process_edf_file(uploaded_file)
                    if features:
                        st.session_state.features = features
                        st.session_state.raw_data = raw_data
                        st.success("EEG data processed successfully!")
                        
                        # Display sample of processed features
                        st.markdown("#### Sample of Extracted Features")
                        sample_df = pd.DataFrame({
                            'Feature': list(features.keys())[:10],
                            'Value': list(features.values())[:10]
                        })
                        st.dataframe(sample_df)
                        
                        # Plot raw EEG data if available
                        if raw_data and 'C3A2' in raw_data:
                            st.markdown("#### Raw EEG Signal Preview")
                            fig, ax = plt.subplots(figsize=(10, 3))
                            ax.plot(raw_data['C3A2'][:1000], label='C3A2')
                            if 'C4A1' in raw_data:
                                ax.plot(raw_data['C4A1'][:1000], label='C4A1', alpha=0.7)
                            ax.set_title("EEG Signal Preview (First 1000 samples)")
                            ax.set_xlabel("Samples")
                            ax.set_ylabel("Amplitude")
                            ax.legend()
                            st.pyplot(fig)
                    else:
                        st.error("Could not process the EEG file. Please try another file.")
            
            elif uploaded_file.name.lower().endswith('.csv'):
                st.success("CSV file detected")
                with st.spinner("Processing CSV data..."):
                    features, _ = process_csv_file(uploaded_file)
                    if features:
                        st.session_state.features = features
                        st.success("CSV data processed successfully!")
                        
                        # Display sample of uploaded features
                        st.markdown("#### Sample of Imported Features")
                        sample_df = pd.DataFrame({
                            'Feature': list(features.keys())[:10],
                            'Value': list(features.values())[:10]
                        })
                        st.dataframe(sample_df)
                    else:
                        st.error("Could not process the CSV file. Please ensure it contains the required features.")
    
    # Tab 2: Manual Input
    with tab2:
        st.markdown("### Manual Input")
        st.write("Input values for key features to assess mental health conditions")
        
        # Create tabs for different feature groups
        manual_tabs = st.tabs(["Brain Activity", "Heart Rate", "Sleep Metrics", "Cognitive"])
        
        manual_features = {}
        
        with manual_tabs[0]:  # Brain Activity
            st.markdown("#### Brain Wave Activity")
            col1, col2 = st.columns(2)
            
            with col1:
                manual_features['powers_C3A2_delta'] = st.slider(
                    "Delta Power (C3A2)", 0.0, 10.0, 5.0,
                    help="Delta waves (0.5-4 Hz) are associated with deep sleep"
                )
                manual_features['powers_C3A2_theta1'] = st.slider(
                    "Theta Power (C3A2)", 0.0, 10.0, 4.0,
                    help="Theta waves (4-8 Hz) are associated with drowsiness and meditation"
                )
                manual_features['powers_C3A2_alpha'] = st.slider(
                    "Alpha Power (C3A2)", 0.0, 10.0, 3.5,
                    help="Alpha waves (8-13 Hz) are associated with relaxed alertness"
                )
                manual_features['powers_C3A2_beta1'] = st.slider(
                    "Beta1 Power (C3A2)", 0.0, 10.0, 2.0,
                    help="Beta1 waves (13-20 Hz) are associated with active thinking"
                )
                manual_features['powers_C3A2_beta2'] = st.slider(
                    "Beta2 Power (C3A2)", 0.0, 10.0, 1.5,
                    help="Beta2 waves (20-30 Hz) are associated with intense focus"
                )
            
            with col2:
                manual_features['powers_C4A1_delta'] = st.slider(
                    "Delta Power (C4A1)", 0.0, 10.0, 4.8,
                    help="Delta waves (0.5-4 Hz) are associated with deep sleep"
                )
                manual_features['powers_C4A1_theta1'] = st.slider(
                    "Theta Power (C4A1)", 0.0, 10.0, 3.8,
                    help="Theta waves (4-8 Hz) are associated with drowsiness and meditation"
                )
                manual_features['powers_C4A1_alpha'] = st.slider(
                    "Alpha Power (C4A1)", 0.0, 10.0, 3.3,
                    help="Alpha waves (8-13 Hz) are associated with relaxed alertness"
                )
                manual_features['powers_C4A1_beta1'] = st.slider(
                    "Beta1 Power (C4A1)", 0.0, 10.0, 1.9,
                    help="Beta1 waves (13-20 Hz) are associated with active thinking"
                )
                manual_features['powers_C4A1_beta2'] = st.slider(
                    "Beta2 Power (C4A1)", 0.0, 10.0, 1.4,
                    help="Beta2 waves (20-30 Hz) are associated with intense focus"
                )
        
        with manual_tabs[1]:  # Heart Rate
            st.markdown("#### Heart Rate & Variability")
            col1, col2 = st.columns(2)
            
            with col1:
                manual_features['ihr'] = st.slider(
                    "Heart Rate (bpm)", 40.0, 120.0, 72.0,
                    help="Average heart rate in beats per minute"
                )
                manual_features['AVNN'] = st.slider(
                    "Average NN Intervals (ms)", 600.0, 1200.0, 850.0,
                    help="Average time between consecutive heartbeats"
                )
                manual_features['SDNN'] = st.slider(
                    "SDNN (ms)", 10.0, 200.0, 50.0,
                    help="Standard deviation of NN intervals, indicator of overall HRV"
                )
            
            with col2:
                manual_features['rMSSD'] = st.slider(
                    "rMSSD (ms)", 10.0, 100.0, 35.0,
                    help="Root mean square of successive differences between heartbeats"
                )
                manual_features['pNN50'] = st.slider(
                    "pNN50 (%)", 0.0, 50.0, 10.0,
                    help="Percentage of successive intervals that differ by more than 50ms"
                )
                manual_features['LF_HF'] = st.slider(
                    "LF/HF Ratio", 0.5, 5.0, 2.0,
                    help="Ratio of low frequency to high frequency power, indicates autonomic balance"
                )
            
            # Additional HRV features with defaults
            manual_features['IHR'] = manual_features['ihr']
            manual_features['SDNNIDX'] = manual_features['SDNN'] * 0.9
            manual_features['PNN50'] = manual_features['pNN50']
            manual_features['baseline_hr'] = manual_features['ihr'] * 0.95
            manual_features['delta_hr'] = manual_features['ihr'] - manual_features['baseline_hr']
            manual_features['dhr_intensity'] = abs(manual_features['delta_hr']) / manual_features['baseline_hr']
            manual_features['change_in_hr_perh_trt'] = manual_features['delta_hr'] / 1.0
        
        with manual_tabs[2]:  # Sleep Metrics
            st.markdown("#### Sleep Pattern Metrics")
            col1, col2 = st.columns(2)
        
            with col1:
                manual_features['avg_orp_wake'] = st.slider(
                    "Odds Ratio Product - Wake", 0.0, 1.0, 0.8,
                    help="ORP during wake periods, higher values indicate more wakefulness"
                )
                manual_features['avg_orp_nonrem'] = st.slider(
                    "Odds Ratio Product - Non-REM", 0.0, 1.0, 0.6,
                    help="ORP during non-REM sleep, lower values indicate deeper sleep"
                )
                manual_features['avg_orp_n1'] = st.slider(
                    "Odds Ratio Product - N1 Sleep", 0.0, 1.0, 0.7,
                    help="ORP during N1 (light) sleep stage"
                )
                manual_features['avg_orp_n3'] = st.slider(
                    "Odds Ratio Product - N3 Sleep", 0.0, 1.0, 0.4,
                    help="ORP during N3 (deep) sleep stage, lower values indicate deeper sleep"
                )
                manual_features['avg_orp_rem'] = st.slider(
                    "Odds Ratio Product - REM Sleep", 0.0, 1.0, 0.7,
                    help="ORP during REM sleep stage"
                )
        
            with col2:
                manual_features['sum_deciles_1and2'] = st.slider(
                    "Sum of Low Power Deciles", 0.0, 200.0, 100.0,
                    help="Sum of power in lowest two deciles, related to insomnia"
                )
                manual_features['alpha_intrusion_pct_3sepochs'] = st.slider(
                    "Alpha Intrusion (%)", 0.0, 100.0, 20.0,
                    help="Percentage of alpha wave intrusion during sleep, higher in insomnia"
                )
        
                # Sleep Apnea Metrics
                sleep_apnea_keys = [
                    'pct_epoch_0to_0_25pct', 'pct_epoch_0_25to0_5pct', 'pct_epoch_0_5to0_75pct', 
                    'pct_epoch_0_75to1pct', 'pct_epoch_1to1_25pct', 'pct_epoch_1_25to1_5pct', 
                    'pct_epoch_1_5to1_75pct', 'pct_epoch_1_75to2pct', 'pct_epoch_2to2_25pct', 
                    'pct_epoch_2_25to2_5pct', 'orpto9'
                ]
                
                for key in sleep_apnea_keys:
                    manual_features[key] = st.slider(
                        key.replace("_", " ").title(), 0.0, 100.0, 0.0,  # Customize the default value as needed
                        help=f"Percentage of epochs with {key.replace('_', ' ').title()} oxygen desaturation"
                    )
                    
            # Fill in remaining sleep apnea metrics with reasonable defaults
            for i in range(3, 10):
                key = f'pct_epoch_{i//4}to{i//4}_25pct' if i % 4 == 0 else f'pct_epoch_{i//4}_{i%4*25}to{i//4}_{(i%4+1)*25}pct'
                manual_features[key] = max(0, 10-i)
        
            manual_features['orpto9'] = 0.85

        
        with manual_tabs[3]:  # Cognitive
            st.markdown("#### Cognitive Function Metrics")
            col1, col2 = st.columns(2)
            
            with col1:
                manual_features['spindle_char_N2_density_C3'] = st.slider(
                    "Sleep Spindle Density (C3)", 0.0, 5.0, 2.5,
                    help="Density of sleep spindles at C3 electrode, important for memory consolidation"
                )
                manual_features['spindle_char_N2_power_C3'] = st.slider(
                    "Sleep Spindle Power (C3)", 0.0, 10.0, 3.0,
                    help="Power of sleep spindles at C3 electrode"
                )
                manual_features['spindle_char_N2_freq_C3'] = st.slider(
                    "Sleep Spindle Frequency (C3)", 11.0, 16.0, 12.5,
                    help="Frequency of sleep spindles at C3 electrode in Hz"
                )
                manual_features['spindle_char_N2_pctfast_C3'] = st.slider(
                    "% Fast Spindles (C3)", 0.0, 100.0, 45.0,
                    help="Percentage of fast sleep spindles at C3 electrode"
                )
            
            with col2:
                manual_features['spindle_char_N2_density_C4'] = st.slider(
                    "Sleep Spindle Density (C4)", 0.0, 5.0, 2.3,
                    help="Density of sleep spindles at C4 electrode, important for memory consolidation"
                )
                manual_features['spindle_char_N2_power_C4'] = st.slider(
                    "Sleep Spindle Power (C4)", 0.0, 10.0, 2.8,
                    help="Power of sleep spindles at C4 electrode"
                )
                manual_features['spindle_char_N2_freq_C4'] = st.slider(
                    "Sleep Spindle Frequency (C4)", 11.0, 16.0, 12.7,
                    help="Frequency of sleep spindles at C4 electrode in Hz"
                )
                manual_features['spindle_char_N2_pctfast_C4'] = st.slider(
                    "% Fast Spindles (C4)", 0.0, 100.0, 48.0,
                    help="Percentage of fast sleep spindles at C4 electrode"
                )
            
            # Additional cognitive metrics
            manual_features['diff_orp'] = manual_features['avg_orp_wake'] - manual_features['avg_orp_n3']
            manual_features['avg_normalized_EEG_power'] = (manual_features['powers_C3A2_delta'] + manual_features['powers_C4A1_delta']) / 2
            manual_features['avg_orp_n3_z'] = (manual_features['avg_orp_n3'] - 0.4) / 0.15
            manual_features['avg_orp_n3_inv'] = 1 / (manual_features['avg_orp_n3'] + 0.1)
            manual_features['powers_C3A2_theta1_z'] = (manual_features['powers_C3A2_theta1'] - 1.0) / 0.5
            manual_features['powers_C4A1_theta1_z'] = (manual_features['powers_C4A1_theta1'] - 1.0) / 0.5
            manual_features['powers_C3A2_theta1_beta_ratio'] = manual_features['powers_C3A2_theta1'] / (manual_features['powers_C3A2_beta1'] + manual_features['powers_C3A2_beta2'] + 1e-10)
            manual_features['powers_C4A1_theta1_beta_ratio'] = manual_features['powers_C4A1_theta1'] / (manual_features['powers_C4A1_beta1'] + manual_features['powers_C4A1_beta2'] + 1e-10)
        
        # Button to use manual input
        if st.button("Use Manual Input", key="manual_input_button"):
            st.session_state.features = manual_features
            st.success("Manual input processed successfully!")
    
    # Tab 3: Information
    with tab3:
        st.markdown("### About Mental Health Assessment")
        
        st.markdown("""
        #### The Brain and Mental Health
        
        The brain is the command center of our nervous system and plays a critical role in mental health. 
        Different brain regions and patterns of activity are associated with various mental health conditions.
        
        #### EEG and Mental Health
        
        Electroencephalography (EEG) measures electrical activity in the brain and can reveal patterns 
        associated with various mental health conditions:
        
        - **Anxiety**: Often shows increased beta wave activity and heightened arousal
        - **Depression**: May show frontal asymmetry and alpha wave abnormalities
        - **ADHD**: Typically shows increased theta/beta ratio in frontal regions
        - **Sleep Disorders**: Show disruptions in normal sleep architecture and wave patterns
        - **Cognitive Decline**: May show reduced alpha and increased delta/theta activity
        
        #### Heart Rate Variability (HRV)
        
        HRV measures the variation in time between consecutive heartbeats and provides insights into the autonomic nervous system:
        
        - **Low HRV**: Associated with anxiety, depression, and stress
        - **High HRV**: Associated with good cardiac health and emotional regulation
        """)
        
        st.markdown("#### Features Used in this Application")
        st.dataframe(pd.DataFrame({
            'Feature': list(feature_descriptions.keys()),
            'Description': list(feature_descriptions.values())
        }))
    
    # Run analysis if features are available
    if st.session_state.features is not None:
        st.markdown("---")
        st.markdown("## Analysis Results")
        
        # Check if predictions already exist
        if st.session_state.predictions is None:
            # Prepare data for model prediction
            model_inputs = prepare_data_for_models(st.session_state.features)
            
            # Make predictions
            st.session_state.predictions = run_predictions(models, model_inputs)
            
            # Generate recommendations
            st.session_state.recommendations = generate_recommendations(st.session_state.predictions)
            
            # Create visualizations
            st.session_state.visualizations = create_visualizations(st.session_state.features, st.session_state.predictions)
        
        # Display results
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Display radar chart of predictions
            if "radar_chart" in st.session_state.visualizations:
                st.plotly_chart(st.session_state.visualizations["radar_chart"], use_container_width=True)
        
        with col2:
            # Display prediction probabilities
            st.markdown("### Assessment Results")
            
            # Sort predictions by probability (descending)
            sorted_predictions = sorted(st.session_state.predictions.items(), key=lambda x: x[1], reverse=True)
            
            for condition, prob in sorted_predictions:
                # Use color coding for probabilities
                if prob >= 0.7:
                    color = "#d9534f"  # Red for high probability
                    emoji = "🔴"
                elif prob >= 0.4:
                    color = "#f0ad4e"  # Yellow for moderate probability
                    emoji = "🟡"
                else:
                    color = "#5cb85c"  # Green for low probability
                    emoji = "🟢"
                
                # Display in a metric card with styling
                st.markdown(f"""
                <div class="metric-card">
                    <h4>{emoji} {condition}</h4>
                    <h3 style="color: {color};">{prob:.1%}</h3>
                    <p>Probability</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Display detailed visualizations
        st.markdown("### Detailed Analysis")
        viz_tabs = st.tabs(["Brain Activity", "Sleep Patterns", "Heart & Cognitive", "Recommendations"])
        
        with viz_tabs[0]:  # Brain Activity
            col1, col2 = st.columns(2)
            
            with col1:
                if "eeg_waves" in st.session_state.visualizations:
                    st.plotly_chart(st.session_state.visualizations["eeg_waves"], use_container_width=True)
            
            with col2:
                if "brain_map" in st.session_state.visualizations:
                    st.plotly_chart(st.session_state.visualizations["brain_map"], use_container_width=True)
                else:
                    st.markdown("""
                    #### Brain Wave Types
                    
                    - **Delta (0.5-4 Hz)**: Deep sleep, healing
                    - **Theta (4-8 Hz)**: Drowsiness, meditation
                    - **Alpha (8-13 Hz)**: Relaxed alertness
                    - **Beta (13-30 Hz)**: Active thinking, focus
                    - **Gamma (30+ Hz)**: Complex cognitive processing
                    """)
        
        with viz_tabs[1]:  # Sleep Patterns
            col1, col2 = st.columns(2)
            
            with col1:
                if "sleep_stages" in st.session_state.visualizations:
                    st.plotly_chart(st.session_state.visualizations["sleep_stages"], use_container_width=True)
            
            with col2:
                st.markdown("#### Sleep Stage Information")
                st.markdown("""
                - **Wake**: Brain is active and aware
                - **N1**: Light sleep, transition between wake and sleep
                - **N2**: Deeper sleep, body temperature drops
                - **N3**: Deep sleep, vital for physical restoration
                - **REM**: Rapid Eye Movement sleep, associated with dreaming
                
                Sleep architecture abnormalities are common in many mental health conditions.
                """)
                
                # Display some sleep metrics
                sleep_metrics = {}
                for key in st.session_state.features:
                    if "orp" in key or "spindle" in key:
                        sleep_metrics[key] = st.session_state.features[key]
                
                if sleep_metrics:
                    sleep_df = pd.DataFrame({
                        'Metric': list(sleep_metrics.keys())[:5],
                        'Value': list(sleep_metrics.values())[:5]
                    })
                    st.dataframe(sleep_df)
        
        with viz_tabs[2]:  # Heart & Cognitive
            col1, col2 = st.columns(2)
            
            with col1:
                if "hrv_metrics" in st.session_state.visualizations:
                    st.plotly_chart(st.session_state.visualizations["hrv_metrics"], use_container_width=True)
            
            with col2:
                if "cognitive" in st.session_state.visualizations:
                    st.plotly_chart(st.session_state.visualizations["cognitive"], use_container_width=True)
                
                st.markdown("#### HRV & Cognitive Health")
                st.markdown("""
                **Heart Rate Variability (HRV)** reflects autonomic nervous system function:
                
                - Higher HRV generally indicates better stress resilience
                - Lower HRV is associated with stress, anxiety, and depression
                
                **Theta/Beta Ratio** is an important cognitive metric:
                
                - Higher ratios may indicate attention issues (common in ADHD)
                - Imbalances between hemispheres can indicate mood disorders
                """)
        
        with viz_tabs[3]:  # Recommendations
            if st.session_state.recommendations:
                # Display recommendations for each condition
                for condition, recs in st.session_state.recommendations.items():
                    if condition == "Healthy" and len(st.session_state.recommendations) > 1:
                        # Skip Healthy recommendations if we have other condition recommendations
                        continue
                    
                    st.markdown(f"#### {condition} Recommendations")
                    
                    # Determine emoji based on condition
                    emoji = "✅" if condition == "Healthy" else "ℹ️"
                    
                    for rec in recs:
                        st.markdown(f"{emoji} {rec}")
                    
                    st.markdown("---")
            
            # Disclaimer
            st.markdown("""
            ### Important Disclaimer
            
            This tool provides preliminary assessment based on the data provided. It is not a substitute for professional medical diagnosis and treatment. 
            
            **If you're experiencing mental health concerns, please consult with a qualified healthcare professional.**
            """)
        
        # Download section
        st.markdown("---")
        st.markdown("### Download Assessment Report")
        
        # Generate a report as a string
        report = io.StringIO()
        report.write("# Mental Health Assessment Report\n\n")
        report.write(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        report.write("## Assessment Results\n\n")
        
        # Write predictions
        sorted_predictions = sorted(st.session_state.predictions.items(), key=lambda x: x[1], reverse=True)
        for condition, prob in sorted_predictions:
            report.write(f"- {condition}: {prob:.1%}\n")
        
        # Write recommendations
        report.write("\n## Recommendations\n\n")
        for condition, recs in st.session_state.recommendations.items():
            report.write(f"### {condition}\n\n")
            for rec in recs:
                report.write(f"- {rec}\n")
            report.write("\n")
        
        # Add disclaimer
        report.write("\n## Disclaimer\n\n")
        report.write("This assessment is for informational purposes only and does not constitute medical advice. ")
        report.write("Please consult with a healthcare professional for proper diagnosis and treatment.\n")
        
        # Create download button
        report_str = report.getvalue()
        st.download_button(
            label="Download Report (TXT)",
            data=report_str,
            file_name="mental_health_assessment_report.txt",
            mime="text/plain"
        )
        
        # Reset button
        if st.button("Start New Assessment"):
            for key in ['features', 'predictions', 'recommendations', 'visualizations', 'raw_data']:
                st.session_state[key] = None
            st.experimental_rerun()

if __name__ == "__main__":
    main()