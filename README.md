# Mental Health Assessment Platform

## Overview
This application uses advanced machine learning models to analyze various physiological signals including EEG (electroencephalogram), HRV (Heart Rate Variability), and other health metrics to identify potential mental health conditions. The models were trained on the Sleep Heart Health Study (SHHS) dataset, allowing for effective assessment of various neurological and psychological conditions.


## Features

### Conditions Detected
The platform can help identify potential signs of:
- Insomnia
- Anxiety
- Depression
- ADHD
- Sleep Apnea
- Cognitive Decline

### Data Analysis Methods
The application supports multiple input methods:
- EEG (electroencephalogram) signals via EDF files
- HRV (Heart Rate Variability) data
- Manual input of important health metrics via CSV
- Real-time physiological monitoring (where supported)

### Technical Features
- Advanced CNN-LSTM hybrid models for analyzing temporal physiological data
- Comprehensive feature extraction from EEG and HRV signals
- Interactive visualization of assessment results
- Detailed explanations of metrics and their significance
- Professional reports with recommendations

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/mental-health-assessment.git

# Navigate to the project directory
cd mental-health-assessment

# Install required packages
pip install -r requirements.txt
```

## Dependencies
- Streamlit
- NumPy
- Pandas
- Scipy
- PyEDFlib
- Neurokit2
- Joblib
- Matplotlib
- Seaborn

## Usage

```bash
# Run the Streamlit app
streamlit run app.py
```

### Input Options
1. **Upload EDF File**: Upload an EEG recording in EDF format for analysis
2. **Upload CSV File**: Upload a CSV containing necessary health metrics
3. **Manual Entry**: Manually input important health parameters
4. **Connect Device**: Connect compatible devices for real-time monitoring (where available)

## Models
The machine learning models used in this application were developed using the Sleep Heart Health Study (SHHS) dataset. Each condition-specific model employs a CNN-LSTM architecture for time-series analysis of physiological signals, with particular focus on:

- EEG power bands (delta, theta, alpha, beta)
- Sleep architecture metrics
- Heart rate variability measures
- Respiratory pattern analysis

## References
- Sleep Heart Health Study (SHHS) - [https://sleepdata.org/datasets/shhs](https://sleepdata.org/datasets/shhs)

## Disclaimer
This tool is for informational purposes only and does not replace professional medical advice, diagnosis, or treatment. Always consult with a qualified healthcare provider for medical concerns.


## Contributors
- Jaya Soorya

## Contact
For questions or support, please contact [amjayasoorya@gmail.com].
