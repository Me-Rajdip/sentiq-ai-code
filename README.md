# Sentiq AI: Privacy-First Time-Series Anomaly Detection for UPI & Bank Transactions

## Overview

Sentiq AI is a sophisticated anomaly detection system designed to identify fraudulent transactions in UPI and banking systems using an LSTM Autoencoder architecture. The system learns normal spending patterns from transaction data and flags anomalies based on reconstruction error, providing explainable insights for each detected anomaly.

## Key Features

- **Privacy-First Architecture**: Processes transaction data locally without requiring sensitive information to be shared externally
- **LSTM Autoencoder**: Deep learning model that learns normal behavioral patterns from time-series transaction data
- **Real-time Detection**: Identifies anomalies in transaction sequences using reconstruction error analysis
- **Explainable AI**: Provides human-readable explanations for each flagged anomaly
- **Comprehensive Visualization**: 10+ chart types for data exploration and model evaluation

## Technical Architecture

### Data Processing Pipeline

1. **Data Collection**
   - Uses UPI Transactions 2024 Dataset from Kaggle
   - Automatically falls back to synthetic data if external data is unavailable
   - Standardizes column names for consistent processing

2. **Data Validation & Cleaning**
   - Handles missing values and duplicates
   - Validates transaction amounts and balances
   - Removes invalid entries

3. **Feature Engineering**
   - Transaction amount (log-transformed for skew reduction)
   - Time-based features (hour of day, day of week, weekend indicator)
   - Behavioral features:
     - Merchant frequency and novelty scores
     - Transaction velocity
     - Balance changes
     - Transaction type encoding
     - Device and network type encoding

4. **Sequence Creation**
   - Sliding window approach (5 transactions per window)
   - Feature scaling using MinMaxScaler
   - Labels windows as anomalous if any transaction within is fraudulent

5. **Model Architecture**

Input Sequence (5 timesteps × 13 features)
↓
Encoder LSTM (32 units, return sequences=True)
↓
Encoder LSTM (16 units, return sequences=False)
↓
Latent Representation (16 units)
↓
Repeat Vector (5 timesteps)
↓
Decoder LSTM (16 units, return sequences=True)
↓
Decoder LSTM (32 units, return sequences=True)
↓
TimeDistributed Dense (13 features)
↓
Reconstructed Sequence

### Model Training

- **Training Data**: Only normal (non-fraud) transaction windows
- **Loss Function**: Mean Squared Error (MSE)
- **Optimizer**: Adam
- **Early Stopping**: Monitors validation loss with patience of 5 epochs
- **Validation Split**: 20% of normal data

### Anomaly Detection

- **Threshold Selection**: 95th percentile of reconstruction errors on validation data
- **Error Calculation**: Mean MSE across sequence and feature dimensions
- **Prediction**: Windows with error above threshold are flagged as anomalies

### Explainability Layer

For each flagged anomaly, the system analyzes:
- Transaction amount deviations from normal patterns
- Unusual transaction times (late-night transactions)
- First-time or rare merchant interactions
- Abnormal balance changes
- Overall behavioral pattern deviations

## Visualizations

The system generates 11 different visualization types:

1. **Histogram**: Transaction amount distribution
2. **Boxplot**: Amount by transaction type
3. **Bar Chart**: Transaction count by type
4. **Pie Chart**: Fraud vs Normal share
5. **Correlation Heatmap**: Numeric feature relationships
6. **Line Chart**: Transaction volume over time
7. **Violin Plot**: Log(amount) by fraud label
8. **Scatter Plot**: Amount vs Balance Change (colored by fraud)
9. **KDE Plot**: Log(amount) density by class
10. **Sunburst**: Transaction type and fraud composition
11. **Model Performance**: Reconstruction error distribution, ROC/PR curves, confusion matrix

## Results

- **Window-level Fraud Detection**: Achieves high accuracy in identifying anomalous transaction sequences
- **Explainable Predictions**: Each anomaly comes with human-readable reasons
- **Privacy-Preserving**: All processing done locally, no data sent externally

## Model Evaluation Metrics

- **Precision**: 0.99 for normal transactions
- **Recall**: 0.95 for normal transactions
- **Accuracy**: 94%
- **F1-Score**: 0.97 for normal transactions

## Installation

```bash
pip install tensorflow scikit-learn pandas numpy matplotlib seaborn plotly kagglehub tf2onnx


## Usage

1. **Data Loading**: The system automatically downloads the UPI Transactions 2024 Dataset
2. **Feature Engineering**: Behavioral features are automatically extracted
3. **Training**: Model is trained on normal transaction patterns
4. **Detection**: New transactions are evaluated for anomalies
5. **Explanation**: Flagged anomalies come with human-readable explanations

## Technical Requirements

- Python 3.8+
- TensorFlow 2.0+
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- plotly
- kagglehub

## Model Persistence

- Saves trained model in `.keras` format
- Supports ONNX export for browser/local inference
- Stores trained model as `financeguard_best.keras`

## Future Enhancements

- Real-time transaction monitoring
- Mobile app integration
- Enhanced explainability features
- Support for additional transaction types

## Acknowledgments

- Dataset: UPI Transactions 2024 Dataset (Kaggle)
- Architecture inspired by FinanceGuard AI
- Built with TensorFlow and scikit-learn
