
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
