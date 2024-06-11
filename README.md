# Predictive Maintenance using Machine Learning
Machinery failure prediction using multiple Machine Learning Models

## Goal of the Project
The goal of this project is to develop predictive maintenance models to estimate the Remaining Useful Life (RUL) of engines using the NASA Turbofan Engine Degradation Simulation Data Set. Different models are evaluated, including a Long Short-Term Memory (LSTM) neural network and a Random Forest regressor, to identify the best approach for this task.

## Dataset
The dataset used in this project is the NASA Turbofan Engine Degradation Simulation Data Set. It includes multiple files:

- `train_FD001.txt`: Training data for the FD001 scenario.
- `test_FD001.txt`: Test data for the FD001 scenario.
- `RUL_FD001.txt`: Remaining Useful Life for the test data in the FD001 scenario.
- `train_FD002.txt`: Training data for the FD002 scenario.
- `test_FD002.txt`: Test data for the FD002 scenario.
- `RUL_FD002.txt`: Remaining Useful Life for the test data in the FD002 scenario.
- `train_FD003.txt`: Training data for the FD003 scenario.
- `test_FD003.txt`: Test data for the FD003 scenario.
- `RUL_FD003.txt`: Remaining Useful Life for the test data in the FD003 scenario.
- `train_FD004.txt`: Training data for the FD004 scenario.
- `test_FD004.txt`: Test data for the FD004 scenario.
- `RUL_FD004.txt`: Remaining Useful Life for the test data in the FD004 scenario.

## Data Tables
Each file contains multiple columns representing various operational settings and sensor measurements over time for multiple engines:

1. `Unit Number`: Identifier for each engine.
2. `Time in Cycles`: Time cycle of the operation.
2. `Operational Setting 1`: Operational setting parameter 1.
4. `Operational Setting 2`: Operational setting parameter 2.
5. `Operational Setting 3`: Operational setting parameter 3.
6. `Sensor Measurement 1-21`: Various sensor measurements.

## Files
### Notebooks
- `FD001 Predictive Maintenance LSTM.ipynb`: Contains the implementation of the LSTM neural network for predicting RUL, including data preprocessing, model training, and evaluation.
- `FD001 Predictive Maintenance Random Forest.ipynb`: Contains the implementation of the Random Forest regressor for predicting RUL, including data preprocessing, feature engineering, model training, and evaluation.

## Models Used
### Long Short-Term Memory (LSTM) Neural Network
#### Strengths:
- Capable of capturing temporal dependencies in sequential data.
- Effective in learning complex patterns over time.
#### Weaknesses:
- Requires extensive hyperparameter tuning.
- Higher computational cost and longer training times.
#### Results:
- Provided robust predictions of RUL, capturing the time-series nature of the data effectively.
- **Test RMSE**: 51.15
- **Test R²**: -0.50

### Random Forest Regressor
#### Strengths:
- Simple to implement and interpret.
- Handles tabular data efficiently with minimal preprocessing.
- Robust against overfitting due to ensemble nature.
#### Weaknesses:
- May not capture temporal dependencies as effectively as LSTM.
- Performance can degrade with very high-dimensional data.
#### Results:
- Achieved good predictive performance with aggregated features.
- **Validation RMSE**: 63.88
- **Validation R²**: 0.106
- **Test RMSE**: 49.42
- **Test R²**: -0.41

## Analysis of Results
The results of the predictive maintenance models using both the Long Short-Term Memory (LSTM) neural network and the Random Forest regressor indicate that while both approaches offer some insights into predicting the Remaining Useful Life (RUL) of engines, there is significant room for improvement. The LSTM model achieved a test RMSE of 51.15 with an R² of -0.50, demonstrating its ability to capture some temporal dependencies but struggling with overall accuracy and variance explanation. The Random Forest model performed similarly, with a test RMSE of 49.42 and an R² of -0.41, highlighting its ease of implementation and interpretability but falling short in capturing the complex patterns required for precise predictions. These results underscore the need for further enhancements to improve model performance.

## Potential Improvements
1. Increase Training Epochs: Extend the number of training epochs for the LSTM model to allow it to learn more effectively from the data.
2. Hyperparameter Tuning: Conduct comprehensive hyperparameter tuning for both models, including adjustments to learning rates, layer sizes, and dropout rates for the LSTM, and tree depth, number of trees, and sample splits for the Random Forest.
3. Regularization Techniques: Apply regularization methods such as dropout in the LSTM model to reduce overfitting and enhance generalization.
4. Feature Engineering: Enhance feature engineering by incorporating additional aggregate features, lag features, and other transformations to better capture the temporal and operational dynamics in the dataset.