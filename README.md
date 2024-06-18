# Predictive Maintenance using Machine Learning
Machinery failure prediction using multiple Machine Learning Models using Python, Tensorflow, Keras, Scikit-Learn, Pandas and NumPy in Jupyter Notebooks.

## Goal of the Project
The goal of this project is to predict the remaining useful life (RUL) of turbofan engines using machine learning models. This project evaluates different models to find the most effective approach for predictive maintenance. The models used include:
- Long Short-Term Memory (LSTM)
- Random Forest
- Gradient Boosting Machine (GBM) using XGBoost

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

### Data Set: FD001

**Train trajectories**: 100  
**Test trajectories**: 100  
**Conditions**: ONE (Sea Level)  
**Fault Modes**: ONE (HPC Degradation)

### Data Set: FD002

**Train trajectories**: 260  
**Test trajectories**: 259  
**Conditions**: SIX  
**Fault Modes**: ONE (HPC Degradation)

### Data Set: FD003

**Train trajectories**: 100  
**Test trajectories**: 100  
**Conditions**: ONE (Sea Level)  
**Fault Modes**: TWO (HPC Degradation, Fan Degradation)

### Data Set: FD004

**Train trajectories**: 248  
**Test trajectories**: 249  
**Conditions**: SIX  
**Fault Modes**: TWO (HPC Degradation, Fan Degradation)

### Citation
`Saxena, A., Goebel, K., Simon, D., & Eklund, N. (2008). "Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation." In Proceedings of the 1st International Conference on Prognostics and Health Management (PHM08), Denver, CO, October 2008.`

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
- `FD001 Predictive Maintenance GBM.ipynb`: Contains the implementation of the Gradient Boost Machine model for predicting RUL, including data preprocessing, feature engineering, model training, and evaluation.

## Running the Project
1. Clone the repository:
    ```bash
    git clone git@github.com:KawalpreetDeol/machinery-failure-prediction.git
    ```
2. Install the necessary libraries:
    ```bash
    pip install -r requirements.txt
    ```
3. Open the Jupyter Notebooks to run the models:
    ```bash
    jupyter notebook
    ```
4. Navigate to the respective notebook and run the cells to see the results.

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

### Gradient Boosting Machine (GBM) using XGBoost

#### Strengths:
- High predictive accuracy.
- Handles various data types well.
- Provides several hyperparameters for tuning.

#### Weaknesses:
- Computationally intensive.
- Prone to overfitting if not properly tuned.

#### Results:
- The XGBoost model achieved the following results:
  - Test RMSE: 47.51
  - Test R²: -0.31
- Feature importance plots and prediction-residuals analysis indicate areas for further improvement.

## Analysis of Results
The results of the predictive maintenance models using the Long Short-Term Memory (LSTM) neural network, the Random Forest regressor, and the Gradient Boosting Machine (GBM) using XGBoost indicate that while all approaches offer some insights into predicting the Remaining Useful Life (RUL) of engines, there is significant room for improvement.

- **LSTM**: The LSTM model achieved a test RMSE of 51.15 with an R² of -0.50, demonstrating its ability to capture some temporal dependencies but struggling with overall accuracy and variance explanation.
- **Random Forest**: The Random Forest model performed similarly, with a test RMSE of 49.42 and an R² of -0.41, highlighting its ease of implementation and interpretability but falling short in capturing the complex patterns required for precise predictions.
- **GBM (XGBoost)**: The XGBoost model showed promising results with a test RMSE of 47.51 and an R² of -0.31, leveraging its ability to handle various data types and hyperparameter tuning to improve predictive accuracy.

## Potential Improvements
1. Increase Training Epochs: Extend the number of training epochs for the LSTM model to allow it to learn more effectively from the data.
2. Hyperparameter Tuning: Conduct comprehensive hyperparameter tuning for both models, including adjustments to learning rates, layer sizes, and dropout rates for the LSTM, and tree depth, number of trees, and sample splits for the Random Forest.
3. Regularization Techniques: Apply regularization methods such as dropout in the LSTM model to reduce overfitting and enhance generalization.
4. Feature Engineering: Enhance feature engineering by incorporating additional aggregate features, lag features, and other transformations to better capture the temporal and operational dynamics in the dataset.
5. Model Ensembling: Combine predictions from multiple models to improve accuracy.