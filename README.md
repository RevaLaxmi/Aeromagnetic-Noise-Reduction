# Aeromagnetic Noise Reduction - MLP Model

## Overview

This repository contains a machine learning model designed to predict the target variable from aeromagnetic data using a **Multilayer Perceptron (MLP)** architecture. The model is trained to predict a continuous target variable, with the aim of reducing noise in aeromagnetic data and providing high accuracy in predictions.

## Model Results

### Target Variable

- **Min of True Target**: 52,948.136
- **Max of True Target**: 55,282.793
- **Mean of True Target**: 53,711.41

The true target values lie within the range of approximately 52,948 to 55,282, with a mean value of 53,711.41.

### Performance Metrics

- **Mean Squared Error (MSE)**: 1.3289
- **Root Mean Squared Error (RMSE)**: 1.1532
- **Percentage Error**: 0.0021%

The **RMSE** value of approximately 1.1532 indicates that the model’s average prediction deviates from the true target value by around 1.15 units. When expressed as a percentage of the mean target value, the model achieves an **extremely low error of 0.0021%**. This highlights that the model’s predictions are highly accurate and well-calibrated with the original data scale.

### Conclusion

- The **low RMSE** and corresponding **percentage error** indicate that the model captures the underlying data patterns effectively, with minimal overfitting or underfitting.
- The model has been optimized to predict the target variable with negligible deviation, suggesting its high performance in predicting unseen data.

## MLP Architecture

### Layer Configuration

1. **Input Layer**: 
   - Number of neurons equal to the number of input features (magnetometer data).
   
2. **Hidden Layer 1**:
   - 50 neurons with **tanh** activation function for non-linear relationship modeling.
   
3. **Hidden Layer 2**:
   - 30 neurons with **tanh** activation for further feature refinement.

4. **Hidden Layer 3**:
   - 10 neurons with **tanh** activation to extract deeper patterns.

5. **Output Layer**:
   - 1 neuron (regression task, predicting a continuous value). No activation (linear output).

### Compilation

- **Loss Function**: Mean Squared Error (MSE), suitable for regression tasks.
- **Optimizer**: Adam (adaptive gradient descent).

### Training

- **Epochs**: 100 iterations through the dataset.
- **Batch Size**: 32 samples per batch for weight updates.
- **Validation**: Performance is evaluated on the validation set during training to monitor for overfitting or underfitting.

### Activation Functions

- The **tanh** activation function is used throughout the hidden layers to model non-linear relationships in the data. This was chosen after experimentation, particularly when some weights were going negative during feature selection.

## Notes

- **Model Variability**: Performance may vary slightly between training sessions due to factors like the number of epochs and dropout usage during training. This is typical in deep learning models, and the results depend on various training parameters.







# -x-x-x-x-x-x-x-

# Aeromagnetic-Noise-Reduction

src3
MAE: 153.466, RMSE: 191.474, Correlation: 1.000


---

## **Step 1: Data Preprocessing (`preprocess.py`)**
**Files Used:**
- **Training Data** → `Flt1003_train.h5`
- **Testing Data** → `Flt1005_train.h5`

### **🔹 What Happens Here?**
1. **Extracts data from HDF5 files**  
   - Reads **three sensor values** (`flux_c_x`, `flux_c_y`, `flux_c_z`) as **input (X)**  
   - Reads **three true flux values** (`flux_b_x`, `flux_b_y`, `flux_b_z`) as **output (y)**  

2. **Standardizes the input (`X`) using a scaler**
   - The model works better when values are in a similar range, so we normalize `X`.

   - Training Data: (160030, 3) → 160,030 samples with 3 features each.
   - Testing Data: (81731, 3) → 81,731 samples with 3 features each.
  
   - If you train on all the data, the model might just memorize it instead of learning general patterns. Testing on unseen data helps check if the model can make accurate predictions on real-world cases. -> problem we saw around epoch 15 in src1



**Final Output:**  
- **Scaler file:** `scaler.pkl`  -> also using model.keras in src3

---

## **Step 2: Model Training (`train.py`)**
**Files Used:**
- `train_X.npy`, `train_y.npy` (processed training data)

### **🔹 What Happens Here?**
1. **Loads the preprocessed training data (`train_X`, `train_y`)**
2. **Reshapes `train_X`** so it works with an LSTM 
3. **Builds a neural network model using LSTMs**
   - **First LSTM layer** → Learns time-based patterns. -> src3 had an improvement jump compared to src1
   - **Second LSTM layer** → can play around with this layer - work on optimizing if we add more layers 
4. **Trains the model**
   - Uses **Mean Squared Error (MSE)** as the loss function.
   - Trains for **50 epochs** with batch size **64**. -> compared to 16/20 for src1
5. **Saves the trained model** as `lstm_model.h5`.

**Final Output:**  
- **Trained LSTM model:** `lstm_model.h5`

---

## **Step 3: Model Evaluation (`evaluate.py`)**
**Files Used:**
- `test_X.npy`, `test_y.npy` (processed testing data)
- `lstm_model.h5` (trained model)

### **🔹 What Happens Here?**
1. **Loads the test data (`test_X`, `test_y`)**
2. **Reshapes `test_X`** to match LSTM input format.
3. **Loads the trained model (`lstm_model.h5`)**
4. **Makes predictions on test data**
5. **Evaluates performance** using:
   - **Mean Absolute Error (MAE)** → Measures average error.
   - **Root Mean Squared Error (RMSE)** → Measures squared error.
   - **Correlation Coefficient** → Checks how well predictions match actual values.

**Final Output:**  
- **Evaluation metrics (MAE, RMSE, Correlation)**

---

