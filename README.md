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

