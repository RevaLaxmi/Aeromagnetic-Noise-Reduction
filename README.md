# Aeromagnetic-Noise-Reduction

src3
MAE: 153.466, RMSE: 191.474, Correlation: 1.000

Alright, let me break it down step by step in **simple terms**, keeping it focused on **what each script does** and how it relates to the **flight data (Flt1003_train.h5 & Flt1005_train.h5)**.

---

## **Step 1: Data Preprocessing (`preprocess.py`)**
📂 **Files Used:**
- **Training Data** → `Flt1003_train.h5`
- **Testing Data** → `Flt1005_train.h5`

### **🔹 What Happens Here?**
1. **Extracts data from HDF5 files**  
   - Reads **three sensor values** (`flux_c_x`, `flux_c_y`, `flux_c_z`) as **input (X)**  
   - Reads **three true flux values** (`flux_b_x`, `flux_b_y`, `flux_b_z`) as **output (y)**  

2. **Standardizes the input (`X`) using a scaler**
   - The model works better when values are in a similar range, so we normalize `X`.
   - For **training**, we **fit the scaler** and save it for later use.
   - For **testing**, we **use the same scaler** to transform the data.

3. **Splits the dataset**
   - **66% of the data → Training (`train_X`, `train_y`)**
   - **34% of the data → Testing (`test_X`, `test_y`)**
   - Saves these as `.npy` files for later use.

📌 **Final Output:**  
- **Preprocessed data files:** `train_X.npy`, `train_y.npy`, `test_X.npy`, `test_y.npy`
- **Scaler file:** `scaler.pkl`  

---

## **Step 2: Model Training (`train.py`)**
📂 **Files Used:**
- `train_X.npy`, `train_y.npy` (processed training data)

### **🔹 What Happens Here?**
1. **Loads the preprocessed training data (`train_X`, `train_y`)**
2. **Reshapes `train_X`** so it works with an LSTM (which expects time series data).
3. **Builds a neural network model using LSTMs**
   - **First LSTM layer** → Learns time-based patterns.
   - **Second LSTM layer** → Extracts deeper relationships.
   - **Dense layers** → Refine and map to the three output values.
4. **Trains the model**
   - Uses **Mean Squared Error (MSE)** as the loss function.
   - Trains for **50 epochs** with batch size **64**.
5. **Saves the trained model** as `lstm_model.h5`.

📌 **Final Output:**  
- **Trained LSTM model:** `lstm_model.h5`

---

## **Step 3: Model Evaluation (`evaluate.py`)**
📂 **Files Used:**
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

📌 **Final Output:**  
- **Predictions (`pred_y`)**
- **Evaluation metrics (MAE, RMSE, Correlation)**

---

## **💡 TL;DR (What We Did)**
1. **Preprocessing** → Extracted flight sensor data, scaled it, and split it into training (`Flt1003_train.h5`) and testing (`Flt1005_train.h5`).
2. **Training** → Built an LSTM model, trained it on the processed `Flt1003_train.h5` data, and saved the trained model.
3. **Evaluation** → Used `Flt1005_train.h5` to test the model and computed performance metrics.

---

### **🚀 Next Steps?**
- Do you need help debugging `evaluate.py`?
- Are you looking for improvements in training/testing?
- Need to upload this to GitHub without hitting the file size limit?

Let me know how you want to proceed! 🚀
