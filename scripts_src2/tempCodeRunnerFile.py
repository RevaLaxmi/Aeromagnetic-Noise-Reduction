import os
import sys
from data_loader import load_data, preprocess_data
from model import build_lstm
from sklearn.model_selection import train_test_split
import pickle

# ✅ Ensure the script can find everything
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# ✅ Load and preprocess training data
df_train = load_data("Flt1003_train.h5", data_dir="data")  # 🔹 Fixed path
X_train, y_train, scaler = preprocess_data(df_train)

# ✅ Build and train the model
model = build_lstm()
model.fit(X_train, y_train, epochs=20, batch_size=16)

# ✅ Save the trained model
model.save("lstm_flux_model.h5")

# ✅ Save the scaler for future use
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("Model and scaler saved successfully!")
