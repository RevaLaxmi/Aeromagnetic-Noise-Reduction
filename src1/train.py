from data_loader import load_data, preprocess_data
from model import build_lstm
from sklearn.model_selection import train_test_split

# Load and preprocess data
df = load_data()
X, y, scaler = preprocess_data(df)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build and train the model
model = build_lstm()
model.fit(X_train, y_train, epochs=20, batch_size=16, validation_data=(X_test, y_test))

# Save model
model.save("lstm_flux_model.h5")
print("Model saved successfully!")
