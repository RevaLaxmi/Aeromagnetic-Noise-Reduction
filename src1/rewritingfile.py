# code to work on redoing the file

from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError

model = load_model("lstm_flux_model.h5", compile=False)
model.save("model.keras", save_format="keras")
