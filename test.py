import os
import pickle
import pandas as pd

temp, moisture = 32.0, 123.0

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(BASE_DIR, "model.pkl"), "rb") as f:
    bundle = pickle.load(f)

model = bundle["model"] if isinstance(bundle, dict) else bundle

print("Model :", type(model).__name__)
print("Fitur :", getattr(model, "feature_names_in_", model.n_features_in_))

data = {"temp": temp, "moisture": moisture, "interaction": temp * moisture}
cols = list(model.feature_names_in_)
X = pd.DataFrame([[data[c] for c in cols]], columns=cols)

print("BDV   :", round(float(model.predict(X)[0]), 2))