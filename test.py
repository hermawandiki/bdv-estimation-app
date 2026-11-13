import os
import pickle
import pandas as pd
import warnings
from sklearn.exceptions import InconsistentVersionWarning

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

temp, moisture = 32.0, 123.0

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FILES = {
    "Mineral Oil": "model_mo.pkl",
    "Natural Ester Oil": "model_neo.pkl",
    "Synthetic Ester Oil": "model_seo.pkl",
}

data = {"temp": temp, "moisture": moisture, "interaction": temp * moisture}

for oil, filename in FILES.items():
    print("=" * 50)
    print(oil, f"({filename})")
    try:
        with open(os.path.join(BASE_DIR, filename), "rb") as f:
            bundle = pickle.load(f)
        model = bundle["model"] if isinstance(bundle, dict) else bundle
        cols = list(model.feature_names_in_)
        X = pd.DataFrame([[data[c] for c in cols]], columns=cols)
        print("BDV   :", round(float(model.predict(X)[0]), 2), "kV")
    except Exception as e:
        print("ERROR :", type(e).__name__, e)