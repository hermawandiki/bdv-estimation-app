import os
import pickle

import pandas as pd
from flask import Flask, render_template_string, request

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# kode -> (nama tampilan, file model)
OILS = {
    "mo": ("Mineral Oil", "model_mo.pkl"),
    "neo": ("Natural Ester Oil", "model_neo.pkl"),
    "seo": ("Synthetic Ester Oil", "model_seo.pkl"),
}


def pick_model(obj):
    if hasattr(obj, "predict"):
        return obj
    if isinstance(obj, dict):
        if hasattr(obj.get("model"), "predict"):
            return obj["model"]
        if isinstance(obj.get("models"), dict):
            for m in obj["models"].values():
                if hasattr(m, "predict"):
                    return m
        for m in obj.values():
            if hasattr(m, "predict"):
                return m
    raise ValueError("No model (object with .predict) found in the pkl file")


# Muat semua model saat start. File yang hilang/rusak dicatat, bukan mematikan app.
MODELS = {}
LOAD_ERRORS = {}
for code, (label, filename) in OILS.items():
    try:
        with open(os.path.join(BASE_DIR, filename), "rb") as f:
            MODELS[code] = pick_model(pickle.load(f))
    except Exception as e:
        LOAD_ERRORS[code] = f"Failed to load {filename}: {e}"


def predict_bdv(model, temp, moist):
    feats = {"temp": temp, "moisture": moist, "interaction": temp * moist}

    names = getattr(model, "feature_names_in_", None)
    if names is not None:
        cols = [str(c) for c in names]
    else:
        n = getattr(model, "n_features_in_", 2)
        cols = ["temp", "moisture", "interaction"][:n]

    missing = [c for c in cols if c not in feats]
    if missing:
        raise ValueError(f"Unknown features: {missing}")

    row = [feats[c] for c in cols]
    X = pd.DataFrame([row], columns=cols) if names is not None else [row]
    return float(model.predict(X)[0])


HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>BDV Prediction</title>
<style>
  :root { --bg:#e8ecee; --panel:#f6f8f9; --ink:#1c2a33; --muted:#5a6b76;
          --line:#b9c4cb; --accent:#1f5f7a; --error:#a62b1f;
          --meter-bg:#15212a; --meter-ink:#f2b53a; }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#0f171d; --panel:#18232b; --ink:#e4ebef; --muted:#94a5b0;
            --line:#33444f; --accent:#5fb4d6; --error:#ff8a7a;
            --meter-bg:#0a1117; }
  }
  * { box-sizing: border-box; }
  body { margin:0; min-height:100vh; background:var(--bg); color:var(--ink);
         font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
         display:flex; align-items:center; justify-content:center; padding:24px; }
  main { width:100%; max-width:420px; background:var(--panel);
         border:1px solid var(--line); border-radius:6px; padding:28px; }
  h1 { margin:0 0 4px; font-size:1.3rem; }
  .sub { margin:0 0 22px; color:var(--muted); font-size:.92rem; }
  label { display:block; font-size:.9rem; font-weight:500; margin-bottom:6px; }
  .field { margin-bottom:14px; }
  input, select { width:100%; font:inherit; color:var(--ink); background:var(--bg);
          border:1px solid var(--line); border-radius:4px; padding:10px 12px; }
  button { width:100%; font:inherit; font-weight:600; color:#fff;
           background:var(--accent); border:0; border-radius:4px;
           padding:12px; cursor:pointer; }
  @media (prefers-color-scheme: dark) { button { color:#0a1117; } }
  .err { color:var(--error); font-size:.88rem; margin:10px 0 0; }
  .meter { margin-top:22px; background:var(--meter-bg); border-radius:4px;
           padding:16px 18px; }
  .cap { display:block; font-size:.9rem; color:#9fb0ba; margin-bottom:2px; }
  .reading { color:var(--meter-ink); font-family: ui-monospace, Consolas, monospace;
             font-size:2.6rem; }
  .reading .u { font-size:1rem; color:#9fb0ba; margin-left:8px; }
</style>
</head>
<body>
<main>
  <h1>Breakdown Voltage (BDV) Prediction</h1>
  <p class="sub">Select the oil type, then enter temperature and moisture.</p>

  <form method="post" action="/">
    <div class="field">
      <label for="oil">Oil Type</label>
      <select id="oil" name="oil">
        {% for code, label in oils %}
        <option value="{{ code }}" {{ 'selected' if code == oil else '' }}>{{ label }}</option>
        {% endfor %}
      </select>
    </div>
    <div class="field">
      <label for="temperature">Temperature (&deg;C)</label>
      <input id="temperature" name="temperature" type="number" step="any"
             value="{{ temperature }}" required>
    </div>
    <div class="field">
      <label for="moisture">Moisture (ppm)</label>
      <input id="moisture" name="moisture" type="number" step="any"
             value="{{ moisture }}" required>
    </div>
    <button type="submit">Calculate BDV</button>
  </form>

  {% if error %}<p class="err" role="alert">{{ error }}</p>{% endif %}

  <div class="meter">
    <span class="cap">BDV</span>
    <div class="reading">{{ bdv if bdv else "–" }}<span class="u">kV</span></div>
  </div>
</main>
</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def index():
    oil = "mo"
    temperature = moisture = bdv = error = ""

    if request.method == "POST":
        oil = request.form.get("oil", "mo")
        temperature = request.form.get("temperature", "").strip()
        moisture = request.form.get("moisture", "").strip()

        if oil not in OILS:
            error = "Invalid oil type."
        else:
            try:
                temp = float(temperature)
                moist = float(moisture)
            except ValueError:
                error = "Temperature and moisture must be numbers."
            else:
                if oil not in MODELS:
                    error = LOAD_ERRORS.get(oil, "Model is not available.")
                else:
                    try:
                        bdv = f"{predict_bdv(MODELS[oil], temp, moist):.2f}"
                    except Exception as e:
                        error = f"Prediction failed: {e}"

    return render_template_string(
        HTML_PAGE,
        oils=[(c, l) for c, (l, _) in OILS.items()],
        oil=oil,
        temperature=temperature,
        moisture=moisture,
        bdv=bdv,
        error=error,
    )


if __name__ == "__main__":
    for msg in LOAD_ERRORS.values():
        print("WARNING:", msg)
    app.run(host="0.0.0.0", port=5000, debug=False)