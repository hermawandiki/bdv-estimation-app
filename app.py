import os
import pickle

import pandas as pd
from flask import Flask, jsonify, request

app = Flask(__name__)

# 1. Load model saat server pertama kali menyala
#    (pakai path absolut supaya aman di lingkungan Vercel)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(BASE_DIR, "svr-model.pkl"), "rb") as f:
    model_bundle = pickle.load(f)
best_estimator = model_bundle["model"]


# 2. Halaman web UI
HTML_PAGE = """<!DOCTYPE html>
<html lang="id">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Prediksi BDV</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@500&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #e8ecee;
    --panel: #f6f8f9;
    --ink: #1c2a33;
    --muted: #5a6b76;
    --line: #b9c4cb;
    --accent: #1f5f7a;
    --meter-bg: #15212a;
    --meter-ink: #f2b53a;
    --error: #a62b1f;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #0f171d;
      --panel: #18232b;
      --ink: #e4ebef;
      --muted: #94a5b0;
      --line: #33444f;
      --accent: #5fb4d6;
      --meter-bg: #0a1117;
      --meter-ink: #f2b53a;
      --error: #ff8a7a;
    }
  }
  * { box-sizing: border-box; }
  html, body { height: 100%; }
  body {
    margin: 0;
    background: var(--bg);
    color: var(--ink);
    font-family: "IBM Plex Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 24px;
  }
  main {
    width: 100%;
    max-width: 420px;
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 6px;
    padding: 28px;
  }
  h1 {
    margin: 0 0 4px;
    font-size: 1.35rem;
    font-weight: 600;
    letter-spacing: -0.01em;
  }
  .sub {
    margin: 0 0 24px;
    color: var(--muted);
    font-size: 0.92rem;
    line-height: 1.5;
  }
  label {
    display: block;
    font-size: 0.9rem;
    font-weight: 500;
    margin-bottom: 6px;
  }
  .field { margin-bottom: 16px; }
  .input-wrap { position: relative; }
  input {
    width: 100%;
    font: inherit;
    font-size: 1rem;
    color: var(--ink);
    background: var(--bg);
    border: 1px solid var(--line);
    border-radius: 4px;
    padding: 11px 52px 11px 12px;
  }
  .unit {
    position: absolute;
    right: 12px;
    top: 50%;
    transform: translateY(-50%);
    color: var(--muted);
    font-size: 0.85rem;
    pointer-events: none;
  }
  input:focus-visible, button:focus-visible {
    outline: 3px solid var(--accent);
    outline-offset: 2px;
  }
  button {
    width: 100%;
    font: inherit;
    font-weight: 600;
    color: #fff;
    background: var(--accent);
    border: 0;
    border-radius: 4px;
    padding: 12px;
    cursor: pointer;
    margin-top: 4px;
  }
  @media (prefers-color-scheme: dark) { button { color: #0a1117; } }
  button:disabled { opacity: 0.6; cursor: progress; }
  .meter {
    margin-top: 24px;
    background: var(--meter-bg);
    border-radius: 4px;
    padding: 16px 18px 14px;
  }
  .meter label { color: #9fb0ba; margin-bottom: 2px; }
  .reading {
    display: flex;
    align-items: baseline;
    gap: 10px;
    color: var(--meter-ink);
    font-family: "IBM Plex Mono", ui-monospace, Consolas, monospace;
    font-weight: 500;
  }
  #bdv { font-size: 2.6rem; line-height: 1.15; }
  .reading .u { font-size: 1rem; color: #9fb0ba; }
  .msg {
    min-height: 1.3em;
    margin: 10px 0 0;
    font-size: 0.88rem;
    color: var(--error);
  }
</style>
</head>
<body>
<main>
  <h1>Prediksi BDV</h1>
  <p class="sub">Masukkan temperature dan moisture, lalu hitung nilai BDV.</p>

  <form id="form" novalidate>
    <div class="field">
      <label for="temperature">Temperature</label>
      <div class="input-wrap">
        <input id="temperature" name="temperature" type="number" step="any" inputmode="decimal" required>
        <span class="unit">&deg;C</span>
      </div>
    </div>
    <div class="field">
      <label for="moisture">Moisture</label>
      <div class="input-wrap">
        <input id="moisture" name="moisture" type="number" step="any" inputmode="decimal" required>
        <span class="unit">ppm</span>
      </div>
    </div>
    <button id="btn" type="submit">Hitung BDV</button>
  </form>

  <p id="msg" class="msg" role="alert"></p>

  <div class="meter">
    <label for="bdv">BDV</label>
    <div class="reading">
      <output id="bdv" aria-live="polite">&ndash;</output>
      <span class="u">kV</span>
    </div>
  </div>
</main>

<script>
  const form = document.getElementById("form");
  const btn = document.getElementById("btn");
  const bdv = document.getElementById("bdv");
  const msg = document.getElementById("msg");

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    msg.textContent = "";

    const temperature = form.temperature.value.trim();
    const moisture = form.moisture.value.trim();
    if (temperature === "" || moisture === "") {
      msg.textContent = "Isi temperature dan moisture terlebih dahulu.";
      return;
    }

    btn.disabled = true;
    btn.textContent = "Menghitung...";
    try {
      const res = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ temperature: temperature, moisture: moisture })
      });
      const data = await res.json();
      if (res.ok && data.status === "success") {
        bdv.textContent = Number(data.bdv).toFixed(2);
      } else {
        bdv.textContent = "\u2013";
        msg.textContent = data.message || "Prediksi gagal. Periksa input Anda.";
      }
    } catch (err) {
      bdv.textContent = "\u2013";
      msg.textContent = "Tidak dapat terhubung ke server. Coba lagi.";
    } finally {
      btn.disabled = false;
      btn.textContent = "Hitung BDV";
    }
  });
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return HTML_PAGE


# 3. Endpoint API (tetap sama, bisa dipakai AppSheet maupun web UI)
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        temp = float(data["temperature"])
        moisture = float(data["moisture"])

        interaction = temp * moisture

        df_input = pd.DataFrame(
            [[temp, moisture, interaction]],
            columns=["temp", "moisture", "interaction"],
        )

        hasil_prediksi = best_estimator.predict(df_input)[0]

        return jsonify({"status": "success", "bdv": float(hasil_prediksi)}), 200

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)