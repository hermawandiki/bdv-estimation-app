from flask import Flask, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# 1. Load model saat server pertama kali menyala
with open("svr-model.pkl", "rb") as f:
    model_bundle = pickle.load(f)
best_estimator = model_bundle["model"]

# 2. Buat rute endpoint untuk menerima data
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Menangkap data JSON yang dikirim (nanti dari AppSheet)
        data = request.get_json()
        
        # Ekstrak nilai suhu dan kelembapan
        temp = float(data['temperature'])
        moisture = float(data['moisture'])
        
        # Hitung fitur interaksi
        interaction = temp * moisture
        
        # Susun ke dalam format DataFrame (seperti saat Colab)
        df_input = pd.DataFrame(
            [[temp, moisture, interaction]], 
            columns=["temp", "moisture", "interaction"]
        )
        
        # Lakukan prediksi
        hasil_prediksi = best_estimator.predict(df_input)[0]
        
        # Kembalikan hasil dalam bentuk JSON
        return jsonify({
            "status": "success",
            "bdv": float(hasil_prediksi)
        }), 200

    except Exception as e:
        # Jika ada error (misal format data salah), kembalikan pesan error
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 400

if __name__ == '__main__':
    # Jalankan server API
    app.run(host='0.0.0.0', port=5000)