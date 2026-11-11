import requests

# Alamat lokal API Anda (sesuaikan port jika berbeda)
url = 'https://bdv-estimation-app.vercel.app/predict'

# Data dummy yang dikirim (seolah-olah dari AppSheet)
data_kirim = {
    "temperature": 31.9,
    "moisture": 363
}

print("Mengirim data ke API...")
response = requests.post(url, json=data_kirim)

print("Status Code:", response.status_code)
print("Balasan dari API:", response.json())