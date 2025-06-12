# Capstone Project – Team ID: CC25-CF197 | Deployment Preparation for the "WisataPas" Model

**Hybrid Travel Recommendation System** adalah sistem rekomendasi tempat wisata yang menggabungkan pendekatan **Collaborative Filtering** dan **Content-Based Filtering**. Sistem ini dirancang untuk memberikan rekomendasi yang relevan dan personal kepada pengguna berdasarkan preferensi pengguna dan karakteristik destinasi wisata.

Sebagai fitur tambahan opsional, sistem juga menyediakan fungsi sederhana untuk menghasilkan deskripsi rekomendasi menggunakan model generatif **T5**.

---

## Fitur Utama

- **Collaborative Filtering**  
  Menggunakan model `RecommenderNet` (TensorFlow) untuk mempelajari interaksi historis pengguna dan memprediksi destinasi yang mungkin disukai.

- **Content-Based Filtering**  
  Menerapkan teknik **TF-IDF** dan **cosine similarity** berdasarkan atribut destinasi seperti nama, kategori, lokasi, dan deskripsi.

- **Pencarian Berbasis Teks**  
  Mendukung pencarian menggunakan kalimat alami, misalnya:  
  `"wisata sejarah di Jakarta"` untuk menghasilkan rekomendasi yang relevan.

---

## Fitur Tambahan (Opsional)

- **Deskripsi Rekomendasi Otomatis**  
  Menggunakan model generatif T5 (fine-tuned) untuk menghasilkan kalimat deskriptif dari hasil rekomendasi.  
  > Fitur ini bersifat pelengkap dan tidak memengaruhi sistem rekomendasi utama.

---

## Struktur Proyek

```
.
├── data/
│   ├── destinasi_df.csv
│   ├── rating_df.csv
│   └── cb_df.csv
├── models
│   ├── collab_model.keras
│   ├── cosine_sim_df.pkl
│   ├── tfidf_vectorizer.pkl
│   ├── user_to_user_encoded.pkl
│   └── place_to_place_encoded.pkl
├── t5-finetuned-recommendation-final/ <- harus di unduh terlbih dahulu
│   └── [model generatif T5]
├── main.py
└── requirements.txt
````

---

## Download Model T5 (Opsional)

Untuk menggunakan fitur deskripsi otomatis, unduh model T5 melalui tautan berikut:

[Google Drive – T5 Fine-Tuned Model](https://drive.google.com/drive/folders/19uD1UfR2xSmOVEC0ulhwsxvNsn73evMD?usp=share_link)

Letakkan isi folder hasil unduhan ke dalam:  
`t5-finetuned-recommendation-final/`

---

## Contoh Penggunaan

```python
# Menghasilkan rekomendasi hybrid
recommendations = get_travel_recommendations(user_id=3, favorite_place="Monumen Nasional")

# Rekomendasi berdasarkan input teks
results = infer_cbf_search("wisata alam daerah yogyakarta", top_k=5)

# (Opsional) Deskripsi hasil rekomendasi menggunakan T5
text_output = generate_natural_recommendation(user_id=1, favorite_place="Monumen Nasional")
print(text_output)
````

---

## Instalasi Dependensi

```bash
pip install -r requirements.txt
```

---

## Model & Teknologi

| Komponen                   | Teknologi                     |
| -------------------------- | ----------------------------- |
| Collaborative Filtering    | TensorFlow (`RecommenderNet`) |
| Content-Based Filtering    | TF-IDF, Cosine Similarity     |
| Text Generation (Opsional) | T5 (HuggingFace Transformers) |

---

## Note

* Model collaborative memerlukan file encoding pengguna dan tempat wisata.
* Fitur T5 hanya aktif jika model telah diunduh dan dimuat.
* Dataset dan model disimpan dalam direktori `data/` dan `models/`.