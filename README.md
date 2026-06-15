# Laporan Proyek Machine Learning - Nurul Azmi Muharom

## Deteksi Website Phishing Menggunakan Machine Learning Berbasis Feature URL pada Dataset Kaggle

## Project Overview

Website phishing merupakan salah satu bentuk ancaman keamanan siber yang digunakan untuk menipu pengguna melalui halaman web palsu. Website ini biasanya dibuat menyerupai website resmi agar pengguna memasukkan informasi sensitif seperti username, password, email, atau data akun lainnya.

Deteksi phishing secara manual sulit dilakukan karena URL phishing sering dibuat mirip dengan website asli. Oleh karena itu, machine learning dapat digunakan untuk membantu mengklasifikasikan website berdasarkan fitur yang terdapat pada URL dan karakteristik halaman web.

Proyek ini bertujuan untuk membangun model klasifikasi website phishing menggunakan dataset dari Kaggle. Model yang digunakan adalah Decision Tree dan XGBoost. Hasil model terbaik kemudian di-deploy menggunakan Hugging Face Spaces dengan Gradio.

💡 Manfaat Proyek:

✔ Membantu mendeteksi website phishing secara otomatis.
✔ Mengurangi risiko pengguna mengakses website berbahaya.
✔ Membandingkan performa model klasifikasi yang sudah dipelajari dan model eksplorasi tambahan.
✔ Menghasilkan aplikasi sederhana yang dapat digunakan untuk prediksi website legitimate atau phishing.

## Business Understanding

### 📝 Problem Statements

* Bagaimana machine learning dapat digunakan untuk mendeteksi website phishing berdasarkan feature URL?
* Bagaimana performa Decision Tree dan XGBoost dalam mengklasifikasikan website legitimate dan phishing?
* Model mana yang memberikan hasil terbaik berdasarkan metrik evaluasi?
* Bagaimana model terbaik dapat diterapkan dalam bentuk aplikasi deployment sederhana?

### 🎯 Goals

* Membuat model klasifikasi untuk mendeteksi website phishing.
* Menggunakan dataset phishing dari Kaggle.
* Membandingkan model Decision Tree dan XGBoost.
* Menentukan model terbaik berdasarkan accuracy, precision, recall, dan F1 Score.
* Melakukan deployment model menggunakan Hugging Face Spaces.

### 🛠 Solution Approach

✔ Decision Tree
Decision Tree digunakan sebagai model klasifikasi yang sudah dipelajari di kelas. Model ini cocok untuk klasifikasi karena dapat membentuk aturan keputusan berdasarkan fitur tertentu.

✔ XGBoost
XGBoost digunakan sebagai model eksplorasi tambahan. Model ini dipilih karena belum dipelajari di kelas dan memiliki kemampuan yang baik dalam menangani masalah klasifikasi.

## Data Understanding

Dataset yang digunakan adalah Web Page Phishing Detection Dataset dari Kaggle. Dataset ini berisi fitur website yang digunakan untuk mengklasifikasikan website menjadi legitimate atau phishing.

Link dataset:
https://www.kaggle.com/datasets/shashwatwork/web-page-phishing-detection-dataset

📂 Informasi Dataset:

| Informasi         | Keterangan              |
| ----------------- | ----------------------- |
| Jumlah Data       | 11.430 baris            |
| Jumlah Kolom      | 89 kolom                |
| Target            | status                  |
| Kelas Target      | legitimate dan phishing |
| Jumlah legitimate | 5.715 data              |
| Jumlah phishing   | 5.715 data              |

📌 Beberapa fitur yang digunakan:

| Fitur           | Keterangan                           |
| --------------- | ------------------------------------ |
| length_url      | Panjang URL                          |
| length_hostname | Panjang hostname                     |
| nb_dots         | Jumlah titik pada URL                |
| nb_hyphens      | Jumlah tanda hubung pada URL         |
| phish_hints     | Indikasi kata atau pola mencurigakan |
| page_rank       | Nilai page rank website              |
| web_traffic     | Trafik website                       |
| google_index    | Status indeks Google                 |
| domain_age      | Usia domain                          |

🔍 Kondisi Data:

1. Missing Value
   Tidak ditemukan missing value pada dataset.

2. Data Duplikat
   Tidak ditemukan data duplikat.

3. Distribusi Target
   Distribusi kelas target seimbang, yaitu 5.715 data legitimate dan 5.715 data phishing.

Distribusi target yang seimbang membantu model belajar dari kedua kelas dengan proporsi yang sama.

## Exploratory Data Analysis

EDA dilakukan untuk memahami pola awal pada dataset sebelum modeling. Visualisasi yang digunakan meliputi:

✔ Distribusi target legitimate dan phishing.
✔ Perbandingan rata-rata beberapa fitur URL berdasarkan status website.
✔ Scatter plot hubungan panjang URL dan jumlah titik pada URL.

Hasil EDA menunjukkan bahwa beberapa fitur URL memiliki pola yang berbeda antara website legitimate dan phishing. Perbedaan ini dapat membantu model dalam proses klasifikasi.

## Data Preparation

Tahapan data preparation dilakukan untuk menyiapkan data sebelum digunakan pada proses modeling.

📌 Langkah yang dilakukan:

✔ Mengecek missing value.
✔ Mengecek data duplikat.
✔ Melakukan encoding target.
✔ Menghapus kolom url karena berbentuk teks mentah.
✔ Memisahkan fitur dan target.
✔ Membagi data menjadi data training dan testing.

Target status diubah menjadi bentuk numerik:

| Label      | Nilai |
| ---------- | ----: |
| legitimate |     0 |
| phishing   |     1 |

Kode encoding target:

```python
df["status_encoded"] = df["status"].map({
    "legitimate": 0,
    "phishing": 1
})
```

Kolom `url` dihapus karena pada proyek ini model menggunakan extracted features, bukan teks URL mentah.

```python
X = df.drop(columns=["url", "status", "status_encoded"])
y = df["status_encoded"]
```

Data dibagi menjadi data training dan testing dengan rasio 80:20.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

Penggunaan `stratify=y` bertujuan menjaga proporsi kelas legitimate dan phishing agar tetap seimbang pada data training dan testing.

## Modeling and Results

Pada tahap modeling, digunakan dua model klasifikasi yaitu Decision Tree dan XGBoost.

### 1. Decision Tree

Decision Tree merupakan model klasifikasi berbasis struktur pohon keputusan. Model ini membagi data berdasarkan fitur tertentu hingga menghasilkan keputusan akhir.

✔ Kelebihan:

* Mudah dipahami.
* Cocok untuk klasifikasi.
* Tidak membutuhkan scaling data.

✔ Kekurangan:

* Rentan overfitting jika pohon terlalu kompleks.
* Performa dapat lebih rendah dibanding model ensemble.

Kode model:

```python
decision_tree = DecisionTreeClassifier(
    random_state=42,
    max_depth=5
)

decision_tree.fit(X_train, y_train)
y_pred_dt = decision_tree.predict(X_test)
```

### 2. XGBoost

XGBoost merupakan model ensemble berbasis boosting. Model ini membangun beberapa pohon keputusan secara bertahap untuk memperbaiki kesalahan dari model sebelumnya.

✔ Kelebihan:

* Performa tinggi untuk klasifikasi.
* Cocok untuk dataset tabular.
* Mampu menangani pola data yang lebih kompleks.

✔ Kekurangan:

* Lebih kompleks dibanding Decision Tree.
* Membutuhkan pengaturan parameter yang lebih teliti.

Kode model:

```python
xgboost_model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42,
    eval_metric="logloss"
)

xgboost_model.fit(X_train, y_train)
y_pred_xgb = xgboost_model.predict(X_test)
```

## Evaluation Model

Evaluasi dilakukan untuk melihat performa model dalam mengklasifikasikan website legitimate dan phishing.

Metrik yang digunakan:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix
* Classification Report

📊 Hasil Evaluasi Model:

| Model         | Accuracy | Precision | Recall | F1 Score |
| ------------- | -------: | --------: | -----: | -------: |
| Decision Tree |   0.9243 |    0.9369 | 0.9099 |   0.9232 |
| XGBoost       |   0.9602 |    0.9582 | 0.9624 |   0.9603 |

### Evaluasi Decision Tree

Confusion Matrix Decision Tree:

| Actual / Predicted | Legitimate | Phishing |
| ------------------ | ---------: | -------: |
| Legitimate         |       1073 |       70 |
| Phishing           |        103 |     1040 |

Decision Tree menghasilkan accuracy sebesar 0.9243 dan F1 Score sebesar 0.9232. Hasil ini menunjukkan bahwa Decision Tree sudah cukup baik dalam membedakan website legitimate dan phishing.

### Evaluasi XGBoost

Confusion Matrix XGBoost:

| Actual / Predicted | Legitimate | Phishing |
| ------------------ | ---------: | -------: |
| Legitimate         |       1095 |       48 |
| Phishing           |         43 |     1100 |

XGBoost menghasilkan accuracy sebesar 0.9602 dan F1 Score sebesar 0.9603. Hasil ini lebih tinggi dibanding Decision Tree. XGBoost juga memiliki jumlah kesalahan prediksi yang lebih sedikit berdasarkan confusion matrix.

📌 Model Terbaik:

Berdasarkan hasil evaluasi, model terbaik adalah XGBoost karena memiliki F1 Score tertinggi, yaitu 0.9603.

F1 Score digunakan sebagai acuan utama karena metrik ini menyeimbangkan precision dan recall. Pada kasus phishing, recall penting karena model perlu mengenali website phishing sebanyak mungkin.

## Feature Importance

Feature importance digunakan untuk melihat fitur yang paling berpengaruh terhadap hasil prediksi model. Hasil ini membantu menjelaskan fitur mana yang paling banyak digunakan model dalam membedakan website legitimate dan phishing.

Beberapa fitur penting yang muncul pada proses model antara lain:

* google_index
* page_rank
* nb_hyperlinks
* web_traffic
* ratio_intHyperlinks
* domain_age
* phish_hints

Fitur tersebut berhubungan dengan reputasi website, struktur URL, dan karakteristik halaman web.

## Deployment

Model terbaik disimpan dalam format pickle agar dapat digunakan kembali tanpa melakukan training ulang.

File deployment yang digunakan:

| File                    | Fungsi                               |
| ----------------------- | ------------------------------------ |
| app.py                  | File aplikasi Gradio                 |
| requirements.txt        | Daftar library yang dibutuhkan       |
| best_phishing_model.pkl | Model terbaik hasil training         |
| feature_names.pkl       | Daftar fitur yang digunakan model    |
| feature_defaults.pkl    | Nilai default fitur untuk deployment |

Deployment dilakukan menggunakan Hugging Face Spaces dengan Gradio.

Link deployment:
https://nurulazmiii98-phishing-website-detection.hf.space/?__theme=system&deep_link=CVX-OJb60b4

Aplikasi menerima beberapa input fitur utama, yaitu:

* Length URL
* Length Hostname
* Number of Dots
* Number of Hyphens
* Page Rank

Fitur lain yang tidak dimasukkan pengguna diisi menggunakan nilai default dari data training.

## Repository Structure

```text
.
├── README.md
├── Deteksi_Website_Phishing_Menggunakan_Machine_Learning_Berbasis_Feature_URL.ipynb
├── app.py
├── requirements.txt
├── best_phishing_model.pkl
├── feature_names.pkl
└── feature_defaults.pkl
```

## Cara Menjalankan Aplikasi

Install library yang dibutuhkan:

```bash
pip install -r requirements.txt
```

Jalankan aplikasi:

```bash
python app.py
```

Atau buka langsung melalui link deployment Hugging Face Spaces.

## Kesimpulan

Proyek ini berhasil membangun model machine learning untuk mendeteksi website phishing berdasarkan feature URL. Dataset yang digunakan memiliki 11.430 data dengan distribusi kelas yang seimbang antara legitimate dan phishing.

Dua model digunakan dalam proyek ini, yaitu Decision Tree dan XGBoost. Decision Tree digunakan sebagai model yang sudah dipelajari di kelas, sedangkan XGBoost digunakan sebagai model eksplorasi tambahan. Hasil evaluasi menunjukkan bahwa XGBoost menjadi model terbaik dengan accuracy 0.9602 dan F1 Score 0.9603.

Model terbaik berhasil di-deploy menggunakan Hugging Face Spaces dengan Gradio. Deployment ini memungkinkan pengguna melakukan prediksi sederhana untuk mengetahui apakah suatu website termasuk legitimate atau phishing.
