# Titanic Classification with MLflow

## Deskripsi Proyek
Proyek ini merupakan implementasi *machine learning pipeline* untuk melakukan klasifikasi kelangsungan hidup penumpang Titanic menggunakan algoritma **Logistic Regression**.  
Seluruh proses pelatihan, evaluasi, dan pencatatan eksperimen dilakukan menggunakan **MLflow** sebagai sarana *experiment tracking* dan *model management*.

Dataset yang digunakan adalah dataset Titanic yang telah melalui tahap preprocessing.

---

## Tujuan
- Membangun model klasifikasi biner (Survived / Not Survived)
- Menerapkan MLflow untuk pencatatan eksperimen machine learning
- Melakukan evaluasi model menggunakan metrik klasifikasi
- Menerapkan *automatic logging* dan *manual logging* metrik sebagai nilai tambah

---


---

## Dataset
- **Nama file**: `titanic_clean.csv`
- **Lokasi**: `namadataset_preprocessing/`
- **Label target**: `Survived`
- Dataset telah melalui tahap preprocessing (cleaning, encoding, dan transformasi fitur)

---

## Teknologi yang Digunakan
- Python
- Pandas
- Scikit-learn
- MLflow

---

## Instalasi dan Setup Environment

### 1. Membuat Virtual Environment
```bash
python -m venv env

Windows
env\Scripts\activate

Linux / MacOS
source env/bin/activate

Instalasi Dependensi

pip install pandas scikit-learn mlflow

