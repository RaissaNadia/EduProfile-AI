# -*- coding: utf-8 -*-
"""DS_CAPSTONE_INTEGRATED
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

"""### Load Datasets"""

try:
    df_education = pd.read_csv('sourcedataset/student_education_dataset.csv')
    df_vak = pd.read_csv('sourcedataset/dataset.csv')
    df_performance = pd.read_csv('sourcedataset/student_performance.csv')
    print("✅ Berhasil memuat semua dataset.")
except FileNotFoundError:
    print("❌ Error: File CSV tidak ditemukan. Pastikan path 'sourcedataset/' sudah benar.")

"""### Data Pre-Integration & Feature Engineering Per Dataset
Mengubah pemrosesan data kuesioner ke biner (0/1) untuk Dataset 1 & 3, serta normalisasi 0-1 untuk Dataset 2.
"""

# ==============================================================================
# === PREPROCESSING DATASET 1: Education Dataset (Berubah ke Biner) ===
# ==============================================================================
df_edu_proc = df_education.copy()

# A. Normalisasi Fitur Kontinu Akademik & Performa (Skala 0 sampai 1)
df_edu_proc['Edutech_Norm'] = df_edu_proc['AcademicScore'] / df_edu_proc['AcademicScore'].max()
df_edu_proc['Resources_Norm'] = df_edu_proc['AttendanceRate']  # Asumsi sudah dalam rasio desimal 0-1
df_edu_proc['StudentPerformance'] = df_edu_proc['Edutech_Norm']

# B. Binary Mapping untuk Jawaban Kuesioner (A, B, C)
# Aturan kelompok: Jawaban 'A' merepresentasikan karakteristik Visual, 'B'/'C' bernilai 0
df_edu_proc['DeviceUsage_Norm'] = df_education['DeviceUsage'].map({'A': 1, 'B': 0, 'C': 0}).fillna(0)
df_edu_proc['CourseParticipation_Norm'] = df_education['CourseParticipation'].map({'A': 1, 'B': 0, 'C': 0}).fillna(0)
df_edu_proc['EmotionEngagement_Norm'] = df_education['EmotionEngagement'].map({'A': 1, 'B': 0, 'C': 0}).fillna(0)

# Rekayasa fitur diskusi berbasis biner
df_edu_proc['Discussion_Norm'] = (df_edu_proc['CourseParticipation_Norm'] * 0.8) + (df_edu_proc['EmotionEngagement_Norm'] * 0.2)

# Jawaban 'A' pada fitur fisik (dianggap tidak dominan kinestetik), Extracurricular default nilai tengah biner (0.5)
df_edu_proc['PhysicalActivity_Norm'] = df_education['PhysicalActivity'].map({'A': 1, 'B': 0, 'C': 0}).fillna(0)
df_edu_proc['Extracurricular_Norm'] = 0.5

# C. Kalkulasi Indikator Rata-rata Terbobot Utama (Otomatis menghasilkan rentang skala 0 sampai 1)
df_edu_proc['Indikator_Visual'] = ((df_edu_proc['Edutech_Norm'] * 3) + (df_edu_proc['DeviceUsage_Norm'] * 1) + (df_edu_proc['Resources_Norm'] * 3)) / 7
df_edu_proc['Indikator_Auditory'] = ((df_edu_proc['Discussion_Norm'] * 3) + (df_edu_proc['CourseParticipation_Norm'] * 1) + (df_edu_proc['EmotionEngagement_Norm'] * 1)) / 5
df_edu_proc['Indikator_Kinestetik'] = ((df_edu_proc['PhysicalActivity_Norm'] * 1) + (df_edu_proc['Extracurricular_Norm'] * 3)) / 4


# ==============================================================================
# === PREPROCESSING DATASET 2: VAK Dataset (Tidak Berubah - Skala Likert / Kontinu) ===
# ==============================================================================
# Tetap menggunakan data aslinya namun dibawa ke rentang 0 sampai 1 menggunakan pembagian Max murni
df_vak_proc = df_vak.copy()

df_vak_proc['Discussions_Norm'] = df_vak_proc['Discussions'] / df_vak_proc['Discussions'].max() if 'Discussions' in df_vak_proc.columns else 0.5
df_vak_proc['CourseParticipation_Norm'] = df_vak_proc['CourseParticipation'] / df_vak_proc['CourseParticipation'].max() if 'CourseParticipation' in df_vak_proc.columns else 0.5
df_vak_proc['EmotionEngagement_Norm'] = df_vak_proc['EmotionEngagement'] / df_vak_proc['EmotionEngagement'].max() if 'EmotionEngagement' in df_vak_proc.columns else 0.5

# Rumus Pembobotan Kustom dengan Pembagi 5 (Rentang Akhir 0-1)
df_vak_proc['Indikator_Auditory'] = ((df_vak_proc['Discussions_Norm'] * 3) + (df_vak_proc['CourseParticipation_Norm'] * 1) + (df_vak_proc['EmotionEngagement_Norm'] * 1)) / 5


# ==============================================================================
# === PREPROCESSING DATASET 3: Performance Dataset (Berubah ke Biner) ===
# ==============================================================================
df_perf_proc = df_performance.copy()

# A. Penyesuaian Fitur Akademik Kontinu (Skala 0 sampai 1)
df_perf_proc['AttendanceRate'] = df_perf_proc['Attendance'] / 100
df_perf_proc = df_perf_proc.rename(columns={
    'ExamScore': 'AcademicScore',
    'FinalGrade': 'StudentPerformance'
})
df_perf_proc['StudentPerformance'] = df_perf_proc['StudentPerformance'] / df_perf_proc['StudentPerformance'].max()
df_perf_proc['Edutech_Norm'] = (df_perf_proc['AcademicScore'] / df_perf_proc['AcademicScore'].max())
df_perf_proc['Resources_Norm'] = df_perf_proc['AttendanceRate']

# B. Imputasi Default Biner (Nilai tengah biner adalah 0.5)
df_perf_proc['DeviceUsage_Norm'] = 0.5
df_perf_proc['DeviceUsage'] = 0.5
df_perf_proc['EmotionEngagement_Norm'] = 0.5
df_perf_proc['Discussion_Norm'] = (df_perf_proc['Discussions'] / df_perf_proc['Discussions'].max()) if 'Discussions' in df_perf_proc.columns else 0.5
df_perf_proc['CourseParticipation_Norm'] = (df_perf_proc['OnlineCourses'] / df_perf_proc['OnlineCourses'].max()) if 'OnlineCourses' in df_perf_proc.columns else 0.5
df_perf_proc['PhysicalActivity_Norm'] = 0.5

# C. Binary Mapping Jawaban Kuesioner Dataset 3
# Aturan kelompok: Jawaban 'C' merepresentasikan karakteristik Kinestetik, 'A'/'B' bernilai 0
df_perf_proc['Extracurricular_Norm'] = df_performance['Extracurricular'].map({'A': 0, 'B': 0, 'C': 1}).fillna(0)

# D. Mapping gaya belajar dari angka ke label string
mapping_styles = {0: 'Auditory', 1: 'Kinesthetic', 2: 'Visual'}
df_perf_proc['LearningStyle'] = df_perf_proc['LearningStyle'].map(mapping_styles)

# E. Kalkulasi Indikator Rata-rata Terbobot Utama
df_perf_proc['Indikator_Visual'] = ((df_perf_proc['Edutech_Norm'] * 3) + (df_perf_proc['DeviceUsage_Norm'] * 1) + (df_perf_proc['Resources_Norm'] * 3)) / 7
df_perf_proc['Indikator_Auditory'] = ((df_perf_proc['Discussion_Norm'] * 3) + (df_perf_proc['CourseParticipation_Norm'] * 1) + (df_perf_proc['EmotionEngagement_Norm'] * 1)) / 5
df_perf_proc['Indikator_Kinestetik'] = ((df_perf_proc['PhysicalActivity_Norm'] * 1) + (df_perf_proc['Extracurricular_Norm'] * 3)) / 4

""" Dokumentasi Imputasi Nilai Default Baru 0.5 """
fitur_imputasi = {
    'DeviceUsage_Norm': 'Tidak ada kolom DeviceUsage di dataset 3, diisi nilai tengah biner 0.5',
    'EmotionEngagement_Norm': 'Tidak ada kolom EmotionEngagement di dataset 3, diisi nilai tengah biner 0.5',
    'PhysicalActivity_Norm': 'Tidak ada kolom PhysicalActivity di dataset 3, diisi nilai tengah biner 0.5',
}
print(f"\n📝 Catatan Imputasi Dataset 3 Baru (Skala 0-1):")
for fitur, alasan in fitur_imputasi.items():
    print(f"    • {fitur:<30} → {alasan}")


"""### Data Integration"""
cols_to_combine = [
    'AcademicScore', 'AttendanceRate', 'StudentPerformance', 'DeviceUsage', 'LearningStyle',
    'Indikator_Visual', 'Indikator_Auditory', 'Indikator_Kinestetik'
]

# Setarakan fitur dasar ke format akhir sebelum ditumpuk secara vertikal
df_edu_ready = df_edu_proc.copy()
df_edu_ready['AcademicScore'] = df_edu_ready['Edutech_Norm']
df_edu_ready['AttendanceRate'] = df_edu_ready['Resources_Norm']
df_edu_ready['DeviceUsage'] = df_edu_ready['DeviceUsage_Norm']

df_perf_ready = df_perf_proc.copy()
df_perf_ready['AcademicScore'] = df_perf_ready['Edutech_Norm']
df_perf_ready['AttendanceRate'] = df_perf_ready['Resources_Norm']
df_perf_ready['DeviceUsage'] = df_perf_ready['DeviceUsage_Norm']

# Menggabungkan data secara vertikal (Seluruh baris aman, skala 0-1 harmonis)
df_combined = pd.concat([df_edu_ready[cols_to_combine], df_perf_ready[cols_to_combine]], axis=0, ignore_index=True)

# Pembersihan Akhir Data Gabungan
df_combined.dropna(subset=['LearningStyle'], inplace=True)
df_combined.fillna(df_combined.median(numeric_only=True), inplace=True)
df_combined['AttendanceRate'] = df_combined['AttendanceRate'].round(2)
df_combined.drop_duplicates(inplace=True)
df_combined['LearningStyle'] = df_combined['LearningStyle'].str.strip().str.capitalize()
df_combined = df_combined[df_combined['AcademicScore'] >= 0]

print(f"✅ Total data gabungan bersih: {len(df_combined)} baris.")


"""### IMPLEMENTASI FITUR BARU: Learning Pace (Unsupervised K-Means)"""
pace_features = ['AcademicScore', 'AttendanceRate', 'DeviceUsage']
scaler_pace = StandardScaler()
df_pace_scaled = scaler_pace.fit_transform(df_combined[pace_features])

# Latih K-Means untuk membentuk 3 kelompok kecepatan belajar (Slow, Medium, Fast)
kmeans_pace = KMeans(n_clusters=3, random_state=42, n_init=10)
df_combined['LearningPace_Cluster'] = kmeans_pace.fit_predict(df_pace_scaled)

# Urutkan cluster agar 0=Slow, 1=Medium, 2=Fast secara konsisten berdasarkan rata-rata AcademicScore dari yang terendah ke tertinggi
cluster_means = df_combined.groupby('LearningPace_Cluster')['AcademicScore'].mean().sort_values().index
cluster_mapping = {cluster_means[0]: 0, cluster_means[1]: 1, cluster_means[2]: 2}
df_combined['LearningPace_Cluster'] = df_combined['LearningPace_Cluster'].map(cluster_mapping)
print("✅ Fitur 'LearningPace_Cluster' berhasil dibuat dan diurutkan secara konsisten!")


"""### Cleaning Dataset NLP VAK (`dataset.csv`)"""
df_vak.columns = df_vak.columns.str.strip()
df_vak = df_vak.rename(columns={'Type': 'LearningStyle'})
df_vak['Sentence'] = df_vak['Sentence'].str.replace('"', '').str.strip()
df_vak['LearningStyle'] = df_vak['LearningStyle'].str.strip().str.capitalize()
df_vak = df_vak[~df_vak['Sentence'].str.contains('Show More', case=False, na=False)]
df_vak.dropna(subset=['Sentence'], inplace=True)
df_vak.drop_duplicates(subset=['Sentence'], inplace=True)
df_vak['Sentence'] = df_vak['Sentence'].str.lower().str.strip()

print(f"✅ Proses cleaning selesai. Data siap: {len(df_combined)} data siswa & {len(df_vak)} data teks kalimat.")


"""### Preprocessing untuk Pemodelan AI Klasifikasi VAK"""
le = LabelEncoder()
df_combined['LearningStyle_Encoded'] = le.fit_transform(df_combined['LearningStyle'])
mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print("\nHandoff info - Label Mapping Gaya Belajar:", mapping)

# Seleksi Fitur Utama Berbasis Skala Baru 0-1
selected_features = [
    'AcademicScore',
    'AttendanceRate',
    'Indikator_Visual',
    'Indikator_Auditory',
    'Indikator_Kinestetik'
]

X = df_combined[selected_features]
y = df_combined['LearningStyle_Encoded']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalisasi Akhir Terstandardisasi untuk Model AI
scaler_final = StandardScaler()
X_train_scaled = pd.DataFrame(scaler_final.fit_transform(X_train), columns=X_train.columns)
X_test_scaled = pd.DataFrame(scaler_final.transform(X_test), columns=X_test.columns)
print("✅ Pemisahan dan scaling data klasifikasi AI selesai.")

"""### Export Data untuk Web Developer & AI Engineer"""
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

X_train_scaled.to_csv(os.path.join(output_dir, 'X_train_scaled.csv'), index=False)
y_train.to_csv(os.path.join(output_dir, 'y_train.csv'), index=False)
X_test_scaled.to_csv(os.path.join(output_dir, 'X_test_scaled.csv'), index=False)
y_test.to_csv(os.path.join(output_dir, 'y_test.csv'), index=False)
df_combined.to_csv(os.path.join(output_dir, 'final_dataset_cleaned_with_pace.csv'), index=False)
df_vak.to_csv(os.path.join(output_dir, 'master_vak_nlp.csv'), index=False)

print("\n🚀 ALL PROCESSES COMPLETED SUCCESSFULLY! SEMUA FILE SIAP DIUNDUH.")
