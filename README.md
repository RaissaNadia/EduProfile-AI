# 🎓 EduProfile AI — Learning Analytics Dashboard

EduProfile AI is an interactive learning analytics dashboard designed to analyze students' learning behavior using **VAK Learning Style (Visual, Auditory, Kinesthetic)** and **Learning Pace Clustering**.

This project was developed for **Coding Camp 2026 powered by DBS Foundation** by the Data Science Team:

* Nadia Raissa R
* Charlene Manuella Angkadjaja

---
Laporan Komperhensif Link Drive:
https://drive.google.com/file/d/1U5koF2SiqelCAozfM8tZFfGlX1vm4lib/view?usp=sharing
---

## 🚀 Live Demo

🔗 Streamlit App:
https://eduprofile-ai-capstone.streamlit.app/

🔗 GitHub Repository:
https://github.com/RaissaNadia/EduProfile-AI

---

## 📌 Project Overview

EduProfile AI helps identify:

* Students’ dominant learning styles (VAK)
* Learning pace categories (Fast, Medium, Slow)
* Academic performance patterns
* Correlation between behavioral indicators and learning outcomes
* Statistical validation through A/B Testing

The dashboard provides interactive visualizations and analytics for educators, researchers, and educational platforms.

---

## 🧠 Features

### 📊 Distribution & Overview

* Learning style distribution
* Learning pace distribution
* Numerical feature distributions
* Key student metrics

### 🔬 VAK Indicator Analysis

* Violin plots
* Boxplots
* Radar charts
* Learning style profiling

### 🚀 Learning Pace Analysis

* K-Means clustering
* PCA 2D visualization
* AcademicScore vs AttendanceRate scatter plot

### 📐 Correlation & Feature Analysis

* Correlation heatmap
* Feature importance analysis
* Learning style relationship analysis

### 🧪 Statistical A/B Testing

* Independent t-test
* Mann-Whitney U Test
* One-Way ANOVA
* Kruskal-Wallis Test
* Chi-Square Test
* Cramér’s V

---

## 🛠️ Technologies Used

* Python
* Streamlit
* Pandas
* NumPy
* Matplotlib
* Seaborn
* SciPy
* Scikit-learn

---

## 📂 Project Structure

```bash
EduProfile-AI/
│
├── app.py
├── capstone.py
├── requirements.txt
├── README.md
│
├── output/
│   └── final_dataset_cleaned_with_pace.csv
│
└── assets/
```

---

## ⚙️ Installation & Run Locally

### 1️⃣ Clone Repository

```bash
git clone https://github.com/RaissaNadia/EduProfile-AI.git
cd EduProfile-AI
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run Streamlit App

```bash
streamlit run app.py
```

---

## 📈 Machine Learning & Analytics

### Learning Pace Clustering

The project uses **K-Means Clustering** to categorize students into:

* Fast Learners
* Medium Learners
* Slow Learners

### Dimensionality Reduction

* PCA (Principal Component Analysis) for 2D cluster visualization.

### Statistical Validation

The dashboard validates findings using:

* ANOVA
* Kruskal-Wallis
* Chi-Square
* t-test
* Mann-Whitney U

---

## ❓Business Question
*  Sejauh mana profil belajar siswa saat ini terdistribusi secara merata di dalam sistem?
*  Apakah sistem pengujian kita mampu menangkap variasi gaya belajar secara objektif untuk mendukukng personalisasi?
*  Faktor perilaku apa yang memiliki pengaruh paling kuat terhadap performa akademik siswa?
*  Bagaimana cara sistem mengelompokkan kecepatan belajar siswa untuk meningkatkan efisiensi kurikulum?
---

## 🎯 Business Impact

EduProfile AI can help:

* Personalize learning recommendations
* Improve teaching strategies
* Identify students needing intervention
* Support adaptive learning systems
* Assist educational decision-making using data-driven insights

---

## 📸 Dashboard Preview

### Main Dashboard

* KPI Metrics
* Interactive Filters
* Learning Analytics Visualization

### Analytics Tabs

* Distribution & Overview
* VAK Indicators
* Learning Pace
* Correlation & Features
* A/B Testing

---

## 👩‍💻 Authors

### Nadia Raissa R

Data Science & Dashboard Development

* Repository Result : (https://github.com/RaissaNadia/EduProfile-AI)

### Charlene Manuella Angkadjaja

Data Analysis & Statistical Validation 

*Repository Utama (Data Processing): (https://github.com/charleneangkadjaja/capstone-project)

---

## 📜 License

This project was created for educational and academic purposes under Coding Camp 2026 powered by DBS Foundation.
