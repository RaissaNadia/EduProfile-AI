from tensorflow.keras.models import load_model

# WAJIB: Masukkan kelas CustomDense kamu ke parameter custom_objects
# agar Keras tahu cara merakit kembali layer buatanmu tersebut.
model_loaded = load_model(
    'eduprofile_multimodal_v1.keras', 
    custom_objects={'CustomDense': CustomDense}
)

print("✅ Model dan Custom Layer berhasil dimuat ulang ke memori!")
# Cek apakah strukturnya masih sama
model_loaded.summary()