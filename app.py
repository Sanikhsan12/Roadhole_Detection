import os
import streamlit as st
import cv2
import tempfile
from ultralytics import YOLO


path = 'Hasil/adam/best.pt'
# * Model Cheking
if not os.path.exists(path):
    st.error(f"Tidak Ada Model Terdeteksi Di {path}. Harap Train Model Terlebih Dahulu.")
    st.stop()

# * Load Model
try:
    model = YOLO(path)
except Exception as e:
    st.error(f"Gagal Memuat Model Dari {path}. Error: {e}")
    st.stop()


# * Frame Processing
def frame_processing(frame):
    results = model(frame)
    annotated_frame = results[0].plot()
    return annotated_frame

# ! UI
st.title("Deteksi Lubang Jalan Menggunakan YOLOv8")
st.write("Aplikasi ini menggunakan model YOLOv8 untuk mendeteksi lubang jalan pada gambar atau video.")

st.sidebar.title("Opsi Deteksi")
detection_choice = st.sidebar.radio(
    "Pilih mode deteksi yang Anda inginkan:",
    ("Deteksi dari Kamera", "Deteksi dari Video")
)

if detection_choice == "Deteksi dari Kamera":
    st.header("Deteksi Real-time dari Kamera")
    run = st.checkbox('Mulai Kamera')
    
    FRAME_WINDOW = st.image([])
    
    if run:
        cap = cv2.VideoCapture(0)
        
        while run:
            ret, frame = cap.read()
            if not ret:
                st.error("Gagal mengakses kamera. Silakan periksa koneksi kamera Anda.")
                break
            
            # * Konversi warna dari BGR (OpenCV) ke RGB (untuk ditampilkan)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # * Proses frame untuk deteksi
            processed_frame = frame_processing(frame_rgb)
            
            # * Tampilkan frame yang sudah diproses
            FRAME_WINDOW.image(processed_frame)
        else:
            # * Hentikan kamera jika checkbox tidak dicentang
            cap.release()
            
elif detection_choice == "Deteksi dari Video":
    st.header("Deteksi dari File Video")
    uploaded_file = st.file_uploader("Unggah file video Anda", type=["mp4", "mov", "avi", "mkv"])
    
    if uploaded_file is not None:
        # Buat file temporer untuk menyimpan video yang diunggah
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
            tfile.write(uploaded_file.read())
            video_path = tfile.name

        st.video(video_path)
        st.write("---")
        
        if st.button('Mulai Deteksi pada Video'):
            stframe = st.empty()
            cap = cv2.VideoCapture(video_path)
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # * Konversi BGR ke RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # * Proses frame
                processed_frame = frame_processing(frame_rgb)

                # * Tampilkan hasilnya
                stframe.image(processed_frame)
            
            cap.release()
            os.remove(video_path) # * Hapus file temporer setelah selesai
            st.success("Deteksi video selesai!")