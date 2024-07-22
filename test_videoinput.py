import cv2
import os
import streamlit as st
from ultralytics import YOLO
import uuid

def app():
    st.header('SafeFall: Fall Detection Web App')
    st.subheader('Made by SIC5 Kelompok 32')
    st.write('To use: Import video, set the minimum confidence, then start!')

    # Menggunakan custom trained model kami
    custom_model_path = 'best.pt'
    model = YOLO(custom_model_path)
    object_names = list(model.names.values())


    # Form input video
    with st.form("my_form"):
        uploaded_file = st.file_uploader("Upload video", type=['mp4', 'avi'])
        min_confidence = st.slider('Confidence score', 0.0, 1.0)
        st.form_submit_button(label='Submit')
            
    if uploaded_file is not None: 
        # Generate unique filenames to avoid conflicts
        unique_id = str(uuid.uuid4().hex)[:8]
        input_path = os.path.join(os.getcwd(), f"temp_{unique_id}.mp4")
        output_path = os.path.join(os.getcwd(), f"output_{unique_id}.mp4")

        try:
            with open(input_path, "wb") as temp_file:
                temp_file.write(uploaded_file.read())

            video_stream = cv2.VideoCapture(input_path)
            width = int(video_stream.get(cv2.CAP_PROP_FRAME_WIDTH)) 
            height = int(video_stream.get(cv2.CAP_PROP_FRAME_HEIGHT)) 
            fourcc = cv2.VideoWriter_fourcc(*'h264') 
            fps = int(video_stream.get(cv2.CAP_PROP_FPS)) 

            out_video = cv2.VideoWriter(output_path, int(fourcc), fps, (width, height)) 

            with st.spinner('Processing video...'): 
                # Beritahu bahwa video sedang diproses
                while True:
                    ret, frame = video_stream.read()
                    if not ret:
                        # Prediksi apakah video frame yang dibaca adalah video terakhir
                        break
                    result = model(frame)
                    for detection in result[0].boxes.data:
                        # Buat bounding box untuk lokasi objek dalam video
                        x0, y0 = (int(detection[0]), int(detection[1])) # Coordinate Top Left of bounding box
                        x1, y1 = (int(detection[2]), int(detection[3])) # Top Right of bounding box
                        score = round(float(detection[4]), 2)
                        cls = int(detection[5])
                        object_name =  model.names[cls]
                        label = f'{object_name} {score}'
                        
                        # Jika prediksi lebih tinggi dari min_confidence
                        if score > min_confidence:
                            cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 0, 0), 5) # Bikin bounding box dengan sintaks : cv2.rectangle( gambar , (koordinat pojok kiri atas) , (koordinat pojok kanan bawah) , (warna bounding box) , (lebar garis bounding box))
                            cv2.putText(frame, label, (x0, y0 - 10), # Bikin text di atas bounding box
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                        else:
                            continue
                    
                    detections = result[0].verbose()
                    cv2.putText(frame, detections, (10, 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    out_video.write(frame) 

            video_stream.release()
            out_video.release()

            # Delete temporary files setelah processing
            if os.path.exists(input_path):
                os.remove(input_path)
            if os.path.exists(output_path):
                st.video(output_path)

        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    app()