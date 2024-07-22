import cv2
import streamlit as st
from ultralytics import YOLO
from datetime import datetime, timedelta
import pandas as pd

def app():
    st.header('SafeFall: Fall Detection Web App')
    st.subheader('Made by SIC5 Kelompok 32')
    st.write('To use: Set the minimum confidence, then start the webcam!')

    # Update this path to your custom model
    custom_model_path = 'best.pt'
    model = YOLO(custom_model_path)
    object_names = list(model.names.values())

    min_confidence = st.slider('Confidence Score', 0.0, 1.0)

    # Button to start webcam
    start_button = st.button('Start Webcam')
    stop_button = st.button('Stop Webcam')

    # Initialize fall history in session state if not already
    if 'fall_history' not in st.session_state:
        st.session_state.fall_history = []

    # Initialize last fall detection time in session state if not already
    if 'last_fall_time' not in st.session_state:
        st.session_state.last_fall_time = datetime.min

    if start_button:
        # Access the webcam
        cap = cv2.VideoCapture(0)

        # Create a placeholder for the video frame
        frame_placeholder = st.empty()

        # Start the video capture loop
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.write("Failed to capture image")
                break

            result = model(frame)
            for detection in result[0].boxes.data:
                x0, y0 = (int(detection[0]), int(detection[1]))
                x1, y1 = (int(detection[2]), int(detection[3]))
                score = round(float(detection[4]), 2)
                cls = int(detection[5])
                object_name = model.names[cls]
                label = f'{object_name} {score}'

                if score > min_confidence:
                    cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 0, 0), 2)
                    cv2.putText(frame, label, (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    
                    # If a fall is detected, log the date, time, & confidence score (with a 10 second interval)
                    if object_name == 'Fall Detected':
                        current_time = datetime.now()
                        if current_time - st.session_state.last_fall_time > timedelta(seconds=10):
                            fall_date = current_time.strftime('%d/%m/%Y')
                            fall_time = current_time.strftime('%H:%M:%S')
                            st.session_state.fall_history.append({
                                'No': len(st.session_state.fall_history) + 1,
                                'Date': fall_date,
                                'Time (WIB)': fall_time,
                                'Confidence Score': f'{score:.2f}'
                            })
                            st.session_state.last_fall_time = current_time

            # Display the frame in the Streamlit app
            frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if stop_button:
                break

        cap.release()

    # Display Fall history
    st.subheader('Fall History')
    if st.session_state.fall_history:
        df_fall_history = pd.DataFrame(st.session_state.fall_history)
        df_fall_history.set_index('No', inplace=True)
        st.table(df_fall_history)
    else:
        st.write('No falls detected yet.')

    # Display Caregiver data
    st.subheader('Caregiver Information')
    caregiver_data = {
        'No': ['1', '2', '3'],
        'Name': ['Lebron James', 'Stephen Curry', 'Lionel Messi'],
        'Phone Number': ['+62 812-3456-7890', '+62 811-2345-6789', '+62 813-4567-8901']
    }
    df_caregivers = pd.DataFrame(caregiver_data)
    df_caregivers.set_index('No', inplace=True)
    st.table(df_caregivers)

if __name__ == "__main__":
    app()
