import streamlit as st
from PIL import Image
import numpy as np
import plotly.graph_objects as go
import os
import tempfile
import shutil
import time

# Import YOLO trial
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

# Page Configuration (Only once, at the top)
st.set_page_config(page_title="Garbage Image Detection", layout="centered")

# YOLO Library Availability check
def yolo_check():
    if not YOLO_AVAILABLE:
        st.error("Ultralytics not installed. Please install using:")
        st.code("pip install ultralytics")
        return False
    return True

# Custom CSS
st.markdown("""
    <style>
        .stButton > button {
            width: 100%;
            padding: 10px;
            font-size: 18px;
            background-color: #007bff;
            color: white;
            border-radius: 8px;
            border: none;
        }
        .stButton > button:hover {
            background-color: #0056b3;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div style="background-color:#2D85E3; padding:20px; border-radius:10px; text-align:center;">
        <h1 style="color:white; margin-bottom:5px;">Garbage Image Detection</h1>
        <p style="color:white; font-size:18px;">Identify types of waste using AI</p>
    </div>
""", unsafe_allow_html=True)


# Check YOLO installation
if yolo_check():
    st.markdown("### Upload an Image 📷")
    uploaded_file = st.file_uploader("", type=['jpg', 'jpeg', 'png'], help="Upload an image of waste for AI classification.")

    # Model Selection Dropdown
    model_options = ["best.pt", "other_model.pt"]  # Add more if available
    selected_model = st.selectbox("Select YOLO Model", model_options)

    if uploaded_file:
        # Temporary storage
        temp_dir = tempfile.mkdtemp()
        temp_file = os.path.join(temp_dir, "image.jpg")
        image = Image.open(uploaded_file)
        
        # Resize image
        image = image.resize((300, 300))
        image.save(temp_file)

        # Display image
        st.image(image, caption="Uploaded Image", use_container_width=True)

        if st.button("Classify Image!"):
            with st.spinner("Processing..."):
                progress_bar = st.progress(0)  # Add progress bar
                try:
                    model = YOLO(selected_model)
                    progress_bar.progress(25)  # Update progress

                    result = model(temp_file)
                    progress_bar.progress(50)  # Update progress

                    object_name = result[0].names
                    pred_value = result[0].probs.data.numpy().tolist()
                    object_detected = object_name[np.argmax(pred_value)]

                    progress_bar.progress(75)  # Update progress
                    
                    # Plot confidence bar chart
                    fig = go.Figure([go.Bar(x=list(object_name.values()), y=pred_value)])
                    fig.update_traces(marker_color='skyblue', marker_line_width=2, marker_line_color='black')
                    fig.update_layout(title='🧐 Prediction Confidence Level', xaxis_title='Garbage Type', yaxis_title='Confidence (%)')
                    
                    st.plotly_chart(fig)
                    progress_bar.progress(100)  # Finish progress

                    st.write(f'✅ Garbage Type Detected: **{object_detected.upper()}** (Confidence: {round(np.max(pred_value) * 100, 2)}%)')

                    # Save output image for download
                    output_path = os.path.join(temp_dir, "result.jpg")
                    image.save(output_path)

                    # Add download button
                    with open(output_path, "rb") as file:
                        btn = st.download_button(label="📥 Download Processed Image", data=file, file_name="garbage_detection_result.jpg", mime="image/jpeg")
                
                except Exception as e:
                    st.error('Image unknown')
                    st.error(f'Error: {e}')
                    
            if st.button("🔄 Try Another Image"):
                st.experimental_rerun()

            shutil.rmtree(temp_dir, ignore_errors=True)  # Delete temp files

# Footer
st.markdown("""
    <div style="text-align:center; padding-top:20px; font-size:14px; color:gray;">
        <strong>Garbage Type Image Detection Program © 2025</strong>
    </div>
""", unsafe_allow_html=True)
