import os
import tempfile
import shutil
import streamlit as st
from PIL import Image
import numpy as np
import plotly.graph_objects as go

# Safe OpenCV patching
try:
    import cv2
    cv2.imshow = lambda *args, **kwargs: None
    cv2.waitKey = lambda *args, **kwargs: None
    cv2.destroyAllWindows = lambda *args, **kwargs: None
except ImportError:
    st.warning("OpenCV not available. GUI patch skipped.")

# Try importing YOLO
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except Exception as e:
    st.error("Ultralytics could not be loaded.")
    st.code(str(e))
    YOLO_AVAILABLE = False

# Page setup
st.set_page_config(page_title="Garbage Image Detection", layout="centered")

# Title
st.markdown("""
    <div style="background-color:#2D85E3; padding:20px; border-radius:10px; text-align:center;">
        <h1 style="color:white; margin-bottom:5px;">Garbage Image Detection</h1>
        <p style="color:white; font-size:18px;">Identify types of waste using AI</p>
    </div>
""", unsafe_allow_html=True)

# Only proceed if YOLO loaded
if YOLO_AVAILABLE:
    st.markdown("### 📷 Upload a Garbage Image")
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, "input.jpg")
        image.save(temp_path)

        st.image(image, caption="Uploaded Image", use_container_width=True)

        if st.button("🔍 Classify Image!"):
            with st.spinner("Running YOLO model..."):
                try:
                    model = YOLO("best.pt")  # Ensure best.pt is in the repo
                    result = model(temp_path)
                    probs = result[0].probs.data.numpy().tolist()
                    labels = list(result[0].names.values())

                    detected = labels[np.argmax(probs)]
                    confidence = round(np.max(probs) * 100, 2)

                    st.success(f"✅ Predicted: **{detected.upper()}** ({confidence}%)")

                    # Plot bar chart
                    fig = go.Figure([go.Bar(x=labels, y=probs)])
                    fig.update_layout(title="Prediction Confidence", xaxis_title="Class", yaxis_title="Confidence")
                    st.plotly_chart(fig)

                    # Download
                    output_path = os.path.join(temp_dir, "output.jpg")
                    image.save(output_path)
                    with open(output_path, "rb") as f:
                        st.download_button("📥 Download Processed Image", f, "result.jpg", "image/jpeg")

                except Exception as e:
                    st.error("Prediction failed.")
                    st.code(str(e))

        if st.button("🔄 Try Another Image"):
            st.experimental_rerun()

        shutil.rmtree(temp_dir, ignore_errors=True)

# Footer
st.markdown("""
    <div style="text-align:center; padding-top:20px; font-size:13px; color:gray;">
        <strong>Garbage Classifier App © 2025</strong>
    </div>
""", unsafe_allow_html=True)