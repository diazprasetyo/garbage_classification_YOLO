import streamlit as st
from PIL import Image
import numpy as np
import plotly.graph_objects as go
import os
import tempfile
import shutil

# import YOLO trial
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    
# Page Configuration
st.set_page_config(page_title="Garbage Image Introduction")

# YOLO Library Availability check
def yolo_check():
    if not YOLO_AVAILABLE:
        st.error("Ultralytics not installed. Please install using:")
        st.code("pip install ultralytics")
        return False
    return True

# Streamlit UI
st.markdown("""
    <div style="background-color:#0984e3; padding:20px; text-align:center; border-radius:10px;">
        <h1 style="color:white;">Garbage Image Detection</h1>
        <h5 style="color:white;">Identify types of waste using AI</h5>
    </div>
""", unsafe_allow_html=True)


# YOLO Library Availability check
if yolo_check():
    # image upload
    uploaded_file = st.file_uploader("Garbage Image Upload", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file:
        # Temporary storage
        temp_dir = tempfile.mkdtemp()
        temp_file = os.path.join(temp_dir, "image.jpg")
        image = Image.open(uploaded_file)
        
        # Image resize
        image = image.resize((300, 300))
        image.save(temp_file)
        
        # Image display limited to CSS
        st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
        st.image(image, caption="Uploaded Image")
        st.markdown("</div>", unsafe_allow_html=True)

        
        # Image detection
        if st.button("Image Detection"):
            with st.spinner("Processing..."):
                try:
                    model = YOLO('best.pt')
                    result = model(temp_file)
                    
                    # Get Prediction result
                    object_name = result[0].names
                    pred_value = result[0].probs.data.numpy().tolist()
                    object_detected = object_name[np.argmax(pred_value)]
                    
                    # Chart
                    fig = go.Figure([go.Bar(x=list(object_name.values()), y=pred_value)])
                    fig.update_layout(title='Prediction Confidence Level', xaxis_title= 'Garbage Type', yaxis_title='Confidence')
                    
                    # Show Result
                    st.write(f'Garbage Type detected: **{object_detected}**')
                    st.plotly_chart(fig)
                except Exception as e:
                    st.error('Image unknown')
                    st.error(f'Error: {e}')
            # Delete temporary image
            shutil.rmtree(temp_dir, ignore_errors=True)
            
# Footer
st.markdown("""
    <div style="text-align:center; padding-top:20px; font-size:14px; color:gray;">
        <strong>Garbage Type Image Detection Program © 2025</strong>
    </div>
""", unsafe_allow_html=True)
                    