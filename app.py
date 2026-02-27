import streamlit as st
import cv2
import numpy as np
from engine import SaliencyEngine

st.set_page_config(page_title="AMD Sling Shot", layout="wide")
st.title("Thumbnail A/B Battle")

engine = SaliencyEngine()

visual_filter = st.sidebar.selectbox(
    "Visual Filters",
    ["Normal", "Grayscale (Squint Test)", "High Contrast"]
)

col_a, col_b = st.columns(2)

with col_a:
    file_a = st.file_uploader("Upload Design A", type=['jpg', 'png'], key="a")
with col_b:
    file_b = st.file_uploader("Upload Design B", type=['jpg', 'png'], key="b")

if file_a and file_b:
    for file, col, label in zip([file_a, file_b], [col_a, col_b], ["A", "B"]):
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        
        if visual_filter == "Grayscale (Squint Test)":
            filtered_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            filtered_img = cv2.cvtColor(filtered_img, cv2.COLOR_GRAY2BGR)
        elif visual_filter == "High Contrast":
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l_channel = clahe.apply(l_channel)
            lab = cv2.merge([l_channel, a_channel, b_channel])
            filtered_img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            filtered_img = img

        sal_map = engine.get_map(img)
        score = engine.calculate_score(img, sal_map) # Pass 'img' for face detection
        
        with col:
            heatmap = cv2.applyColorMap(sal_map, cv2.COLORMAP_JET)
            
            overlay_intensity = st.slider(
                "Heatmap Overlay Intensity",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.05,
                key=f"intensity_{label}"
            )
            
            blended = cv2.addWeighted(img, 1 - overlay_intensity, heatmap, overlay_intensity, 0)
            
            st.image(blended, caption=f"Focus Map {label}", use_container_width=True)
            st.metric(label=f"Engagement Score {label}", value=f"{score}%")
            
            suggestions = engine.get_improvement_suggestions(img, sal_map)
            with st.expander("Suggestions to improve this thumbnail"):
                for suggestion in suggestions:
                    st.write(f"- {suggestion}")

            st.write("---")
            st.caption(f"{label} Mobile Preview")
            mobile_img = cv2.resize(filtered_img, (150, int(150 * filtered_img.shape[0]/filtered_img.shape[1])))
            st.image(mobile_img)
    
    st.write("---")
    st.subheader("Understanding the Heatmap")
    st.info("""
    **Red/Yellow areas**: High visual attention - viewers look here first
    
    **Green areas**: Moderate attention
    
    **Blue/Dark areas**: Low attention - often overlooked
    
    The heatmap predicts where viewers' eyes will naturally be drawn. Place your key message, face, or product in the red/yellow zones for maximum impact.
    """)