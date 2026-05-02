import streamlit as st
from streamlit_drawable_canvas import st_canvas
import fitz
import easyocr
import torch
import numpy as np
from PIL import Image

st.set_page_config(layout="wide")
st.title("Editable Quiz Cropper")

# 1. Setup OCR
@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'], gpu=torch.cuda.is_available())

reader = load_ocr()

# 2. File Upload
pdf_file = st.sidebar.file_uploader("Upload PDF", type="pdf")

if pdf_file:
    with open("temp.pdf", "wb") as f:
        f.write(pdf_file.getbuffer())
    
    doc = fitz.open("temp.pdf")
    page_num = st.sidebar.number_input("Page", min_value=1, max_value=len(doc)) - 1
    page = doc[page_num]
    
    # Render page to image
    pix = page.get_pixmap(dpi=150)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    # 3. Initial AI Detection (Pre-filling the canvas)
    if "initial_rects" not in st.session_state:
        img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)
        results = reader.readtext(img_np)
        
        initial_objects = []
        for (bbox, text, prob) in results:
            clean = "".join(filter(str.isdigit, text))
            if clean and bbox[0][0] < 100: # Threshold for left margin
                # Format for streamlit-canvas
                obj = {
                    "type": "rect",
                    "left": bbox[0][0],
                    "top": bbox[0][1],
                    "width": bbox[2][0] - bbox[0][0],
                    "height": bbox[2][1] - bbox[0][1],
                    "fill": "rgba(255, 0, 0, 0.3)",
                    "stroke": "red"
                }
                initial_objects.append(obj)
        st.session_state.initial_rects = {"objects": initial_objects}

    # 4. The Interactive Canvas
    st.subheader("Edit Boxes: Click to select, drag to move/resize, or press 'Del' to remove.")
    
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Box color
        stroke_width=2,
        stroke_color="red",
        background_image=img,
        initial_drawing=st.session_state.initial_rects,
        update_streamlit=True,
        height=pix.height,
        width=pix.width,
        drawing_mode="transform", # This allows moving and resizing
        key="canvas",
    )

    # 5. Export Logic
    if canvas_result.json_data is not None:
        objects = canvas_result.json_data["objects"]
        if st.button(f"Crop and Save {len(objects)} Questions"):
            for idx, obj in enumerate(objects):
                # Convert canvas coordinates back to PDF scale
                # Save images based on these final hand-adjusted boxes
                st.write(f"Cropping Question {idx+1} at Y: {obj['top']}")
