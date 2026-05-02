import streamlit as st
import fitz
import easyocr
import torch
import numpy as np
import os
import re
import shutil
from PIL import Image, ImageDraw

st.set_page_config(page_title="AI Quiz Tool", layout="wide")

# --- UI HEADER ---
st.title("✂️ Interactive PDF Quiz Cropper")
st.markdown("Upload your files, adjust the detection boxes, and download your package.")

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.header("1. Upload")
    pdf_file = st.file_uploader("Exam PDF", type="pdf")
    ans_file = st.file_uploader("Answer Key (txt)", type="txt")
    
    st.header("2. Fine-Tune Detection")
    margin_limit = st.slider("Left Margin (x-axis limit)", 20, 150, 80)
    v_padding = st.slider("Vertical Padding (Crop height)", -20, 20, -10)
    
    st.header("3. Quality")
    dpi_val = st.select_slider("Image Quality (DPI)", options=[150, 200, 300], value=200)

# --- APP LOGIC ---
if pdf_file and ans_file:
    # Save files locally for processing
    with open("temp.pdf", "wb") as f: f.write(pdf_file.read())
    
    # Load Answers
    answers_map = {}
    ans_content = ans_file.read().decode("utf-8")
    for line in ans_content.splitlines():
        match = re.search(r'(\d+)[\s,.]*([A-D])', line.upper())
        if match: answers_map[int(match.group(1))] = match.group(2)

    doc = fitz.open("temp.pdf")
    
    @st.cache_resource
    def load_ocr():
        return easyocr.Reader(['en'], gpu=torch.cuda.is_available())
    
    reader = load_ocr()

    # --- PREVIEW SECTION ---
    page_idx = st.number_input("Preview Page #", min_value=1, max_value=len(doc)) - 1
    page = doc[page_idx]
    
    # Get OCR Preview
    pix = page.get_pixmap(dpi=100) # Low DPI for fast preview
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    draw = ImageDraw.Draw(img)
    
    # Run OCR on current page
    img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)
    results = reader.readtext(img_np)
    
    detected_y = []
    for (bbox, text, prob) in results:
        x0 = bbox[0][0] * (page.rect.width / pix.width)
        y0 = bbox[0][1] * (page.rect.height / pix.height)
        clean = "".join(filter(str.isdigit, text))
        
        if clean and x0 < margin_limit:
            # Draw visual feedback
            draw.rectangle([bbox[0][0], bbox[0][1], bbox[2][0], bbox[2][1]], outline="red", width=3)
            detected_y.append(y0)

    st.image(img, caption="Red boxes show detected Question Numbers. If questions are missing, increase 'Left Margin' in sidebar.", use_container_width=True)

    # --- PROCESSING & DOWNLOAD ---
    if st.button("🚀 Process Full PDF & Download Zip"):
        output_folder = "quiz_package"
        if os.path.exists(output_folder): shutil.rmtree(output_folder)
        os.makedirs(output_folder)
        
        q_num = 1
        markdown_content = ""
        
        progress_bar = st.progress(0)
        
        for p_idx in range(len(doc)):
            p = doc[p_idx]
            # ... (Your existing cropping logic goes here, using the margin_limit variable) ...
            # [Logic simplified for brevity, use your original loop with variables]
            progress_bar.progress((p_idx + 1) / len(doc))

        shutil.make_archive("results", 'zip', output_folder)
        with open("results.zip", "rb") as f:
            st.download_button("📥 Download Quiz Package", f, file_name="quiz_results.zip")
          
