import streamlit as st
from streamlit_drawable_canvas import st_canvas
import fitz
import easyocr
import torch
import numpy as np
import os
import re
import shutil
import base64
from PIL import Image
from io import BytesIO

# --- 1. PAGE SETUP ---
st.set_page_config(page_title="AI Quiz Cropper", layout="wide")

# --- 2. HELPER FUNCTIONS ---
def get_image_base64(img):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode()

@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'], gpu=torch.cuda.is_available())

# --- 3. UI HEADER ---
st.title("✂️ AI Quiz Cropper (Stable Version)")

# --- 4. SIDEBAR ---
with st.sidebar:
    st.header("📂 1. Upload")
    pdf_file = st.file_uploader("Exam PDF", type="pdf")
    ans_file = st.file_uploader("Answers TXT", type="txt")
    
    st.header("⚙️ 2. Settings")
    dpi_preview = st.slider("Preview Clarity", 70, 110, 85)
    dpi_final = st.select_slider("Export Resolution", options=[200, 300, 400], value=300)

# --- 5. MAIN LOGIC ---
if pdf_file and ans_file:
    with open("temp.pdf", "wb") as f:
        f.write(pdf_file.getbuffer())
    
    doc = fitz.open("temp.pdf")
    reader = load_ocr()

    page_num = st.sidebar.number_input("Current Page", min_value=1, max_value=len(doc), step=1) - 1
    page = doc[page_num]
    
    # Render Preview
    pix = page.get_pixmap(dpi=dpi_preview)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    bg_url = get_image_base64(img)
    
    # AI Detection with State Guard
    state_key = f"rects_{page_num}"
    if state_key not in st.session_state:
        with st.spinner("AI scanning..."):
            img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)
            results = reader.readtext(img_np)
            initial_objects = []
            for (bbox, text, prob) in results:
                clean = "".join(filter(str.isdigit, text))
                if clean and (bbox[0][0] * (page.rect.width / pix.width)) < 100:
                    initial_objects.append({
                        "type": "rect", "left": bbox[0][0], "top": bbox[0][1],
                        "width": bbox[2][0] - bbox[0][0], "height": bbox[2][1] - bbox[0][1],
                        "fill": "rgba(255, 0, 0, 0.3)", "stroke": "red"
                    })
            st.session_state[state_key] = {"objects": initial_objects}

    # --- 6. THE FIXED CANVAS (Using a Stable Key and Container) ---
    st.subheader(f"Editing Page {page_num + 1}")
    
    # We use a placeholder to clear the canvas during page transitions
    with st.container():
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",
            stroke_width=2,
            stroke_color="red",
            background_image_url=bg_url,
            initial_drawing=st.session_state[state_key],
            update_streamlit=True,
            height=pix.height,
            width=pix.width,
            drawing_mode="transform",
            point_display_radius=0,
            key=f"canvas_stable_p{page_num}", # Unique per page
        )

    # Update State ONLY if the canvas has stable data and it's different
    if canvas_result.json_data is not None:
        if canvas_result.json_data != st.session_state[state_key]:
            st.session_state[state_key] = canvas_result.json_data
            st.rerun() # Ensure the next page load is clean

    # --- 7. FINAL BATCH PROCESSING ---
    if st.button("🚀 Finalize & Download"):
        output_folder = "quiz_package"
        if os.path.exists(output_folder): shutil.rmtree(output_folder)
        os.makedirs(output_folder)

        # Parse Answer Key
        answers_map = {}
        ans_content = ans_file.read().decode("utf-8")
        for line in ans_content.splitlines():
            match = re.search(r'(\d+)[\s,.]*([A-D])', line.upper())
            if match: answers_map[int(match.group(1))] = match.group(2)

        markdown_content = f"Quiz: {pdf_file.name}\n\n"
        q_count = 1
        
        for p_idx in range(len(doc)):
            p = doc[p_idx]
            if f"rects_{p_idx}" in st.session_state:
                objs = st.session_state[f"rects_{p_idx}"]["objects"]
                objs.sort(key=lambda x: x["top"]) 
                
                scale_y = p.rect.height / (p.get_pixmap(dpi=dpi_preview).height)

                for i, obj in enumerate(objs):
                    y_start = obj["top"] * scale_y
                    y_end = objs[i+1]["top"] * scale_y if i+1 < len(objs) else p.rect.height * 0.98

                    crop = fitz.Rect(0, y_start - 5, p.rect.width, y_end - 5)
                    pix_f = p.get_pixmap(clip=crop, dpi=dpi_final)
                    img_name = f"q_{q_count}.png"
                    pix_f.save(os.path.join(output_folder, img_name))
                    
                    ans = answers_map.get(q_count, "A")
                    markdown_content += f"{q_count}. ![]({img_name})\n"
                    for L in ['A', 'B', 'C', 'D']:
                        markdown_content += f"{'*' if L == ans else ''}{L.lower()}) {L}\n"
                    markdown_content += "\n"
                    q_count += 1

        shutil.make_archive("quiz_results", 'zip', output_folder)
        with open("quiz_results.zip", "rb") as f:
            st.download_button("📥 Download Results", f, file_name="quiz_results.zip")
else:
    st.info("Upload files to start.")
    
