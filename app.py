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
st.title("✂️ AI Quiz Cropper (Hardened Version)")

# --- 4. SIDEBAR ---
with st.sidebar:
    st.header("📂 1. Upload")
    pdf_file = st.file_uploader("Exam PDF", type="pdf")
    ans_file = st.file_uploader("Answers TXT", type="txt")
    
    st.header("⚙️ 2. Settings")
    dpi_preview = st.slider("Preview Quality", 50, 100, 75)
    dpi_final = st.select_slider("Export Resolution", options=[200, 300, 400], value=300)

# --- 5. MAIN LOGIC ---
if pdf_file and ans_file:
    # Save PDF locally
    if "pdf_path" not in st.session_state:
        with open("temp_exam.pdf", "wb") as f:
            f.write(pdf_file.getbuffer())
        st.session_state.pdf_path = "temp_exam.pdf"
    
    doc = fitz.open(st.session_state.pdf_path)
    reader = load_ocr()

    page_num = st.sidebar.number_input("Current Page", min_value=1, max_value=len(doc), step=1) - 1
    page = doc[page_num]
    
    # Render Preview
    pix = page.get_pixmap(dpi=dpi_preview)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    bg_url = get_image_base64(img)
    
    # 6. OCR INITIALIZATION (Separated from Canvas)
    state_key = f"rects_{page_num}"
    if state_key not in st.session_state:
        if st.button(f"🔍 Scan Page {page_num + 1} for Questions"):
            with st.spinner("AI finding question numbers..."):
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
                st.rerun()
    
    # 7. THE CANVAS (Only shows after scanning or if empty)
    if state_key in st.session_state:
        st.subheader(f"Editing Page {page_num + 1}")
        
        # We use a static key to keep it from resetting while editing
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
            key=f"fixed_canvas_{page_num}", 
        )

        # Sync changes back to state
        if canvas_result.json_data is not None:
            if canvas_result.json_data["objects"] != st.session_state[state_key]["objects"]:
                st.session_state[state_key] = canvas_result.json_data

    # --- 8. FINAL BATCH PROCESSING ---
    st.divider()
    if st.button("🚀 Finalize & Generate ZIP"):
        output_folder = "quiz_package"
        if os.path.exists(output_folder): shutil.rmtree(output_folder)
        os.makedirs(output_folder)

        # Answer Map
        ans_content = ans_file.read().decode("utf-8")
        answers_map = {int(m.group(1)): m.group(2) for m in re.finditer(r'(\d+)[\s,.]*([A-D])', ans_content.upper())}

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

        shutil.make_archive("results", 'zip', output_folder)
        with open("results.zip", "rb") as f:
            st.download_button("📥 Download ZIP", f, file_name="quiz_results.zip")
else:
    st.info("Upload files to begin.")
    
