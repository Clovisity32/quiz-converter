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
    """Converts PIL image to base64 to prevent background_image_url errors."""
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode()

@st.cache_resource
def load_ocr():
    """Initializes EasyOCR and caches it for performance."""
    return easyocr.Reader(['en'], gpu=torch.cuda.is_available())

# --- 3. UI HEADER ---
st.title("✂️ Interactive AI Quiz Cropper")
st.markdown("""
**Workflow:** 1. Upload Files 
2. Use 'Current Page' to review each page 
3. Click/Drag boxes to fix AI mistakes (or draw new ones) 
4. Click 'Process' at the bottom to get your ZIP.
""")

# --- 4. SIDEBAR SETTINGS ---
with st.sidebar:
    st.header("📂 1. Upload Files")
    pdf_file = st.file_uploader("Exam PDF", type="pdf")
    ans_file = st.file_uploader("Answers TXT", type="txt")
    
    st.header("⚙️ 2. Configuration")
    dpi_preview = st.slider("Preview Clarity", 70, 150, 100)
    dpi_final = st.select_slider("Export Resolution", options=[200, 300, 400], value=300)
    st.info("Higher Export Resolution produces better images but takes longer.")

# --- 5. MAIN LOGIC ---
if pdf_file and ans_file:
    # Temporary storage for PDF
    with open("temp.pdf", "wb") as f:
        f.write(pdf_file.getbuffer())
    
    doc = fitz.open("temp.pdf")
    reader = load_ocr()

    # Page Navigation
    page_num = st.sidebar.number_input("Current Page", min_value=1, max_value=len(doc), step=1) - 1
    page = doc[page_num]
    
    # Render PDF page to PIL Image
    pix = page.get_pixmap(dpi=dpi_preview)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    bg_url = get_image_base64(img)
    
    # AI Detection Logic (Only runs if we haven't scanned this page yet)
    if f"rects_{page_num}" not in st.session_state:
        with st.spinner(f"AI scanning Page {page_num + 1}..."):
            img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)
            results = reader.readtext(img_np)
            
            initial_objects = []
            for (bbox, text, prob) in results:
                # Basic cleaning of detected text
                clean = "".join(filter(str.isdigit, text))
                # Heuristic: Is it a digit and is it on the left side of the page?
                if clean and (bbox[0][0] * (page.rect.width / pix.width)) < 100:
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
            st.session_state[f"rects_{page_num}"] = {"objects": initial_objects}

    # 6. INTERACTIVE CANVAS
    st.subheader(f"Edit Questions on Page {page_num + 1}")
    st.caption("Click a box to transform/resize. Press 'Delete' to remove. Draw a rectangle for missing questions.")
    
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",
        stroke_width=2,
        stroke_color="red",
        background_image_url=bg_url,
        initial_drawing=st.session_state[f"rects_{page_num}"],
        update_streamlit=True,
        height=pix.height,
        width=pix.width,
        drawing_mode="transform",
        point_display_radius=0,
        key=f"canvas_p{page_num}",
    )

    # Update session state with manual adjustments immediately
    if canvas_result.json_data is not None:
        st.session_state[f"rects_{page_num}"] = canvas_result.json_data

    # --- 7. FINAL BATCH PROCESSING ---
    st.divider()
    if st.button("🚀 Finalize and Generate Package"):
        output_folder = "quiz_package"
        if os.path.exists(output_folder): shutil.rmtree(output_folder)
        os.makedirs(output_folder)

        # Parse Answer Key
        answers_map = {}
        ans_content = ans_file.read().decode("utf-8")
        for line in ans_content.splitlines():
            match = re.search(r'(\d+)[\s,.]*([A-D])', line.upper())
            if match: answers_map[int(match.group(1))] = match.group(2)

        markdown_content = f"Quiz Title: {pdf_file.name}\n\n"
        global_q_count = 1
        
        progress_bar = st.progress(0)
        
        # Loop through all pages to apply user-edited boxes
        for p_idx in range(len(doc)):
            p = doc[p_idx]
            
            # Check if this page was edited/scanned
            if f"rects_{p_idx}" in st.session_state:
                objs = st.session_state[f"rects_{p_idx}"]["objects"]
                objs.sort(key=lambda x: x["top"]) # Sort by Y coordinate
                
                # We need scaling factors to go from Preview Pixels back to PDF Points
                # We use the preview DPI used when the boxes were created (dpi_preview)
                # But PyMuPDF allows us to use points (72 DPI) directly.
                scale_y = p.rect.height / (p.get_pixmap(dpi=dpi_preview).height)

                for i, obj in enumerate(objs):
                    y_start = obj["top"] * scale_y
                    
                    # Logic: Crop until next box or end of page
                    if i + 1 < len(objs):
                        y_end = objs[i+1]["top"] * scale_y
                    else:
                        y_end = p.rect.height * 0.98

                    crop_rect = fitz.Rect(0, y_start - 5, p.rect.width, y_end - 5)
                    
                    # High-res export
                    pix_final = p.get_pixmap(clip=crop_rect, dpi=dpi_final)
                    img_filename = f"q_{global_q_count}.png"
                    pix_final.save(os.path.join(output_folder, img_filename))
                    
                    # Generate Markdown text
                    correct = answers_map.get(global_q_count, "A")
                    markdown_content += f"{global_q_count}. ![]({img_filename})\n"
                    for letter in ['A', 'B', 'C', 'D']:
                        prefix = "*" if letter == correct else ""
                        markdown_content += f"{prefix}{letter.lower()}) {letter}\n"
                    markdown_content += "\n"
                    
                    global_q_count += 1
            
            progress_bar.progress((p_idx + 1) / len(doc))

        # Save Markdown and Zip
        with open(os.path.join(output_folder, "questions.txt"), "w", encoding="utf-8") as f:
            f.write(markdown_content)
        
        shutil.make_archive("final_quiz", 'zip', output_folder)
        
        with open("final_quiz.zip", "rb") as f:
            st.download_button("📥 Download Quiz Results (.zip)", f, file_name="quiz_results.zip")
        st.success(f"Generated {global_q_count-1} question images successfully!")

else:
    st.info("Waiting for PDF and Answer Key upload...")
    
