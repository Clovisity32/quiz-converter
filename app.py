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

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Quiz Cropper", layout="wide")

# --- HELPER FUNCTIONS ---
def get_image_base64(img):
    """Encodes PIL image to base64 to avoid Streamlit URL errors."""
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode()

@st.cache_resource
def load_ocr():
    """Loads OCR model once and keeps it in memory."""
    return easyocr.Reader(['en'], gpu=torch.cuda.is_available())

# --- APP UI ---
st.title("✂️ Interactive Quiz Cropper")
st.markdown("1. Upload Files → 2. Edit Boxes (Move/Resize/Delete) → 3. Download Zip")

with st.sidebar:
    st.header("1. Upload")
    pdf_file = st.file_uploader("Exam PDF", type="pdf")
    ans_file = st.file_uploader("Answer Key (txt)", type="txt")
    
    st.header("2. Settings")
    dpi_preview = st.slider("Preview Quality", 72, 150, 100)
    dpi_final = st.select_slider("Final Crop Quality", options=[200, 300, 400], value=300)

# --- MAIN LOGIC ---
if pdf_file and ans_file:
    # Save PDF locally
    with open("temp.pdf", "wb") as f:
        f.write(pdf_file.getbuffer())
    
    doc = fitz.open("temp.pdf")
    reader = load_ocr()

    # Page Selection
    page_num = st.sidebar.number_input("Current Page", min_value=1, max_value=len(doc), step=1) - 1
    page = doc[page_num]
    
    # 1. Render Page for Display
    pix = page.get_pixmap(dpi=dpi_preview)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    
    # 2. AI Detection (only runs once per page)
    if f"rects_{page_num}" not in st.session_state:
        with st.spinner("AI is finding question numbers..."):
            img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)
            results = reader.readtext(img_np)
            
            initial_objects = []
            for (bbox, text, prob) in results:
                clean = "".join(filter(str.isdigit, text))
                # Logic: Is it a number in the left margin?
                if clean and (bbox[0][0] * (page.rect.width / pix.width)) < 80:
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

    # 3. Interactive Canvas
    st.info("💡 Instructions: Click a box to Move/Resize. Press 'Delete' to remove. Draw new boxes if AI missed any.")
    bg_url = get_image_base64(img)
    
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
        key=f"canvas_{page_num}",
    )

    # 4. Final Processing
    if st.button("🚀 Process & Generate Zip"):
        output_folder = "quiz_package"
        if os.path.exists(output_folder): shutil.rmtree(output_folder)
        os.makedirs(output_folder)

        # Load answers into map
        answers_map = {}
        ans_content = ans_file.read().decode("utf-8")
        for line in ans_content.splitlines():
            match = re.search(r'(\d+)[\s,.]*([A-D])', line.upper())
            if match: answers_map[int(match.group(1))] = match.group(2)

        # We use the boxes from the current canvas for the final crop
        if canvas_result.json_data:
            objects = canvas_result.json_data["objects"]
            # Sort boxes by top-to-bottom
            objects.sort(key=lambda x: x["top"])
            
            markdown_content = f"Quiz: {pdf_file.name}\n\n"
            
            for i, obj in enumerate(objects):
                # Scale coordinates back to PDF points
                scale_x = page.rect.width / pix.width
                scale_y = page.rect.height / pix.height
                
                # Logic: Crop from this box's Y until the next box's Y (or end of page)
                y_start = obj["top"] * scale_y
                if i + 1 < len(objects):
                    y_end = objects[i+1]["top"] * scale_y
                else:
                    y_end = page.rect.height * 0.98

                crop_rect = fitz.Rect(0, y_start - 5, page.rect.width, y_end - 5)
                
                # High-res crop
                pix_high = page.get_pixmap(clip=crop_rect, dpi=dpi_final)
                img_name = f"q_{i+1}.png"
                pix_high.save(os.path.join(output_folder, img_name))
                
                correct = answers_map.get(i+1, "A")
                markdown_content += f"{i+1}. ![]({img_name})\n"
                for letter in ['A', 'B', 'C', 'D']:
                    prefix = "*" if letter == correct else ""
                    markdown_content += f"{prefix}{letter.lower()}) {letter}\n"
                markdown_content += "\n"

            with open(os.path.join(output_folder, "questions.txt"), "w") as f:
                f.write(markdown_content)

            # Zip and Download
            shutil.make_archive("quiz_results", 'zip', output_folder)
            with open("quiz_results.zip", "rb") as f:
                st.download_button("📥 Download Results (.zip)", f, file_name="quiz_results.zip")
            st.success("✅ Package ready!")

else:
    st.warning("Please upload both a PDF and an Answers.txt file to begin.")
    
