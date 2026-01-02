import streamlit as st
import os
import shutil
import zipfile
import uuid
from main import run_all_sheets
from hscope_uat.UAT_config import work_dir, output_dir

# -------------------------------
# App Configuration
# -------------------------------
st.set_page_config(page_title="HSCOPE-PDF Extract", page_icon="📄", layout="wide")
st.markdown(
    "<h1 style='text-align: center; margin-top: 30px;'>HSCOPE-PDF Extract</h1>",
    unsafe_allow_html=True
)

# -------------------------------
# Session-specific directory setup
# -------------------------------
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Create session-specific directories
SESSION_WORK_DIR = os.path.join(work_dir, st.session_state.session_id)
SESSION_DELIVERY_DIR = os.path.join(output_dir, st.session_state.session_id)

# -------------------------------
# Paths
# -------------------------------
WORK_DIR = SESSION_WORK_DIR
DELIVERY_FOLDER = SESSION_DELIVERY_DIR

try:
    BASE_DIR = os.path.dirname(__file__)
except NameError:
    BASE_DIR = os.getcwd()

ICD_DIR = os.path.join(BASE_DIR, "ICD")

# Ensure directories exist
for dir_path in [WORK_DIR, DELIVERY_FOLDER, ICD_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# -------------------------------
# Helpers
# -------------------------------
def clear_dir(dir_path, remove_files=True, remove_dirs=True, exclude_files=None):
    """Remove files and/or directories inside dir_path (not dir_path itself)."""
    exclude_files = exclude_files or []
    if not os.path.exists(dir_path):
        return
    for name in os.listdir(dir_path):
        path = os.path.join(dir_path, name)
        if os.path.isfile(path) and remove_files and name not in exclude_files:
            os.remove(path)
        elif os.path.isdir(path) and remove_dirs:
            shutil.rmtree(path)

def zip_folder(folder_path, zip_path):
    """Zip all files in a folder."""
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                zipf.write(file_path, os.path.relpath(file_path, folder_path))

def cleanup_session_dirs():
    """Remove session-specific directories completely."""
    for dir_path in [SESSION_WORK_DIR, SESSION_DELIVERY_DIR]:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)

# -------------------------------
# Session State
# -------------------------------
if "processed_files" not in st.session_state:
    st.session_state.processed_files = {}  # filename -> metadata dict

# -------------------------------
# UI: File Upload
# -------------------------------
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    st.write("📄 Uploaded:", uploaded_file.name)

    if uploaded_file.name not in st.session_state.processed_files:
        st.session_state.processed_files[uploaded_file.name] = {
            "saved": False,
            "processed": False,
            "zip_path": None,
            "delivery_subfolder": None
        }

    if st.button("Process PDF"):
        with st.spinner("Processing PDF... (this can take some time)"):
            try:
                # Clear session-specific work and delivery directories
                clear_dir(WORK_DIR, remove_files=True, remove_dirs=True)
                clear_dir(DELIVERY_FOLDER, remove_files=True, remove_dirs=True)

                # Save uploaded PDF
                pdf_save_path = os.path.join(WORK_DIR, uploaded_file.name)
                with open(pdf_save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.session_state.processed_files[uploaded_file.name]["saved"] = True

                # Run the heavy processing
                with st.spinner("Running pipeline..."):
                    run_all_sheets(pdf_save_path, WORK_DIR, DELIVERY_FOLDER)

                st.success("✅ PDF processed successfully!")

                # Prepare delivery folder
                pdf_basename = os.path.splitext(uploaded_file.name)[0]
                delivery_subfolder = os.path.join(DELIVERY_FOLDER, pdf_basename)
                os.makedirs(delivery_subfolder, exist_ok=True)

                # Copy XLSX outputs
                for f in os.listdir(DELIVERY_FOLDER):
                    f_path = os.path.join(DELIVERY_FOLDER, f)
                    if os.path.isfile(f_path) and f.endswith(".xlsx"):
                        shutil.copy2(f_path, os.path.join(delivery_subfolder, f))

                # Copy ICD files if any
                icd_files = [f for f in os.listdir(ICD_DIR) if f.endswith(".xlsx")]
                if not icd_files:
                    st.warning("⚠️ No ICD files found in ICD folder – continuing without ICD files.")
                for icd_file in icd_files:
                    shutil.copy2(os.path.join(ICD_DIR, icd_file), os.path.join(delivery_subfolder, icd_file))

                # Zip the folder
                zip_file_path = os.path.join(DELIVERY_FOLDER, f"{pdf_basename}.zip")
                zip_folder(delivery_subfolder, zip_file_path)

                # Update session state
                st.session_state.processed_files[uploaded_file.name].update({
                    "processed": True,
                    "zip_path": zip_file_path,
                    "delivery_subfolder": delivery_subfolder
                })

                st.success("📦 Delivery package prepared!")
                st.write("Files in delivery subfolder:", os.listdir(delivery_subfolder))

            except Exception as e:
                st.error(f"❌ Error while processing PDF: {e}")

# -------------------------------
# Download & Cleanup
# -------------------------------
for fname, meta in st.session_state.processed_files.items():
    if meta.get("processed") and meta.get("zip_path") and os.path.exists(meta["zip_path"]):
        st.subheader(f"Prepared package for {fname}")
        with open(meta["zip_path"], "rb") as f:
            st.download_button(
                label=f"⬇️ Download {os.path.basename(meta['zip_path'])}",
                data=f,
                file_name=os.path.basename(meta["zip_path"]),
                mime="application/zip"
            )
        st.write(f"Delivery folder: {meta.get('delivery_subfolder')}")

if st.button("Cleanup all generated files (WORK_DIR & DELIVERY_FOLDER)"):
    try:
        cleanup_session_dirs()
        st.session_state.processed_files = {}
        st.success("✅ Cleanup complete (ICD folder untouched).")
    except Exception as e:
        st.error(f"❌ Cleanup failed: {e}")
