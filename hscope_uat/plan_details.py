from bs4 import BeautifulSoup
from io import StringIO
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import pandas as pd
import subprocess
import json
import re
import os
from pathlib import Path
import pdfplumber
from collections import Counter
from rapidfuzz import process, fuzz
from hscope_uat.helper.get_mineru import parse_doc
from hscope_uat.helper.get_si import get_sum_insured_from_mineru
from hscope_uat.UAT_config import *
import pymupdf4llm

try:
    from langchain_ollama import ChatOllama
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False


##################################################################################
# =============================================================================
# CORE PROCESSING FUNCTIONS
# =============================================================================

def pdf_to_md(pdf_path):
    """Convert PDF to markdown using PyMuPDF4LLM."""
    try:
        import pymupdf4llm
        md_text = pymupdf4llm.to_markdown(pdf_path)
        return md_text
    except ImportError:
        print("❌ pymupdf4llm not available")
        return ""

def pdf_first_page_to_image(pdf_path: str):
    """Convert the first page of PDF to a PNG image using PyMuPDF."""
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    doc = fitz.open(pdf_path)

    # Take only the first page
    page = doc[0]
    pix = page.get_pixmap(dpi=300)

    png_path = f"{base_name}_page1.png"
    pix.save(png_path)
    doc.close()

    print(f"[pdf_first_page_to_image] Saved: {png_path}")
    return png_path

def pdf_to_dataframe_ocr(pdf_path: str):
    """Convert first page of PDF → Image → OCR → DataFrame."""
    image_path = pdf_first_page_to_image(pdf_path)
    
    try:
        # OCR to text
        text = pytesseract.image_to_string(Image.open(image_path)) or ""

        # Build DataFrame similar to pdfplumber output
        pages_data = [{"order": 1, "text": text, "tables": []}]
        return pd.DataFrame(pages_data)
    
    finally:
        # Always delete the temporary PNG image
        if os.path.exists(image_path):
            os.remove(image_path)
            print(f"🧹 Cleaned up temporary image: {image_path}")

def pdf_to_dataframe_plumber(pdf_path: str):
    """Extract full text + tables using pdfplumber."""
    pages_data = []
    with pdfplumber.open(pdf_path) as pdf:
        for order, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            tables = page.extract_tables() or []
            pages_data.append({
                "order": order,
                "text": text,
                "tables": tables
            })
    return pd.DataFrame(pages_data)

def query_ollama(prompt, model="llama3.2:latest"):
    """Run prompt through Ollama on Ubuntu EC2."""
    result = subprocess.run(
        ["ollama", "run", model],
        input=prompt.encode("utf-8"),
        capture_output=True
    )
    return result.stdout.decode("utf-8").strip()



# =============================================================================
# SCHEMA BUILDING FUNCTION (UPDATED TO USE FIRST SHEET DATA)
# =============================================================================

def build_schema_from_plumber(df: pd.DataFrame, org_name: str, from_date: str, to_date: str, pdf_path: str, output_dir: str, method_used: str):
    """
    Build schema based on pdfplumber extracted text & tables.
    Uses first sheet data for organization name and dates.
    Sum insured options come from get_sum_insured_from_mineru().
    """
    all_text = " ".join(df["text"].dropna().tolist())

    # 1. Sum Insured Type
    floater_regex = re.compile(
        r"(endt\.?\s*no\.?\s*6.*floater cover|end\.?\s*no\.?\s*6.*floater cover|no\.?\s*6.*floater cover)",
        re.IGNORECASE
    )
    sum_insured_type = "Floater" if floater_regex.search(all_text) else "Non-Floater"

    # 2. Sum Insured Options (using MinerU-based extraction with method_used parameter)
    try:
        sum_insured_options = get_sum_insured_from_mineru(pdf_path, method_used)
        print(f"[build_schema_from_plumber] MinerU extracted sum insured: {sum_insured_options}")
    except Exception as e:
        print(f"[build_schema_from_plumber] Error extracting sum insured with MinerU: {e}")
        print("[build_schema_from_plumber] Falling back to empty list")
        sum_insured_options = []

    # 3. Policy Number (extracted from third sheet OCR)
    df_ocr = pdf_to_dataframe_ocr(pdf_path)
    page1_text = df_ocr.loc[df_ocr['order'] == 1, 'text'].values[0] if len(df_ocr) > 0 else ""
    
    policy_number = ""
    if page1_text:
        prompt = f"""
        Extract ONLY the Policy Number from the text below and return it as a simple string.
        
        Text:
        \"\"\"{page1_text}\"\"\"
        
        Policy Number:
        """
        response = query_ollama(prompt)
        policy_number = response.strip()
        print(f"[build_schema_from_plumber] Extracted Policy Number: {policy_number}")

    # Build Excel rows - FIXED: This creates the actual data structure
    rows_data = []
    
    # Handle case when no sum insured options found
    if not sum_insured_options:
        print("[build_schema_from_plumber] No sum insured options found, creating single row with empty option")
        row_data = {
            "Plan Details_Plan Name": org_name,
            "Plan Details_Plan Description": "Group Health Policy",
            "Plan Details_Plan Effective Date": from_date,
            "Plan Details_Plan Expiry Date": to_date,
            "Plan Details_Plan Status": "Active",
            "Plan Details_Sum Insured Type": sum_insured_type,
            "Plan Details_Sum Insured Options": "",
            "Plan Details_Is CB/Indexation Applicable?": "No",
            "Plan Details_Policy Number": policy_number,
            "Payment Configuration_Claim Payment": "Employee",
            "Worldwide Cover_Worldwide Covered": "No",
            "Zone Configuration_Zone Applicable?": "No",
            "Zone Configuration_Add Zone": "",
            "Zone Configuration_Zone Name": "",
            "Metro Configuration_Is Metro Configuration Applicable?": "No",
            "Metro Configuration_Select Cities for Metro": ""
        }
        rows_data.append(row_data)
    else:
        for idx, amt in enumerate(sum_insured_options):
            row_data = {
                "Plan Details_Plan Name": org_name if idx == 0 else "",
                "Plan Details_Plan Description": "Group Health Policy" if idx == 0 else "",
                "Plan Details_Plan Effective Date": from_date if idx == 0 else "",
                "Plan Details_Plan Expiry Date": to_date if idx == 0 else "",
                "Plan Details_Plan Status": "Active" if idx == 0 else "",
                "Plan Details_Sum Insured Type": sum_insured_type if idx == 0 else "",
                "Plan Details_Sum Insured Options": str(amt),
                "Plan Details_Is CB/Indexation Applicable?": "No" if idx == 0 else "",
                "Plan Details_Policy Number": policy_number if idx == 0 else "",
                "Payment Configuration_Claim Payment": "Employee" if idx == 0 else "",
                "Worldwide Cover_Worldwide Covered": "No" if idx == 0 else "",
                "Zone Configuration_Zone Applicable?": "No" if idx == 0 else "",
                "Zone Configuration_Add Zone": "" if idx == 0 else "",
                "Zone Configuration_Zone Name": "" if idx == 0 else "",
                "Metro Configuration_Is Metro Configuration Applicable?": "No" if idx == 0 else "",
                "Metro Configuration_Select Cities for Metro": "" if idx == 0 else ""
            }
            rows_data.append(row_data)

    # Create DataFrame
    df_out = pd.DataFrame(rows_data)

    # Save Excel with method-specific naming
    base_name = Path(pdf_path).stem
    output_folder = output_dir  # define your working folder
    os.makedirs(output_folder, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(pdf_path))[0]  
    excel_filename = os.path.join(output_folder, f"{base_name}_plan_details.xlsx")

    json_filename = os.path.join(output_dir, f"{base_name}_{method_used}_3.json")

    df_out.to_excel(excel_filename, index=False)
    print(f"✅ Excel saved: {excel_filename}")

    # Build the proper JSON schema structure
    final_schema = {
        "Product Setup": {
            "Plans": {
                "Plan Details": rows_data  # This contains the actual filled data
            }
        }
    }

    # Save the main JSON with the actual schema
    with open(json_filename, "w", encoding="utf-8") as f:
        json.dump(final_schema, f, indent=2, ensure_ascii=False)
    print(f"✅ JSON saved: {json_filename}")

    return final_schema

# =============================================================================
# LOAD FIRST SHEET RESULTS FUNCTION (SIMILAR TO SECOND SHEET)
# =============================================================================

def load_first_sheet_results(pdf_path, output_dir):
    """Load first sheet results from the JSON file with detailed debugging."""
    base_name = Path(pdf_path).stem
    
    print(f"🔍 Looking for first sheet results for: {base_name}")
    
    # Try to load method info first
    method_info_path = os.path.join(output_dir, f"{base_name}_method.txt")
    if os.path.exists(method_info_path):
        with open(method_info_path, "r", encoding="utf-8") as f:
            method_used = f.read().strip()
        print(f"📁 Method detected from method.txt: {method_used}")
    else:
        # Fallback: check which JSON files exist
        pymupdf_json = os.path.join(output_dir, f"{base_name}_pymupdf4llm.json")
        mineru_json = os.path.join(output_dir, f"{base_name}_mineru.json")
        
        print(f"🔍 Checking for JSON files:")
        print(f"   PyMuPDF: {pymupdf_json} - Exists: {os.path.exists(pymupdf_json)}")
        print(f"   MinerU: {mineru_json} - Exists: {os.path.exists(mineru_json)}")
        
        if os.path.exists(pymupdf_json):
            method_used = "pymupdf4llm"
        elif os.path.exists(mineru_json):
            method_used = "mineru"
        else:
            print("❌ No first sheet JSON found. Please run first_sheet.py first.")
            return None, None, None, None
    
    # Load the final first sheet JSON (this contains the corrected/updated data)
    first_sheet_json_path = os.path.join(output_dir, f"{base_name}_{method_used}.json")
    print(f"📁 Loading first sheet JSON: {first_sheet_json_path}")
    
    if not os.path.exists(first_sheet_json_path):
        print(f"❌ First sheet JSON not found: {first_sheet_json_path}")
        return None, None, None, None
    
    try:
        with open(first_sheet_json_path, "r", encoding="utf-8") as f:
            first_sheet_data = json.load(f)
        print(f"✅ Successfully loaded first sheet JSON")
        
        # Extract required data from the final updated JSON
        org_name = first_sheet_data.get("Profile", {}).get("Organization Name", "")
        from_date = first_sheet_data.get("From date of the insurance", "")
        to_date = first_sheet_data.get("To date of the insurance", "")
        
        print(f"✅ Data extracted from first sheet JSON:")
        print(f"   Organization Name: '{org_name}'")
        print(f"   From Date: '{from_date}'")
        print(f"   To Date: '{to_date}'")
        print(f"   Method: '{method_used}'")
        
    except Exception as e:
        print(f"❌ Error loading first sheet JSON: {e}")
        return None, None, None, None
    
    return org_name, from_date, to_date, method_used

# =============================================================================
# MAIN THIRD SHEET PROCESSING FUNCTION
# =============================================================================

def process_third_sheet(pdf_path: str, output_dir: str):
    """Main function for third sheet processing."""
    print("=" * 80)
    print("📄 THIRD SHEET PROCESSING")
    print("=" * 80)
    
    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load first sheet results (similar to second sheet approach)
    org_name, from_date, to_date, method_used = load_first_sheet_results(pdf_path, output_dir)
    
    # Check if we have the minimum required data
    if org_name is None or method_used is None:
        print("❌ Failed to load first sheet results. Please ensure first_sheet.py ran successfully.")
        return None
    
    # Check if we have the organization name (this is critical)
    if not org_name:
        print("❌ Organization Name is empty in first sheet results. Cannot proceed.")
        return None
    
    # Warn if dates are missing but proceed
    if not from_date:
        print("⚠️ From Date is empty in first sheet results. Proceeding with empty date.")
    if not to_date:
        print("⚠️ To Date is empty in first sheet results. Proceeding with empty date.")
    
    print(f"✅ Loaded first sheet data:")
    print(f"   Organization: '{org_name}'")
    print(f"   From Date: '{from_date}'")
    print(f"   To Date: '{to_date}'")
    print(f"   Method: '{method_used}'")
    
    # Extract plumber data
    print("\n📊 Extracting PDF data with pdfplumber...")
    try:
        df_plumber = pdf_to_dataframe_plumber(pdf_path)
        print(f"✅ Successfully extracted {len(df_plumber)} pages with pdfplumber")
    except Exception as e:
        print(f"❌ Error extracting PDF with pdfplumber: {e}")
        return None
    
    # Build schema (uses first sheet data and MinerU for sum insured extraction)
    print("\n🏗️ Building schema...")
    try:
        schema = build_schema_from_plumber(df_plumber, org_name, from_date, to_date, pdf_path, output_dir, method_used)
        print(f"✅ Successfully built schema")
    except Exception as e:
        print(f"❌ Error building schema: {e}")
        return None
    
    print(f"\n✅ Third Sheet Processing Completed Successfully!")
    print(f"   Output Files: {Path(pdf_path).stem}_{method_used}_3.json and {Path(pdf_path).stem}_{method_used}_3.xlsx")
    
    return schema

    # =============================================================================
# MAIN EXECUTION
# =============================================================================

def plan_details_run(pdf_path,source_dir):
    """Main function for third sheet processing."""
    output_dir = source_dir
    
    
    result = process_third_sheet(pdf_path, output_dir)
    
    if result:
        print(f"\n🎉 Third sheet processing completed successfully!")
    else:
        
        print(f"\n❌ Third sheet processing failed.")



# -----------------------------
# Example Main Run
# -----------------------------
if __name__ == "__main__":
    # Test with different PDFs
    # pdf_path = "HG00000006000124.pdf"
    # pdf_path = "HG00000007000124.pdf"
    pdf_path = "HG00006654000100.pdf"
    source_dir=r""
    result=plan_details_run(pdf_path=pdf_path,source_dir=source_dir)



    # Extract OCR data (unchanged)
    