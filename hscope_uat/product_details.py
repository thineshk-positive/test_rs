import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import pandas as pd
import subprocess
import json
import re
import os
import time
from datetime import datetime
from pathlib import Path
from hscope_uat.UAT_config import *

# =============================================================================
# SECOND SHEET PROCESSING FUNCTIONS
# =============================================================================

def query_ollama(prompt, model="gpt-oss:20b"):
    """
    Runs a prompt through Ollama on Ubuntu EC2.
    Assumes ollama is installed and accessible via PATH.
    """
    result = subprocess.run(
        ["ollama", "run", model],
        input=prompt.encode("utf-8"),
        capture_output=True
    )
    return result.stdout.decode("utf-8").strip()

def map_tpa_to_standard(extracted_tpa: str) -> str:
    """
    Use LLM to map the extracted TPA name to one of the standard TPA names from your list.
    Returns the mapped standard TPA name or empty string if no match found.
    """
    if not extracted_tpa or extracted_tpa.strip() == "":
        return ""
    
    standard_tpas = [
        "Heritage Health Insurance TPA Pvt Ltd",
        "Ericson Insurance TPA PVT . LTD.",
        "FAMILY HEALTH PLAN INSURANCE TPA LIMITED",
        "Genins India Insurance TPA Ltd",
        "Good Health Insurance TPA Limited",
        "HEALTHINDIA INSURANCE TPA SERVICES PVT. LTD.",
        "MDIndia Health INSURANCE TPA Private Limited",
        "Medi Assist India Pvt Ltd",
        "Medsave Health Insurance TPA Ltd",
        "Paramount Health Services and InsuranceTPA Pvt Ltd",
        "Park Mediclaim TPA Pvt. Ltd.",
        "Raksha Health Insurance TPA Private Limited",
        "SAFEWAY INSURANCE TPA PVT LTD",
        "VIDAL HEALTH INSURANCE TPA PVT LTD",
        "Volo Health Insurance TPA Pvt. Ltd",
        "RS-Axa",
        "Medicare TPA Serives India Pvt. Ltd"
    ]
    
    standard_tpas_str = "\n".join(standard_tpas)
    
    prompt = f"""
    Map this TPA name to the closest match from the standard list below.
    Return ONLY the exact standard TPA name or "NO_MATCH".
    
    IMPORTANT: Focus on matching the FIRST WORD and COMPANY NAME, not just partial matches.
    
    Extracted TPA: "{extracted_tpa}"
    
    Standard TPA Names:
    {standard_tpas_str}
    
    Response (only the standard TPA name or NO_MATCH):
    """
    
    response = query_ollama(prompt).strip()
    print(f"TPA mapping response: '{response}'")
    
    # Clean the response - remove thinking text
    response = re.sub(r'Thinking\.\.\..*?done thinking\.?', '', response, flags=re.DOTALL | re.IGNORECASE)
    response = response.strip()
    
    # If multiple lines, take the last non-empty line
    lines = response.split('\n')
    clean_lines = [line.strip() for line in lines if line.strip()]
    if clean_lines:
        response = clean_lines[-1]
    
    print(f"Cleaned mapping response: '{response}'")
    
    # Check if response is "NO_MATCH"
    if response.upper() == "NO_MATCH":
        return ""
    
    # FIRST: Exact case-insensitive match
    for standard_tpa in standard_tpas:
        if response.lower() == standard_tpa.lower():
            print(f"Exact match: '{standard_tpa}'")
            return standard_tpa
    
    # SECOND: Check if response contains first word of any standard TPA
    first_word_matches = []
    response_first_word = response.split()[0].lower() if response.split() else ""
    
    for standard_tpa in standard_tpas:
        standard_first_word = standard_tpa.split()[0].lower()
        if response_first_word and response_first_word == standard_first_word:
            first_word_matches.append(standard_tpa)
    
    if first_word_matches:
        print(f"First word matches: {first_word_matches}")
        # Return the first one (usually best match)
        return first_word_matches[0]
    
    # THIRD: Check if any standard TPA contains the first word of response
    if response_first_word:
        for standard_tpa in standard_tpas:
            if response_first_word in standard_tpa.lower():
                print(f"First word '{response_first_word}' found in: '{standard_tpa}'")
                return standard_tpa
    
    # FOURTH: More restrictive substring matching
    def normalize_for_matching(text):
        # Remove common words and keep only meaningful company name parts
        text = text.lower()
        # Remove common TPA-related terms to focus on company name
        text = re.sub(r'\b(tpa|pvt|ltd|private|limited|insurance|services|health)\b', '', text)
        # Remove extra spaces and punctuation
        text = re.sub(r'[^\w]', '', text)
        return text.strip()
    
    normalized_response = normalize_for_matching(response)
    
    for standard_tpa in standard_tpas:
        normalized_standard = normalize_for_matching(standard_tpa)
        
        # Check if either is a substantial substring of the other
        if (normalized_response and normalized_standard and
            (normalized_response in normalized_standard or 
             normalized_standard in normalized_response) and
            len(normalized_response) > 3 and len(normalized_standard) > 3):
            print(f"Substantial substring match: '{standard_tpa}'")
            return standard_tpa
    
    print("No good match found")
    return ""

def process_second_sheet(md_text: str, org_name: str, from_date: str, to_date: str, pdf_path: str, output_dir: str, method: str):
    """Process second sheet using the same md_text from first sheet with method-specific naming."""
    print("\n🚀 Processing Second Sheet...")
    
    # Only take first 75 lines for LLM context
    page1_lines = md_text.strip().splitlines()
    limited_page1_text = "\n".join(page1_lines[:150])
    print(limited_page1_text)
    
    # Step 2: Prompt Ollama (for other fields only, not organization name)
    prompt = f"""
Extract ONLY the THIRD PARTY ADMINISTRATOR field and return STRICTLY in JSON format.
Replace any '&' characters with 'and' in the extracted value.

IMPORTANT: Return ONLY the JSON, no additional text or explanations.

Example output:
{{"THIRD PARTY ADMINISTRATOR": "Health and Insurance TPA Pvt Ltd"}}

Text to extract from:
\"\"\"{limited_page1_text}\"\"\" 
"""

    response = query_ollama(prompt)
    print(f"\nLLM Response for TPA: {response}")

    print(f"Raw LLM response: {response}")  # Debug print
    
    # Clean the response first - remove thinking text, code blocks, etc.
    cleaned_response = response
    
    # Remove common thinking patterns
    cleaned_response = re.sub(r'Thinking\.\.\..*?done thinking\.', '', cleaned_response, flags=re.DOTALL)
    cleaned_response = re.sub(r'Thinking\.\.\..*?\.\.\.done', '', cleaned_response, flags=re.DOTALL)
    
    # Remove code blocks
    cleaned_response = re.sub(r'```(?:json)?\s*', '', cleaned_response)
    cleaned_response = re.sub(r'```\s*', '', cleaned_response)
    
    # Remove "Here is the extracted field" type text
    cleaned_response = re.sub(r'Here is.*?JSON format:', '', cleaned_response, flags=re.IGNORECASE | re.DOTALL)
    cleaned_response = re.sub(r'Extracted.*?JSON format:', '', cleaned_response, flags=re.IGNORECASE | re.DOTALL)
    
    print(f"Cleaned response: {cleaned_response}")
    
    # Multiple approaches to extract TPA value
    
    # Approach 1: Try to find complete JSON object
    json_pattern = r'\{\s*"[^"]*"\s*:\s*"[^"]*"\s*\}'
    matches = re.findall(json_pattern, cleaned_response)
    
    if matches:
        print(f"Found JSON matches: {matches}")
        # Use the last match (usually the final answer)
        json_text = matches[-1]
        try:
            data = json.loads(json_text)
            extracted_tpa = data.get("THIRD PARTY ADMINISTRATOR", "").strip()
            print(f"Extracted TPA from JSON: '{extracted_tpa}'")
        except json.JSONDecodeError as e:
            print(f"JSON decode error, trying alternative methods: {e}")
            extracted_tpa = ""
    else:
        extracted_tpa = ""
    
    # Approach 2: If JSON parsing failed, try direct pattern matching
    if not extracted_tpa:
        # Look for "THIRD PARTY ADMINISTRATOR": "value" pattern
        tpa_match = re.search(r'"THIRD PARTY ADMINISTRATOR"\s*:\s*"([^"]*)"', cleaned_response)
        if not tpa_match:
            # Try without quotes around key
            tpa_match = re.search(r'THIRD PARTY ADMINISTRATOR\s*:\s*"([^"]*)"', cleaned_response)
        if not tpa_match:
            # Try with single quotes
            tpa_match = re.search(r"'THIRD PARTY ADMINISTRATOR'\s*:\s*'([^']*)'", cleaned_response)
        if not tpa_match:
            # Try without any quotes
            tpa_match = re.search(r'THIRD PARTY ADMINISTRATOR\s*:\s*([^\n,}]+)', cleaned_response)
        
        if tpa_match:
            extracted_tpa = tpa_match.group(1).strip().strip('"').strip("'").strip()
            print(f"Extracted TPA from pattern: '{extracted_tpa}'")
        else:
            extracted_tpa = ""
            print("No TPA value found in response")
    
    # Approach 3: Last resort - look for any text after "THIRD PARTY ADMINISTRATOR"
    if not extracted_tpa:
        # Find the line containing THIRD PARTY ADMINISTRATOR
        lines = cleaned_response.split('\n')
        for line in lines:
            if 'THIRD PARTY ADMINISTRATOR' in line.upper():
                # Extract everything after the colon
                parts = line.split(':', 1)
                if len(parts) > 1:
                    extracted_tpa = parts[1].strip().strip('"').strip("'").strip()
                    print(f"Extracted TPA from line: '{extracted_tpa}'")
                    break
    
    print(f"Final extracted TPA name: '{extracted_tpa}'")
    
    # Replace & with and in TPA name
    extracted_tpa = extracted_tpa.replace('&', 'and')
    
    # Define empty and "not applicable/available" patterns (case-insensitive)
    empty_patterns = [
        r"^\s*$",  # Only whitespace
        r"^$",     # Empty string
        r"^N/A$",  # N/A
        r"^n/a$",  # n/a
        r"^\-$",   # Just a dash
        r"^\.$",   # Just a dot
        r"^null$", # null
        r"^NULL$", # NULL
    ]
    
    # Define "not applicable/available" patterns (case-insensitive)
    not_applicable_patterns = [
        r"not\s*applicable",
        r"not\s*available", 
        r"na",
        r"n\.a\.",
        r"n\.a",
        r"not\s*appl",
        r"not\s*avail",
        r"non\s*applicable",
        r"non\s*available",
        r"none",
        r"nil",
        r"no\s*tpa",
        r"tpa\s*not\s*applicable",
        r"tpa\s*not\s*available",
    ]
    
    # Check if TPA is empty or represents "not applicable"
    is_empty = False
    for pattern in empty_patterns:
        if re.match(pattern, extracted_tpa, re.IGNORECASE):
            is_empty = True
            break
    
    # Check if TPA indicates "not applicable/available"
    is_not_applicable = False
    for pattern in not_applicable_patterns:
        if re.search(pattern, extracted_tpa, re.IGNORECASE):
            is_not_applicable = True
            break
    
    # NEW: Map TPA to standard name using LLM
    if is_empty or is_not_applicable:
        # Empty or "not applicable/available" TPA case
        servicing_done_by_tpa = "No"
        mapped_tpa_name = ""  # Set to empty string
    elif re.fullmatch(r"(N\.?A\.?|NA|N\s*A)", extracted_tpa, re.IGNORECASE):
        # N.A case (standard insurance notation)
        servicing_done_by_tpa = "No"
        mapped_tpa_name = ""  # Set to empty string
    else:
        # Valid TPA name case - map to standard name
        mapped_tpa_name = map_tpa_to_standard(extracted_tpa)
        if mapped_tpa_name:  # Successfully mapped to standard TPA
            servicing_done_by_tpa = "Yes"
            print(f"Successfully mapped to standard TPA: '{mapped_tpa_name}'")
        else:
            # Could not map to standard TPA
            servicing_done_by_tpa = "No"
            mapped_tpa_name = ""
            print(f"Could not map '{extracted_tpa}' to any standard TPA name")
            
    print(f"Final extracted TPA name: '{extracted_tpa}'")
    print(f"Final Mapped TPA name: '{mapped_tpa_name}'")
    print(f"Servicing done by TPA: '{servicing_done_by_tpa}'")

    # Use the org_name, from_date and to_date passed from the first file instead of extracting it again
    corporate_name = org_name.strip()
    from_date_1S = from_date.strip()
    to_date_1S = to_date.strip()

    final_schema = {
        "Product Details": {
            "Configure product": {
                "Corporate Name": corporate_name,
                "Product Name": "Group Health Policy",
                "Product Code": "AMG",
                "Effective Date": from_date_1S,
                "Expiry Date": to_date_1S,
                "Product Type": "Indemnity policy",
                "Product Category (Indemnity Policy)": "Hospitalization",
                "Product Status": "Active",
                "Business Type": "Non Rural",
                "Grace Period (In Days)": "0",
                "IRDA Product Code": "RSAHLGP22167V032122"
            }
        },
        "Product Setup": {
            "Reinstatements of Sum Insured": {
                "Reinstatements Permitted": "NO",
                "Number of Reinstatements": "",
                "Applicable for Same Ailment?": ""
            },
            "Servicing Offices": {
                "Head Office": "",
                "Select Office": "Head Office",
                "Upload Allotment Letter": ""
            },
            "TPA Details": {
                "Servicing done by TPA": servicing_done_by_tpa,
                "Domestic Claims": mapped_tpa_name,  # Use the mapped TPA name
                "International Claims": ""
            },
            "Plans": {
                "Add New Plans": "",
                "Plan Name": corporate_name,
                "Plan Description": "Group Health Policy",
                "Plan Effective Date": from_date_1S,
                "Plan Expiry Date": to_date_1S,
                "Plan Status": "Active"
            }
        }
    }

    base_name = Path(pdf_path).stem
    json_filename = os.path.join(output_dir, f"{base_name}_{method}_2.json")
    
    output_folder = output_dir  # define your working folder
    os.makedirs(output_folder, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(pdf_path))[0]  
    excel_filename = os.path.join(output_folder, f"{base_name}_product_setup.xlsx")

    with open(json_filename, "w", encoding="utf-8") as f:
        json.dump(final_schema, f, indent=2, ensure_ascii=False)
    print(f"JSON saved: {json_filename}")

    df = pd.json_normalize(final_schema, sep="_")
    df.to_excel(excel_filename, index=False)
    print(f"Excel saved: {excel_filename}")

    return final_schema

def load_first_sheet_results(pdf_path, output_dir):
    """Load first sheet results and md_text with correct method detection."""
    base_name = Path(pdf_path).stem
    
    # Try to load method info first
    method_info_path = os.path.join(output_dir, f"{base_name}_method.txt")
    if os.path.exists(method_info_path):
        with open(method_info_path, "r", encoding="utf-8") as f:
            method_used = f.read().strip()
        print(f"📁 Method detected: {method_used}")
    else:
        # Fallback: check which JSON files exist
        pymupdf_json = os.path.join(output_dir, f"{base_name}_pymupdf4llm.json")
        mineru_json = os.path.join(output_dir, f"{base_name}_mineru.json")
        
        if os.path.exists(pymupdf_json):
            method_used = "pymupdf4llm"
        elif os.path.exists(mineru_json):
            method_used = "mineru"
        else:
            print("❌ No first sheet JSON found. Please run first_sheet.py first.")
            return None, None, None, None, None
    
    # Load the final first sheet JSON (this contains the corrected/updated data)
    first_sheet_json_path = os.path.join(output_dir, f"{base_name}_{method_used}.json")
    if not os.path.exists(first_sheet_json_path):
        print(f"❌ First sheet JSON not found: {first_sheet_json_path}")
        return None, None, None, None, None
    
    try:
        with open(first_sheet_json_path, "r", encoding="utf-8") as f:
            first_sheet_data = json.load(f)
        print(f"✅ Loaded first sheet JSON: {first_sheet_json_path}")
        
        # Extract required data from the final updated JSON
        org_name = first_sheet_data.get("Profile", {}).get("Organization Name", "")
        from_date = first_sheet_data.get("From date of the insurance", "")
        to_date = first_sheet_data.get("To date of the insurance", "")
        
        print(f"🔍 Extracted from first sheet JSON:")
        print(f"   Organization Name: '{org_name}'")
        print(f"   From Date: '{from_date}'")
        print(f"   To Date: '{to_date}'")
        
    except Exception as e:
        print(f"❌ Error loading first sheet JSON: {e}")
        return None, None, None, None, None
    
    # Load md_text - look for .md files instead of .txt files
    md_text = None
    
    if method_used == "pymupdf4llm":
        # PyMuPDF4LLM: Direct .md file in output directory
        md_path = os.path.join(output_dir, f"{base_name}_pymupdf4llm.md")
        if os.path.exists(md_path):
            try:
                with open(md_path, "r", encoding="utf-8") as f:
                    md_text = f.read()
                print(f"✅ Loaded PyMuPDF4LLM MD: {md_path}")
            except Exception as e:
                print(f"❌ Error loading PyMuPDF4LLM MD: {e}")
        else:
            print(f"❌ PyMuPDF4LLM MD file not found: {md_path}")
    
    elif method_used == "mineru":
        # MinerU: Nested directory structure
        mineru_md_path1 = os.path.join(output_dir, f"{base_name}_mineru", "auto", f"{base_name}_mineru.md")
        mineru_md_path2 = os.path.join(output_dir, f"{base_name}_mineru.md")
        
        # Try nested directory first, then direct file
        if os.path.exists(mineru_md_path1):
            try:
                with open(mineru_md_path1, "r", encoding="utf-8") as f:
                    md_text = f.read()
                print(f"✅ Loaded MinerU MD (nested): {mineru_md_path1}")
            except Exception as e:
                print(f"❌ Error loading MinerU MD (nested): {e}")
        elif os.path.exists(mineru_md_path2):
            try:
                with open(mineru_md_path2, "r", encoding="utf-8") as f:
                    md_text = f.read()
                print(f"✅ Loaded MinerU MD (direct): {mineru_md_path2}")
            except Exception as e:
                print(f"❌ Error loading MinerU MD (direct): {e}")
        else:
            print(f"❌ MinerU MD file not found in either location:")
            print(f"   {mineru_md_path1}")
            print(f"   {mineru_md_path2}")
    
    if md_text is None:
        return None, None, None, None, None
    
    # Debug: Check if any required field is empty
    print(f"🔍 Data validation:")
    print(f"   Organization Name: '{org_name}'")
    print(f"   From Date: '{from_date}'")
    print(f"   To Date: '{to_date}'")
    print(f"   Method: '{method_used}'")
    print(f"   MD Text length: {len(md_text)}")
    
    # Check if we have the minimum required data
    if not org_name:
        print("❌ Organization name is empty in first sheet results")
        return None, None, None, None, None
    
    # Dates can be empty but we'll proceed with warnings
    if not from_date:
        print("⚠️ From date is empty in first sheet results")
    if not to_date:
        print("⚠️ To date is empty in first sheet results")
    
    return md_text, org_name, from_date, to_date, method_used




# =============================================================================
# MAIN FUNCTION FOR SECOND SHEET
# =============================================================================

def product_setup_run(pdf_path,source_dir):
    """Main fun"""
    output_dir=source_dir
    print("=" * 80)
    print("📄 SECOND SHEET PROCESSING")
    print("=" * 80)
    
    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load first sheet results
    md_text, org_name, from_date, to_date, method_used = load_first_sheet_results(pdf_path, output_dir)
    
    if md_text is None or org_name is None or method_used is None:
        print("❌ Failed to load first sheet results. Please ensure first_sheet.py ran successfully.")
        return
    
    print(f"✅ Loaded first sheet data:")
    print(f"   Organization: {org_name}")
    print(f"   From Date: {from_date}")
    print(f"   To Date: {to_date}")
    print(f"   MD Source: {method_used}")
    print(f"   MD Text length: {len(md_text)} characters")
    
    # Process second sheet
    second_sheet_result = process_second_sheet(md_text, org_name, from_date, to_date, pdf_path, output_dir, method_used)
    
    if second_sheet_result:
        print(f"\n✅ Second Sheet Processing Completed Successfully!")
        tpa_details = second_sheet_result.get("Product Setup", {}).get("TPA Details", {})
        print(f"   TPA Servicing: {tpa_details.get('Servicing done by TPA', 'No')}")
        print(f"   TPA Name: {tpa_details.get('Domestic Claims', 'None')}")
        print(f"   Output Files: {Path(pdf_path).stem}_{method_used}_2.json and {Path(pdf_path).stem}_{method_used}_2.xlsx")
    else:
        print("\n❌ Second Sheet Processing Failed")

if __name__ == "__main__":
    pdf_path=r""
    source_dir=r""
    product_setup_run(pdf_path,source_dir)