import fitz  # PyMuPDF
import pymupdf4llm
import os
from pathlib import Path
import re
import json
import ast
import time
import subprocess
import pandas as pd
from langchain_ollama import ChatOllama
from langchain.schema import HumanMessage
import copy
from hscope_uat.UAT_config import *

from hscope_uat.helper.get_mineru import parse_doc
# =============================================================================
# CONSTANTS
# =============================================================================
ORG_SCHEMA = {
    "Profile": {
        "Organization Category": "Corporate",
        "Configuration Type": "Others",
        "Organization Type": "Corporate",
        "Organization Name": "",
        "CIN No.": "",
        "Party Code": "",
        "Registered Address": {
            "Country": "",
            "State": "",
            "District": "",
            "City": "",
            "Area": "",
            "Address Line 1": "",
            "Address Line 2": "",
            "Address Line 3": "",
            "Pin Code": ""
        },
        "Corporate Address": {
            "Same as Registered Address": "YES",
            "Country": "",
            "State": "",
            "District": "",
            "City": "",
            "Area": "",
            "Address Line 1": "",
            "Address Line 2": "",
            "Address Line 3": "",
            "Pin Code": ""
        },
        "Mailing Address": {
            "Same as": "YES",
            "Country": "",
            "State": "",
            "District": "",
            "City": "",
            "Area": "",
            "Address Line 1": "",
            "Address Line 2": "",
            "Address Line 3": "",
            "Pin Code": ""
        },
        "Organization Logo": "",
        "Web Site": "",
        "Phone No": "",
        "Fax No": "",
        "Number of Levels Required": "1",
        "Level Id": "1",
        "Level Definition": "Sales Team",
        "Reports To": "Self"
    },
    "License And Registration": {
        "Registration No": "",
        "Registration Authority": "",
        "Registration Date": "",
        "GST No": "",
        "PAN No": "",
        "TAN No": ""
    },
    "SPOC Details": {
        "Name": "Sales Team",
        "Phone No": "",
        "Country": "",
        "State": "",
        "District": "",
        "City": "",
        "Area": "",
        "Address Line 1": "",
        "Address Line 2": "",
        "Address Line 3": "",
        "Pin Code": "",
        "Email Id": "nomailid@gmail.com",
        "Mobile No": "9999999999",
        "Check Add Spoc Then Save": ""
    }
}

# =============================================================================
# CORE PDF PROCESSING FUNCTIONS
# =============================================================================

def pdf_to_md(pdf_path: str, output_dir: str = None, method: str = "pymupdf4llm") -> str:
    """Convert only the first page of a PDF to markdown using PyMuPDF4LLM."""
    try:
        # Ensure output directory exists
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
        else:
            output_dir = str(Path(pdf_path).parent)

        # Load PDF
        doc = fitz.open(pdf_path)
        if len(doc) == 0:
            raise ValueError("PDF has no pages.")
        
        # Create 1-page temporary PDF path
        first_page_pdf = os.path.join(output_dir, Path(pdf_path).stem + "_page1.pdf")

        # Extract first page into that file
        new_doc = fitz.open()
        new_doc.insert_pdf(doc, from_page=0, to_page=0)
        new_doc.save(first_page_pdf)
        new_doc.close()
        doc.close()

        # Convert that first-page PDF into Markdown
        md_text = pymupdf4llm.to_markdown(first_page_pdf)

        # Save markdown output with method name
        output_md_path = os.path.join(output_dir, Path(pdf_path).stem + f"_{method}.md")
        with open(output_md_path, "w", encoding="utf-8") as f:
            f.write(md_text)

        print(f"✅ Converted first page of '{pdf_path}' → '{output_md_path}'")

        # Optional: cleanup temporary 1-page PDF
        try:
            os.remove(first_page_pdf)
        except Exception as e:
            print(f"⚠️ Could not delete temp file: {e}")

        return md_text

    except Exception as e:
        print(f"❌ PDF to Markdown (first page) failed: {e}")
        return ""

def pdf_first_page_to_image(pdf_path: str, output_dir: str, method: str = "mineru"):
    """Convert the first page of PDF to a PNG image using PyMuPDF and save to output_dir."""
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    doc = fitz.open(pdf_path)

    # Take only the first page
    page = doc[0]
    pix = page.get_pixmap(dpi=300)

    # Save PNG to output directory with method name
    png_path = os.path.join(output_dir, f"{base_name}_{method}.png")
    pix.save(png_path)
    doc.close()

    print(f"[pdf_first_page_to_image] Saved: {png_path}")
    return png_path

def image_to_markdown(image_path: str, output_dir: str) -> str:
    """
    Use MinerU parse_doc to convert image -> markdown.
    Returns the first-page markdown text.
    """
    parse_doc([Path(image_path)], output_dir, backend="pipeline")

    file_base_name = Path(image_path).stem
    md_file = Path(output_dir) / file_base_name / "auto" / f"{file_base_name}.md"

    if not md_file.exists():
        raise FileNotFoundError(f"Markdown file not found: {md_file}")

    with open(md_file, "r", encoding="utf-8") as f:
        md_content = f.read()

    # For single image input, whole file is one "page"
    first_page_text = md_content.strip()
    return first_page_text

# =============================================================================
# LLM UTILITIES
# =============================================================================

def query_field(prompt, model="gpt-oss:20b", temperature=0, max_retries=2):
    """Query Ollama with a strict prompt and return clean JSON with retries."""
    llm = ChatOllama(model=model, temperature=temperature)
    
    for attempt in range(max_retries + 1):
        try:
            response = llm.invoke([HumanMessage(content=prompt)])
            out = response.content or ""
            parsed = extract_json_from_text(out)
            
            if parsed is not None:
                return parsed
            else:
                print(f"⚠️ Attempt {attempt + 1}: Failed to parse JSON from response")
                if attempt < max_retries:
                    print("🔄 Retrying...")
                    time.sleep(1)  # Brief pause before retry
                    
        except Exception as e:
            print(f"⚠️ Attempt {attempt + 1}: LLM query failed: {e}")
            if attempt < max_retries:
                print("🔄 Retrying...")
                time.sleep(1)
    
    print("❌ All retry attempts failed")
    return None

def extract_json_from_text(text):
    if not text:
        return None
    
    # Clean the text more aggressively
    cleaned = re.sub(r'```(?:json)?\s*', '', text, flags=re.IGNORECASE)
    cleaned = re.sub(r'```\s*', '', cleaned)
    cleaned = cleaned.strip()
    
    # Try to find JSON object or array
    json_match = None
    
    # First try to find complete JSON object
    start_brace = cleaned.find('{')
    end_brace = cleaned.rfind('}')
    if start_brace != -1 and end_brace != -1 and end_brace > start_brace:
        json_candidate = cleaned[start_brace:end_brace+1]
        try:
            return json.loads(json_candidate)
        except:
            json_match = json_candidate
    
    # Try array
    start_bracket = cleaned.find('[')
    end_bracket = cleaned.rfind(']')
    if start_bracket != -1 and end_bracket != -1 and end_bracket > start_bracket:
        json_candidate = cleaned[start_bracket:end_bracket+1]
        try:
            return json.loads(json_candidate)
        except:
            json_match = json_candidate
    
    # If we found something that looks like JSON but couldn't parse, try literal eval
    if json_match:
        try:
            return ast.literal_eval(json_match)
        except:
            pass
    
    # Last resort: try to parse the entire cleaned text
    try:
        return json.loads(cleaned)
    except:
        try:
            return ast.literal_eval(cleaned)
        except:
            return None

# =============================================================================
# JSON EXTRACTION FUNCTION
# =============================================================================

def extract_json_from_markdown(md_text, pdf_path, output_dir, method: str):
    """Extract organization schema from markdown text using LLM."""
    
    # Only take first 25 lines for LLM context
    page1_lines = md_text.strip().splitlines()
    limited_page1_text = "\n".join(page1_lines[:50])

    # -------------------------------
    # 1. Organization Name
    # -------------------------------
    org_prompt = f"""
You are given insurance policy details in MARKDOWN format below.
Extract ONLY the insured's organization name and OUTPUT STRICTLY in JSON with the single key: "Name of the insured".
If not found, return {{"Name of the insured": ""}}.

Rules:
- Use ONLY text inside the markdown, do NOT invent.
- Organization name usually appears near "Policy Number Name and Address of the Insured" or immediately after the policy number.
- Remove address-like tokens (e.g., "Panchayat", "Plot", "Lane", "Sector", "Block", "Floor", "No", numbers, building/unit identifiers) from the organization name. These belong in the address.
- Handle concatenated organization tokens:  
   1. Example: "VIHAANNETWORKSLIMITED" → "VIHAAN NETWORKS LIMITED".  
   2. Check 2–3 times whether the split produces meaningful organization suffixes (PRIVATE, LIMITED, LTD, PVT, COMPANY, BROKERS, TECHNOLOGIES, SERVICES, VIDYA, BHAVAN, GLOBAL, CONTROLS etc).  
   3. Do not split valid acronyms.  
- Preserve the original case of the text very strictly (no lowercase/uppercase conversion).
- Replace '&' with 'and'.
- Remove unrelated tokens if accidentally inside (GST, PAN, policy numbers, dates, "from", "to").
- Preserve line order if name spans multiple lines.
- The final organization name must be a single continuous line.
   1. Remove only line breaks, tabs, and multiple consecutive spaces, replacing them with a single space.
   2. Example: "POSITIVE INTEGERS\nPVT LTD" → "POSITIVE INTEGERS PVT LTD" 

Output strictly JSON:
{{"Name of the insured": "<cleaned org name>"}}

Text to extract from:
\"\"\"{limited_page1_text}\"\"\""""




    org_json = query_field(org_prompt)
    org_name = org_json.get("Name of the insured", "").strip() # use here instead of down side

    # -------------------------------
    # 2. Address
    # -------------------------------
    addr_prompt = f"""
You are given insurance policy details in MARKDOWN format below.
Extract ONLY the insured's address and OUTPUT STRICTLY in JSON with the single key: "Address of the insured".
If not found, return {{"Address of the insured": ""}}.

Rules:
- Use ONLY text from the markdown, do NOT invent.
- Address usually follows the organization name in the "Policy Number Name and Address of the Insured" section, or appears under headings like "Address:".
- If the organization name is present partially or completely in the address, strictly remove it.  
  Hint: The organization name to cross-check with is: "{org_name}".
- Prepend any address-like parts removed from the organization name (e.g., "Plot", "Sector", "Block", "Floor", "No", numbers, etc).
- Preserve document line order for the address parts.
- Strictly remove only extra commas or special characters that occur more than once continuously.
  Example: "Mumbai,,," → "Mumbai,", "##Street" → "#Street", "LANE......." → "LANE.".
- Preserve at least one instance of the symbol if it is present. Do not drop it completely.
- Do NOT add extra commas or symbols in place of removed ones.

- If any name, location, state, country, locality name, or related word is repeated anywhere in the address (either consecutively or non-consecutively, including spelling variations), remove duplicates and keep only the first occurrence’s position.
- If the first occurrence is misspelled, replace it with the correct spelling, but keep it in the same position.
- Remove all subsequent (either consecutively or non-consecutively, including spelling variations) duplicates or spelling variations.
- Examples: 
  "Pun, pune, punee" → "Pune"
  "Pun, Maharashtraa, Puneee, Maharashtraa, Pune" → "Pune, Maharashtra"
  "Pune, Maharashtra, India, Pune" → "Pune, Maharashtra, India"

Important:
1. Even if repetitions are non-consecutive, they must be removed after the first corrected occurrence.
2. If the repeated token appears:
- as part of a larger phrase (e.g., “Pune Lane”, “Royapettah High Road”), or
- as a standalone locality or sub-area that follows such a larger phrase (either immediately or non-immediately) within the same address block,
then preserve both occurrences.
- Examples:
- “Royapettah High Road, Royapettah” → keep both
- “Pune Lane, Pune Nagar” → keep both
- "Pune Lane, Pune" → keep both
- “Chennai, Tamil Nadu, Chennai 600014” → remove second Chennai
3. Standalone location tokens (e.g., "CHANDIGARH", "Pune", "Delhi") should be deduplicated.
4. But if the token is part of a larger phrase (e.g., "S.O CHANDIGARH", "(CHANDIGARH)", "CHANDIGARH (M CORP.)"), then it must not be stripped out — the whole phrase should be preserved.
5. Do not alter the original sequence/order of other words in the address.

- Deduplicate location tokens (with spelling variation correction).
- Maintain natural hierarchy order: City → State → Country → Pincode.
- If a city and state both appear, do not drop either; preserve them in order.
- If the same city repeats (e.g., "Gurugram, Haryana Gurugram"), keep only the first occurrence.
- Example: "Gurugram, Haryana Gurugram 122050" → "Gurugram, Haryana - 122050", "Pune, Maharashtra, India, Pune" → "Pune, Maharashtra, India".


- Exception: **City–Pincode duplicates**:
   1. Identify the token immediately before a 6-digit pincode. Mostly, it may be a location especially city.
   2. If this token already appears earlier in the address as a standalone word (i.e., not as part of a larger phrase), then remove the token before the pincode.
   3. If this token appears earlier only as part of a nested phrase or larger phrase (e.g., "Howrah Amta Road"), then preserve the token before the pincode.


- Fix concatenated words (check 2–3 times). Example: "GURUGRAM122050" → "Gurugram - 122050".
- Preserve the original case of the text very strictly (no lowercase/uppercase conversion).
- Auto-spacing rule: Carefully scan the full address for concatenated words, and insert spaces to form meaningful tokens (Sector, Nagar, Colony, Road, Lane, Floor, Block, etc.), while preserving acronyms. Use ONLY text from the markdown, do NOT invent.
- Replace '&' with 'and'.
- Remove unrelated fields like GST, PAN, dates, "from", "to".

Pincode handling:
1. Always a 6-digit number. Move it to the END of the address as " - <Pincode>".
2. If pincode not in the address block, look at the **next 4 lines after the org name**, or **3 lines after the address block**.
3. Special concatenation: If letters+6 digits appear (e.g., "GURUGRAM122050", "#Chennai600018"), split into {{Location}}, {{Pincode}}.  
   Example: "GURUGRAM122050" → "GURUGRAM - 122050".
   Example: "#Chennai600018" → "Chennai - 600018". 
4. Double check that any "#" (hashtag) symbol in the markdown is **removed before the location token**.
5. If line with pincode also has state/city/country, include that text, **before the pincode**. Also, preserve the case of that text.
6. Ignore pincodes appearing **before** the org name.

The final address must be a single continuous line:
1. Preserve meaningful punctuation such as commas, hyphens, periods if they exist.
2. Remove only line breaks, tabs, and multiple consecutive spaces, replacing them with a single space.
3. Example: "BLOCK A2 FIRST FLOOR\nSECTOR-5" → "BLOCK A2 FIRST FLOOR SECTOR-5", "Mumbai,,," → "Mumbai," (comma still preserved).

Output strictly JSON:
{{"Address of the insured": "<cleaned single-line address>" }}

Text to extract from:
\"\"\"{limited_page1_text}\"\"\""""

    addr_json = query_field(addr_prompt)

    # -------------------------------
    # 3. From Date
    # -------------------------------
    from_date_prompt = f"""
You are given insurance policy details in MARKDOWN format below.
Extract ONLY the insurance "From" date and OUTPUT STRICTLY in JSON with the single key: "From date of the insurance".
If not found, return {{"From date of the insurance": ""}}.

Rules:
- Use ONLY the markdown text.
- The "From" date usually appears in a line like: "Period of Insurance: From <date> To <date>".
- Return in strict DD/MM/YYYY format (zero-padded).
- Normalize formats: "24/01/2025", "24-01-2025", "24 Jan 2025", "January 24, 2025", "2025-01-24" → "24/01/2025".
- Select the first valid "From" date after "Period of Insurance". Do not confuse with "Inception date".

Output strictly JSON:
{{"From date of the insurance": "DD/MM/YYYY"}}

Text to extract from:
\"\"\"{limited_page1_text}\"\"\""""

    from_date_json = query_field(from_date_prompt)

    # -------------------------------
    # 4. To Date
    # -------------------------------
    to_date_prompt = f"""
You are given insurance policy details in MARKDOWN format below.
Extract ONLY the insurance "To" date and OUTPUT STRICTLY in JSON with the single key: "To date of the insurance".
If not found, return {{"To date of the insurance": ""}}.

Rules:
- Use ONLY the markdown text.
- The "To" date usually appears in a line like: "Period of Insurance: From <date> To <date>".
- Return in strict DD/MM/YYYY format (zero-padded).
- Normalize date formats as in the From-date rules.
- Sometimes OCR misreads "To" as "T0". Handle this case.
- Do not confuse "To date" with Inception date.

Output strictly JSON:
{{"To date of the insurance": "DD/MM/YYYY"}}

Text to extract from:
\"\"\"{limited_page1_text}\"\"\""""

    to_date_json = query_field(to_date_prompt)

    # -------------------------------
    # 5. Inception Date
    # -------------------------------
    inception_prompt = f"""
You are given insurance policy details in MARKDOWN format below.
Extract ONLY the "Inception date of the insurance" and OUTPUT STRICTLY in JSON with the single key: "Inception date of the insurance".
If not found, return {{"Inception date of the insurance": ""}}.

Rules:
- Use ONLY the markdown text.
- Look for explicit "Inception Date" or "signed on <date>" or "Inception" mentions in markdown. This is NOT the From/To date.
- Return in strict DD/MM/YYYY format.
- Normalize formats (24/01/2025, 24 Jan 2025, 2025-01-24 → 24/01/2025).
- Do not confuse with "Period of Insurance" dates.

Output strictly JSON:
{{"Inception date of the insurance": "DD/MM/YYYY"}}

Text to extract from:
\"\"\"{limited_page1_text}\"\"\""""

    inception_json = query_field(inception_prompt)

    # -------------------------------
    # Merge Results with PIN Code in Both Places
    # -------------------------------
    final_schema = json.loads(json.dumps(ORG_SCHEMA))  # deep copy

    address = addr_json.get("Address of the insured", "").strip()
    from_date = from_date_json.get("From date of the insurance", "").strip()
    to_date = to_date_json.get("To date of the insurance", "").strip()
    inception_date = inception_json.get("Inception date of the insurance", "").strip()

    # Fill schema
    final_schema["Profile"]["Organization Name"] = org_name.strip()
    
    # PIN code extraction and handling
    pincode_match = re.search(r"\b(\d{6})\b", address)
    pincode = pincode_match.group(1) if pincode_match else ""
    
    # Ensure PIN code appears at the end of address
    if pincode:
        # Remove the PIN code from the address if it's in the middle
        address_without_pincode = re.sub(r'\b' + pincode + r'\b', '', address).strip()
        # Add PIN code at the end
        final_address = f"{address_without_pincode} {pincode}".strip()
        final_schema["Profile"]["Registered Address"]["Address Line 1"] = final_address
        final_schema["Profile"]["Registered Address"]["Pin Code"] = pincode
    else:
        final_schema["Profile"]["Registered Address"]["Address Line 1"] = address

    # Add dates to the returned result
    final_schema["From date of the insurance"] = from_date
    final_schema["To date of the insurance"] = to_date
    final_schema["Inception date of the insurance"] = inception_date

    return final_schema

# =============================================================================
# IMPROVED JSON CORRECTION FUNCTION
# =============================================================================

def correct_json_with_context(mineru_json, pymupdf4llm_md, pdf_path, output_dir):
    """Correct MinerU JSON using PyMuPDF4LLM markdown as context."""
    
    # Check if PyMuPDF4LLM markdown has relevant content
    if not pymupdf4llm_md.strip():
        print("❌ PyMuPDF4LLM markdown is empty - PDF is likely an image PDF")
        print("📄 This is an IMAGE PDF")
        return mineru_json, "mineru"
    
    # Check for minimum content length and key insurance-related keywords
    insurance_keywords = ["Policy", "Insurance", "Insured", "Premium", "Cover", "Sum", "Date", "Name", "Address"]
    found_keywords = [keyword for keyword in insurance_keywords if keyword.lower() in pymupdf4llm_md.lower()]
    
    if len(pymupdf4llm_md.strip()) < 50 or len(found_keywords) < 3:
        print(f"❌ PyMuPDF4LLM markdown has insufficient context - only found keywords: {found_keywords}")
        print("📄 This is an IMAGE PDF or poor quality text PDF")
        return mineru_json, "mineru"
    
    print("✅ PyMuPDF4LLM markdown has sufficient context")
    print("📄 This is a TEXT/EXTRACTABLE PDF")
    
    # Prepare context for correction - take more lines for better context
    limited_pymupdf_md = "\n".join(pymupdf4llm_md.strip().splitlines()[:50])
    
    # Extract key fields from current MinerU JSON for the prompt
    current_org_name = mineru_json.get("Profile", {}).get("Organization Name", "")
    current_address = mineru_json.get("Profile", {}).get("Registered Address", {}).get("Address Line 1", "")
    current_pincode = mineru_json.get("Profile", {}).get("Registered Address", {}).get("Pin Code", "")
    current_from_date = mineru_json.get("From date of the insurance", "")
    current_to_date = mineru_json.get("To date of the insurance", "")
    current_inception = mineru_json.get("Inception date of the insurance", "")
    
    # Check which fields are missing or empty in MinerU JSON
    missing_fields = []
    if not current_org_name.strip():
        missing_fields.append("Organization Name")
    if not current_address.strip():
        missing_fields.append("Address")
    if not current_pincode.strip():
        missing_fields.append("Pin Code")
    if not current_from_date.strip():
        missing_fields.append("From Date")
    if not current_to_date.strip():
        missing_fields.append("To Date")
    if not current_inception.strip():
        missing_fields.append("Inception Date")
    
    # Step 1: If there are missing fields, first extract them using the same prompt logic as MinerU
    intermediate_json = json.loads(json.dumps(mineru_json))  # Deep copy for intermediate processing
    
    if missing_fields:
        print(f"⚠️ Missing/empty fields in MinerU JSON: {', '.join(missing_fields)}")
        print("🔄 Step 1: Extracting missing fields using PyMuPDF4LLM markdown...")
        
        # Prepare limited text from PyMuPDF4LLM markdown
        page1_lines = pymupdf4llm_md.strip().splitlines()
        limited_page1_text = "\n".join(page1_lines[:50])
        
        # Track org name for address extraction
        org_name_for_addr = current_org_name
        
        # Extract Organization Name if missing
        if "Organization Name" in missing_fields:
            org_prompt = f"""
You are given insurance policy details in MARKDOWN format below.
Extract ONLY the insured's organization name and OUTPUT STRICTLY in JSON with the single key: "Name of the insured".
If not found, return {{"Name of the insured": ""}}.

Rules:
- Use ONLY text inside the markdown, do NOT invent.
- Organization name usually appears near "Policy Number Name and Address of the Insured" or immediately after the policy number.
- Remove address-like tokens (e.g., "Panchayat", "Plot", "Lane", "Sector", "Block", "Floor", "No", numbers, building/unit identifiers) from the organization name. These belong in the address.
- Handle concatenated organization tokens:  
   1. Example: "VIHAANNETWORKSLIMITED" → "VIHAAN NETWORKS LIMITED".  
   2. Check 2–3 times whether the split produces meaningful organization suffixes (PRIVATE, LIMITED, LTD, PVT, COMPANY, BROKERS, TECHNOLOGIES, SERVICES, VIDYA, BHAVAN, GLOBAL, CONTROLS etc).  
   3. Do not split valid acronyms.  
- Preserve the original case of the text very strictly (no lowercase/uppercase conversion).
- Replace '&' with 'and'.
- Remove unrelated tokens if accidentally inside (GST, PAN, policy numbers, dates, "from", "to").
- Preserve line order if name spans multiple lines.
- The final organization name must be a single continuous line.
   1. Remove only line breaks, tabs, and multiple consecutive spaces, replacing them with a single space.
   2. Example: "POSITIVE INTEGERS\nPVT LTD" → "POSITIVE INTEGERS PVT LTD" 

Output strictly JSON:
{{"Name of the insured": "<cleaned org name>"}}

Text to extract from:
\"\"\"{limited_page1_text}\"\"\""""
            org_json = query_field(org_prompt)
            if org_json:
                new_org = org_json.get("Name of the insured", "").strip()
                if new_org:
                    intermediate_json["Profile"]["Organization Name"] = new_org
                    print(f"   ➕ Organization Name EXTRACTED: {new_org}")
                    org_name_for_addr = new_org
        
        # Extract Address if missing (uses org_name_for_addr)
        if "Address" in missing_fields:
            addr_prompt = f"""
You are given insurance policy details in MARKDOWN format below.
Extract ONLY the insured's address and OUTPUT STRICTLY in JSON with the single key: "Address of the insured".
If not found, return {{"Address of the insured": ""}}.

Rules:
- Use ONLY text from the markdown, do NOT invent.
- Address usually follows the organization name in the "Policy Number Name and Address of the Insured" section, or appears under headings like "Address:".
- If the organization name is present partially or completely in the address, strictly remove it.  
  Hint: The organization name to cross-check with is: "{org_name_for_addr}".
- Prepend any address-like parts removed from the organization name (e.g., "Plot", "Sector", "Block", "Floor", "No", numbers, etc).
- Preserve document line order for the address parts.
- Strictly remove only extra commas or special characters that occur more than once continuously.
  Example: "Mumbai,,," → "Mumbai,", "##Street" → "#Street", "LANE......." → "LANE.".
- Preserve at least one instance of the symbol if it is present. Do not drop it completely.
- Do NOT add extra commas or symbols in place of removed ones.

- If any name, location, state, country, locality name, or related word is repeated anywhere in the address (either consecutively or non-consecutively, including spelling variations), remove duplicates and keep only the first occurrence’s position.
- If the first occurrence is misspelled, replace it with the correct spelling, but keep it in the same position.
- Remove all subsequent (either consecutively or non-consecutively, including spelling variations) duplicates or spelling variations.
- Examples: 
  "Pun, pune, punee" → "Pune"
  "Pun, Maharashtraa, Puneee, Maharashtraa, Pune" → "Pune, Maharashtra"
  "Pune, Maharashtra, India, Pune" → "Pune, Maharashtra, India"

Important:
1. Even if repetitions are non-consecutive, they must be removed after the first corrected occurrence.
2. If the repeated token appears:
- as part of a larger phrase (e.g., “Pune Lane”, “Royapettah High Road”), or
- as a standalone locality or sub-area that follows such a larger phrase (either immediately or non-immediately) within the same address block,
then preserve both occurrences.
- Examples:
- “Royapettah High Road, Royapettah” → keep both
- “Pune Lane, Pune Nagar” → keep both
- "Pune Lane, Pune" → keep both
- “Chennai, Tamil Nadu, Chennai 600014” → remove second Chennai
3. Standalone location tokens (e.g., "CHANDIGARH", "Pune", "Delhi") should be deduplicated.
4. But if the token is part of a larger phrase (e.g., "S.O CHANDIGARH", "(CHANDIGARH)", "CHANDIGARH (M CORP.)"), then it must not be stripped out — the whole phrase should be preserved.
5. Do not alter the original sequence/order of other words in the address.

- Deduplicate location tokens (with spelling variation correction).
- Maintain natural hierarchy order: City → State → Country → Pincode.
- If a city and state both appear, do not drop either; preserve them in order.
- If the same city repeats (e.g., "Gurugram, Haryana Gurugram"), keep only the first occurrence.
- Example: "Gurugram, Haryana Gurugram 122050" → "Gurugram, Haryana - 122050", "Pune, Maharashtra, India, Pune" → "Pune, Maharashtra, India".


- Exception: **City–Pincode duplicates**:
   1. Identify the token immediately before a 6-digit pincode. Mostly, it may be a location especially city.
   2. If this token already appears earlier in the address as a standalone word (i.e., not as part of a larger phrase), then remove the token before the pincode.
   3. If this token appears earlier only as part of a nested phrase or larger phrase (e.g., "Howrah Amta Road"), then preserve the token before the pincode.


- Fix concatenated words (check 2–3 times). Example: "GURUGRAM122050" → "Gurugram - 122050".
- Preserve the original case of the text very strictly (no lowercase/uppercase conversion).
- Auto-spacing rule: Carefully scan the full address for concatenated words, and insert spaces to form meaningful tokens (Sector, Nagar, Colony, Road, Lane, Floor, Block, etc.), while preserving acronyms. Use ONLY text from the markdown, do NOT invent.
- Replace '&' with 'and'.
- Remove unrelated fields like GST, PAN, dates, "from", "to".

Pincode handling:
1. Always a 6-digit number. Move it to the END of the address as " - <Pincode>".
2. If pincode not in the address block, look at the **next 4 lines after the org name**, or **3 lines after the address block**.
3. Special concatenation: If letters+6 digits appear (e.g., "GURUGRAM122050", "#Chennai600018"), split into {{Location}}, {{Pincode}}.  
   Example: "GURUGRAM122050" → "GURUGRAM - 122050".
   Example: "#Chennai600018" → "Chennai - 600018". 
4. Double check that any "#" (hashtag) symbol in the markdown is **removed before the location token**.
5. If line with pincode also has state/city/country, include that text, **before the pincode**. Also, preserve the case of that text.
6. Ignore pincodes appearing **before** the org name.

The final address must be a single continuous line:
1. Preserve meaningful punctuation such as commas, hyphens, periods if they exist.
2. Remove only line breaks, tabs, and multiple consecutive spaces, replacing them with a single space.
3. Example: "BLOCK A2 FIRST FLOOR\nSECTOR-5" → "BLOCK A2 FIRST FLOOR SECTOR-5", "Mumbai,,," → "Mumbai," (comma still preserved).

Output strictly JSON:
{{"Address of the insured": "<cleaned single-line address>" }}

Text to extract from:
\"\"\"{limited_page1_text}\"\"\""""
            addr_json = query_field(addr_prompt)
            if addr_json:
                new_addr = addr_json.get("Address of the insured", "").strip()
                if new_addr:
                    intermediate_json["Profile"]["Registered Address"]["Address Line 1"] = new_addr
                    print(f"   ➕ Address EXTRACTED: {new_addr}")
        
        # Extract dates if missing
        date_mappings = {
            "From Date": "From date of the insurance",
            "To Date": "To date of the insurance", 
            "Inception Date": "Inception date of the insurance"
        }
        
        for missing_field, json_field in date_mappings.items():
            if missing_field in missing_fields:
                if missing_field == "From Date":
                    date_prompt = f"""
You are given insurance policy details in MARKDOWN format below.
Extract ONLY the insurance "From" date and OUTPUT STRICTLY in JSON with the single key: "From date of the insurance".
If not found, return {{"From date of the insurance": ""}}.

Rules:
- Use ONLY the markdown text.
- The "From" date usually appears in a line like: "Period of Insurance: From <date> To <date>".
- Return in strict DD/MM/YYYY format (zero-padded).
- Normalize formats: "24/01/2025", "24-01-2025", "24 Jan 2025", "January 24, 2025", "2025-01-24" → "24/01/2025".
- Select the first valid "From" date after "Period of Insurance". Do not confuse with "Inception date".

Output strictly JSON:
{{"From date of the insurance": "DD/MM/YYYY"}}

Text to extract from:
\"\"\"{limited_page1_text}\"\"\""""
                elif missing_field == "To Date":
                    date_prompt = f"""
You are given insurance policy details in MARKDOWN format below.
Extract ONLY the insurance "To" date and OUTPUT STRICTLY in JSON with the single key: "To date of the insurance".
If not found, return {{"To date of the insurance": ""}}.

Rules:
- Use ONLY the markdown text.
- The "To" date usually appears in a line like: "Period of Insurance: From <date> To <date>".
- Return in strict DD/MM/YYYY format (zero-padded).
- Normalize date formats as in the From-date rules.
- Sometimes OCR misreads "To" as "T0". Handle this case.
- Do not confuse "To date" with Inception date.

Output strictly JSON:
{{"To date of the insurance": "DD/MM/YYYY"}}

Text to extract from:
\"\"\"{limited_page1_text}\"\"\""""
                elif missing_field == "Inception Date":
                    date_prompt = f"""
You are given insurance policy details in MARKDOWN format below.
Extract ONLY the "Inception date of the insurance" and OUTPUT STRICTLY in JSON with the single key: "Inception date of the insurance".
If not found, return {{"Inception date of the insurance": ""}}.

Rules:
- Use ONLY the markdown text.
- Look for explicit "Inception Date" or "signed on <date>" or "Inception" mentions in markdown. This is NOT the From/To date.
- Return in strict DD/MM/YYYY format.
- Normalize formats (24/01/2025, 24 Jan 2025, 2025-01-24 → 24/01/2025).
- Do not confuse with "Period of Insurance" dates.

Output strictly JSON:
{{"Inception date of the insurance": "DD/MM/YYYY"}}

Text to extract from:
\"\"\"{limited_page1_text}\"\"\""""
                
                date_json = query_field(date_prompt)
                if date_json:
                    new_date = date_json.get(json_field, "").strip()
                    if new_date:
                        intermediate_json[json_field] = new_date
                        print(f"   ➕ {missing_field} EXTRACTED: {new_date}")
        
        # Handle Pin Code extraction if missing (direct regex as fallback, but since it's selective, we can skip full prompt here as regex follows)
        # Pin Code will be handled in the next block anyway
        
        print("✅ Missing fields extraction completed")
    else:
        print("❌ Failed to extract missing fields - proceeding with original MinerU data")
    
    # Update current fields after missing field extraction
    current_org_name = intermediate_json.get("Profile", {}).get("Organization Name", "")
    current_address = intermediate_json.get("Profile", {}).get("Registered Address", {}).get("Address Line 1", "")
    current_pincode = intermediate_json.get("Profile", {}).get("Registered Address", {}).get("Pin Code", "")
    current_from_date = intermediate_json.get("From date of the insurance", "")
    current_to_date = intermediate_json.get("To date of the insurance", "")
    current_inception = intermediate_json.get("Inception date of the insurance", "")
    
    # Limit pincode search to only the first 35 lines of the OCR markdown
    limited_md_context = "\n".join(pymupdf4llm_md.splitlines()[:35])
    
    # Check if PIN code is still missing after extraction and try to extract it directly
    if not current_pincode.strip():
        print("🔄 PIN Code still missing - attempting direct PIN code extraction...")
        pincode_match = re.search(r"\b(\d{6})\b", limited_md_context)
        if pincode_match:
            direct_pincode = pincode_match.group(1)
            intermediate_json["Profile"]["Registered Address"]["Pin Code"] = direct_pincode
            print(f"   ➕ Pin Code EXTRACTED (direct): {direct_pincode}")
            
            # Also add to address if address exists
            if current_address.strip() and not current_address.endswith(direct_pincode):
                intermediate_json["Profile"]["Registered Address"]["Address Line 1"] = f"{current_address} {direct_pincode}".strip()
                print(f"   Address updated with PIN: {intermediate_json['Profile']['Registered Address']['Address Line 1']}")
    
    # Step 2: Now perform final JSON correction with enhanced generic prompt
    print("🔄 Step 2: Performing final JSON correction with enhanced generic prompt...")
    
    # Update current fields again after direct PIN code extraction
    current_org_name = intermediate_json.get("Profile", {}).get("Organization Name", "")
    current_address = intermediate_json.get("Profile", {}).get("Registered Address", {}).get("Address Line 1", "")
    current_pincode = intermediate_json.get("Profile", {}).get("Registered Address", {}).get("Pin Code", "")
    
    correction_prompt = f"""
You are given two versions of extracted information from the same PDF document:

1. CURRENT EXTRACTION (from image-based OCR - may have errors or missing fields):
   - Organization Name: "{current_org_name}" {"" if current_org_name.strip() else "[MISSING]"}
   - Address: "{current_address}" {"" if current_address.strip() else "[MISSING]"}
   - Pin Code: "{current_pincode}" {"" if current_pincode.strip() else "[MISSING]"}
   - From Date: "{current_from_date}" {"" if current_from_date.strip() else "[MISSING]"}
   - To Date: "{current_to_date}" {"" if current_to_date.strip() else "[MISSING]"}
   - Inception Date: "{current_inception}" {"" if current_inception.strip() else "[MISSING]"}

2. REFERENCE TEXT (from direct text extraction - more accurate and complete):
\"\"\"{limited_pymupdf_md}\"\"\"

INTEGRATED CORRECTION AND COMPLETION TASKS:

A. PIN CODE EXTRACTION (HIGHEST PRIORITY - CRITICAL):
   - If missing or empty: MUST extract 6-digit PIN code from reference text
   - If present but incomplete/wrong: Correct using reference text
   - Look for 6-digit numbers in address blocks, near city names, at end of addresses
   - Ensure PIN code appears at the end of address AND in dedicated PIN code field
   - Common patterns: "City PIN", "Area PIN", "Location - PIN", standalone 6-digit numbers

B. ADDRESS COMPLETION AND CORRECTION:
   - If missing or empty: Extract complete address from reference text including PIN code
   - If present but has errors: Fix location names, split words, spacing issues
   - Ensure proper address hierarchy (area, city, state, PIN code)
   - Fix OCR errors: concatenation, splitting, character recognition issues
   - Remove duplicate words or phrases, correct spelling of locations

C. ORGANIZATION NAME CORRECTIONS (ENHANCED FOR MULTI-LINE AND COMPLETENESS):
   - If missing or empty: Extract from reference text
   - If present but has errors: Fix concatenated words, split words, spelling mistakes
   - Handle concatenation: "COMPANYNAME" → "COMPANY NAME", "FEDERATIONOF" → "FEDERATION OF"
   - Handle splitting: "COMP ANY" → "COMPANY", "ORGANI SATION" → "ORGANISATION"
   
   **CRITICAL: MULTI-LINE ORGANIZATION NAME RECONSTRUCTION**
   - If organization name appears split across multiple lines in reference text, COMBINE them in proper order
   - Example: "BHAVYABHANU ELECTRONICS PRIVATE\\nLIMITED" → "BHAVYA BHANU ELECTRONICS PRIVATE LIMITED"
   - Look for continuation patterns: check if next line contains missing organization suffixes (PRIVATE, LIMITED, LTD, PVT, COMPANY, CORPORATION, INC, etc.)
   - Preserve ALL parts of the organization name from reference text
   - NEVER drop valid organization suffixes (LIMITED, LTD, PRIVATE, PVT, COMPANY, CORP, INC)
   
   **COMPLETENESS VERIFICATION:**
   - Cross-check current organization name against reference text for missing parts
   - If reference text has more complete organization name, use the complete version
   - Common organization suffixes that MUST be preserved: LIMITED, LTD, PRIVATE, PVT, COMPANY, CORP, INC, CORPORATION
   - If current name is partial (e.g., "BHAVYA BHANU ELECTRONICS PRIVATE") but reference has full name (e.g., "BHAVYABHANU ELECTRONICS PRIVATE LIMITED"), use the complete version
   
   - Preserve the original case style (uppercase/lowercase)
   - Remove address-like tokens from organization name

D. DATE CORRECTIONS:
   - If missing or empty: Extract from reference text in DD/MM/YYYY format
   - If present but has errors: Correct OCR errors in dates
   - Normalize formats to DD/MM/YYYY
   - Handle common OCR errors: "0" vs "O", "1" vs "I", "5" vs "S" in dates

CRITICAL FIELD HANDLING RULES:

1. MISSING FIELD RECOVERY:
   - If any field is marked as [MISSING], you MUST extract it from reference text
   - Do not leave required fields empty if information is available
   - Perform fresh extraction for missing fields, correction for existing fields

2. PIN CODE SPECIFIC RULES (EXTREMELY IMPORTANT):
   - PIN code extraction is MANDATORY if present in reference text
   - Scan entire reference text for 6-digit numbers that could be PIN codes
   - PIN code must appear at the end of address string AND in PIN code field
   - If address exists but PIN is missing, append extracted PIN to address

3. OCR ERROR CORRECTION PATTERNS:
   - Concatenation: "WORD1WORD2" → "WORD1 WORD2" (check 2-3 times for meaningful splits)
   - Splitting: "WO RD" → "WORD" (remove extra spaces between word fragments)
   - Character errors: "0" vs "O", "1" vs "I", "5" vs "S", "8" vs "B"
   - Spacing issues: Remove extra spaces, add missing spaces between words
   - Spelling mistakes: Correct based on context and common words

4. ORGANIZATION NAME SPECIFIC RULES (NEW - CRITICAL):
   - **MULTI-LINE RECONSTRUCTION**: Scan for organization name fragments across consecutive lines and combine them
   - **SUFFIX PRESERVATION**: Never drop valid organization suffixes (LIMITED, LTD, PRIVATE, etc.)
   - **COMPLETENESS CHECK**: Always prefer the most complete organization name found in reference text
   - **LINE CONTINUATION**: If a line ends with organization-related words (PRIVATE, COMPANY, CORPORATION, etc.), check if next line contains continuation suffixes (LIMITED, LTD, INC, etc.)

5. SYMBOL NORMALIZATION RULE (VERY IMPORTANT):
   - ALWAYS convert "&" to "and".
   - NEVER convert "and" back into "&" under any circumstances.
   - If reference text contains "&", convert it to "and".
   - If reference text contains "and", keep it as "and".
   - The final output MUST NOT contain the "&" character anywhere.

6. DATA VALIDATION AND CONSISTENCY:
   - Use ONLY information from reference text - no invention
   - Preserve original text case where possible
   - Ensure dates follow DD/MM/YYYY format consistently
   - Maintain address hierarchy and natural reading order

COMMON INSURANCE DOCUMENT PATTERNS TO LOOK FOR:
- Organization name near "Name and Address of the Insured" or after policy number
- **Organization names often span multiple lines** - look for continuation patterns
- Address following organization name, often with locality, city, state, PIN
- PIN codes typically 6-digit numbers at end of address lines
- Dates in "Period of Insurance: From DD/MM/YYYY To DD/MM/YYYY" format
- Inception date near "signed on" or "Inception Date" mentions

SPECIAL ATTENTION FOR ORGANIZATION NAMES:
- Look for organization name fragments in consecutive lines
- Common pattern: First line has main name, second line has "LIMITED" or "PRIVATE LIMITED"
- Example: "BHAVYABHANU ELECTRONICS PRIVATE\\nLIMITED" → reconstruct as "BHAVYA BHANU ELECTRONICS PRIVATE LIMITED"
- Always preserve the complete legal entity name from reference text

Return ONLY a JSON object with ALL fields in this exact format:
{{
  "Profile": {{
    "Organization Name": "corrected/extracted organization name",
    "Registered Address": {{
      "Address Line 1": "corrected/extracted complete address with PIN code at the end",
      "Pin Code": "corrected/extracted pincode"
    }}
  }},
  "From date of the insurance": "corrected/extracted from date in DD/MM/YYYY",
  "To date of the insurance": "corrected/extracted to date in DD/MM/YYYY", 
  "Inception date of the insurance": "corrected/extracted inception date in DD/MM/YYYY"
}}

IMPORTANT: Return ONLY the JSON object, no additional text or explanations.
"""

    corrected_json = query_field(correction_prompt)
    
    if corrected_json and isinstance(corrected_json, dict):
        print("✅ Successfully performed final JSON correction")
        
        # Merge the corrections back into the intermediate JSON structure
        final_json = json.loads(json.dumps(intermediate_json))  # Deep copy
        
        # Update organization name if corrected
        if "Profile" in corrected_json and "Organization Name" in corrected_json["Profile"]:
            new_org_name = corrected_json["Profile"]["Organization Name"].strip()
            if new_org_name:
                final_json["Profile"]["Organization Name"] = new_org_name
                print(f"   Organization Name CORRECTED: {final_json['Profile']['Organization Name']}")
        
        # Update address and PIN code - prioritize these fields
        if "Profile" in corrected_json and "Registered Address" in corrected_json["Profile"]:
            addr_corrections = corrected_json["Profile"]["Registered Address"]
            
            # Update PIN Code first (most important)
            if "Pin Code" in addr_corrections:
                new_pincode = addr_corrections["Pin Code"].strip()
                if new_pincode:
                    final_json["Profile"]["Registered Address"]["Pin Code"] = new_pincode
                    print(f"   Pin Code CORRECTED: {final_json['Profile']['Registered Address']['Pin Code']}")
            
            # Update Address Line 1
            if "Address Line 1" in addr_corrections:
                new_address = addr_corrections["Address Line 1"].strip()
                if new_address:
                    final_json["Profile"]["Registered Address"]["Address Line 1"] = new_address
                    print(f"   Address CORRECTED: {final_json['Profile']['Registered Address']['Address Line 1']}")
            
            # Ensure PIN code is at the end of address and matches PIN code field
            current_pincode = final_json["Profile"]["Registered Address"]["Pin Code"]
            current_address = final_json["Profile"]["Registered Address"]["Address Line 1"]
            
            if current_pincode and current_address:
                # Remove any existing pincode from address
                address_without_pincode = re.sub(r'\b' + current_pincode + r'\b', '', current_address).strip()
                # Add current pincode at the end
                final_json["Profile"]["Registered Address"]["Address Line 1"] = f"{address_without_pincode} {current_pincode}".strip()
                print(f"   Final address with PIN: {final_json['Profile']['Registered Address']['Address Line 1']}")
        
        # Update dates if corrected
        date_fields = ["From date of the insurance", "To date of the insurance", "Inception date of the insurance"]
        for field in date_fields:
            if field in corrected_json:
                new_date = corrected_json[field].strip()
                if new_date:
                    final_json[field] = new_date
                    print(f"   {field} CORRECTED: {final_json[field]}")
        
        # Final verification - if PIN code is still missing, try one more direct extraction
        final_pincode = final_json["Profile"]["Registered Address"]["Pin Code"]
        if not final_pincode.strip():
            print("🔄 Final PIN code verification - still missing, attempting regex extraction...")
            pincode_match = re.search(r"\b(\d{6})\b", limited_md_context)
            if pincode_match:
                final_pincode = pincode_match.group(1)
                final_json["Profile"]["Registered Address"]["Pin Code"] = final_pincode
                # Also add to address
                current_addr = final_json["Profile"]["Registered Address"]["Address Line 1"]
                if current_addr and not current_addr.endswith(final_pincode):
                    final_json["Profile"]["Registered Address"]["Address Line 1"] = f"{current_addr} {final_pincode}".strip()
                print(f"   ➕ Pin Code EXTRACTED (final): {final_pincode}")
                print(f"   Final address: {final_json['Profile']['Registered Address']['Address Line 1']}")
        
        return final_json, "pymupdf4llm"
    else:
        print("❌ Final JSON correction failed")
        # If correction failed but we extracted missing fields, return the intermediate JSON
        if missing_fields:
            print("🔄 Using intermediate JSON with extracted missing fields")
            return intermediate_json, "pymupdf4llm"
        else:
            print("🔄 Using original MinerU JSON")
            return mineru_json, "mineru"
        
        
# =============================================================================
# MAIN PROCESSING FLOW
# =============================================================================

def process_pdf_new_flow(pdf_path, output_dir):
    """New flow: MinerU first, then PyMuPDF4LLM for correction if possible."""
    
    print("=" * 80)
    print("🔄 NEW PROCESSING FLOW: MinerU → PyMuPDF4LLM Correction")
    print("=" * 80)
    
    base_name = Path(pdf_path).stem
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Step 1: Convert first page to image and run MinerU
    print("\n📸 Step 1: Converting first page to image and running MinerU...")
    image_path = pdf_first_page_to_image(pdf_path, output_dir, method="mineru")
    mineru_md = image_to_markdown(image_path, output_dir)
    
    if not mineru_md.strip():
        print("❌ MinerU markdown extraction failed")
        return None, "mineru"
    
    # Step 2: Extract JSON from MinerU markdown
    print("\n🔍 Step 2: Extracting JSON from MinerU markdown...")
    mineru_json = extract_json_from_markdown(mineru_md, pdf_path, output_dir, method="mineru")
    
    # Save initial MinerU JSON
    mineru_json_path = os.path.join(output_dir, f"{base_name}_mineru.json")
    with open(mineru_json_path, "w", encoding="utf-8") as f:
        json.dump(mineru_json, f, indent=2, ensure_ascii=False)
    print(f"✅ MinerU JSON saved: {mineru_json_path}")
    
    # Step 3: Run PyMuPDF4LLM on first page
    print("\n📄 Step 3: Running PyMuPDF4LLM on first page...")
    pymupdf4llm_md = pdf_to_md(pdf_path, output_dir, method="pymupdf4llm")
    
    # Step 4: Correct MinerU JSON using PyMuPDF4LLM context
    print("\n✏️  Step 4: Correcting JSON using PyMuPDF4LLM context...")
    final_json, final_method = correct_json_with_context(mineru_json, pymupdf4llm_md, pdf_path, output_dir)
    
    # Step 5: Save final outputs
    print(f"\n💾 Step 5: Saving final outputs ({final_method})...")
    
    # Save JSON (with dates for second sheet use)
    json_filename = os.path.join(output_dir, f"{base_name}_{final_method}.json")
    with open(json_filename, "w", encoding="utf-8") as f:
        json.dump(final_json, f, indent=2, ensure_ascii=False)
    print(f"✅ Final JSON saved: {json_filename}")
    
    # Create a copy of final_json for Excel (without dates)
    excel_json = json.loads(json.dumps(final_json))  # Deep copy
    
    # Remove date fields from Excel JSON
    date_fields = ["From date of the insurance", "To date of the insurance", "Inception date of the insurance"]
    for field in date_fields:
        if field in excel_json:
            del excel_json[field]
    
    # Save Excel (without dates)
    
    output_folder = output_dir  # define your working folder
    os.makedirs(output_folder, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(pdf_path))[0]  
    excel_filename = os.path.join(output_folder, f"{base_name}_org_sheet.xlsx")
    df = pd.json_normalize(excel_json, sep="_")
    df.to_excel(excel_filename, index=False)
    print(f"✅ Final Excel saved (without dates): {excel_filename}")
    
    # Cleanup temporary files
    if os.path.exists(image_path):
        os.remove(image_path)
        print(f"🧹 Cleaned up temporary image: {image_path}")
    
    print(f"\n🎯 Processing completed using method: {final_method}")
    print(f"   Organization: {final_json['Profile']['Organization Name']}")
    print(f"   Address: {final_json['Profile']['Registered Address']['Address Line 1']}")
    print(f"   Pin Code: {final_json['Profile']['Registered Address']['Pin Code']}")
    
    # Print dates for reference (but they won't be in Excel)
    print(f"   From Date: {final_json.get('From date of the insurance', 'Not found')}")
    print(f"   To Date: {final_json.get('To date of the insurance', 'Not found')}")
    print(f"   Inception Date: {final_json.get('Inception date of the insurance', 'Not found')}")
    
    return final_json, final_method


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def run_org(pdf_path,source_dir):
    """Main function using the new processing flow."""
    output_dir = source_dir
    
    final_json, method_used = process_pdf_new_flow(pdf_path, output_dir)
    
    if final_json:
        print(f"\n✅ Processing Completed Successfully!")
        print(f"   Final method: {method_used}")
        print(f"   Output files saved in: {output_dir}")
    else:
        print("\n❌ Processing Failed")

if __name__ == "__main__":
    pdf_path=r""
    source_dir=r""
    run_org(pdf_path,source_dir)