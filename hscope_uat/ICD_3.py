import re
import os
import ast
import json
import difflib
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List
import pandas as pd
from langchain_ollama import ChatOllama
from langchain.schema import HumanMessage
import pymupdf4llm
from hscope_uat.UAT_config import *
from hscope_uat.helper.get_si import load_sum_insured_from_json
# =========================================================
# ✅ PDF to Markdown Conversion (PyMuPDF4LLM)
# =========================================================
def pdf_to_md(pdf_path: str, output_dir: str = None) -> str:
    """Convert PDF to markdown text using PyMuPDF4LLM and save in the specified output directory."""
    try:
        md_text = pymupdf4llm.to_markdown(pdf_path)

        # Define .md output path (either same folder or output_dir)
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            output_md_path = os.path.join(output_dir, Path(pdf_path).stem + ".md")
        else:
            output_md_path = os.path.splitext(pdf_path)[0] + ".md"

        with open(output_md_path, "w", encoding="utf-8") as f:
            f.write(md_text)

        print(f"✅ Converted '{pdf_path}' → '{output_md_path}'")
        return md_text
    except Exception as e:
        print(f"❌ PDF to Markdown failed: {e}")
        return ""


# =========================================================
# ✅ Generate MD Path from PDF (NEW FUNCTION)
# =========================================================
def generate_md_path_from_pdf(pdf_path: str, base_output_dir: str = "/home/ubuntu/rspdf/newtest") -> str:
    """
    Generate the markdown file path based on PDF input name following the pattern:
    base_output_dir / file_base_name / file_base_name / "auto" / file_base_name.md
   
    Args:
        pdf_path: Path to the input PDF file
        base_output_dir: Base directory where output is stored (default: /home/ubuntu/rspdf/newtest)
   
    Returns:
        Full path to the generated markdown file
    """
    pdf_path_obj = Path(pdf_path)
    file_base_name = pdf_path_obj.stem
    temp_output_dir = pdf_path_obj.parent / "mineru_temp"
    temp_output_dir.mkdir(exist_ok=True)
    
    # Construct the MD path following the pattern
    md_path = temp_output_dir / file_base_name / "auto" / f"{file_base_name}.md"
   
    return str(md_path)


# =========================================================
# ✅ Read Markdown File (REPLACES pdf_to_md)
# =========================================================
def read_md_file(md_path: str) -> str:
    """Read markdown file and return its content."""
    try:
        with open(md_path, "r", encoding="utf-8") as f:
            md_text = f.read()
        print(f"✅ Read markdown from '{md_path}'")
        return md_text
    except FileNotFoundError:
        print(f"❌ Markdown file not found: {md_path}")
        return ""
    except Exception as e:
        print(f"❌ Failed to read markdown: {e}")
        return ""


# =========================================================
# Normalize IDs (UPDATED)
# =========================================================
def normalize_id(eid: str) -> str:
    """Convert endorsement IDs to canonical form."""
    m = re.match(r"^(\d+)(?:\(?([a-zivx]+)\)?)?$", str(eid).strip(), re.IGNORECASE)
    if not m:
        return eid
    num, suffix = m.groups()
    if suffix:
        return f"{num}({suffix.lower()})"
    return num


# =========================================================
# Clean endorsement content (NEW FUNCTION)
# =========================================================
def clean_endorsement_content(content: str) -> str:
    """
    Remove member data tables and other redundant information from endorsement content.
    """
    # Remove lines that look like member data tables (with age ranges, etc.)
    lines = content.split('\n')
    cleaned_lines = []
    
    # Skip lines that contain typical table headers or data rows
    skip_patterns = [
        r'^\s*(Name|Age|Gender|Relationship)\s*:',
        r'^\s*\d+\s+years',
        r'^\s*\d+\s*-\s*\d+\s*years',
        r'^\s*Age\s*:\s*\d+',
        r'^\s*Table\s+\d+',
        r'^\s*Member\s+\d+',
        r'^\s*SI\.\s*No',
        r'^\s*\|\s*Name\s*\|\s*Age\s*\|\s*Gender',
        r'^\s*[-=]+\s*$',  # Table separators
        r'^\s*\(?in\s+years\)?',
        r'^\s*Above\s+\d+\s+years',
    ]
    
    # Also skip if line is just numbers or very short with numbers
    for line in lines:
        skip = False
        # Skip if line matches any skip pattern
        for pattern in skip_patterns:
            if re.search(pattern, line, re.IGNORECASE):
                skip = True
                break
        
        # Skip lines that are just numbers or very short
        if re.match(r'^\s*\d+\s*$', line) or (len(line.strip()) < 5 and re.search(r'\d', line)):
            skip = True
        
        if not skip:
            cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)


# =========================================================
# Extract Endorsements (UPDATED with cleaning)
# =========================================================
def extract_endorsements(md_text: str, debug: bool = False):
    lines = md_text.splitlines()

    # More flexible pattern to catch various endorsement formats
    heading_pattern = re.compile(
        r"(?:^|\s)(?:\*{0,2}\s*)?(?:Endorsement|Endt\.?|ENDORSEMENT|ENDT\.?)\.?\s*(?:No\.?|NUMBER|#)?\s*"
        r"(\d+)"
        r"(?:\s*[\(\[]\s*([A-Za-z0-9ivxIVX]{1,3})\s*[\)\]]"  # (a) or [a]
        r"|-\s*([A-Za-z0-9ivxIVX]{1,3})"  # -a
        r"|([A-Za-z]{1,3})(?=\s|[-–:]|$))?",  # plain suffix
        re.IGNORECASE,
    )

    # Remove page headers/footers & junk lines
    junk_patterns = [
        re.compile(r"^group health policy", re.IGNORECASE),
        re.compile(r"^uin[:\s]", re.IGNORECASE),
        re.compile(r"^irda", re.IGNORECASE),
        re.compile(r"^policy number", re.IGNORECASE),
        re.compile(r"^name of the insured", re.IGNORECASE),
        re.compile(r"^period of insurance", re.IGNORECASE),
        re.compile(r"endorsements attached", re.IGNORECASE),
        re.compile(r"Group Health Policy – Endorsements", re.IGNORECASE),
        re.compile(r"^Page\s+\*\*\d+\*\*\s+of\s+\*\*\d+\*\*$", re.IGNORECASE),
        re.compile(r"^\*\*Policy Number.*\*\*$", re.IGNORECASE),
        re.compile(r"^\*\*Name of the Insured.*\*\*$", re.IGNORECASE),
        re.compile(r"^\*\*Period of Insurance.*\*\*$", re.IGNORECASE),
        re.compile(r"^//\s*\d+\s*//$", re.IGNORECASE),
        re.compile(r"^royal sundaram.*", re.IGNORECASE),
        re.compile(r"^regd office.*", re.IGNORECASE),
        re.compile(r"^corporate office.*", re.IGNORECASE),
        re.compile(r"^email[:\s].*", re.IGNORECASE),
        re.compile(r"^website[:\s].*", re.IGNORECASE),
        re.compile(r"^ph[:\s].*", re.IGNORECASE),
        re.compile(r"^.*irda regn.*", re.IGNORECASE),
        re.compile(r"^.*cin[-:\s].*", re.IGNORECASE),
        re.compile(r".*\bchennai\s*\d{3}\s*\d{3}.*", re.IGNORECASE),
    ]

    endorsements = OrderedDict()
    current_id, buffer = None, []

    def clean_buffer(buf):
        """Remove unwanted header/footer lines."""
        return [line for line in buf if not any(p.search(line) for p in junk_patterns)]

    for i, line in enumerate(lines, 1):
        m = heading_pattern.search(line)
        if m:
            num = m.group(1)
            suffix = m.group(2) or m.group(3) or m.group(4)
            canonical_id = num
            if suffix and len(suffix) <= 3:
                canonical_id = f"{num}({suffix.lower()})"

            if debug:
                print(f"LINE {i}: ✅ {line}")
                print(f"       -> canonical id: {canonical_id}")

            if current_id and buffer:
                # Clean the buffer content before storing
                clean_content = clean_endorsement_content("\n".join(clean_buffer(buffer)).strip())
                endorsements[current_id] = clean_content

            current_id = canonical_id
            buffer = [line]

        elif current_id:
            buffer.append(line)

    if current_id and buffer:
        clean_content = clean_endorsement_content("\n".join(clean_buffer(buffer)).strip())
        endorsements[current_id] = clean_content

    # Print endorsements with cleaner output
    print("\n" + "="*80)
    print("🔍 EXTRACTED ENDORSEMENTS (CLEANED):")
    print("="*80)
    for eid, content in endorsements.items():
        print(f"\n📋 Endorsement ID: {eid}")
        
        # Get the first few non-empty lines for preview
        lines = content.split('\n')
        preview_lines = []
        for line in lines:
            line = line.strip()
            if line and len(line) > 10:  # Only show meaningful lines
                preview_lines.append(line)
                if len(preview_lines) >= 5:  # Show first 5 meaningful lines
                    break
        
        if preview_lines:
            print(f"Preview:")
            for i, line in enumerate(preview_lines, 1):
                print(f"  {i}. {line[:100]}{'...' if len(line) > 100 else ''}")
        
        print(f"Total length: {len(content)} characters")
    print("="*80 + "\n")
    
    return endorsements


# =========================================================
# Get Endorsements Content (MODIFIED)
# =========================================================
def get_endorsements_content(pdf_path, desired_endorsements):
    md_path = generate_md_path_from_pdf(pdf_path)
    md_text = read_md_file(md_path)
    
    if not md_text:
        print(f"❌ No markdown content available from {md_path}")
        return ""
    
    endorsements = extract_endorsements(md_text)
    desired_endorsements = [normalize_id(eid) for eid in desired_endorsements]

    matching_endorsements = {
        eid: endorsements[eid]
        for eid in desired_endorsements
        if eid in endorsements
    }

    matched_endorsements = list(matching_endorsements.values())

    endorsements_context = ""
    for endnt in matched_endorsements:
        endorsements_context += endnt + "\n\n"

    return endorsements_context.strip()


# =========================================================
# Extract Special Conditions (UPDATED)
# =========================================================
def extract_special_conditions(md_text: str) -> Dict[str, List[str]]:
    """
    Extract blocks starting from bold markdown headers that contain keywords:
    'special' or 'other' + ('condition'/'conditions'/'clause'/'clauses'/'coverage').
    Captures from that header until the next bold header.
    """
    header_pattern = re.compile(r"\*\*(.+?)\*\*", re.IGNORECASE)
    matches = list(header_pattern.finditer(md_text))
    results = []

    for i, match in enumerate(matches):
        header_text = match.group(1).strip().lower()

        if (("special" in header_text or "other" in header_text)
            and any(word in header_text for word in ["condition", "conditions", "clause", "clauses", "coverage", "Endorsements", " Endorsement", "Note", "Notes"])):
            
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(md_text)
            block = md_text[start:end].strip()
            block_cleaned = re.sub(r"\*\*(.+?)\*\*", r"\1", block)
            results.append(block_cleaned)

    endorsements_context = ""
    for endnt in results:
        endorsements_context += endnt + "\n\n"

    return {"special condition": endorsements_context.strip()}


# =========================================================
# Overlay Default if No Special Condition
# =========================================================
def overaly_specail_condtion(md_text):
    endnt_context = extract_special_conditions(md_text=md_text)
    if endnt_context['special condition'] == "":
        spl_cond_context = (
            "There is no special condition provided, so you can consider "
            "no context from the special conditions section."
        )
        endnt_context['special condition'] = spl_cond_context
    return endnt_context


# =========================================================
# Get all endorsements (NEW FUNCTION)
# =========================================================
def get_all_endorsements(md_path):
    """
    Extract all endorsements once and return them with special conditions.
    
    Args:
        md_path: Path to the markdown file
        
    Returns:
        tuple: (endorsements_dict, special_conditions_dict)
    """
    md_text = read_md_file(md_path)
    
    if not md_text:
        print(f"❌ No markdown content available from {md_path}")
        return {}, {}
    
    endorsements = extract_endorsements(md_text)
    special_conditions_context = overaly_specail_condtion(md_text)
    return endorsements, special_conditions_context


# =========================================================
# Get specific endorsements (NEW FUNCTION)
# =========================================================
def get_specific_endorsements(endorsements_dict, desired_ids):
    """
    Get specific endorsements from the already extracted endorsements dictionary.
    
    Args:
        endorsements_dict: Dictionary of all extracted endorsements
        desired_ids: List of endorsement IDs to retrieve
        
    Returns:
        str: Combined text of matching endorsements
    """
    desired_endorsements = [normalize_id(eid) for eid in desired_ids]
    matching_endorsements = {
        eid: endorsements_dict[eid]
        for eid in desired_endorsements
        if eid in endorsements_dict
    }

    # Debug output
    print(f"\n🎯 Requested endorsements: {desired_endorsements}")
    print(f"✅ Found endorsements: {list(matching_endorsements.keys())}")
    if not matching_endorsements:
        print("⚠️  No matching endorsements found!")

    # Combine matching endorsements into a single string
    endorsements_context = ""
    for endnt in matching_endorsements.values():
        endorsements_context += endnt + "\n\n"
    
    return endorsements_context.strip()


def extract_conditions_from_5a(result_text: str) -> List[str]:
    """
    Extract the list of medical conditions mentioned in Endorsement 5a
    (Removal of Limitation of Benefits).
    Returns a list of condition names found in that section.
    """
    conditions_5a = []
    
    # Find the Endt. 5a section
    lines = result_text.split('\n')
    in_5a_section = False
    section_text = []
    
    for line in lines:
        # Check if we've entered the 5a section
        if re.search(r'endt\.?\s*(?:no\.?)?\s*5\s*(?:\(?\s*a\s*\)?)', line, re.IGNORECASE):
            in_5a_section = True
            section_text.append(line)
            continue
        
        # Check if we've entered a different endorsement section (end of 5a)
        if in_5a_section and re.search(r'endt\.?\s*(?:no\.?)?\s*\d+', line, re.IGNORECASE):
            if not re.search(r'endt\.?\s*(?:no\.?)?\s*5\s*(?:\(?\s*a\s*\)?)', line, re.IGNORECASE):
                break
        
        if in_5a_section:
            section_text.append(line)
    
    # Join the section text
    full_5a_text = '\n'.join(section_text).lower()
    
    # Common condition patterns to look for
    all_possible_conditions = [
        "arthritis", "gout", "rheumatism", "spondylosis", "spondylitis", "ivdp",
        "benign prostatic hypertrophy", "cataract", "congenital internal anamoly",
        "dub", "fibroids", "prolapse uterus", "endometriosis",
        "fissure", "fistula", "haemorrhoid", "gastric and duodenal ulcers",
        "hernia", "hydrocele", "hysterectomy",
        "lumps", "cysts", "nodules", "polyps", "internal tumours",
        "maternity", "caesarean", "mental and behavioural disorders",
        "osteoarthritis", "osteoporosis",
        "sinusitis", "dns", "tympanoplasty", "csom",
        "stones in biliary and urinary systems",
        "surgery on tonsils", "surgery on adenoids", "varicose veins",
        "psychiatric ailment", "mental", "tonsillectomy", "septoplasty", "adenoidectomy"
    ]
    
    # Check which conditions are mentioned in the 5a section
    for condition in all_possible_conditions:
        # Create flexible pattern to match condition
        pattern = r'\b' + re.escape(condition.lower()) + r'\b'
        if re.search(pattern, full_5a_text):
            conditions_5a.append(condition)
    
    return conditions_5a


# =========================================================
# Combined Extractor for Maternity (USES PyMuPDF4LLM)
# =========================================================
def get_endorsements_and_spl_cond(pdf_path, desired_endorsements=None, output_dir=None):
    """
    Extract endorsements using PyMuPDF4LLM conversion.
    Used specifically for maternity/caesarean extraction (Endt. 11b).
    
    Args:
        pdf_path: Path to PDF file
        desired_endorsements: List of endorsement IDs to extract
        output_dir: Directory to save output markdown
        
    Returns:
        str: Combined endorsement text with special conditions
    """
    if desired_endorsements is None:
        desired_endorsements = []

    # ✅ Convert PDF to markdown using PyMuPDF4LLM
    md_text = pdf_to_md(pdf_path, output_dir=output_dir)
    
    if not md_text:
        print(f"❌ No markdown content available from {pdf_path}")
        return ""

    # Extract endorsements and special conditions from markdown
    endorsements = extract_endorsements(md_text)
    desired_endorsements = [normalize_id(eid) for eid in desired_endorsements]

    matching_endorsements = {
        eid: endorsements[eid]
        for eid in desired_endorsements
        if eid in endorsements
    }

    endnt_context = "\n\n".join(matching_endorsements.values())
    spl_cond = overaly_specail_condtion(md_text)

    # Combine both into one markdown output
    if isinstance(spl_cond, dict) and len(spl_cond) > 0:
        key, value = list(spl_cond.items())[0]
        combined_text = f"{endnt_context.strip()}\n\n**{key}:**\n{value.strip()}"
    else:
        combined_text = endnt_context.strip()

    # ✅ Save combined markdown output alongside Excel/JSON
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        combined_md_path = os.path.join(output_dir, Path(pdf_path).stem + "_Maternity_Extracted.md")
        with open(combined_md_path, "w", encoding="utf-8") as f:
            f.write(combined_text)
        print(f"✅ Maternity extraction saved → {combined_md_path}")

    return combined_text


# -----------------------------
# ✅ Ollama LLM Query Utility
# -----------------------------
def query_field(prompt, model="gpt-oss:20b", temperature=0):
    llm = ChatOllama(model=model, temperature=temperature)
    response = llm.invoke([HumanMessage(content=prompt)])
    return extract_json_from_text(response.content or "")


def extract_json_from_text(text):
    if not text:
        return None
    cleaned = re.sub(r'```(?:json)?\n', '', text, flags=re.IGNORECASE).replace('```', '')
    match = re.search(r'(\{.*\}|\[.*\])', cleaned, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except Exception:
        try:
            return ast.literal_eval(match.group(0))
        except Exception:
            return None


# -----------------------------
# ✅ Universal Condition Extractor
# -----------------------------
def extract_medical_condition(result_text: str, condition: str, endt_no: str) -> Dict:
    """
    Robust extractor for medical condition limits.
    Handles 'Nil Capping', numeric extraction, absence, grouped conditions, and fuzzy names.
    """
    clean_text = re.sub(r"<br\s*/?>", " ", result_text, flags=re.I)
    clean_text = re.sub(r"\s+", " ", clean_text)
    lowered_text = clean_text.lower()

    # --- Fuzzy match ---
    normalized_condition = re.sub(r"[^a-z]", "", condition.lower())
    words_in_text = list(set(re.findall(r"[A-Za-z]+", lowered_text)))
    possible_matches = [w for w in words_in_text if abs(len(w) - len(normalized_condition)) < 6]
    close_match = difflib.get_close_matches(
        normalized_condition,
        [re.sub(r"[^a-z]", "", w) for w in possible_matches],
        n=1,
        cutoff=0.8
    )
    matched_term = condition
    if close_match:
        for w in words_in_text:
            if re.sub(r"[^a-z]", "", w) == close_match[0]:
                matched_term = w
                break

    # --- Check if condition exists ---
    if not re.search(rf"\b{re.escape(condition)}\b", lowered_text, flags=re.I) and \
       not re.search(rf"\b{re.escape(matched_term)}\b", lowered_text, flags=re.I):
        related_word = condition.split()[-1] if " " in condition else condition
        if not re.search(rf"{related_word}\s*/", lowered_text, flags=re.I) and \
           not re.search(rf"/\s*{related_word}", lowered_text, flags=re.I):
            return {f"{condition}Percentage": "", f"{condition}Maximum": ""}

    # --- Check explicitly for Nil Capping first ---
    if re.search(rf"{matched_term}.*nil\s*capping", lowered_text, flags=re.I):
        return {f"{condition}Percentage": "100", f"{condition}Maximum": "sum_insured"}

    # --- Build the strict LLM prompt ---
    prompt = f"""
You are given extracted insurance endorsement text (not full markdown).
Focus ONLY on the section corresponding to **Endt. No. {endt_no}** (may appear as "Endorsement {endt_no}", etc.).
The text contains limits for diseases/conditions like:
|Condition|Limit|
|Cataract|20% of the Sum Insured subject to a maximum of INR 70000/-|

Now, extract the values ONLY for **{condition}** (including minor OCR spelling variants like "{matched_term}" or concatenated forms).
If {condition} is not found exactly or as part of a clear combined entry (e.g., "Hernia/ Hydrocele"), return empty strings.

---
### Extraction Rules:
1. If line mentions **"Nil Capping"** or any variant (`nil capping`, `nillcaping`, etc.), both values = "Nil Capping".
2. Otherwise, extract:
   - The numeric percentage before "% of the Sum Insured".
   - The numeric maximum after "maximum of" or "subject to a maximum of".
3. If no percentage is found, but a fixed amount is mentioned like "Rs. X /- per Eye", "Rs. X /- @ each Eye", "Rs. X,XXX/-" or similar direct limits associated with the condition, extract the numeric value as the maximum, and leave percentage empty.
4. If {condition} is not explicitly listed, or context is unclear — return both fields empty.
5. Do NOT infer or assume data.
6. Output only plain numbers (no %, Rs, INR, commas, etc.) unless Nil Capping applies.
7. If grouped conditions appear (e.g. "Hernia/ Hydrocele"), extract values for all conditions in that row.

---
### Output Format (strict JSON)
{{
  "{condition}Percentage": "<number or 'Nil Capping' or empty>",
  "{condition}Maximum": "<number or 'Nil Capping' or empty>"
}}

Text to extract from:
\"\"\"{result_text}\"\"\"
"""
    response = query_field(prompt)
    pct = response.get(f"{condition}Percentage", "").strip()
    maxv = response.get(f"{condition}Maximum", "").strip()

    # --- Validation: enforce Nil Capping if found in text ---
    if re.search(rf"{matched_term}.*nil\s*capping", lowered_text, flags=re.I):
        pct, maxv = "100", "sum_insured"

    # --- Fallback regex only if not Nil Capping ---
    if pct == "" and maxv == "":
        pattern = rf"{matched_term}.*?(\d+(?:\.\d+)?)\s*%[^0-9]+?(\d+[,.]?\d*)"
        match = re.search(pattern, lowered_text, flags=re.I)
        if match:
            pct, maxv = match.group(1), re.sub(r"[^\d.]", "", match.group(2))

    # --- Final safety ---
    if pct.lower() == "nil capping":
        pct, maxv = "100", "sum_insured"
    return {f"{condition}Percentage": pct, f"{condition}Maximum": maxv}


# ───────────────────────────────
# Calculation functions
# ───────────────────────────────
def calculate_final_limit_for_conditions(extracted_data: dict, sum_insured, conditions_in_5a: List[str] = None):
    """
    Calculate final payable limits for multiple conditions based on extracted % and max values.
    Properly handles 'Nil Capping' cases by returning empty amounts and percentages.
    
    NEW RULE: If a condition is in Endt. 5a (Removal of Limitation of Benefits),
    force Percentage = 100% and Amount = sum_insured (ignoring extracted values).
    """
    if conditions_in_5a is None:
        conditions_in_5a = []
    
    # Normalize conditions_in_5a for case-insensitive matching
    conditions_in_5a_lower = [c.lower().strip() for c in conditions_in_5a]
    
    results = {}
    conditions = set(
        key.replace("Percentage", "").replace("Maximum", "")
        for key in extracted_data.keys()
        if any(x in key for x in ["Percentage", "Maximum"])
    )

    for condition in conditions:
        # ---- NEW RULE: Check if condition is in 5a ----
        if condition.lower().strip() in conditions_in_5a_lower:
            results[f"{condition}FinalLimit"] = sum_insured
            results[f"{condition}FinalPercentage"] = 100.0
            print(f"✅ Condition '{condition}' found in Endt. 5a → Forcing 100% coverage: {sum_insured}")
            continue
        
        pct_key = f"{condition}Percentage"
        max_key = f"{condition}Maximum"
        percent_str = (extracted_data.get(pct_key) or "").strip().lower()
        max_str = (extracted_data.get(max_key) or "").strip().lower()

        # ---- Case 1: Nil Capping ----
        if percent_str.lower() == "nil capping" or max_str.lower() == "nil capping":
            results[f"{condition}FinalLimit"] = sum_insured
            results[f"{condition}FinalPercentage"] = 100.0
            print(f"✅ Nil Capping detected for '{condition}' → Forcing 100% coverage: {sum_insured}")
            continue

        # ---- Case 2: Both empty ----
        if not percent_str and not max_str:
            results[f"{condition}FinalLimit"] = ""
            continue

        # ---- Case 3: Numeric calculation ----
        try:
            if percent_str:
                percentage = float(percent_str)
                calculated = (percentage / 100) * sum_insured
                if max_str:
                    if max_str == "sum_insured":
                        maximum = sum_insured
                    else:
                        maximum = float(max_str)
                    final_value = min(calculated, maximum)
                else:
                    final_value = calculated
            else:
                if max_str == "sum_insured":
                    final_value = sum_insured
                else:
                    final_value = float(max_str)

            results[f"{condition}FinalLimit"] = int(final_value)
        except ValueError:
            results[f"{condition}FinalLimit"] = ""

    return results


def calculate_final_percentage_for_conditions(final_limits: dict, sum_insured):
    """Calculate final percentages, using pre-calculated 100% if already set."""
    if not final_limits:
        return {}
    results = {}
    for key, value in final_limits.items():
        cond = key.replace("FinalLimit", "")
        
        existing_pct_key = f"{cond}FinalPercentage"
        if existing_pct_key in final_limits:
            results[existing_pct_key] = final_limits[existing_pct_key]
        elif isinstance(value, (int, float)):
            results[existing_pct_key] = round((value / sum_insured) * 100, 2)
        else:
            results[existing_pct_key] = ""
    return results


def update_medical_conditions(final_schema, final_limits, sum_insured):
    """Update medical conditions in schema, handling pre-set percentages from 5a rule."""
    final_percentages = {}
    
    for key, value in final_limits.items():
        if "FinalPercentage" in key:
            final_percentages[key] = value
    
    regular_limits = {k: v for k, v in final_limits.items() if "FinalPercentage" not in k}
    calculated_percentages = calculate_final_percentage_for_conditions(regular_limits, sum_insured)
    final_percentages.update(calculated_percentages)
    
    updates = {}
    for key_limit, amount_value in final_limits.items():
        if "FinalPercentage" in key_limit:
            continue
            
        cond = key_limit.replace("FinalLimit", "")
        pct_key = f"{cond}FinalPercentage"
        updates[cond] = {
            f"{cond}_amount": amount_value,
            f"{cond}_percentage": final_percentages.get(pct_key, "")
        }
    
    if "medicalConditions" not in final_schema:
        final_schema["medicalConditions"] = {}
    final_schema["medicalConditions"].update(updates)
    return final_schema


def update_maternity_and_caesarean(final_schema, maternity_json, sum_insured):
    def compute_percentage(amount):
        try:
            if isinstance(amount, (int, float)):
                return round((amount / sum_insured) * 100, 2)
            elif isinstance(amount, str) and amount.strip().isdigit():
                return round((float(amount.strip()) / sum_insured) * 100, 2)
            else:
                return ""
        except Exception:
            return ""

    normal_amount = maternity_json.get("NormalDeliveryLimit", "")
    caesarean_amount = maternity_json.get("CaesareanDeliveryLimit", "")
    normal_percentage = compute_percentage(normal_amount)
    caesarean_percentage = compute_percentage(caesarean_amount)

    final_schema.setdefault("medicalConditions", {}).setdefault("Maternity", {})
    final_schema.setdefault("medicalConditions", {}).setdefault("Caesarean", {})

    final_schema["medicalConditions"]["Maternity"].update({
        "Maternity_amount": normal_amount,
        "Maternity_percentage": normal_percentage
    })
    final_schema["medicalConditions"]["Caesarean"].update({
        "Caesarean_amount": caesarean_amount,
        "Caesarean_percentage": caesarean_percentage
    })

    return final_schema


# -----------------------------
# ✅ Medical Schema Template
# -----------------------------
MED_SCHEMA = {
    "medicalConditions": {
        "Arthritis": {"Arthritis_amount": "", "Arthritis_percentage": ""},
        "gout": {"gout_amount": "", "gout_percentage": ""},
        "rheumatism": {"rheumatism_amount": "", "rheumatism_percentage": ""},
        "spondylosis": {"spondylosis_amount": "", "spondylosis_percentage": ""},
        "spondylitis": {"spondylitis_amount": "", "spondylitis_percentage": ""},
        "IVDP": {"IVDP_amount": "", "IVDP_percentage": ""},
        "Benign Prostatic Hypertrophy": {"Benign Prostatic Hypertrophy_amount": "", "Benign Prostatic Hypertrophy_percentage": ""},
        "Cataract": {"Cataract_amount": "", "Cataract_percentage": ""},
        "Congenital Internal Anamoly": {"Congenital Internal Anamoly_amount": "", "Congenital Internal Anamoly_percentage": ""},
        "DUB": {"DUB_amount": "", "DUB_percentage": ""},
        "Fibroids": {"Fibroids_amount": "", "Fibroids_percentage": ""},
        "Prolapse uterus": {"Prolapse uterus_amount": "", "Prolapse uterus_percentage": ""},
        "Endometriosis": {"Endometriosis_amount": "", "Endometriosis_percentage": ""},
        "Fissure": {"Fissure_amount": "", "Fissure_percentage": ""},
        "Fistula": {"Fistula_amount": "", "Fistula_percentage": ""},
        "Haemorrhoid": {"Haemorrhoid_amount": "", "Haemorrhoid_percentage": ""},
        "Gastric and Duodenal Ulcers": {"Gastric and Duodenal Ulcers_amount": "", "Gastric and Duodenal Ulcers_percentage": ""},
        "Hernia": {"Hernia_amount": "", "Hernia_percentage": ""},
        "Hydrocele": {"Hydrocele_amount": "", "Hydrocele_percentage": ""},
        "Hysterectomy": {"Hysterectomy_amount": "", "Hysterectomy_percentage": ""},
        "Lumps": {"Lumps_amount": "", "Lumps_percentage": ""},
        "Cysts": {"Cysts_amount": "", "Cysts_percentage": ""},
        "Nodules": {"Nodules_amount": "", "Nodules_percentage": ""},
        "Polyps": {"Polyps_amount": "", "Polyps_percentage": ""},
        "Internal Tumours": {"Internal Tumours_amount": "", "Internal Tumours_percentage": ""},
        "Maternity": {"Maternity_amount": "", "Maternity_percentage": ""},
        "Caesarean": {"Caesarean_amount": "", "Caesarean_percentage": ""},
        "Mental and behavioural disorders": {"Mental and behavioural disorders_amount": "", "Mental and behavioural disorders_percentage": ""},
        "Osteoarthritis": {"Osteoarthritis_amount": "", "Osteoarthritis_percentage": ""},
        "Osteoporosis": {"Osteoporosis_amount": "", "Osteoporosis_percentage": ""},
        "Sinusitis": {"Sinusitis_amount": "", "Sinusitis_percentage": ""},
        "DNS": {"DNS_amount": "", "DNS_percentage": ""},
        "Tympanoplasty": {"Tympanoplasty_amount": "", "Tympanoplasty_percentage": ""},
        "CSOM": {"CSOM_amount": "", "CSOM_percentage": ""},
        "Stones in biliary and urinary systems": {"Stones in biliary and urinary systems_amount": "", "Stones in biliary and urinary systems_percentage": ""},
        "Surgery on Tonsils": {"Surgery on Tonsils_amount": "", "Surgery on Tonsils_percentage": ""},
        "Surgery on Adenoids": {"Surgery on Adenoids_amount": "", "Surgery on Adenoids_percentage": ""},
        "Varicose veins": {"Varicose veins_amount": "", "Varicose veins_percentage": ""}
    }
}


# -----------------------------
# ✅ Main Processor (MODIFIED)
# -----------------------------
def process_pdf_to_medschema(pdf_path: str, output_dir: str) -> dict:
    """
    Main processor with NEW RULE for Endt. 5a conditions.
    Now reads from pre-generated markdown files instead of converting PDFs.
    
    Args:
        pdf_path: Path to the PDF file
        output_dir: Directory to save outputs
        sum_insured: Base sum insured amount (default 100000)
    """
    # Check if markdown file
    
    md_path = generate_md_path_from_pdf(pdf_path)
    #############
    si_list = load_sum_insured_from_json(pdf_path)

# If Sum Insured is missing or empty → return ICD sheet directly
    if not si_list:
        print(f"❌ Sum Insured not found for {pdf_path}. Skipping SI-based logic.")

        master_icd = pd.read_excel(ICD_3_file, header=1)
        os.makedirs(output_dir, exist_ok=True)

        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        excel_filename = os.path.join(
            output_dir, f"{base_name}_ICD_3_sheet.xlsx"
        )

        master_icd.to_excel(excel_filename, index=False)

        return  # ⬅️ important: stop further processing

    # If SI exists → safe to calculate max
    sum_insured = max(si_list)






    #############
    if not os.path.exists(md_path):
        print(f"❌ Markdown file not found at: {md_path}")
        print("⚠️  Please ensure the markdown file is generated before running this script.")
        return {}
    
    # Desired endorsements for main extraction (5a and 5(ii))
    desired_endorsements_main = ['5a', '5(ii)', '5(i)']
    
    # Desired endorsement for maternity (11b) - uses PyMuPDF4LLM
    desired_endorsements_maternity = ['11(b)']

    # ✅ Extract all endorsements once from pre-generated markdown
    print("📊 Extracting all endorsements from pre-generated markdown...")
    all_endorsements, special_conditions = get_all_endorsements(md_path)
    
    # Get specific endorsements for medical conditions (5a and 5(ii))
    endorsements_text = get_specific_endorsements(all_endorsements, desired_endorsements_main)
    
    # Combine endorsements with special conditions
    if isinstance(special_conditions, dict) and len(special_conditions) > 0:
        key, value = list(special_conditions.items())[0]
        result_text = f"{endorsements_text.strip()}\n\n**{key}:**\n{value.strip()}"
    else:
        result_text = endorsements_text.strip()
    
    # Save the extracted content
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    combined_md_path = os.path.join(output_dir, Path(pdf_path).stem + "_Extracted.md")
    with open(combined_md_path, "w", encoding="utf-8") as f:
        f.write(result_text)
    print(f"✅ Combined extracted markdown saved → {combined_md_path}")
    
    print("\n==================== Extracted Content (5a & 5(ii)) ====================\n")
    print(result_text)

    # ✅ NEW: Extract conditions from Endt. 5a
    conditions_in_5a = extract_conditions_from_5a(result_text)
    print(f"\n📋 Conditions found in Endt. 5a (Removal of Limitation): {conditions_in_5a}\n")
    
    # ============================================================
    # Extract Maternity/Caesarean using PyMuPDF4LLM (Endt. 11b)
    # ============================================================
    print("\n📊 Extracting maternity data using PyMuPDF4LLM conversion...")
    maternity_text = get_endorsements_and_spl_cond(pdf_path, desired_endorsements_maternity, output_dir=output_dir)
    print("\n==================== Extracted Maternity Content (11b) ====================\n")
    print(maternity_text)

    # ✅ Extract multiple medical conditions
    conditions = ["Arthritis", "gout", "rheumatism", "spondylosis", "spondylitis", "IVDP", 
                  "Benign Prostatic Hypertrophy", "Cataract", "Congenital Internal Anamoly", 
                  "DUB", "Fibroids", "Prolapse uterus", "Endometriosis", "Fissure", "Fistula", 
                  "Haemorrhoid", "Gastric and Duodenal Ulcers", "Hernia", "Hydrocele", 
                  "Hysterectomy", "Lumps", "Cysts", "Nodules", "Polyps", "Internal Tumours", 
                  "Maternity", "Caesarean", "Mental and behavioural disorders", "Osteoarthritis", 
                  "Osteoporosis", "Sinusitis", "DNS", "Tympanoplasty", "CSOM", 
                  "Stones in biliary and urinary systems", "Surgery on Tonsils", 
                  "Surgery on Adenoids", "Varicose veins", "Psychiatric ailment", "Mental", 
                  "Tonsillectomy", "Septoplasty", "Adenoidectomy"]
    
    extracted_data = {}

    for cond in conditions:
        extracted = extract_medical_condition(result_text, cond, "5(ii)")
        extracted_data.update(extracted)
    
    # ---------- Maternity Prompt ----------
    maternity_prompt = f"""
You are given extracted insurance endorsement text (not full markdown). 
Focus ONLY on the section corresponding to **Endt. No. 11(b)**, which may appear as 
"Endorsement 11(b)", "Endt No 11(b)", "Endt. 11 (b)", etc., in any formatting.

This section always covers **Maternity Expenses / Maternity Treatment Charges Benefit** and contains 
the limits for **Normal** and **Caesarean** deliveries. The relevant sentence typically looks like:
"The maximum benefit under this Benefit is limited to Rs. 40000/- for Normal 100000/- for Caesarean per Family."

### Extraction Goals
1. Extract the **limit for Normal delivery** — e.g., from "Rs. 40000/- for Normal" → `40000`
2. Extract the **limit for Caesarean delivery** — e.g., from "100000/- for Caesarean" → `100000`

### Extraction & Formatting Rules
- Output **only numeric values** (no Rs, INR, commas, dots, or slashes).
- The words "Normal" and "Caesarean" may appear in any order, on separate lines, or with punctuation; still extract both.
- Ignore any "per Family", "subject to", or descriptive text.
- Ignore other amounts (premium, IRDA, UIN, etc.).
- Treat line breaks, `<br>`, and markdown tables as continuous text.

### Output Format (STRICT JSON only, no extra commentary)
{{
  "NormalDeliveryLimit": "<numeric amount only>",
  "CaesareanDeliveryLimit": "<numeric amount only>"
}}

Text to extract from:
\"\"\"{maternity_text}\"\"\"
"""

    maternity_json = query_field(maternity_prompt)

    # ✅ Step 1: Compute final rupee limits WITH 5a RULE
    final_limits = calculate_final_limit_for_conditions(
        extracted_data, 
        sum_insured=sum_insured,
        conditions_in_5a=conditions_in_5a
    )

    # ✅ Merge into schema
    final_schema = json.loads(json.dumps(MED_SCHEMA))
    final_schema = update_medical_conditions(final_schema, final_limits, sum_insured=sum_insured)

    # Add maternity values
    final_schema = update_maternity_and_caesarean(final_schema, maternity_json, sum_insured=sum_insured)

    # ---------- Save Outputs ----------
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    base = Path(pdf_path).stem
    json_path = os.path.join(output_dir, f"{base}_MedicalSchema.json")
    excel_path = os.path.join(output_dir, f"{base}_MedicalSchema.xlsx")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(final_schema, f, indent=2, ensure_ascii=False)

    pd.json_normalize(final_schema, sep="_").to_excel(excel_path, index=False)

    print(f"✅ JSON saved → {json_path}")
    print(f"✅ Excel saved → {excel_path}")

    # =========================================================
    # ✅ Generate NEW_MED_SCHEMA
    # =========================================================
    NEW_MED_SCHEMA = {
        "newmedicalConditions": {
            "Arthritis/gout/rheumatism/spondylosis/spondylitis/IVDP": {"amount": "", "percentage": ""},
            "Benign Prostatic Hypertrophy": {"amount": "", "percentage": ""},
            "Cataract": {"amount": "", "percentage": ""},
            "Congenital Internal Anamoly": {"amount": "", "percentage": ""},
            "DUB/Fibroids/Prolapse uterus/Endometriosis": {"amount": "", "percentage": ""},
            "Fissure/Fistula/Haemorrhoid": {"amount": "", "percentage": ""},
            "Gastric and Duodenal Ulcers": {"amount": "", "percentage": ""},
            "Hernia/Hydrocele": {"amount": "", "percentage": ""},
            "Hysterectomy": {"amount": "", "percentage": ""},
            "Lumps/Cysts/Nodules/Polyps/Internal Tumours": {"amount": "", "percentage": ""},
            "Maternity": {"amount": "", "percentage": ""},
            "Maternity/Caesarean": {"amount": "", "percentage": ""},
            "Mental and behavioural disorders": {"amount": "", "percentage": ""},
            "Osteoarthritis/Osteoporosis": {"amount": "", "percentage": ""},
            "Sinusitis/DNS/Tympanoplasty/CSOM": {"amount": "", "percentage": ""},
            "Stones in biliary and urinary systems": {"amount": "", "percentage": ""},
            "Surgery on Tonsils/Adenoids": {"amount": "", "percentage": ""},
            "Varicose veins": {"amount": "", "percentage": ""}
        }
    }

    def get_condition_data(cond_name: str):
        """Fetch amount & percentage safely from MED_SCHEMA (case-insensitive)."""
        medconds = final_schema.get("medicalConditions", {})
        normalized_lookup = cond_name.strip().lower()
        
        for k, v in medconds.items():
            if k.strip().lower() == normalized_lookup:
                amt_key = f"{k}_amount"
                pct_key = f"{k}_percentage"
                amt = v.get(amt_key, "")
                pct = v.get(pct_key, "")
                return amt, pct
        return "", ""

    print("\n==================== NEW_MED_SCHEMA Mapping Summary ====================")
    for combo_key, combo_val in NEW_MED_SCHEMA["newmedicalConditions"].items():
        subconds = [c.strip() for c in re.split(r"[\/]", combo_key) if c.strip()]
        filled_amount = ""
        filled_pct = ""

        for sc in subconds:
            map_exceptions = {
                "mental": "Mental and behavioural disorders",
                "psychiatric ailment": "Mental and behavioural disorders",
                "tonsillectomy": "Surgery on Tonsils/Adenoids",
                "adenoidectomy": "Surgery on Tonsils/Adenoids",
                "septoplasty": "Sinusitis/DNS/Tympanoplasty/CSOM"
            }

            lookup_name = map_exceptions.get(sc.strip().lower(), sc)
            amt, pct = get_condition_data(lookup_name)

            if amt == "" and pct == "":
                base_med_schema = final_schema.get("medicalConditions", {})
                if sc in base_med_schema:
                    amt = base_med_schema[sc].get(f"{sc}_amount", "")
                    pct = base_med_schema[sc].get(f"{sc}_percentage", "")
                else:
                    mapped_key = map_exceptions.get(sc.strip().lower(), "")
                    if mapped_key and mapped_key in base_med_schema:
                        amt = base_med_schema[mapped_key].get(f"{mapped_key}_amount", "")
                        pct = base_med_schema[mapped_key].get(f"{mapped_key}_percentage", "")

            if filled_amount == "" and filled_pct == "" and (amt != "" or pct != ""):
                filled_amount, filled_pct = amt, pct
            elif (filled_amount != "" or filled_pct != "") and ((amt != "" and amt != filled_amount) or (pct != "" and pct != filled_pct)):
                print(f"⚠️ Rewrite notice: mismatch detected in '{combo_key}' → using '{sc}' values ({amt}, {pct})")
                filled_amount, filled_pct = amt, pct

        NEW_MED_SCHEMA["newmedicalConditions"][combo_key]["amount"] = filled_amount
        NEW_MED_SCHEMA["newmedicalConditions"][combo_key]["percentage"] = filled_pct

    extra_mappings = {
        "Tonsillectomy": "Surgery on Tonsils/Adenoids",
        "Adenoidectomy": "Surgery on Tonsils/Adenoids",
        "Septoplasty": "Sinusitis/DNS/Tympanoplasty/CSOM",
        "Psychiatric ailment": "Mental and behavioural disorders",
        "Mental": "Mental and behavioural disorders",
    }
    for extra_cond, target_group in extra_mappings.items():
        if target_group in NEW_MED_SCHEMA["newmedicalConditions"]:
            amt, pct = get_condition_data(extra_cond)
            current_amt = NEW_MED_SCHEMA["newmedicalConditions"][target_group]["amount"]
            current_pct = NEW_MED_SCHEMA["newmedicalConditions"][target_group]["percentage"]
            if not current_amt and not current_pct and (amt or pct):
                NEW_MED_SCHEMA["newmedicalConditions"][target_group]["amount"] = amt
                NEW_MED_SCHEMA["newmedicalConditions"][target_group]["percentage"] = pct
                print(f"✅ Mapped extra '{extra_cond}' to '{target_group}': {amt}, {pct}")

    new_json_path = os.path.join(output_dir, f"{base}_ICD_3_sheet.json")
    new_excel_path = os.path.join(output_dir, f"{base}_ICD_3_sheet.xlsx")

    with open(new_json_path, "w", encoding="utf-8") as f:
        json.dump(NEW_MED_SCHEMA, f, indent=2, ensure_ascii=False)
    

    master_icd = pd.read_excel(ICD_3_file, header=1)
    conditions = NEW_MED_SCHEMA['newmedicalConditions']

    def fill_limits(row):
        cond = row['Group']
        if cond in conditions:
            amount = conditions[cond]['amount']
            perc = conditions[cond]['percentage']
            if amount != '' and perc != '':
                perc_Limit_Applicable_On = "Sum Insured"
                applicability = 'Lower'
            else:
                perc_Limit_Applicable_On = ''
                applicability = ''
            return pd.Series([perc_Limit_Applicable_On, perc, amount, applicability])
        else:
            return pd.Series(['', '', '', ''])

    master_icd[['% Limit Applicable On', '% Limit','Limit Amount', 'Applicability']] = master_icd.apply(fill_limits, axis=1)
    
    print(f"Saving Excel file to: {new_excel_path}")
    master_icd.to_excel(new_excel_path, index=False)


    #pd.json_normalize(NEW_MED_SCHEMA, sep="_").to_excel(new_excel_path, index=False)


    print(f"✅ NEW_MED_SCHEMA JSON saved → {new_json_path}")
    print(f"✅ NEW_MED_SCHEMA Excel saved → {new_excel_path}")
    print("=====================================================================")

    return NEW_MED_SCHEMA


def run_and_save_ICD(pdf_path,source_dir):
    output_dir=source_dir
    return process_pdf_to_medschema(pdf_path,output_dir)


# -----------------------------
# ✅ Example Run (MODIFIED)
# -----------------------------
if __name__ == "__main__":
    # Input PDF path
    #pdf_path = "/home/ubuntu/rspdf/newtest/1&2&3/HG00007893000100.pdf"

    pdf_path = "/home/ubuntu/rspdf/newtest/1&2&3/GMC0001160000100.pdf"
    #output_dir = "/home/ubuntu/rspdf/newtest/1&2&3/GMC0001160000100"
    
    # Output directory for final results
    output_dir = "/home/ubuntu/rspdf/newtest/ICD/output_1160_newtest/"
    
    # Sum insured value
    sum_insured = 500000
    
    # Check if markdown file exists before processing
    md_path = generate_md_path_from_pdf(pdf_path)
    print(f"📄 Expected markdown file location: {md_path}")
    
    if not os.path.exists(md_path):
        print(f"\n❌ ERROR: Markdown file does not exist at {md_path}")
        print("⚠️  Please run the PDF-to-markdown conversion process first.")
        print("    The markdown file should be generated at:")
        print(f"    {md_path}")
    else:
        print(f"✅ Markdown file found, proceeding with extraction...\n")
        schema = process_pdf_to_medschema(pdf_path, output_dir)
        print("\nExtracted Schema:")
        print(json.dumps(schema, indent=2))