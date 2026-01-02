from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
import pymupdf4llm
import re
from collections import OrderedDict
import json
import pandas as pd
from typing import Dict, Any, List, Optional
from openpyxl import Workbook
from openpyxl.utils import get_column_letter
from hscope_uat.helper.get_si import load_sum_insured_from_json
import os
from hscope_uat.UAT_config import *

# ============================================================
# Initialize Chat
# ============================================================
chat = ChatOllama(
    model="gpt-oss:20b",
    temperature=0.0,
    top_p=1.0,
    max_tokens=2048,
    verbose=False
)

# ============================================================
# PDF → Markdown
# ============================================================
def pdf_to_md(pdf_path: str) -> str:
    return pymupdf4llm.to_markdown(pdf_path)

# ============================================================
# Normalize IDs
# ============================================================
def normalize_id(eid: str) -> str:
    m = re.match(r"^(\d+)(?:\(?([a-zivx]+)\)?)?$", eid, re.IGNORECASE)
    if not m:
        return eid
    num, suffix = m.groups()
    if suffix:
        return f"{num}({suffix.lower()})"
    return num

# ============================================================
# Extract Endorsements
# ============================================================
def extract_endorsements(md_text: str, debug: bool = False) -> OrderedDict[str, str]:
    lines = md_text.splitlines()

    heading_pattern = re.compile(
        r"^\s*\*{0,2}\s*(?:Endorsement|Endt\.?)\.?\s*No\.?\s*"
        r"(\d+)"
        r"(?:\s*\(\s*([A-Za-z0-9ivxIVX]{1,3})\s*\)"
        r"|-(?:([A-Za-z0-9ivxIVX]{1,3}))"
        r"|([A-Za-z]{1,3})(?=\s|[-–]|$))?",
        re.IGNORECASE,
    )

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
        return [line for line in buf if not any(p.search(line) for p in junk_patterns)]

    for i, line in enumerate(lines, 1):
        m = heading_pattern.search(line)
        if m:
            num = m.group(1)
            suffix = m.group(2) or m.group(3) or m.group(4)
            canonical_id = f"{num}({suffix.lower()})" if suffix else num
            if debug:
                print(f"LINE {i}: {line} -> canonical id: {canonical_id}")

            if current_id and buffer:
                endorsements[current_id] = "\n".join(clean_buffer(buffer)).strip()

            current_id = canonical_id
            buffer = [line]
        elif current_id:
            buffer.append(line)

    if current_id and buffer:
        endorsements[current_id] = "\n".join(clean_buffer(buffer)).strip()

    return endorsements

# # ============================================================
# # Extract Special Conditions
# # ============================================================
# def extract_special_conditions(md_text: str) -> Dict[str, str]:
#     header_pattern = re.compile(r"\*\*(.+?)\*\*", re.IGNORECASE)
#     matches = list(header_pattern.finditer(md_text))
#     results = []
#     for i, match in enumerate(matches):
#         header_text = match.group(1).strip().lower()
#         if (("special" in header_text or "other" in header_text) and
#             any(word in header_text for word in ["condition", "conditions", "clause", "clauses", "coverage"])):
#             start = match.start()
#             end = matches[i + 1].start() if i + 1 < len(matches) else len(md_text)
#             block = md_text[start:end].strip()
#             block_cleaned = re.sub(r"\*\*(.+?)\*\*", r"\1", block)
#             results.append(block_cleaned)
#     endorsements_context = ""
#     for endnt in results:
#         endorsements_context += endnt + "\n\n"
#     return {"special condition": endorsements_context}

# def overlay_special_condition(md_text: str) -> Dict[str, str]:
#     endnt_context = extract_special_conditions(md_text=md_text)
#     if endnt_context.get('special condition', "") == "":
#         endnt_context['special condition'] = "There is no special condition provided, so you can consider No context from special conditions section."
#     return endnt_context
# ============================================================
# Extract Special Conditions
# ============================================================
def extract_special_conditions(md_text: str) -> Dict[str, str]:
    header_pattern = re.compile(r"\*\*(.+?)\*\*", re.IGNORECASE)
    matches = list(header_pattern.finditer(md_text))
    results = []

    for i, match in enumerate(matches):
        header_text = match.group(1).strip().lower()
        if (("special" in header_text or "other" in header_text)
            and any(word in header_text for word in ["condition", "conditions", "clause", "clauses", "coverage", " Endorsement", " Endorsements", " Coverages"])):
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(md_text)
            block = md_text[start:end].strip()
            block_cleaned = re.sub(r"\*\*(.+?)\*\*", r"\1", block)
            results.append(block_cleaned)

    endorsements_context = "\n\n".join(results)
    return {"special condition": endorsements_context}

def overlay_special_condition(md_text: str) -> Dict[str, str]:
    endnt_context = extract_special_conditions(md_text)
    if not endnt_context.get("special condition", "").strip():
        endnt_context["special condition"] = (
            "There is no special condition provided, so you can consider "
            "No context from special conditions section."
        )
    return endnt_context

# ============================================================
# Combine Endorsements + Special Conditions - UPDATED FUNCTION
# ============================================================
def get_endorsements_content(pdf_path: str, desired_endorsements: List[str]) -> str:
    md_text = pdf_to_md(pdf_path)
    endorsements = extract_endorsements(md_text)
    desired_endorsements_norm = [normalize_id(eid) for eid in desired_endorsements]
    matching = {eid: endorsements[eid] for eid in desired_endorsements_norm if eid in endorsements}
    matched_values = list(matching.values())
    endorsements_context = ""
    for endnt in matched_values:
        endorsements_context += endnt + "\n\n"
    return endorsements_context

def get_endorsements_and_spl_cond(pdf_path: str, desired_endorsements: List[str] = None) -> str:
    if desired_endorsements is None:
        desired_endorsements = ['11b']  # default
    endnt_context = get_endorsements_content(pdf_path=pdf_path, desired_endorsements=desired_endorsements)
    md_text = pdf_to_md(pdf_path)
    spl_cond = overlay_special_condition(md_text)
    key, value = list(spl_cond.items())[0] if isinstance(spl_cond, dict) and len(spl_cond) > 0 else (None, "")
    if key:
        combined_text = f"{endnt_context.strip()}\n\n**{key}:**\n{value.strip()}"
    else:
        combined_text = endnt_context.strip()
    return combined_text

# ============================================================
# Dummy Sum Insured List
# ============================================================
def get_dummy_sum_insured() -> List[int]:
    return [300000, 400000, 500000]

# ============================================================
# Helper Functions
# ============================================================
def numeric(v: Optional[Any]) -> Optional[float]:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        s = v.replace('%', '').replace(',', '').strip()
        try:
            return float(s)
        except Exception:
            return None
    return None

def replace_none_with_empty(obj):
    if isinstance(obj, dict):
        return {k: replace_none_with_empty(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [replace_none_with_empty(item) for item in obj]
    elif obj is None:
        return ""
    else:
        return obj

# ============================================================
# PART 1: Pre & Post Hospitalization
# ============================================================
def pre_and_post_hospitalization(sample_text: str, chat, special_conditions: str,si_list) -> Dict[str, Any]:
    combined_context = f"{sample_text}\n\n---\n\nAdditional Special Conditions Context:\n{special_conditions}"

    prompt_check = ChatPromptTemplate.from_messages([
         ("system",
         "You are an expert insurance policy analyzer."
         "Return Benifit Applicable Feild default as : Yes"
        ),
        ("human",
         """
         Analyze the following endorsement text and special conditions:

         {context}

         Answer strictly in JSON: {{"Benefit Applicable?": "Yes"}}
         """
        )
    ])

    chain_check = prompt_check | chat
    result_check = chain_check.invoke({"context": combined_context})
    text_check = (result_check.content or "").strip()
    print(text_check)##

    try:
        if text_check.startswith(""):
            text_check = re.sub(r"^(?:json)?\s*", "", text_check).rstrip("`\n\r ")
            text_check = re.sub(r"\s*$", "", text_check).strip()
        first, last = text_check.find("{"), text_check.rfind("}")
        json_text_check = text_check[first:last + 1] if first != -1 and last != -1 else text_check
        benefit_dict = json.loads(json_text_check)
    except Exception:
        benefit_dict = {"Benefit Applicable?": "No"}

    if benefit_dict.get("Benefit Applicable?", "").strip().lower() != "yes":
        return {
            "Benefit Applicable?": "No",
            "Is Pre and Post Combined?": None,
            "Combined_Type Of Expense": None,
            "Combined_No. Of Days": None,
            "Combined_% Limit Applicable On": None,
            "Combined_% Limit": None,
            "Combined_Limit": None,
            "Combined_Applicability": None,
            "Pre Hospitalization Days_Type of expense": None,
            "Pre Hospitalization Days_No. Of Days": None,
            "Pre Hospitalization Days_% Limit Applicable": None,
            "Pre Hospitalization Days_Limit Percentage": None,
            "Pre Hospitalization Days_Limit Amount": None,
            "Pre Hospitalization Days_Applicability": None,
            "Post Hospitalization Days_Type of expense": None,
            "Post Hospitalization Days_No. Of Days": None,
            "Post Hospitalization Days_% Limit Applicable": None,
            "Post Hospitalization Days_Limit Percentage": None,
            "Post Hospitalization Days_Limit Amount": None,
            "Post Hospitalization Days_Applicability": None
        }

    #si_list = load_sum_insured_from_json(pdf_path)
    si_text = json.dumps(si_list)

    full_context = (
        f"{combined_context}\n\n"
        f"IMPORTANT NOTE FOR EXTRACTION:\n"
        f"- The following are the available Sum Insured values: {si_text}\n"
        f"- Whenever 'Limit Amount' fields are required, use these values directly "
        f"for both Pre and Post Hospitalization sections."
    )

    prompt_full = ChatPromptTemplate.from_messages([
        ("system",
         "You are an expert insurance policy analyzer. "
         "Extract Pre and Post Hospitalization details in valid JSON format. "
         "Use the given Sum Insured values exactly as provided. Do not replace them with empty arrays."
         "Fix this as default (Pre Hospitalization Days_No. Of Days: 30),(Post Hospitalization Days_No. Of Days: 60)."
        ),
        ("human",
         """
         Analyze the following endorsement text:

         {context}

         Return strictly valid JSON in this exact format:
         {{
           "Benefit Applicable?": "Yes", 
           "Is Pre and Post Combined?": "No",
           "Combined_Type Of Expense": "",
           "Combined_No. Of Days": "",
           "Combined_% Limit Applicable On": "",
           "Combined_% Limit": "",
           "Combined_Limit": "",
           "Combined_Applicability": "",
           "Pre Hospitalization Days_Type of expense": "",
           "Pre Hospitalization Days_No. Of Days": 30,
           "Pre Hospitalization Days_% Limit Applicable": "Sum Insured",
           "Pre Hospitalization Days_Limit Percentage": 100,
           "Pre Hospitalization Days_Limit Amount": {si_list},
           "Pre Hospitalization Days_Applicability": "Lower",
           "Post Hospitalization Days_Type of expense": "",
           "Post Hospitalization Days_No. Of Days": 60,
           "Post Hospitalization Days_% Limit Applicable": "Sum Insured",
           "Post Hospitalization Days_Limit Percentage": 100,
           "Post Hospitalization Days_Limit Amount": {si_list},
           "Post Hospitalization Days_Applicability": "Lower"
         }}
         """)
    ])

    chain_full = prompt_full | chat
    result_full = chain_full.invoke({
        "context": full_context,
        "si_list": si_text
    })
    text_full = (result_full.content or "").strip()

    try:
        if text_full.startswith(""):
            text_full = re.sub(r"^(?:json)?\s*", "", text_full).rstrip("`\n\r ")
            text_full = re.sub(r"\s*$", "", text_full).strip()
        first, last = text_full.find("{"), text_full.rfind("}")
        json_text_full = text_full[first:last + 1] if first != -1 and last != -1 else text_full
        result = json.loads(json_text_full)
        
        # Force Type of expense fields to be empty
        result["Pre Hospitalization Days_Type of expense"] = ""
        result["Post Hospitalization Days_Type of expense"] = ""
        
        return result
    except Exception:
        return {k: None for k in [
            "Benefit Applicable?",
            "Is Pre and Post Combined?",
            "Combined_Type Of Expense",
            "Combined_No. Of Days",
            "Combined_% Limit Applicable On",
            "Combined_% Limit",
            "Combined_Limit",
            "Combined_Applicability",
            "Pre Hospitalization Days_Type of expense",
            "Pre Hospitalization Days_No. Of Days",
            "Pre Hospitalization Days_% Limit Applicable",
            "Pre Hospitalization Days_Limit Percentage",
            "Pre Hospitalization Days_Limit Amount",
            "Pre Hospitalization Days_Applicability",
            "Post Hospitalization Days_Type of expense",
            "Post Hospitalization Days_No. Of Days",
            "Post Hospitalization Days_% Limit Applicable",
            "Post Hospitalization Days_Limit Percentage",
            "Post Hospitalization Days_Limit Amount",
            "Post Hospitalization Days_Applicability"
        ]}

# ============================================================
# PART 2: Maternity Extraction
# ============================================================
def get_ordered_keys_maternity() -> List[str]:
    return [
        "Maternity_Benefit Applicable?",
        "Maternity_Waiting Period(In Days)",
        "Maternity_Limit On Number Of Live Children",
        "Maternity_Member Contribution Applicable?",
        "Maternity_Copay or deductible Applicable?",
        "Maternity_Is Maternity Combined?",
        "Maternity_Sum Insured",
        "Maternity_% Limit",
        "Maternity_Limit_Decimal",
        "Maternity_Limit_Numeric",
        "Maternity_Applicability",
        "Maternity_Copay",
        "Maternity_Deductible",
        "Maternity_Is Maternity Combined?2",
        "Normal_Sum Insured",
        "Normal_% Limit",
        "Normal_Limit_Decimal",
        "Normal_Limit_Numeric",
        "Normal_Applicability",
        "Normal_Copay",
        "Normal_Deductible",
        "Caesarian_Sum Insured",
        "Caesarian_% Limit",
        "Caesarian_Limit_Decimal",
        "Caesarian_Limit_Numeric",
        "Caesarian_Applicability",
        "Caesarian_Copay",
        "Caesarian_Deductible",
        "Critical_Sum Insured",
        "Critical_% Limit",
        "Critical_Limit_Decimal",
        "Critical_Limit_Numeric",
        "Critical_Applicability",
        "Critical_Copay",
        "Critical_Deductible",
    ]

def manual_extract_maternity(context: str, si_list: List[int]) -> Dict[str, Any]:
    print("Using manual extraction fallback...")
    
    result = {
        "Maternity_Benefit Applicable?": "Yes",
        "Maternity_Waiting Period(In Days)": 0,
        "Maternity_Limit On Number Of Live Children": 2,
        "Maternity_Member Contribution Applicable?": "No",
        "Maternity_Copay or deductible Applicable?": "",
        "Maternity_Is Maternity Combined?": "No",
        "Maternity_Sum Insured": "",
        "Maternity_% Limit": "",
        "Maternity_Limit_Decimal": "",
        "Maternity_Limit_Numeric": "",
        "Maternity_Applicability": "",
        "Maternity_Copay": "",
        "Maternity_Deductible": "",
        "Maternity_Is Maternity Combined?2": "Yes",
        "Normal_Sum Insured": si_list,
        "Normal_% Limit": "Sum Insured",
        "Normal_Limit_Decimal": [],
        "Normal_Limit_Numeric": "",
        "Normal_Applicability": "Lower",
        "Normal_Copay": "",
        "Normal_Deductible": "",
        "Caesarian_Sum Insured": si_list,
        "Caesarian_% Limit": "Sum Insured",
        "Caesarian_Limit_Decimal": [],
        "Caesarian_Limit_Numeric": "",
        "Caesarian_Applicability": "Lower",
        "Caesarian_Copay": "",
        "Caesarian_Deductible": "",
        "Critical_Sum Insured": "",
        "Critical_% Limit": "",
        "Critical_Limit_Decimal": "",
        "Critical_Limit_Numeric": "",
        "Critical_Applicability": "",
        "Critical_Copay": "",
        "Critical_Deductible": "",
    }
    
    normal_patterns = [
	r'Rs\.?\s*([\d,]+)(?:/-)?\s*for\s+Normal\s*(?:&|and)',  # More specific
	r'Rs\.?\s*([\d,]+)(?:/-)?\s*(?:for|per)?\s*Normal(?:\s|,|&)',
	r'Normal\s*[&]?\s*Rs\.?\s*([\d,]+)',]
    for pattern in normal_patterns:
        match = re.search(pattern, context, re.IGNORECASE)
        if match:
            result["Normal_Limit_Numeric"] = float(match.group(1).replace(',', ''))
            print(f"Extracted Normal: {result['Normal_Limit_Numeric']}")
            break
    
    caesarean_patterns = [
	r'(?:&|and)\s*Rs?\.?\s*([\d,]+)(?:/-)?\s*(?:for|per)?\s*(?:Caesarean|Caesarian|C-section)',
	r'Rs\.?\s*([\d,]+)(?:/-)?\s*(?:for|per)?\s*(?:Caesarean|Caesarian)',
	r'(?:Caesarean|Caesarian)\s*Rs\.?\s*([\d,]+)',]	
    for pattern in caesarean_patterns:
        match = re.search(pattern, context, re.IGNORECASE)
        if match:
            result["Caesarian_Limit_Numeric"] = float(match.group(1).replace(',', ''))
            print(f"Extracted Caesarean: {result['Caesarian_Limit_Numeric']}")
            break
    
    if re.search(r'combined|single limit|overall limit', context, re.IGNORECASE):
        result["Maternity_Is Maternity Combined?"] = "Yes"
        result["Maternity_Is Maternity Combined?2"] = ""
    
    return result

def maternity_extraction(pdf_path: str, chat) -> Dict[str, Any]:
    md_text = pdf_to_md(pdf_path)
    endorsements = extract_endorsements(md_text)
    si_list = load_sum_insured_from_json(pdf_path)

    if "11(b)" in endorsements:
        maternity_benefit = "Yes"
    else:
        maternity_benefit = "No"

    if maternity_benefit == "No":
        keys = get_ordered_keys_maternity()
        result = {k: "" for k in keys}
        result["Maternity_Benefit Applicable?"] = "No"
        return result

    combined_context = get_endorsements_and_spl_cond(pdf_path, desired_endorsements=["11b"])
    
    messages = [
        SystemMessage(content="You are a JSON extraction expert. Return ONLY valid JSON, no explanations."),
        HumanMessage(content=f"""Extract maternity information from this text and return as JSON.

Context:
{combined_context}
CRITICAL EXTRACTION RULES:
1. Look for the FIRST amount mentioned before "Normal" - that's the Normal delivery limit
2. Look for the SECOND amount mentioned before/after "Caesarean" - that's the Caesarean limit
3. Example: "Rs. 45000/- for Normal & 50000/- for Caesarean" means:
   - Normal_Limit_Numeric: 45000
   - Caesarian_Limit_Numeric: 50000

Extract these values:
1. Is maternity combined or separate for Normal/Caesarean?
2. Normal delivery limit amount (extract the FIRST number associated with Normal)
3. Caesarean delivery limit amount (extract the number associated with Caesarean)

Rules:
- If text says "Rs. 50,000/- for Normal & Rs. 50,000/- for Caesarean" → NOT combined
- Extract ONLY the numeric amount (remove Rs., /-, commas)

Return this exact JSON structure (fill in actual values):
{{
  "Maternity_Benefit Applicable?": "Yes",
  "Maternity_Waiting Period(In Days)": 0,
  "Maternity_Limit On Number Of Live Children": 2,
  "Maternity_Member Contribution Applicable?": "No",
  "Maternity_Copay or deductible Applicable?": null,
  "Maternity_Is Maternity Combined?": "No",
  "Maternity_Sum Insured": null,
  "Maternity_% Limit": null,
  "Maternity_Limit_Decimal": null,
  "Maternity_Limit_Numeric": null,
  "Maternity_Applicability": null,
  "Maternity_Copay": null,
  "Maternity_Deductible": null,
  "Maternity_Is Maternity Combined?2": "Yes",
  "Normal_Sum Insured": [300000, 400000, 500000],
  "Normal_% Limit": "Sum Insured",
  "Normal_Limit_Decimal": [],
  "Normal_Limit_Numeric": 50000,
  "Normal_Applicability": "Lower",
  "Normal_Copay": null,
  "Normal_Deductible": null,
  "Caesarian_Sum Insured": [300000, 400000, 500000],
  "Caesarian_% Limit": "Sum Insured",
  "Caesarian_Limit_Decimal": [],
  "Caesarian_Limit_Numeric": 50000,
  "Caesarian_Applicability": "Lower",
  "Caesarian_Copay": null,
  "Caesarian_Deductible": null,
  "Critical_Sum Insured": null,
  "Critical_% Limit": null,
  "Critical_Limit_Decimal": null,
  "Critical_Limit_Numeric": null,
  "Critical_Applicability": null,
  "Critical_Copay": null,
  "Critical_Deductible": null
}}

IMPORTANT: Return ONLY the JSON above with actual extracted values. No other text.""")
    ]
    
    result_full = chat.invoke(messages)
    text_full = (result_full.content or "").strip()
    
    try:
        if text_full.startswith("```"):
            text_full = re.sub(r"^```(?:json)?\s*", "", text_full).rstrip("`\n\r ")
            text_full = re.sub(r"\s*```$", "", text_full).strip()
        
        first, last = text_full.find("{"), text_full.rfind("}")
        
        if first == -1 or last == -1 or last <= first:
            print("ERROR: No valid JSON found in LLM response")
            parsed = manual_extract_maternity(combined_context, si_list)
        else:
            json_text_full = text_full[first:last + 1]
            parsed = json.loads(json_text_full)
        
        is_combined = parsed.get("Maternity_Is Maternity Combined?")
        
        if isinstance(is_combined, str) and is_combined.strip().lower() == "yes":
            combined_patterns = [
                r'(?:limited to|limit of)\s*Rs\.?\s*([\d,]+)(?:/-)?\s*per\s*Family',
                r'maximum benefit.*?Rs\.?\s*([\d,]+)(?:/-)?\s*per\s*Family',
                r'Rs\.?\s*([\d,]+)(?:/-)?\s*per\s*Family',
            ]
            for pattern in combined_patterns:
                match = re.search(pattern, combined_context, re.IGNORECASE)
                if match:
                    maternity_amount = float(match.group(1).replace(',', ''))
                    parsed["Maternity_Limit_Numeric"] = maternity_amount
                    print(f"Regex fallback: Extracted Combined Maternity amount = {maternity_amount}")
                    break
        else:
            normal_match = re.search(r'(?:Rs\.?\s*|INR\s*)?([\d,]+)(?:/-)?\s*(?:for|per)?\s*Normal', combined_context, re.IGNORECASE)
            caesarean_match = re.search(r'(?:Rs\.?\s*|INR\s*)?([\d,]+)(?:/-)?\s*(?:for|per)?\s*(?:Caesarean|Caesarian|C-section)', combined_context, re.IGNORECASE)
            
            if normal_match and (not parsed.get("Normal_Limit_Numeric") or parsed.get("Normal_Limit_Numeric") == ""):
                normal_amount = float(normal_match.group(1).replace(',', ''))
                parsed["Normal_Limit_Numeric"] = normal_amount
                print(f"Regex fallback: Extracted Normal amount = {normal_amount}")
            
            if caesarean_match and (not parsed.get("Caesarian_Limit_Numeric") or parsed.get("Caesarian_Limit_Numeric") == ""):
                caesarean_amount = float(caesarean_match.group(1).replace(',', ''))
                parsed["Caesarian_Limit_Numeric"] = caesarean_amount
                print(f"Regex fallback: Extracted Caesarean amount = {caesarean_amount}")
        
        out = {k: parsed.get(k) for k in get_ordered_keys_maternity()}
        
        mat_app = out.get("Maternity_Benefit Applicable?")
        if isinstance(mat_app, str) and mat_app.strip().lower() == "yes":
            is_combined = out.get("Maternity_Is Maternity Combined?")
            if isinstance(is_combined, str) and is_combined.strip().lower() == "yes":
                out["Maternity_Sum Insured"] = si_list
                
                mat_pct_val = out.get("Maternity_% Limit")
                if isinstance(mat_pct_val, str) and mat_pct_val.strip().lower() in ["sum insured", "si"]:
                    mat_pct = 100.0
                else:
                    mat_pct = numeric(mat_pct_val)
                
                if mat_pct is not None and mat_pct > 0:
                    out["Maternity_Limit_Numeric"] = [round(si * mat_pct / 100.0, 2) for si in si_list]
                    out["Maternity_Limit_Decimal"] = [round(mat_pct / 100.0, 4) for si in si_list]
                    out["Maternity_% Limit"] = "Sum Insured"
                    out["Maternity_Applicability"]="Lower"####
                else:
                    mnum = numeric(out.get("Maternity_Limit_Numeric"))
                    if mnum is not None:
                        out["Maternity_Limit_Numeric"] = [mnum for _ in si_list]
                        out["Maternity_Limit_Decimal"] = [round(mnum / si, 4) if si else "" for si in si_list]
                        out["Maternity_% Limit"] = "Sum Insured"
                        out["Maternity_Applicability"]="Lower"####
                    else:
                        out["Maternity_Limit_Decimal"] = ["" for _ in si_list]
                        out["Maternity_Limit_Numeric"] = ["" for _ in si_list]
                
                out["Maternity_Is Maternity Combined?2"] = "No"
                for name in ("Normal", "Caesarian", "Critical"):
                    out[f"{name}_Sum Insured"] = ""
                    out[f"{name}_% Limit"] = ""
                    out[f"{name}_Limit_Decimal"] = ""
                    out[f"{name}_Limit_Numeric"] = ""
                    out[f"{name}_Applicability"] = ""
                    out[f"{name}_Copay"] = ""
                    out[f"{name}_Deductible"] = ""
            else:
                out["Maternity_Sum Insured"] = ""
                out["Maternity_% Limit"] = ""
                out["Maternity_Limit_Decimal"] = ""
                out["Maternity_Limit_Numeric"] = ""
                out["Maternity_Applicability"] = ""
                out["Maternity_Copay"] = ""
                out["Maternity_Deductible"] = ""
                
                out["Maternity_Is Maternity Combined?2"] = "Yes"
                for name in ("Normal", "Caesarian"):
                    out[f"{name}_Sum Insured"] = si_list
                    pct_key = f"{name}_% Limit"
                    num_key = f"{name}_Limit_Numeric"
                    dec_key = f"{name}_Limit_Decimal"
                    p = numeric(out.get(pct_key))
                    if p is not None:
                        out[num_key] = [round(si * p / 100.0, 2) for si in si_list]
                        #out[dec_key] = [round(p / 100.0, 4) for si in si_list]
                        out[dec_key] = [round(p , 4) for si in si_list]
                    else:
                        v = numeric(out.get(num_key))
                        if v is not None:
                            out[num_key] = [v for _ in si_list]
                            out[dec_key] = [round((v / si) *100, 4) if si else "" for si in si_list]
                        else:
                            out[dec_key] = ["" for _ in si_list]
                            out[num_key] = ["" for _ in si_list]
        else:
            for k in list(out.keys()):
                if k.startswith("Maternity_") or k.startswith("Normal_") or k.startswith("Caesarian_"):
                    out[k] = ""
            out["Maternity_Benefit Applicable?"] = "No" if not mat_app else out["Maternity_Benefit Applicable?"]
        
        out = replace_none_with_empty(out)
        
        return out
        
    except Exception as e:
        print(f"Error parsing JSON: {e}")
        print("Using manual extraction as final fallback...")
        parsed = manual_extract_maternity(combined_context, si_list)
        
        out = {k: parsed.get(k) for k in get_ordered_keys_maternity()}
        
        for name in ("Normal", "Caesarian", "Critical"):
            v = numeric(out.get(f"{name}_Limit_Numeric"))
            if v is not None:
                out[f"{name}_Limit_Numeric"] = [v for _ in si_list]
                out[f"{name}_Limit_Decimal"] = [round(v / si, 4) if si else "" for si in si_list]
            else:
                out[f"{name}_Limit_Decimal"] = ["" for _ in si_list]
                out[f"{name}_Limit_Numeric"] = ["" for _ in si_list]
        
        out = replace_none_with_empty(out)
        
        return out

# ============================================================
# PART 3: Pre & Post Natal Extraction
# ============================================================
def pre_and_post_natal(sample_text: str, chat, special_conditions: str,si_list) -> Dict[str, Any]:
    combined_context = f"{sample_text}\n\n---\n\nAdditional Special Conditions Context:\n{special_conditions}"

    prompt_check = ChatPromptTemplate.from_messages([
        ("system",
         "You are an expert insurance policy analyzer. "
         "Check if the 'special condition' is 'extracted' with endorsements (especially endorsement 11b). "
         "Check if Pre-Natal, Post-Natal, or Pre & Post Natal expenses exists in the extracted special condition or endorsement 11b. "
         "CRITICAL: Check if an ACTUAL AMOUNT/LIMIT is mentioned for Pre & Post Natal. "
         "If Pre & Post Natal text exists BUT NO AMOUNT → 'Pre&Post Natal Applicable' is 'No'. "
         "If Pre & Post Natal text exists AND AMOUNT is present → 'Pre&Post Natal Applicable' is 'Yes'."
        ),
        ("human",
         """
         Analyze the following endorsement text and special conditions:

         {context}
        IMPORTANT RULES:
         - If text mentions "Pre-natal and post-natal expenses are covered" but NO specific amount/limit → return "No"
         - If text mentions specific amount like "Rs. 5000 for Pre & Post Natal" → return "Yes"
        Answer strictly in JSON: {{"Pre&Post Natal Applicable": "Yes" or "No"}}
         """
        )
    ])

    chain_check = prompt_check | chat
    result_check = chain_check.invoke({"context": combined_context})
    text_check = (result_check.content or "").strip()

    try:
        if text_check.startswith("```"):
            text_check = re.sub(r"^```(?:json)?\s*", "", text_check).rstrip("`\n\r ")
            text_check = re.sub(r"\s*```$", "", text_check).strip()
        first, last = text_check.find("{"), text_check.rfind("}")
        json_text_check = text_check[first:last + 1] if first != -1 and last != -1 else text_check
        benefit_dict = json.loads(json_text_check)
    except Exception:
        benefit_dict = {"Pre&Post Natal Applicable": "No"}

    if benefit_dict.get("Pre&Post Natal Applicable", "").strip().lower() != "yes":
        return {
            "Pre&Post Natal Applicable": "No",
            "Over & Above Maternity Limit": " ",
            "Is Pre&Post Natal Combined?": " ",
            "Pre And Post Natal Combined Expenses_Maternity":" ",
            "Pre And Post Natal Combined Expenses_No. Of Days": " ",
            "Pre And Post Natal Combined Expenses_% Limit Applicable On": " ",
            "Pre And Post Natal Combined Expenses_% Limit": " ",
            "Pre And Post Natal Combined Expenses_Limit": " ",
            "Pre And Post Natal Combined Expenses_Applicability": " ",
            "Pre-Natal Expenses_Maternity": " ",
            "Pre-Natal Expenses_No. Of Days": " ",
            "Pre-Natal Expenses_% Limit Applicable On": " ",
            "Pre-Natal Expenses_% Limit": " ",
            "Pre-Natal Expenses_Limit": " ",
            "Pre-Natal Expenses_Applicability": " ",
            "Post-Natal Expenses_Maternity": " ",
            "Post-Natal Expenses_No. Of Days": " ",
            "Post-Natal Expenses_% Limit Applicable On": " ",
            "Post-Natal Expenses_% Limit": " ",
            "Post-Natal Expenses_Limit": " ",
            "Post-Natal Expenses_Applicability": " "
        }

    #si_list = get_dummy_sum_insured()
    si_text = json.dumps(si_list)

    full_context = (
        f"{combined_context}\n\n"
        f"IMPORTANT NOTE FOR EXTRACTION:\n"
        f"- Extract ONLY the actual Pre-Natal and Post-Natal amounts mentioned in the endorsement text\n"
        f"- Do NOT use Sum Insured values for Limit calculations\n"
        f"- Return the exact amount found in the text as a single numeric value"
    )

    prompt_full = ChatPromptTemplate.from_messages([
        ("system",
         "You are an expert insurance policy analyzer. "
         "Extract Pre & Post Natal details in valid JSON format. "
         "Fix these as defaults: (Pre-Natal Expenses_No. Of Days: 30), (Post-Natal Expenses_No. Of Days: 60). "
         "Default 'Over & Above Maternity Limit' is 'No' unless explicitly stated. "
         "Default 'Is Pre&Post Natal Combined?' is 'No' unless explicitly stated."
        ),
        ("human",
         """
         Analyze the following endorsement text:

         {context}

         Return strictly valid JSON in this exact format:
         {{
           "Pre&Post Natal Applicable": "Yes",
           "Over & Above Maternity Limit": "No",
           "Is Pre&Post Natal Combined?": "No",
           "Pre And Post Natal Combined Expenses_Maternity": null,
           "Pre And Post Natal Combined Expenses_No. Of Days": null,
           "Pre And Post Natal Combined Expenses_% Limit Applicable On": null,
           "Pre And Post Natal Combined Expenses_% Limit": null,
           "Pre And Post Natal Combined Expenses_Limit": null,
           "Pre And Post Natal Combined Expenses_Applicability": null,
           "Pre-Natal Expenses_Maternity": null,
           "Pre-Natal Expenses_No. Of Days": 30,
           "Pre-Natal Expenses_% Limit Applicable On": "Sum Insured",
           "Pre-Natal Expenses_% Limit": null,
           "Pre-Natal Expenses_Limit": null,
           "Pre-Natal Expenses_Applicability": "lower",
           "Post-Natal Expenses_Maternity": null,
           "Post-Natal Expenses_No. Of Days": 60,
           "Post-Natal Expenses_% Limit Applicable On": "Sum Insured",
           "Post-Natal Expenses_% Limit": null,
           "Post-Natal Expenses_Limit": null,
           "Post-Natal Expenses_Applicability": "lower"
         }}
         
        IMPORTANT RULES: (Follow the Rules Strictly)

        - Check if Pre-Natal and Post-Natal have a COMBINED/SINGLE amount mentioned together in the text
        - If COMBINED text is found but NO amount is present:
            * Set "Pre&Post Natal Applicable": "No"
            * Leave ALL Pre-Natal and Post-Natal fields EMPTY (null)
            * Example: Text says "Pre and Post Natal expenses are covered" but no amount → Applicable = "No"
        - If COMBINED amount is found AND amount IS present(e.g., "Pre and Post Natal OPD Expenses: Subject to a sublimit of INR. 5000"):
            * Set "Pre&Post Natal Applicable": "Yes"
            * Use the SAME single amount for BOTH "Pre-Natal Expenses_Limit" AND "Post-Natal Expenses_Limit"
            * Example: If text says Rs. 5000 for Pre & Post combined, set BOTH limits to 5000    
        - Extract ONLY the EXACT amounts mentioned in the endorsement text (single numeric value only)
        - DO NOT use Sum Insured values for Limit fields - only use actual amounts from the text
        - Leave "Pre-Natal Expenses_% Limit" and "Post-Natal Expenses_% Limit" as null initially
        - Since Pre&Post Natal Combined is always "No", always fill fields from 'Pre-Natal Expenses_Maternity' to 'Post-Natal Expenses_Applicability'
         """
        )
    ])

    chain_full = prompt_full | chat
    result_full = chain_full.invoke({
        "context": full_context,
        "si_list": si_text
    })
    text_full = (result_full.content or "").strip()

    try:
        if text_full.startswith("```"):
            text_full = re.sub(r"^```(?:json)?\s*", "", text_full).rstrip("`\n\r ")
            text_full = re.sub(r"\s*```$", "", text_full).strip()
        first, last = text_full.find("{"), text_full.rfind("}")
        json_text_full = text_full[first:last + 1] if first != -1 and last != -1 else text_full
        parsed_result = json.loads(json_text_full)
        
        #si_list = get_dummy_sum_insured()
        
        pre_limit = parsed_result.get("Pre-Natal Expenses_Limit")
        if pre_limit is not None and not isinstance(pre_limit, list):
            pre_pct_list = [round((pre_limit / si) * 100, 2) for si in si_list]
            parsed_result["Pre-Natal Expenses_% Limit"] = pre_pct_list
            parsed_result["Pre-Natal Expenses_Limit"] = [float(pre_limit) for _ in si_list]
        
        post_limit = parsed_result.get("Post-Natal Expenses_Limit")
        if post_limit is not None and not isinstance(post_limit, list):
            post_pct_list = [round((post_limit / si) * 100, 2) for si in si_list]
            parsed_result["Post-Natal Expenses_% Limit"] = post_pct_list
            parsed_result["Post-Natal Expenses_Limit"] = [float(post_limit) for _ in si_list]
        
        combined_limit = parsed_result.get("Pre And Post Natal Combined Expenses_Limit")
        if combined_limit is not None and not isinstance(combined_limit, list):
            combined_pct_list = [round((combined_limit / si) * 100, 2) for si in si_list]
            parsed_result["Pre And Post Natal Combined Expenses_% Limit"] = combined_pct_list
            parsed_result["Pre And Post Natal Combined Expenses_Limit"] = [float(combined_limit) for _ in si_list]
        
        return parsed_result
        
    except Exception as e:
        print(f"Error parsing JSON: {e}")
        return {k: None for k in [
            "Pre&Post Natal Applicable",
            "Over & Above Maternity Limit",
            "Is Pre&Post Natal Combined?",
            "Pre And Post Natal Combined Expenses_Maternity",
            "Pre And Post Natal Combined Expenses_No. Of Days",
            "Pre And Post Natal Combined Expenses_% Limit Applicable On",
            "Pre And Post Natal Combined Expenses_% Limit",
            "Pre And Post Natal Combined Expenses_Limit",
            "Pre And Post Natal Combined Expenses_Applicability",
            "Pre-Natal Expenses_Maternity",
            "Pre-Natal Expenses_No. Of Days",
            "Pre-Natal Expenses_% Limit Applicable On",
            "Pre-Natal Expenses_% Limit",
            "Pre-Natal Expenses_Limit",
            "Pre-Natal Expenses_Applicability",
            "Post-Natal Expenses_Maternity",
            "Post-Natal Expenses_No. Of Days",
            "Post-Natal Expenses_% Limit Applicable On",
            "Post-Natal Expenses_% Limit",
            "Post-Natal Expenses_Limit",
            "Post-Natal Expenses_Applicability"
        ]}

# ============================================================
# PART 4: New Born Cover Extraction
# ============================================================
def check_endorsements_12_12a(endorsements: OrderedDict[str, str]) -> Dict[str, bool]:
    has_12 = "12" in endorsements
    has_12a = "12(a)" in endorsements
    
    return {
        "has_endorsement_12": has_12,
        "has_endorsement_12a": has_12a,
        "any_endorsement_12_variant": has_12 or has_12a
    }

def new_born_cover_extraction(endorsement_check: Dict[str, bool]) -> Dict[str, Any]:
    found_12 = endorsement_check.get("has_endorsement_12", False)
    found_12a = endorsement_check.get("has_endorsement_12a", False)

    if not found_12 and not found_12a:
        return {
            "New Born Covered?": "No",
            "New Born Covered_Covered From": "",
            "New Born Covered_Is New Born Limit Applicable": "",
            "New Born Covered_Sum Insured": "",
            "New Born Covered_% Limit Applicable On": "",
            "New Born Covered_Limit Percentage": "",
            "New Born Covered_Limit Amount": "",
            "New Born Covered_Applicability": ""
        }

    return {
        "New Born Covered?": "Yes",
        "New Born Covered_Covered From": 0,
        "New Born Covered_Is New Born Limit Applicable": "No",
        "New Born Covered_Sum Insured": "",
        "New Born Covered_% Limit Applicable On": "",
        "New Born Covered_Limit Percentage": "",
        "New Born Covered_Limit Amount": "",
        "New Born Covered_Applicability": ""
    }

# ============================================================
# FULL SCHEMA OUTPUT ORDER
# ============================================================
def get_full_schema_order() -> List[str]:
    return [
        "Benefit Applicable?",
        "Is Pre and Post Combined?",
        "Combined_Type Of Expense",
        "Combined_No. Of Days",
        "Combined_% Limit Applicable On",
        "Combined_% Limit",
        "Combined_Limit",
        "Combined_Applicability",
        "Pre Hospitalization Days_Type of expense",
        "Pre Hospitalization Days_No. Of Days",
        "Pre Hospitalization Days_% Limit Applicable",
        "Pre Hospitalization Days_Limit Percentage",
        "Pre Hospitalization Days_Limit Amount",
        "Pre Hospitalization Days_Applicability",
        "Post Hospitalization Days_Type of expense",
        "Post Hospitalization Days_No. Of Days",
        "Post Hospitalization Days_% Limit Applicable",
        "Post Hospitalization Days_Limit Percentage",
        "Post Hospitalization Days_Limit Amount",
        "Post Hospitalization Days_Applicability",
        "Maternity_Benefit Applicable?",
        "Maternity_Waiting Period(In Days)",
        "Maternity_Limit On Number Of Live Children",
        "Maternity_Member Contribution Applicable?",
        "Maternity_Copay or deductible Applicable?",
        "Maternity_Is Maternity Combined?",
        "Maternity_Sum Insured",
        "Maternity_% Limit",
        "Maternity_Limit_Decimal",
        "Maternity_Limit_Numeric",
        "Maternity_Applicability",
        "Maternity_Copay",
        "Maternity_Deductible",
        "Maternity_Is Maternity Combined?2",
        "Normal_Sum Insured",
        "Normal_% Limit",
        "Normal_Limit_Decimal",
        "Normal_Limit_Numeric",
        "Normal_Applicability",
        "Normal_Copay",
        "Normal_Deductible",
        "Caesarian_Sum Insured",
        "Caesarian_% Limit",
        "Caesarian_Limit_Decimal",
        "Caesarian_Limit_Numeric",
        "Caesarian_Applicability",
        "Caesarian_Copay",
        "Caesarian_Deductible",
        "Critical_Sum Insured",
        "Critical_% Limit",
        "Critical_Limit_Decimal",
        "Critical_Limit_Numeric",
        "Critical_Applicability",
        "Critical_Copay",
        "Critical_Deductible",
        "Pre&Post Natal Applicable",
        "Over & Above Maternity Limit",
        "Is Pre&Post Natal Combined?",
        "Pre And Post Natal Combined Expenses_Maternity",
        "Pre And Post Natal Combined Expenses_No. Of Days",
        "Pre And Post Natal Combined Expenses_% Limit Applicable On",
        "Pre And Post Natal Combined Expenses_% Limit",
        "Pre And Post Natal Combined Expenses_Limit",
        "Pre And Post Natal Combined Expenses_Applicability",
        "Pre-Natal Expenses_Maternity",
        "Pre-Natal Expenses_No. Of Days",
        "Pre-Natal Expenses_% Limit Applicable On",
        "Pre-Natal Expenses_% Limit",
        "Pre-Natal Expenses_Limit",
        "Pre-Natal Expenses_Applicability",
        "Post-Natal Expenses_Maternity",
        "Post-Natal Expenses_No. Of Days",
        "Post-Natal Expenses_% Limit Applicable On",
        "Post-Natal Expenses_% Limit",
        "Post-Natal Expenses_Limit",
        "Post-Natal Expenses_Applicability",
        "New Born Covered?",
        "New Born Covered_Covered From",
        "New Born Covered_Is New Born Limit Applicable",
        "New Born Covered_Sum Insured",
        "New Born Covered_% Limit Applicable On",
        "New Born Covered_Limit Percentage",
        "New Born Covered_Limit Amount",
        "New Born Covered_Applicability"
    ]

# ============================================================
# MAIN INTEGRATION FUNCTION
# ============================================================
def process_pdf_to_full_schema(pdf_path: str, chat) -> Dict[str, Any]:
    """
    Process a PDF and extract all information into the full schema.
    """
    print(f"Processing PDF: {pdf_path}")
    
    # Get markdown and endorsements
    md_text = pdf_to_md(pdf_path)
    endorsements = extract_endorsements(md_text)
    spl_cond = overlay_special_condition(md_text)
    special_conditions_text = spl_cond.get("special condition", "")
    si_list=load_sum_insured_from_json(pdf_path)
    
    # PART 1: Pre & Post Hospitalization
    print("\n--- Extracting Pre & Post Hospitalization ---")
    try:
        context_text = get_endorsements_and_spl_cond(pdf_path, ["11b"])
        #  PRINT THE EXTRACTED CONTEXT 
        print("="*60)
        print("Extracted Endorsement and Special Condition Context (Sent to LLM):")
        print("="*60)
        print(context_text)
        print("="*60)
        
    except Exception as e:
        print(f"Warning: endorsement extraction failed: {e}")
        context_text = md_text
    
    pre_post_hosp = pre_and_post_hospitalization(context_text, chat, special_conditions_text,si_list)
    pre_post_hosp = replace_none_with_empty(pre_post_hosp)
    
    # PART 2: Maternity
    print("\n--- Extracting Maternity ---")
    maternity = maternity_extraction(pdf_path, chat)
    
    # PART 3: Pre & Post Natal
    print("\n--- Extracting Pre & Post Natal ---")
    try:
        context_text_natal = get_endorsements_and_spl_cond(pdf_path, ["11b"])
    except Exception as e:
        print(f"Warning: endorsement extraction failed: {e}")
        context_text_natal = md_text
    
    pre_post_natal = pre_and_post_natal(context_text_natal, chat, special_conditions_text,si_list)
    pre_post_natal = replace_none_with_empty(pre_post_natal)
    
    # PART 4: New Born Cover
    print("\n--- Extracting New Born Cover ---")
    endorsement_check = check_endorsements_12_12a(endorsements)
    new_born = new_born_cover_extraction(endorsement_check)
    
    # Combine all results
    full_result = {}
    full_result.update(pre_post_hosp)
    full_result.update(maternity)
    full_result.update(pre_post_natal)
    full_result.update(new_born)
    
    # Reorder according to schema
    ordered_result = {}
    for key in get_full_schema_order():
        ordered_result[key] = full_result.get(key, "")
    
    return ordered_result 

def repeat_fields_based_on_array_length(
    data: Dict[str, Any],
    field_mappings: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Repeat specified fields based on the length of array fields.
    
    Args:
        data: Dictionary containing the extracted data
        field_mappings: List of dictionaries with structure:
            {
                'main_field': 'field_name_with_array',
                'repeat_fields': ['field1', 'field2', 'field3']
            }
    
    Returns:
        Modified data dictionary with repeated fields
    """
    modified_data = data.copy()
    
    for mapping in field_mappings:
        main_field = mapping['main_field']
        repeat_fields = mapping['repeat_fields']
        
        # Check if main field exists and has data
        if main_field not in data:
            continue
            
        main_value = data[main_field]
        
        # Skip if main field is empty, None, or not a list
        if not main_value or main_value == "" or main_value is None:
            continue
        
        # Convert to list if it's a string representation
        if isinstance(main_value, str):
            try:
                import json
                main_value = json.loads(main_value)
            except:
                continue
        
        # Must be a list with elements
        if not isinstance(main_value, list) or len(main_value) == 0:
            continue
        
        # Get the length of the main field array
        array_length = len(main_value)
        
        # Repeat each field in repeat_fields
        for field in repeat_fields:
            if field in modified_data:
                original_value = modified_data[field]
                
                # Create a list with the same value repeated
                if isinstance(original_value, list):
                    # If already a list, ensure it matches the length
                    if len(original_value) == 1:
                        modified_data[field] = original_value * array_length
                    elif len(original_value) < array_length:
                        # Pad with last value
                        modified_data[field] = original_value + [original_value[-1]] * (array_length - len(original_value))
                else:
                    # Repeat the single value
                    modified_data[field] = [original_value] * array_length
    
    return modified_data


def export_to_excel_with_field_repetition(pdf_path,
    data: Dict[str, Any],
    schema_order: List[str],
    field_mappings: List[Dict[str, Any]],source_dir
):
    """
    Export data to Excel with field repetition based on array fields.
    
    Args:
        data: Dictionary containing the extracted data
        output_path: Path for output Excel file
        schema_order: List of field names in desired order
        field_mappings: List of dictionaries defining which fields to repeat
    """
    # Apply field repetition
    modified_data = repeat_fields_based_on_array_length(data, field_mappings)
    
    # Identify all array fields (fields that should be expanded row-wise)
    array_fields = set()
    for mapping in field_mappings:
        array_fields.add(mapping['main_field'])
        array_fields.update(mapping['repeat_fields'])
    
    # Determine the number of rows (max length of any array field)
    max_length = 1
    for key, value in modified_data.items():
        if isinstance(value, list) and len(value) > 0:
            max_length = max(max_length, len(value))
    
    # Prepare rows
    rows = []
    for i in range(max_length):
        row = {}
        for key in schema_order:
            if key not in modified_data:
                row[key] = ""
                continue
            
            value = modified_data[key]
            
            if isinstance(value, list) and len(value) > 0:
                # Extract the i-th element from the array
                row[key] = value[i] if i < len(value) else ""
            elif isinstance(value, str):
                # Try to parse JSON string
                try:
                    import json
                    parsed = json.loads(value)
                    if isinstance(parsed, list) and len(parsed) > 0:
                        row[key] = parsed[i] if i < len(parsed) else ""
                    else:
                        row[key] = value if i == 0 else ""
                except:
                    row[key] = value if i == 0 else ""
            else:
                # Non-array fields: show value only in first row
                row[key] = value if i == 0 else ""
        
        rows.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    
    # Reorder columns according to schema
    existing_cols = [col for col in schema_order if col in df.columns]
    df = df[existing_cols]
    
    output_folder = source_dir  # define your working folder
    os.makedirs(output_folder, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(pdf_path))[0]  
    excel_filename = os.path.join(output_folder, f"{base_name}_primay_cover.xlsx")

    # Write to Excel
    df.to_excel(excel_filename, index=False, engine='openpyxl')
    
    print(f"\n✓ Excel file created: {excel_filename}")
    print(f"✓ Total rows: {max_length}")


# Example usage configuration
FIELD_REPETITION_MAPPINGS = [
    {
        'main_field': 'Pre Hospitalization Days_Limit Amount',
        'repeat_fields': [
            'Pre Hospitalization Days_No. Of Days',
            'Pre Hospitalization Days_% Limit Applicable',
            'Pre Hospitalization Days_Limit Percentage',
            'Pre Hospitalization Days_Applicability'
        ]
    },
    {
        'main_field': 'Post Hospitalization Days_Limit Amount',
        'repeat_fields': [
            'Post Hospitalization Days_No. Of Days',
            'Post Hospitalization Days_% Limit Applicable',
            'Post Hospitalization Days_Limit Percentage',
            'Post Hospitalization Days_Applicability'
        ]
    },
    {
        'main_field': 'Normal_Limit_Numeric',
        'repeat_fields': [
            'Normal_Sum Insured',
            'Normal_% Limit',
            'Normal_Limit_Decimal',
            'Normal_Applicability',
            'Normal_Copay',
            'Normal_Deductible'
        ]
    },
    {
        'main_field': 'Caesarian_Limit_Numeric',
        'repeat_fields': [
            'Caesarian_Sum Insured',
            'Caesarian_% Limit',
            'Caesarian_Limit_Decimal',
            'Caesarian_Applicability',
            'Caesarian_Copay',
            'Caesarian_Deductible'
        ]
    },
    {
        'main_field': 'Maternity_Limit_Numeric',
        'repeat_fields': [
            'Maternity_Sum Insured',
            'Maternity_% Limit',
            'Maternity_Limit_Decimal',
            'Maternity_Applicability',
            'Maternity_Copay',
            'Maternity_Deductible'
        ]
    },
    {
        'main_field': 'Pre-Natal Expenses_Limit',
        'repeat_fields': [
            'Pre-Natal Expenses_Maternity',
            'Pre-Natal Expenses_No. Of Days',
            'Pre-Natal Expenses_% Limit Applicable On',
            'Pre-Natal Expenses_% Limit',
            'Pre-Natal Expenses_Applicability'
        ]
    },
    {
        'main_field': 'Post-Natal Expenses_Limit',
        'repeat_fields': [
            'Post-Natal Expenses_Maternity',
            'Post-Natal Expenses_No. Of Days',
            'Post-Natal Expenses_% Limit Applicable On',
            'Post-Natal Expenses_% Limit',
            'Post-Natal Expenses_Applicability'
        ]
    }
]

# ============================================================
# EXPORT TO EXCEL WITH ROW-WISE EXPANSION
# ============================================================
def export_to_excel(data: Dict[str, Any],source_dir):
    """
    Export the extracted data to an Excel file with row-wise expansion for array fields.
    """
    si_list = load_sum_insured_from_json(pdf_path)
    num_rows = len(si_list)
    
    # Fields that should be expanded row-wise (contain arrays)
    array_fields = [
        "Pre Hospitalization Days_Limit Amount",
        "Post Hospitalization Days_Limit Amount",
        "Normal_Sum Insured",
        "Normal_Limit_Decimal",
        "Normal_Limit_Numeric",
        "Caesarian_Sum Insured",
        "Caesarian_Limit_Decimal",
        "Caesarian_Limit_Numeric",
        "Maternity_Sum Insured",
        "Maternity_Limit_Decimal",
        "Maternity_Limit_Numeric",
        "Pre-Natal Expenses_% Limit",
        "Pre-Natal Expenses_Limit",
        "Post-Natal Expenses_% Limit",
        "Post-Natal Expenses_Limit",
        "Pre And Post Natal Combined Expenses_% Limit",
        "Pre And Post Natal Combined Expenses_Limit"
    ]
    
    # Prepare rows
    rows = []
    for i in range(num_rows):
        row = {}
        for key, value in data.items():
            if key in array_fields and isinstance(value, list) and len(value) > 0:
                # Extract the i-th element from the array
                row[key] = value[i] if i < len(value) else ""
            elif key in array_fields and isinstance(value, str):
                # If it's a JSON string, parse it
                try:
                    parsed = json.loads(value)
                    if isinstance(parsed, list) and len(parsed) > 0:
                        row[key] = parsed[i] if i < len(parsed) else ""
                    else:
                        row[key] = value if i == 0 else ""
                except:
                    row[key] = value if i == 0 else ""
            else:
                # Non-array fields: show value only in first row
                row[key] = value if i == 0 else ""
        
        rows.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    
    # Reorder columns according to schema
    schema_order = get_full_schema_order()
    existing_cols = [col for col in schema_order if col in df.columns]
    df = df[existing_cols]
    
    # Write to Excel
    output_folder = source_dir  # define your working folder
    os.makedirs(output_folder, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(pdf_path))[0]  
    excel_filename = os.path.join(output_folder, f"{base_name}_primay_cover.xlsx")

    df.to_excel(excel_filename, index=False, engine='openpyxl')
    
    print(f"\n✓ Excel file created: {excel_filename}")
    print(f"✓ Total rows: {num_rows} (one per Sum Insured value)")

# ============================================================
# MAIN EXECUTION
# ============================================================
# if __name__ == "__main__":
#     # Input PDF path
#     pdf_path = "/home/ubuntu/rspdf/Primarycover_Sheet/HG00007391000100.pdf"
    
#     # Output Excel path
#     output_excel = "insurance_extraction_output.xlsx"
    
#     # Process PDF
#     result = process_pdf_to_full_schema(pdf_path, chat)
    
#     # Print JSON result
#     print("\n" + "="*80)
#     print("FINAL JSON OUTPUT:")
#     print("="*80)
#     print(json.dumps(result, indent=2))
    
#     # Export to Excel
#     export_to_excel(result, output_excel)
    
#     print("\n" + "="*80)
#     print("PROCESSING COMPLETE")
#     print("="*80)


def run_primary_cover(pdf_path,chat,source_dir):



        result = process_pdf_to_full_schema(pdf_path, chat)

        print("\n" + "="*80)
        print("FINAL JSON OUTPUT:")
        print("="*80)
        print(json.dumps(result, indent=2))

        # Use new Excel export with field repetition
        export_to_excel_with_field_repetition(pdf_path,
            data=result,
            schema_order=get_full_schema_order(),
            field_mappings=FIELD_REPETITION_MAPPINGS,source_dir=source_dir
        )

        print("\n" + "="*80)
        print("PROCESSING COMPLETE")
        print("="*80)
        


if __name__ == "__main__":
    pdf_path = "/home/ubuntu/rspdf/Primarycover_Sheet/input/23-input/HG00006789000100.pdf"
    #output_excel = "/home/ubuntu/rspdf/Primarycover_Sheet/UATOP/insurance_extraction_output.xlsx"
    #output_excel = pdf_path.rsplit("/", 1)[-1].replace(".pdf", ".xlsx")
    output_excel = pdf_path.replace(".pdf", ".xlsx")
    source_dir=r""

    run_primary_cover(pdf_path,chat,source_dir=source_dir)


    