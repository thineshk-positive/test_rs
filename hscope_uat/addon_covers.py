# %%
import pymupdf4llm
import re
from collections import OrderedDict
import json
import re
from typing import Dict, Any
import subprocess
import json
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass
import pandas as pd
import os
from hscope_uat.UAT_config import *


# ----------------------------
# Normalize IDs (🔥 New)
# ----------------------------
def normalize_id(eid: str) -> str:
    """
    Convert endorsement IDs to canonical form:
    - 5a -> 5(a)
    - 5d -> 5(d)
    - 5(i) -> 5(i)
    """
    m = re.match(r"^(\d+)(?:\(?([a-zivx]+)\)?)?$", eid, re.IGNORECASE)
    if not m:
        return eid
    num, suffix = m.groups()
    if suffix:
        return f"{num}({suffix.lower()})"
    return num

# %%


# ----------------------------
# Extract Endorsements (with cleanup)
# ----------------------------
def extract_endorsements(md_text: str, debug: bool = False):
    lines = md_text.splitlines()

    heading_pattern = re.compile(
        r"^\s*\*{0,2}\s*(?:Endorsement|Endt\.?)\.?\s*No\.?\s*"
        r"(\d+)"                                  # number
        r"(?:\s*\(\s*([A-Za-z0-9ivxIVX]{1,3})\s*\)"   # (a), (i), etc.
        r"|-(?:([A-Za-z0-9ivxIVX]{1,3}))"             # -a, -i, etc.
        r"|([A-Za-z]{1,3})(?=\s|[-–]|$)"              # attached suffix
        r")?",
        re.IGNORECASE,
    )

    # Unwanted boilerplate (headers/footers)
    junk_patterns = [
        re.compile(r"^group health policy", re.IGNORECASE),
        re.compile(r"^uin[:\s]", re.IGNORECASE),
        re.compile(r"^irda", re.IGNORECASE),
        re.compile(r"^policy number", re.IGNORECASE),
        re.compile(r"^name of the insured", re.IGNORECASE),
        re.compile(r"^period of insurance", re.IGNORECASE),
        re.compile(r"endorsements attached", re.IGNORECASE),
        re.compile(r"Group Health Policy – Endorsements", re.IGNORECASE),
        re.compile(r"^Page\s+\*\*\d+\*\*\s+of\s+\*\*\d+\*\*$", re.IGNORECASE),   # Page **4** of **18**
        re.compile(r"^\*\*Policy Number.*\*\*$", re.IGNORECASE),
        re.compile(r"^\*\*Name of the Insured.*\*\*$", re.IGNORECASE),
        re.compile(r"^\*\*Period of Insurance.*\*\*$", re.IGNORECASE),
        re.compile(r"^//\s*\d+\s*//$", re.IGNORECASE),   # 🔥 catches //2// and // 2 //
        re.compile(r"^royal sundaram.*", re.IGNORECASE),
        re.compile(r"^regd office.*", re.IGNORECASE),
        re.compile(r"^corporate office.*", re.IGNORECASE),
        re.compile(r"^email[:\s].*", re.IGNORECASE),
        re.compile(r"^website[:\s].*", re.IGNORECASE),
        re.compile(r"^ph[:\s].*", re.IGNORECASE),             # phone numbers
        re.compile(r"^.*irda regn.*", re.IGNORECASE),         # IRDA lines
        re.compile(r"^.*cin[-:\s].*", re.IGNORECASE),         # CIN lines
        re.compile(r".*\bchennai\s*\d{3}\s*\d{3}.*", re.IGNORECASE)

    ]

    endorsements = OrderedDict()
    current_id, buffer = None, []

    def clean_buffer(buf):
        """Remove unwanted header/footer lines."""
        return [
            line for line in buf
            if not any(p.search(line) for p in junk_patterns)
        ]

    for i, line in enumerate(lines, 1):
        m = heading_pattern.search(line)
        if m:
            # ---- New endorsement found ----
            num = m.group(1)
            suffix = m.group(2) or m.group(3) or m.group(4)

            canonical_id = num
            if suffix and len(suffix) <= 3:
                canonical_id = f"{num}({suffix.lower()})"

            if debug:
                print(f"LINE {i}: ✅ {line}")
                print(f"       -> canonical id: {canonical_id}")

            # Save previous endorsement
            if current_id and buffer:
                endorsements[current_id] = "\n".join(clean_buffer(buffer)).strip()

            current_id = canonical_id
            buffer = [line]

        elif current_id:
            buffer.append(line)

    # Save last one
    if current_id and buffer:
        endorsements[current_id] = "\n".join(clean_buffer(buffer)).strip()

    return endorsements




from langchain_ollama import ChatOllama

chat = ChatOllama(
    model="gpt-oss:20b",
    temperature=0.0,
    top_p=1.0,
    max_tokens=1024,
    verbose=False
)





def pdf_to_md(pdf_path):
    md_text = pymupdf4llm.to_markdown(pdf_path)
    return md_text



def get_endorsements_content(pdf_path,desired_endorsements):
    md_text=pdf_to_md(pdf_path)
    endorsements=extract_endorsements(md_text)
    desired_endorsements = [normalize_id(eid) for eid in desired_endorsements]
    matching_endorsements = {
            eid: endorsements[eid]
            for eid in desired_endorsements
            if eid in endorsements
        }
    matched_endorsements=matching_endorsements.values()
    matched_endorsements=list(matched_endorsements)

    total_endoresments=len(matched_endorsements)
    endorsements_context = ""
    for endnt in matched_endorsements:
        endorsements_context += endnt + "\n\n"
    return endorsements_context




from typing import Dict, List

def extract_special_conditions(md_text: str) -> Dict[str, List[str]]:
    """
    Extract blocks starting from bold markdown headers that contain keywords:
    'special' or 'other' + ('condition'/'conditions'/'clause'/'clauses'/'coverage').
    
    Captures from that header until the next bold header.
    """
    # Regex for bold headers (**Header**)
    header_pattern = re.compile(r"\*\*(.+?)\*\*", re.IGNORECASE)

    matches = list(header_pattern.finditer(md_text))
    results = []

    for i, match in enumerate(matches):
        header_text = match.group(1).strip().lower()
        
        # Check if it contains "special" or "other" AND one of the target keywords
        if (("special" in header_text or "other" in header_text) and 
            any(word in header_text for word in ["condition", "conditions", "clause", "clauses", "coverage"])):
            
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(md_text)
            
            block = md_text[start:end].strip()
            
            # Clean bold markers (**..**)
            block_cleaned = re.sub(r"\*\*(.+?)\*\*", r"\1", block)
            
            results.append(block_cleaned)
    endorsements_context = ""
    for endnt in results:
        endorsements_context += endnt + "\n\n"

    return {"special condition": endorsements_context}




def overaly_specail_condtion(md_text):
    endnt_context=extract_special_conditions(md_text=md_text)
    if endnt_context['special condition']=="":
        spl_cond_context= """There is no special conditon is provided, so you can consider No context from specail conditons section.
            """
        endnt_context['special condition']= spl_cond_context
        return endnt_context
    return endnt_context




def get_endorsements_and_spl_cond(pdf_path,desired_endorsements = ['16', '15', '14', '17','7', '20']
):    
    endnt_context = get_endorsements_content(pdf_path=pdf_path, desired_endorsements=desired_endorsements)
    md_text = pdf_to_md(pdf_path)
    spl_cond = overaly_specail_condtion(md_text)

    # ✅ Unpack dict properly (key + value)
    if isinstance(spl_cond, dict) and len(spl_cond) > 0:
        key, value = list(spl_cond.items())[0]   # take first key-value pair
        combined_text = f"{endnt_context.strip()}\n\n**{key}:**\n{value.strip()}"
    else:
        combined_text = endnt_context.strip()

    return combined_text




import re
import json

def extract_addon_covers(end_and_spl_cond) -> dict:
    """
    Generate a prompt for the LLM to extract Yes/No for six add-on covers
    based on endorsements and special conditions.
    """

    prompt = f"""
You are an expert insurance policy analyst. Extract the presence of the following add-on covers 
from the given endorsements and special conditions text. Return only Yes or No for each field.

Fields to extract:
- Ambulance Cover
- Convalescence Benefit
- Daily/Hospital Cash Benefit
- Doctor & Nurse Home Visit Cover
- Out Patient Cover
- Critical Illness Benefit

Rules:
1. If the endorsement corresponding to the cover is present, return Yes.
2. If the special conditions text mentions or paraphrases the cover, return Yes.
3. Default value is No if neither endorsement nor special conditions indicate the cover.
4. Only respond with JSON containing the above fields with values Yes/No.

Endorsements  and specail conditions provided: {end_and_spl_cond}


Respond ONLY with JSON like this:
{{
  "Ambulance Cover": "",
  "Convalescence Benefit": "",
  "Daily/Hospital Cash Benefit": "",
  "Doctor & Nurse Home Visit Cover": "",
  "Out Patient Cover": "",
  "Critical Illness Benefit": ""
}}
"""

    try:
        # Query the LLM (replace 'chat.invoke' with your LLM call)
        raw_response = chat.invoke(prompt)  # your LLM call
        if hasattr(raw_response, "content"):
            raw_response = raw_response.content

        # Extract JSON from LLM response
        match = re.search(r"\{[\s\S]*\}", raw_response)
        if not match:
            # fallback defaults
            return {
                "Ambulance Cover": "No",
                "Convalescence Benefit": "No",
                "Daily/Hospital Cash Benefit": "No",
                "Doctor & Nurse Home Visit Cover": "No",
                "Out Patient Cover": "No",
                "Critical Illness Benefit": "No"
            }

        data = json.loads(match.group())

        # Ensure only Yes/No values and all fields are present
        result = {}
        for field in [
            "Ambulance Cover",
            "Convalescence Benefit",
            "Daily/Hospital Cash Benefit",
            "Doctor & Nurse Home Visit Cover",
            "Out Patient Cover",
            "Critical Illness Benefit"
        ]:
            val = data.get(field, "No")
            result[field] = "Yes" if str(val).strip().lower() == "yes" else "No"

        return result

    except Exception as e:
        # On error, return defaults
        return {
            "Ambulance Cover": "No",
            "Convalescence Benefit": "No",
            "Daily/Hospital Cash Benefit": "No",
            "Doctor & Nurse Home Visit Cover": "No",
            "Out Patient Cover": "No",
            "Critical Illness Benefit": "No"
        }



def get_addon_covers(pdf_path):

    addon_covers = {
    "Addon Covers": {
        "Ambulance Cover": "",
        "Anyone Illness": "No",
        "Attendant Care": "No",
        "Cancer Cover": "No",
        "Convalescence Benefit": "",
        "Critical Illness Benefit": "",
        "Daily/Hospital Cash Benefit": "",
        "Dental Cover": "No",
        "Diabetic Cover": "No",
        "Doctor & Nurse Home Visit Cover": "",
        "Education Fund": "No",
        "Funeral": "No",
        "Getwell Benefit": "No",
        "Hardship Critical Illness Cover": "No",
        "Health Check up": "No",
        "Hypertension Cover": "No",
        "Intensive Care Benefit": "No",
        "Loss Of Pay Cover": "No",
        "Medical Evacuation Cover": "No",
        "Medical Second Opinion": "No",
        "Non Medical Expense Cover": "No",
        "Out Patient Cover": "",
        "Optical Cover": "No",
        "Organ Donor Medical Expense Cover": "No",
        "Personal Accident Cover": "No",
        "Pre Existing Disease Benefit": "No",
        "Psychiatric Cover": "No",
        "Recovery Benefit": "No",
        "Referral Hospital Care": "No",
        "Surgical Benefit": "No",
        "Top Up Cover": "No",
        "Vaccination/Immunization Cover": "No"
    }
}

    
    #end_and_spl_cond=get_endorsements_and_spl_cond(pdf_path)
    end_and_spl_cond=get_endorsements_content(pdf_path,desired_endorsements = ['16', '15', '14', '17','7', '20'])
    extracted_addons=extract_addon_covers(end_and_spl_cond)
    for key, value in extracted_addons.items():
     if key in addon_covers["Addon Covers"]:
        addon_covers["Addon Covers"][key] = value
    
    return addon_covers




def run_addon_covers(pdf_path,source_dir):
    addon_covers=get_addon_covers(pdf_path)
    df = pd.DataFrame([addon_covers['Addon Covers']])

    # Save to Excel


    output_folder = source_dir  # define your working folder
    os.makedirs(output_folder, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(pdf_path))[0]  
    excel_filename = os.path.join(output_folder, f"{base_name}_addon_covers.xlsx")
    json_filename=os.path.join(output_folder, f"{base_name}_addon_covers.json")
    df.to_json(json_filename, orient="records", indent=4)
    

    # Save to Excel
    df.to_excel(excel_filename, index=False)




if __name__ =="__main__":
    pdf_path=r""
    source_dir=r""
    run_addon_covers(pdf_path,source_dir)
    