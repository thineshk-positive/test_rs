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
from hscope_uat.helper.get_si import load_sum_insured_from_json
from hscope_uat.UAT_config import *
import os



def pdf_to_md(pdf_path):
    md_text = pymupdf4llm.to_markdown(pdf_path)
    return md_text


from langchain_ollama import ChatOllama

chat = ChatOllama(
    model="gpt-oss:20b",
    temperature=0.0,
    top_p=1.0,
    max_tokens=1024,
    verbose=False
)



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


import os
import json

def load_TPA(pdf_path):
    """
    Loads the corresponding JSON for a given PDF.
    Prefers <basename>_mineru_2.json, falls back to <basename>_pymupdf4llm_2.json.
    """
    # Step 1: Extract base name (without extension)
    base_name = os.path.splitext(pdf_path)[0]
    
    # Step 2: Form two JSON filenames
    mineru_json = f"{base_name}_mineru_2.json"
    pymupdf_json = f"{base_name}_pymupdf4llm_2.json"

    # Step 3: Choose which file exists
    json_path = None
    if os.path.exists(mineru_json):
        json_path = mineru_json
    elif os.path.exists(pymupdf_json):
        json_path = pymupdf_json
    else:
        raise FileNotFoundError(f"No matching JSON found for {pdf_path}")

    # Step 4: Load JSON file
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"✅ Loaded JSON file: {json_path}")
    return data




def find_TPA_presence(pdf_path):
    data=load_TPA(pdf_path)
    if data['Product Setup']['TPA Details']['Domestic Claims']!="":
        return "YES"
    else:
        return "NO"
    


def extract_claims_feild(pdf_path):


    claims={"Claims Condition": {
        "Claim Lock In Period Applicable": "",
        "Lock In Period (In Days)": "",
        "Claim Lock In Period Applicable for Critical Illness": "",
        "Lock In Period for Critical Illness (In Days)": "",
        "Cashless Applicable?": "",
        "Reimbursement Applicable?": ""
        }
    } 


    TPA_YES_OR_NO=find_TPA_presence(pdf_path)
    claims['Claims Condition']['Claim Lock In Period Applicable']="NO"
    claims['Claims Condition']['Lock In Period (In Days)']=""
    claims['Claims Condition']['Claim Lock In Period Applicable for Critical Illness']="NO"
    claims['Claims Condition']['Lock In Period for Critical Illness (In Days)']= ""
    claims['Claims Condition']['Cashless Applicable?']=TPA_YES_OR_NO
    if claims['Claims Condition']['Cashless Applicable?']=="YES":
        claims['Claims Condition']['Reimbursement Applicable?']='NO'
    else:
        claims['Claims Condition']['Reimbursement Applicable?']='YES'
    return claims

"""
def cashless_and_reimbursement_fields(claims):
    

    # -------------------- Cashless --------------------
    Cashless_Normal = {
        "Cashless_Normal": {
            "Claim Intimation Required?": "",
            "Before Date Of Admission": "",
            "After Date Of Admission": "",
            "Is Copay Applicable for delay in Intimation?": "",
            "From Day": "",
            "To Day": "",
            "Copay Type": "",
            "Copay": ""
        }
    }

    Cashless_Emergency = {
        "Cashless_Emergency": {
            "Claim Intimation Required?": "",
            "Before Date Of Admission": "",
            "After Date Of Admission": "",
            "Is Copay Applicable for delay in Intimation?": "",
            "From Day": "",
            "To Day": "",
            "Copay Type": "",
            "Copay": ""
        }
    }

    Cashless_Claim_Submission_Within = {
        "Cashless_Claim_Submission_Within":"", 
            "Is Copay Applicable for Cashless submission?": "",
            "From Day": "",
            "To Day": "",
            "Copay Type": "",
            "Copay": ""
        
    }

    # -------------------- Reimbursement --------------------
    Reimbursement_Normal = {
        "Reimbursement_Normal": {
            "Claim Intimation Required?": "",
            "Before Date Of Admission": "",
            "After Date Of Admission": "",
            "Is Copay Applicable for delay in Intimation?": "",
            "From Day": "",
            "To Day": "",
            "Copay Type": "",
            "Copay": ""
        }
    }

    Reimbursement_Emergency = {
        "Reimbursement_Emergency": {
            "Claim Intimation Required?": "",
            "Before Date Of Admission": "",
            "After Date Of Admission": "",
            "Is Copay Applicable for delay in Intimation?": "",
            "From Day": "",
            "To Day": "",
            "Copay Type": "",
            "Copay": ""
        }
    }

    Reimbursement_Claim_Submission_Within = {
        "Reimbursement_Claim_Submission_Within": "",
            "Is Copay Applicable for delay in Reimbursement claim submission?": "",
            "From Day": "",
            "To Day": "",
            "Copay Type": "",
            "Copay": ""
        
    }

    # -------------------- Logic --------------------
    cashless_applicable = claims.get('Claims Condition', {}).get('Cashless Applicable?', 'NO').upper()

    if cashless_applicable == "YES":
        # Cashless Normal
        Cashless_Normal["Cashless_Normal"].update({
            "Claim Intimation Required?": "YES",
            "Before Date Of Admission": 7,
            "After Date Of Admission": 2,
            "Is Copay Applicable for delay in Intimation?": "NO",
        })

        # Cashless Emergency
        Cashless_Emergency["Cashless_Emergency"].update({
            "Claim Intimation Required?": "YES",
            "Before Date Of Admission": 0,
            "After Date Of Admission": 2,
            "Is Copay Applicable for delay in Intimation?": "NO",
        })

        # Cashless Claim Submission
        Cashless_Claim_Submission_Within["Cashless_Claim_Submission_Within"].update({
            "Is Copay Applicable for Cashless submission?": "NO"
        })

    else:
        # Reimbursement Normal
        Reimbursement_Normal["Reimbursement_Normal"].update({
            "Claim Intimation Required?": "YES",
            "Before Date Of Admission": 7,
            "After Date Of Admission": 2,
            "Is Copay Applicable for delay in Intimation?": "NO",

        })

        # Reimbursement Emergency
        Reimbursement_Emergency["Reimbursement_Emergency"].update({
            "Claim Intimation Required?": "YES",
            "Before Date Of Admission": 0,
            "After Date Of Admission": 2,
            "Is Copay Applicable for delay in Intimation?": "NO",
 
        })

        # Reimbursement Claim Submission
        Reimbursement_Claim_Submission_Within["Reimbursement_Claim_Submission_Within"].update({
            "Is Copay Applicable for delay in Reimbursement claim submission?": "NO"
        })

    # -------------------- Return All --------------------
    return (
        Cashless_Normal,
        Cashless_Emergency,
        Cashless_Claim_Submission_Within,
        Reimbursement_Normal,
        Reimbursement_Emergency,
        Reimbursement_Claim_Submission_Within
    )

#okay version.
"""



# -------------------- Cashless --------------------
def create_dict(var):
    fields = {
        f"{var}_Claim Intimation Required?": "",
        f"{var}_Before Date Of Admission": "",
        f"{var}_After Date Of Admission": "",
        f"{var}_Is Copay Applicable for delay in Intimation?": "",
        f"{var}_From Day": "",
        f"{var}_To Day": "",
        f"{var}_Copay Type": "",
        f"{var}_Copay": ""
    }
    return fields


def cashless_and_reimbursement_fields(claims):
    """
    Create structured field dictionaries for Cashless and Reimbursement claim processes
    using the create_dict() template, and fill according to Cashless/Reimbursement logic.
    """

    # -------------------- Generate field templates --------------------
    cashless_normal_fields = create_dict('Cashless_Normal')
    cashless_emergency_fields = create_dict('Cashless_Emergency')
    reimb_normal_fields = create_dict('Reimbursement_Normal')
    reimb_emergency_fields = create_dict('Reimbursement_Emergency')

    # -------------------- Initialize dicts --------------------
    Cashless_Normal = {"Cashless_Normal": "", **cashless_normal_fields}
    Cashless_Emergency = {"Cashless_Emergency": "", **cashless_emergency_fields}
    Cashless_Claim_Submission_Within = {
        "Cashless_Claim_Submission_Within": "",
        "Is Copay Applicable for Cashless submission?": "",
        "From Day": "",
        "To Day": "",
        "Copay Type": "",
        "Copay": ""
    }

    Reimbursement_Normal = {"Reimbursement_Normal": "", **reimb_normal_fields}
    Reimbursement_Emergency = {"Reimbursement_Emergency": "", **reimb_emergency_fields}
    Reimbursement_Claim_Submission_Within = {
        "Reimbursement_Claim_Submission_Within": "",
        "Is Copay Applicable for delay in Reimbursement claim submission?": "",
        "From Day": "",
        "To Day": "",
        "Copay Type": "",
        "Copay": ""
    }

    # -------------------- Logic --------------------
    cashless_applicable = claims.get('Claims Condition', {}).get('Cashless Applicable?', 'NO').upper()

    if cashless_applicable == "YES":
        # --- Cashless Normal ---
        Cashless_Normal['Cashless_Normal'] = "YES"
        Cashless_Normal["Cashless_Normal_Claim Intimation Required?"] = "YES"
        Cashless_Normal["Cashless_Normal_Before Date Of Admission"] = 7
        Cashless_Normal["Cashless_Normal_After Date Of Admission"] = 2
        Cashless_Normal["Cashless_Normal_Is Copay Applicable for delay in Intimation?"] = "NO"

        # --- Cashless Emergency ---
        Cashless_Emergency['Cashless_Emergency'] = "YES"
        Cashless_Emergency["Cashless_Emergency_Claim Intimation Required?"] = "YES"
        Cashless_Emergency["Cashless_Emergency_Before Date Of Admission"] = 0
        Cashless_Emergency["Cashless_Emergency_After Date Of Admission"] = 2
        Cashless_Emergency["Cashless_Emergency_Is Copay Applicable for delay in Intimation?"] = "NO"

        # --- Cashless Claim Submission ---
        Cashless_Claim_Submission_Within["Cashless_Claim_Submission_Within"] = 30
        Cashless_Claim_Submission_Within["Is Copay Applicable for Cashless submission?"] = "NO"

    else:
        # --- Reimbursement Normal ---
        Reimbursement_Normal["Reimbursement_Normal"] = "YES"
        Reimbursement_Normal["Reimbursement_Normal_Claim Intimation Required?"] = "YES"
        Reimbursement_Normal["Reimbursement_Normal_Before Date Of Admission"] = 7
        Reimbursement_Normal["Reimbursement_Normal_After Date Of Admission"] = 2
        Reimbursement_Normal["Reimbursement_Normal_Is Copay Applicable for delay in Intimation?"] = "NO"

        # --- Reimbursement Emergency ---
        Reimbursement_Emergency["Reimbursement_Emergency"] = "YES"
        Reimbursement_Emergency["Reimbursement_Emergency_Claim Intimation Required?"] = "YES"
        Reimbursement_Emergency["Reimbursement_Emergency_Before Date Of Admission"] = 0
        Reimbursement_Emergency["Reimbursement_Emergency_After Date Of Admission"] = 2
        Reimbursement_Emergency["Reimbursement_Emergency_Is Copay Applicable for delay in Intimation?"] = "NO"

        # --- Reimbursement Claim Submission ---
        Reimbursement_Claim_Submission_Within["Reimbursement_Claim_Submission_Within"] = 30
        Reimbursement_Claim_Submission_Within["Is Copay Applicable for delay in Reimbursement claim submission?"] = "NO"

    # -------------------- Return all dicts --------------------
    return (
        Cashless_Normal,
        Cashless_Emergency,
        Cashless_Claim_Submission_Within,
        Reimbursement_Normal,
        Reimbursement_Emergency,
        Reimbursement_Claim_Submission_Within
    )




def get_claims_and_conditions(pdf_path):
    print('10')
    md_text=pdf_to_md(pdf_path)
    claims=extract_claims_feild(pdf_path)
    Cashless_Normal,Cashless_Emergency,Cashless_Claim_Submission_Within,Reimbursement_Normal,Reimbursement_Emergency,Reimbursement_Claim_Submission_Within=cashless_and_reimbursement_fields(claims)
    filter_dict = claims["Claims Condition"].copy()
    
    # store before popping
    reimbursement_applicable = filter_dict.get("Reimbursement Applicable?", "")
    
    # remove only if it's not reimburse

# Remove 'Reimbursement Applicable?' if it exists
    reimbursement_applicable = filter_dict.pop("Reimbursement Applicable?", None)
    print(Cashless_Normal,Cashless_Emergency)

    


    return [
        filter_dict,
        Cashless_Normal,
        Cashless_Emergency,
        Cashless_Claim_Submission_Within,
        {'reimbursement_applicable':reimbursement_applicable},
        Reimbursement_Normal,
        Reimbursement_Emergency,
        Reimbursement_Claim_Submission_Within
    ]





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





import re
import json

def extract_ayush_limits(snippet: str, chat) -> dict:
    """
    Extract Ayush Limit details from an insurance policy snippet using ChatOllama.

    Returns a dict with:
        - Is Ayush Limit Applicable? (Yes/No)
        - Is Hospital Type Applicable? (Yes/No)

    Works by prompting LLM, then fallback to local keyword detection if LLM fails.
    """

    prompt = f"""
You are an expert insurance policy analyst. Extract the following information
from the provided text snippet regarding AYUSH limits.

Target JSON:
{{
  "Ayush Limit": {{
    "Is Ayush Limit Applicable?": "",
    "Is Hospital Type Applicable?": ""
  }}
}}

Instructions:
1. Determine "Is Ayush Limit Applicable?" as follows:
   - Mark as "Yes" if **any** of these conditions are true:
       a) The word "Ayush" appears anywhere in the text.
       b) The endorsement number "25" is explicitly mentioned.
       c) Any Non-Allopathic medicine is mentioned, including:
          Ayurveda, Yoga, Unani, Siddha, Homoeopathy.
   - Otherwise, mark as "No".

2. For "Is Hospital Type Applicable?":
   - If "Is Ayush Limit Applicable?" is "Yes", set this as "Yes".
   - Otherwise, set this as "No".

3. Respond strictly in valid JSON format as shown:
{{
  "Ayush Limit": {{
    "Is Ayush Limit Applicable?": "Yes" or "No",
    "Is Hospital Type Applicable?": "Yes" or "No"
  }}
}}

Text Snippet:
\"\"\"{snippet}\"\"\"
"""

    try:
        # ✅ Call the LLM
        raw_response = chat.invoke(prompt)

        # Robustly extract JSON from LLM output
        json_match = re.search(r"\{(?:[^{}]|(?R))*\}", raw_response, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            ayush_data = data.get("Ayush Limit", {})
            applicable = ayush_data.get("Is Ayush Limit Applicable?", "No")
            applicable = "Yes" if applicable.strip().lower() == "yes" else "No"
            hospital_applicable = "Yes" if applicable == "Yes" else "No"

            return {
                "Ayush Limit": {
                    "Is Ayush Limit Applicable?": applicable,
                    "Is Hospital Type Applicable?": hospital_applicable
                }
            }

        # Fallback: local detection if LLM output is unusable
        raise ValueError("No valid JSON detected from LLM")

    except Exception:
        # Local keyword-based fallback
        ayush_terms = ["ayush", "ayurveda", "yoga", "unani", "siddha", "homoeopathy"]
        ayush_applicable = any(term.lower() in snippet.lower() for term in ayush_terms) or "25" in snippet
        return {
            "Ayush Limit": {
                "Is Ayush Limit Applicable?": "Yes" if ayush_applicable else "No",
                "Is Hospital Type Applicable?": "Yes" if ayush_applicable else "No"
            }
        }




def get_endorsements_and_spl_cond(pdf_path,desired_endorsements = ['25']
):    
    print("11")
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



def get_ayush_feilds(pdf_path,chat):
    print('5')

    combined_text=get_endorsements_and_spl_cond(pdf_path,desired_endorsements=['25'])
    extracted_ayush_feilds=extract_ayush_limits(combined_text,chat)
    return extracted_ayush_feilds



def no_ayush_limit():
    Hospital_types=["In Government Hospital","In NABH Hospital","In Non-NABH Hospital"]
    Non_Alpothaic_medicine=["Ayurveda","Yoga","Unani","Siddha","Homoeopathy"]
    feilds_to_fiil= {
        "Sum Insured": "",
        "% Limit Applicable On": "",
        "Limit Percentage": "",
        "Limit Amount": "",
        "Applicability": ""
      }
    output_dict = {medicine: feilds_to_fiil.copy() for medicine in Non_Alpothaic_medicine}
    hospital_list={ hospital_type:output_dict.copy() for hospital_type in Hospital_types}
    return hospital_list



import re
import json

def extract_ayush_treatment_limits(snippet: str, si_list: list) -> dict:
    """
    Extract AYUSH (Non-Allopathic) Treatment limits and Sum Insured exclusions using only LLM.
    Handles % or ₹ values, or cases like 'Full Floater', 'Actuals', etc. (interpreted as 100%).
    Ensures stable structure and fills missing data safely.
    """

    prompt = f"""
You are an expert health insurance policy analyst.

Read the endorsement or special condition text below and extract:
1. AYUSH / Non-Allopathic Treatment limit (either % or ₹ per year/family)
2. Which Sum Insured (SI) amounts are excluded.

---
### Strict JSON Output Format:
{{
  "Sum Insured": [],
  "AYUSH Treatment": {{"Percentage": [], "Amount": []}},
  "Sum Insured Excluded": []
}}
---

### Rules:
- Use this reference Sum Insured list: {si_list}
- Extract only numeric values (e.g., 50 for 50%, 25000 for ₹25,000).
- If the text says **"Full Floater", "Actuals", "Full Sum Insured", "Full SI", or "up to Sum Insured"**, treat as **100%**.
- If the exclusion says "up to ₹3,00,000", exclude all ≤ 300000.
- If it says "not applicable for ₹4,00,000 and ₹5,00,000", exclude those specifically.
- For all remaining SIs, apply the extracted AYUSH limits.

### IMPORTANT DEFAULT RULE:
- If AYUSH or any alternative medicine term appears (Ayush, Ayurveda, Yoga, Unani, Siddha, Homoeopathy),
  but the text DOES NOT provide:
    - a percentage,
    - a fixed amount,
    - or words indicating full cover like:
      "Actuals", "Full", "Full SI", "Full Floater", "up to Sum Insured",
  → then assume AYUSH limit = 100%.

Respond strictly in valid JSON only. Do not provide explanations or extra text.


---
### Endorsement / Special Condition Text:
\"\"\"{snippet}\"\"\"
"""

    # ---- LLM Call ----
    response = chat.invoke(prompt)
    raw_response = getattr(response, "content", str(response))

    # Extract JSON safely
    match = re.search(r"\{[\s\S]*\}", raw_response)
    if not match:
        raise ValueError("No valid JSON found in LLM response.")
    data = json.loads(match.group())

    # ---- Ensure stable structure ----
    data.setdefault("Sum Insured Excluded", [])
    included_si = [si for si in si_list if si not in data["Sum Insured Excluded"]]
    data["Sum Insured"] = included_si

    # Guarantee nested structure exists
    data.setdefault("AYUSH Treatment", {})
    for field in ["Percentage", "Amount"]:
        vals = data["AYUSH Treatment"].get(field)
        if not isinstance(vals, list):
            vals = [] if vals is None else [vals]
        data["AYUSH Treatment"][field] = vals

    # ---- Normalize and clean numbers ----
    def clean_num(val):
        if val in [None, "", "null"]:
            return None
        try:
            # Treat "Actuals" / "Full Floater" / "Full SI" as 100%
            text_val = str(val).lower()
            if any(k in text_val for k in ["actual", "floater", "full", "sum insured"]):
                return 100.0
            v = re.search(r"(\d+(\.\d+)?)", str(val).replace(",", ""))
            return float(v.group(1)) if v else None
        except Exception:
            return None

    for field in ["Percentage", "Amount"]:
        vals = data["AYUSH Treatment"][field]
        cleaned = [clean_num(v) for v in vals]
        data["AYUSH Treatment"][field] = cleaned

    n = len(data["Sum Insured"])
    # Pad lists to match Sum Insured count
    for field in ["Percentage", "Amount"]:
        vals = data["AYUSH Treatment"][field]
        if len(vals) == 0:
            vals = [None] * n
        elif len(vals) == 1:
            vals = vals * n
        elif len(vals) < n:
            vals += [vals[-1]] * (n - len(vals))
        data["AYUSH Treatment"][field] = vals

    # ---- Fill forward safely ----
    def fill_forward(lst):
        if not lst:
            return []
        filled = []
        last_val = None
        for x in lst:
            if x is not None:
                last_val = x
            filled.append(last_val)
        # backfill first if missing
        if filled and filled[0] is None:
            first_val = next((x for x in filled if x is not None), None)
            filled = [first_val if v is None else v for v in filled]
        return filled

    for field in ["Percentage", "Amount"]:
        data["AYUSH Treatment"][field] = fill_forward(data["AYUSH Treatment"][field])

    return data


def calculate_missing_ayush(data: dict) -> dict:
    """
    Fill missing AYUSH % or Amount using the corresponding Sum Insured.
    Assumes 'Sum Insured' are strings like '1,00,000'.
    """
    def parse_si(si_str):
        """Convert '1,00,000' → 100000.0"""
        return float(si_str.replace(",", "").strip())

    #si_values = [parse_si(s) for s in data["Sum Insured"]]
    si_values = [s for s in data["Sum Insured"]]

    perc_list = data["AYUSH Treatment"]["Percentage"]
    amt_list = data["AYUSH Treatment"]["Amount"]

    for i in range(len(si_values)):
        si = si_values[i]
        perc = perc_list[i]
        amt = amt_list[i]

        # --- Case 1: % missing, Amount present ---
        if perc is None and amt is not None:
            perc_list[i] = round((amt / si) * 100, 2)

        # --- Case 2: Amount missing, % present ---
        elif amt is None and perc is not None:
            amt_list[i] = int(round(si * perc / 100, 0))

    data["AYUSH Treatment"]["Percentage"] = perc_list
    data["AYUSH Treatment"]["Amount"] = amt_list

    return data



def map_ayush_limits_to_schema(extracted_result: dict) -> dict:
    """
    Maps the extracted AYUSH (Non-Allopathic) Treatment limit details 
    into the standard schema format.
    
    Parameters:
        extracted_result (dict): The dictionary output from extract_ayush_limits().
    
    Returns:
        dict: Mapped schema with AYUSH Treatment limits.
    """

    si_list = extracted_result.get('Sum Insured', [])
    ayush_percentage = extracted_result.get('AYUSH Treatment', {}).get('Percentage', [])
    ayush_amount = extracted_result.get('AYUSH Treatment', {}).get('Amount', [])

    # Build AYUSH Treatment schema
    ayush_treatment = {
        "Sum Insured": si_list,
        "% Limit Applicable On": ["Sum Insured"] * len(si_list),
        "Limit Percentage": ayush_percentage,
        "Limit Amount": ayush_amount,
        "Applicability": ["Lower"] * len(si_list)
    }

    # Final standardized schema
    schema_mapped_result = {
        "AYUSH Treatment": ayush_treatment
    }

    return schema_mapped_result



def get_ayush_treatment_fields(pdf_path,si_list):

    print('4')
    Hospital_types=["In Government Hospital","In NABH Hospital","In Non-NABH Hospital"]
    Non_Alpothaic_medicine=["Ayurveda","Yoga","Unani","Siddha","Homoeopathy"]
    endnt_and_spl_context=get_endorsements_and_spl_cond(pdf_path,desired_endorsements = ['25'])
    result=extract_ayush_treatment_limits(endnt_and_spl_context,si_list)
    calc_result=calculate_missing_ayush(result)
    map_result=map_ayush_limits_to_schema(calc_result)
    
    output_dict = {medicine: map_result['AYUSH Treatment'].copy() for medicine in Non_Alpothaic_medicine}
    hospitals={ hospital_type:output_dict.copy() for hospital_type in Hospital_types}

    required_hospitals=hospitals.copy()

    non_nabh=["In Non-NABH Hospital"]

    feilds_to_fiil= {
        "Sum Insured": "",
        "% Limit Applicable On": "",
        "Limit Percentage": "",
        "Limit Amount": "",
        "Applicability": ""
      }
    
    output_dict = {medicine: feilds_to_fiil.copy() for medicine in Non_Alpothaic_medicine}
    hospital_list={ hospital_type:output_dict.copy() for hospital_type in non_nabh}
    

    non_nabh_list=hospital_list.copy()

    hospital_list={**required_hospitals,**non_nabh_list}

    return hospital_list    


def run_ayush(pdf_path,si_list,chat):
    print('3')

    result=get_ayush_feilds(pdf_path,chat)

    if result['Ayush Limit']['Is Ayush Limit Applicable?']=='Yes':
        ayush_result=get_ayush_treatment_fields(pdf_path,si_list)
        ayush={"ayush_applicable":result,"ayush_result":ayush_result}


    else:
        Ayush_limits={"Ayush Limit": {
        "Is Ayush Limit Applicable?": "No",
        "Is Hospital Type Applicable?": "No"}}
        
        ayush_result=no_ayush_limit()
        
        ayush={"ayush_applicable":Ayush_limits,"ayush_result":ayush_result}
    return ayush




def run_claim_conditons(pdf_path,si_list,chat):

    claim_and_conditons_result=get_claims_and_conditions(pdf_path)
    print("1")
    ayush=run_ayush(pdf_path,si_list,chat)
    print("2")

    merged_dict = {}
    for d in claim_and_conditons_result:
        merged_dict.update(d)

    all_feilds=(merged_dict,ayush)
    
    return all_feilds





import pandas as pd

def flatten_policy_dict(policy_dict):
    flat = {key: policy_dict.get(key, '') for key in policy_dict}

    # Convert to DataFrame
    df = pd.DataFrame([flat])
    return df

def save_ayush_excel(data):

    # Prepare empty list for rows
    rows = []

    hospitals = ['In Government Hospital', 'In NABH Hospital', 'In Non-NABH Hospital']
    ayush_types = ['Ayurveda','Yoga','Unani','Siddha','Homoeopathy']

    # Loop through hospitals
    for i in range(5):  # max 5 rows per AYUSH type
        row = {}
        row['Is Ayush Limit Applicable?'] = data['ayush_applicable']['Ayush Limit']['Is Ayush Limit Applicable?']
        row['Is Hospital type Applicable?'] = data['ayush_applicable']['Ayush Limit']['Is Hospital Type Applicable?']
        
        for hosp in hospitals:
            hosp_data = data['ayush_result'][hosp]
            for ayush in ayush_types:
                ayush_data = hosp_data[ayush]
                # Check if list or empty scalar
                if isinstance(ayush_data['Sum Insured'], list):
                    idx = i if i < len(ayush_data['Sum Insured']) else -1
                    row[f'{hosp}_{ayush}_Sum insured'] = ayush_data['Sum Insured'][idx] if idx >= 0 else ''
                    row[f'{hosp}_{ayush}_% Limit Applicable On'] = ayush_data['% Limit Applicable On'][idx] if idx >= 0 else ''
                    row[f'{hosp}_{ayush}_Limit Percentage'] = ayush_data['Limit Percentage'][idx] if idx >= 0 else ''
                    row[f'{hosp}_{ayush}_Limit Amount'] = ayush_data['Limit Amount'][idx] if idx >= 0 else ''
                    row[f'{hosp}_{ayush}_Applicability'] = ayush_data['Applicability'][idx] if idx >= 0 else ''
                else:
                    row[f'{hosp}_{ayush}_Sum insured'] = ayush_data['Sum Insured']
                    row[f'{hosp}_{ayush}_% Limit Applicable On'] = ayush_data['% Limit Applicable On']
                    row[f'{hosp}_{ayush}_Limit Percentage'] = ayush_data['Limit Percentage']
                    row[f'{hosp}_{ayush}_Limit Amount'] = ayush_data['Limit Amount']
                    row[f'{hosp}_{ayush}_Applicability'] = ayush_data['Applicability']
        
        rows.append(row)

    # Create DataFrame
    df = pd.DataFrame(rows)

    single_value_cols = ['Is Ayush Limit Applicable?', 'Is Hospital type Applicable?']

    # Fill those columns only in the first row, and make rest rows blank
    for col in single_value_cols:
        df.loc[1:, col] = ''   # Blank out rows except first one

    # Save to Excel
    #df.to_excel('ayush_policy.xlsx', index=False)


    #print("Excel file created successfully!")
    return df



def save_claim_codntion_as_excel(data,pdf_path,source_dir):
        

    df=flatten_policy_dict(data[0])
    df_2=save_ayush_excel(data[1])
    # Concatenate them (stack one below another)


    df1 = df.reset_index(drop=True)
    df2 = df_2.reset_index(drop=True)

    # Merge them side-by-side (same row level)
    final_df = pd.concat([df1, df2], axis=1)

    output_folder = source_dir # define your working folder
    os.makedirs(output_folder, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(pdf_path))[0]  
    excel_filename = os.path.join(output_folder, f"{base_name}_claim_and_conditons.xlsx")

    # Save to Excel
    final_df.to_excel(excel_filename, index=False)




def run_and_save_claim_conditons(pdf_path,chat,source_dir):
    si_list=load_sum_insured_from_json(pdf_path)
    print(1)
    data=run_claim_conditons(pdf_path,si_list,chat)
    save_claim_codntion_as_excel(data,pdf_path,source_dir)
    

if __name__ =="__main__":
    si_list = ['1,00,000', '2,00,000', '3,00,000', '4,00,000', '5,00,000']
    pdf_path=r"/home/ubuntu/THINESH_WS/royal_sunadaram_hscope/samples/HG Cases/HG00000006000124.pdf"
    source_dir=r""

    run_and_save_claim_conditons(pdf_path,chat,source_dir)


