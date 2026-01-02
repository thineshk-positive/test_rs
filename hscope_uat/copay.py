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


import os
from hscope_uat.UAT_config import *


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





def get_endorsements_and_spl_cond(pdf_path,desired_endorsements = ['8']
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




# new version

import re
import json

def extract_copay_category(endorsement_text: str) -> dict:
    """
    Identify which ONE Co-Pay category (Relationship, Ailment, Age Based, etc.)
    applies based on the endorsement text using an LLM.

    Returns a dict with exactly ONE 'Yes' (the highest match) and all others 'No'.
    """

    prompt = f"""
You are an expert insurance policy analyst.
Your task is to identify which **ONE** Co-Pay category best applies from the given endorsement text.

Co-Pay categories to choose from:
- Relationship
- Claim Frequency
- Ailment
- Type of Provider
- Relationship and Ailment
- Age Based
- Claim Amount
- Capped Ailment

Rules:
1. Only one category can be "Yes" — the one that best fits the text.
2. All other categories must be "No".
3. If nothing clearly matches, return all "No".
4. Examples:
   - If text says "20% co-pay for parents" → Relationship = Yes
   - If it says "based on age" → Age Based = Yes
   - If it says "per claim basis or depends on number of claims" → Claim Frequency = Yes
   - If it mentions specific ailments (like "cardiac" or "diabetes") → Ailment = Yes
   - If it mentions "network hospital vs non-network" → Type of Provider = Yes
   - If both relationship and ailment factors appear (e.g., "parents with heart disease") → Relationship and Ailment = Yes
   - If it mentions claim amount thresholds ("claims above ₹50,000") → Claim Amount = Yes
   - If it caps co-pay by ailment ("max ₹20,000 for knee replacement") → Capped Ailment = Yes

Respond ONLY with JSON like this:
{{
  "Relationship": "",
  "Claim Frequency": "",
  "Ailment": "",
  "Type of Provider": "",
  "Relationship and Ailment": "",
  "Age Based": "",
  "Claim Amount": "",
  "Capped Ailment": ""
}}

Set exactly ONE field to "Yes" — the most relevant one — and all others to "No".

Endorsement text:
{endorsement_text}
"""

    try:
        # Query the LLM (replace this with your LLM API call)
        raw_response = chat.invoke(prompt)
        if hasattr(raw_response, "content"):
            raw_response = raw_response.content

        # Extract JSON portion from LLM response
        match = re.search(r"\{[\s\S]*\}", raw_response)
        if not match:
            raise ValueError("No JSON found in LLM response")

        data = json.loads(match.group())

        # Sanitize and ensure only one "Yes"
        categories = [
            "Relationship",
            "Claim Frequency",
            "Ailment",
            "Type of Provider",
            "Relationship and Ailment",
            "Age Based",
            "Claim Amount",
            "Capped Ailment"
        ]

        # Normalize outputs
        result = {cat: "No" for cat in categories}
        yes_fields = [cat for cat, val in data.items() if str(val).strip().lower() == "yes"]

        # Keep only one 'Yes' (first if multiple)
        if yes_fields:
            result[yes_fields[0]] = "Yes"

        return result

    except Exception:
        # Default fallback
        return {
            "Relationship": "No",
            "Claim Frequency": "No",
            "Ailment": "No",
            "Type of Provider": "No",
            "Relationship and Ailment": "No",
            "Age Based": "No",
            "Claim Amount": "No",
            "Capped Ailment": "No"
        }


import re
import json
import re
import json

def extract_all_ailments(text: str) -> dict:
    """
    LLM-based extraction for multiple ailment names, their % or flat limits.
    Ensures all ailments are captured and 'etc.' is removed.
    Ignores generic terms like "robotic surgery" but captures specific treatments like "Cyberknife".
    """

    prompt = f"""
You are an expert health insurance policy analyst.
Read the policy text below **line by line** and extract **all explicit ailment/treatment names** and their co-pay or limits.

Return JSON ONLY in this format:

{{
  "Ailment Name": [],
  "Limit Applicable On": "",
  "% Limit": [],
  "Limit": [],
  "Applicability": ""
}}

Rules:
1. Extract every visible ailment or treatment name. Remove 'etc.' from the names.
2. If a line lists multiple treatments (e.g., 'like Cyberknife, Gamma Knife'), split them into separate names.
3. Generic or broad terms like "robotic surgery" can be ignored.
4. Specific treatment names (e.g., Cyberknife, Gamma Knife, Stent Placement) should always be included.
5. "Limit Applicable On" should always be "Sum Insured".
6. Extract % co-pay as numeric values and place them in the same order as the ailments.
7. Extract flat amounts (e.g., Rs 50,000) into "Limit" list, aligned with the ailments.
8. "Applicability" is always "Lower".
9. If no % or limit is mentioned for an ailment, leave empty string in that position.
10. **Ensure every specific ailment in the text is captured**, not just the last one.

Text to analyze:
{text}

Respond ONLY in valid JSON in the format above.
"""

    try:
        # Call your LLM here (replace 'chat.invoke' with your model call)
        raw_response = chat.invoke(prompt)
        if hasattr(raw_response, "content"):
            raw_response = raw_response.content

        # Extract JSON portion
        match = re.search(r"\{[\s\S]*\}", raw_response)
        if not match:
            return {
                "Ailment Name": [],
                "Limit Applicable On": "Sum Insured",
                "% Limit": [],
                "Limit": [],
                "Applicability": "Lower"
            }

        data = json.loads(match.group())

        # Normalize fields and apply defaults
        result = {
            "Ailment Name": data.get("Ailment Name", []),
            "Limit Applicable On": data.get("Limit Applicable On", "Sum Insured") or "Sum Insured",
            "% Limit": data.get("% Limit", []),
            "Limit": data.get("Limit", []),
            "Applicability": data.get("Applicability", "Lower") or "Lower"
        }

        return result

    except Exception:
        return {
            "Ailment Name": [],
            "Limit Applicable On": "Sum Insured",
            "% Limit": [],
            "Limit": [],
            "Applicability": "Lower"
        }



def expand_ailment_limits(ailment_json: dict, si_list: list) -> list:
    """
    Expands ailment co-pay details based on sum insured values.
    Each ailment will repeat for every sum insured with calculated limit.
    """

    expanded_records = []

    ailment_names = ailment_json.get("Ailment Name", [])
    percent_limits = ailment_json.get("% Limit", [])
    flat_limits = ailment_json.get("Limit", [])
    limit_on = ailment_json.get("Limit Applicable On", "Sum Insured")
    applicability = ailment_json.get("Applicability", "Lower")

    for i, ailment in enumerate(ailment_names):
        percent = percent_limits[i] if i < len(percent_limits) else ""
        limit = flat_limits[i] if i < len(flat_limits) else ""

        for si in si_list:
            if percent and str(percent).strip():
                calc_limit = round((float(percent) / 100) * si, 2)
            elif limit and str(limit).strip():
                try:
                    calc_limit = float(str(limit).replace(",", ""))
                except:
                    calc_limit = ""
            else:
                calc_limit = ""

            expanded_records.append({
                "Ailment Name": ailment,
                "Limit Applicable On": limit_on,
                "% Limit": percent,
                "Limit": calc_limit,
                "Applicability": applicability
            })

    return expanded_records


def copay_ailment(endnt_and_spl_cond,si_list):
    result=extract_all_ailments(endnt_and_spl_cond)
    result=expand_ailment_limits(result,si_list)

    return result
import re
import json

def detect_endorsement_8(endorsement_text: str) -> dict:
    """
    Detect whether 'Endorsement No. 8' (or equivalent phrases) is mentioned
    in the given endorsement text using LLM + regex fallback.

    Returns:
        {
            "Endorsement No. 8 Applicable": "Yes" or "No"
        }
    """

    prompt = f"""
You are an expert in insurance policy interpretation.
Your task is to determine whether the given endorsement text refers to
**Endorsement No. 8** or any equivalent form (like "Endt No. 8", "Endorsement 8", "Endt-8", etc.)

Rules:
1. Return "Yes" only if there is a clear mention of endorsement number 8 in any textual form.
2. Even if it appears as part of a phrase like "applicable as per Endt No. 8" or "refer Endorsement 8", mark it as Yes.
3. If no such reference exists or the number is different, return "No".
4. Return only JSON with this exact structure:
{{
  "Endorsement No. 8 Applicable": ""
}}

Endorsement text:
{endorsement_text}
"""

    try:
        # Query the LLM (replace this with your actual LLM call, e.g., chat.invoke)
        raw_response = chat.invoke(prompt)
        if hasattr(raw_response, "content"):
            raw_response = raw_response.content

        # Try to parse JSON from response
        match = re.search(r"\{[\s\S]*\}", raw_response)
        if match:
            result = json.loads(match.group())
            val = result.get("Endorsement No. 8 Applicable", "No").strip().lower()
            return {"Endorsement No. 8 Applicable": "Yes" if val == "yes" else "No"}

        # If LLM fails, fallback to regex detection
        pattern = r"\b(endorsement|endt)[\s\.\-]*(no\.?|number)?[\s\-]*8\b"
        if re.search(pattern, endorsement_text, re.IGNORECASE):
            return {"Endorsement No. 8 Applicable": "Yes"}
        else:
            return {"Endorsement No. 8 Applicable": "No"}

    except Exception:
        # Regex fallback in case of any failure
        pattern = r"\b(endorsement|endt)[\s\.\-]*(no\.?|number)?[\s\-]*8\b"
        if re.search(pattern, endorsement_text, re.IGNORECASE):
            return {"Endorsement No. 8 Applicable": "Yes"}
        return {"Endorsement No. 8 Applicable": "No"}
def check_copay_applicable_and_copay_category(pdf_path):
    desired_endorsements=['8']
    endt_context=get_endorsements_content(pdf_path,desired_endorsements)
    data=detect_endorsement_8(endt_context)

    endnt_and_spl_cond=get_endorsements_and_spl_cond(pdf_path)
    result=extract_copay_category(endnt_and_spl_cond)

    copay_applicable={"copay_applicable":""}

    if data['Endorsement No. 8 Applicable']=="Yes":
        all_no = all(v == "No" for v in list(result.values()))

        if all_no==True:
            copay_applicable['copay_applicable']='No'

        else:
            copay_applicable['copay_applicable']='Yes'
    else:
            copay_applicable['copay_applicable']='No'

    return copay_applicable, result


from collections import defaultdict

def convert_ailment_to_dict_of_lists(data):
    if "Ailment" not in data or not isinstance(data["Ailment"], list):
        return data  # nothing to transform

    ailment_list = data["Ailment"]
    grouped = defaultdict(list)

    for entry in ailment_list:
        for key, value in entry.items():
            grouped[key].append(value)

    data["Ailment"] = dict(grouped)
    return data


def run_copay(pdf_path,si_list):

    copay_applicable,copay_cat=check_copay_applicable_and_copay_category(pdf_path)
    endnt_and_spl_cond=get_endorsements_and_spl_cond(pdf_path,desired_endorsements=['8'])


    co_pay_schema = {
    "Co-Pay": {
    "Relationship": {
        "Relationship": "",
        "Co-pay Applicable": "",
        "Co-pay Type": "",
        "Co-pay": "",
    },
    "Claim Frequency": {
        "Frequency From": "",
        "Frequency To": "",
        "Co-pay Applicable": "",
        "Co-pay Type": "",
        "Co-pay": "",
    },
    "Ailment": {
        "Ailment Name": "",
        "Limit Applicable On": "",
        "% Limit": "",
        "Limit": "",
        "Applicability": "",
    },
    "Type of Provider": {
        "Reimbursement in network hospital": {
            "% Limit Applicable On": "",
            "Limit Percentage": "",
            "Limit Amount": "",
            "Applicability": ""
        },
        "Reimbursement in non network hospital": {
            "% Limit Applicable On": "",
            "Limit Percentage": "",
            "Limit Amount": "",
            "Applicability": ""
        }
    },
    "Relationship and Ailment": {
        "Relationship": "",
        "Ailment Name": "",
        "Limit Applicable On": "",
        "% Limit": "",
        "Limit": "",
        "Applicability": ""
    },
    "Age Based": {
        "Age From": "",
        "Age To": "",
        "Co-Pay Applicable?": "",
        "Co-pay Type": "",
        "Co-Pay": ""
    },
    "Claim Amount": {
        "Claim Amount From": "",
        "Claim Amount To": "",
        "Co-Pay Applicable": "",
        "Co-pay Type": "",
        "Co-Pay": ""
    },
    "Capped Ailment": {
        "Ailment Name": "",
        "% Limit": "",
        "Limit": "",
        "Applicability": ""
    },
    
}}


    if copay_applicable['copay_applicable']=="Yes":
        
        yes_keys = [k for k, v in copay_cat.items() if v.lower() == "yes"]
        yes_keys=yes_keys[0]
        copay_feilds={
                "Co-Pay Applicable?": copay_applicable['copay_applicable'],
                "Co-Pay Based On": yes_keys,
            }


        if copay_cat['Ailment']=="Yes":
            result=copay_ailment(endnt_and_spl_cond,si_list)
            result={'Ailment':result}
            result=convert_ailment_to_dict_of_lists(result)
            #result=result['Ailment']

            copay_cat_type = [
        'Relationship', 'Claim Frequency', 
        'Type of Provider', 'Relationship and Ailment',
        'Age Based', 'Claim Amount', 'Capped Ailment'
    ]

            copay_not_applicable = {
        cat: co_pay_schema["Co-Pay"][cat]
        for cat in copay_cat_type
        if cat in co_pay_schema["Co-Pay"]
    }

                        

            return {**copay_feilds,**result,**copay_not_applicable}
            

        else:
            copay_cat_type = [
        'Relationship', 'Claim Frequency', 'Ailment',
        'Type of Provider', 'Relationship and Ailment',
        'Age Based', 'Claim Amount', 'Capped Ailment'
    ]

            copay_not_applicable = {
        cat: co_pay_schema["Co-Pay"][cat]
        for cat in copay_cat_type
        if cat in co_pay_schema["Co-Pay"]
    }
    
            return {**copay_feilds,**copay_not_applicable}

    else:
        copay_feilds={
                "Co-Pay Applicable?": copay_applicable['copay_applicable'],
                "Co-Pay Based On": "",
            }
        
        copay_cat_type = [
        'Relationship', 'Claim Frequency', 'Ailment',
        'Type of Provider', 'Relationship and Ailment',
        'Age Based', 'Claim Amount', 'Capped Ailment'
    ]

        copay_not_applicable = {
        cat: co_pay_schema["Co-Pay"][cat]
        for cat in copay_cat_type
        if cat in co_pay_schema["Co-Pay"]
    }
    
        return {**copay_feilds,**copay_not_applicable}
        
        
import re
import json

def detect_deductible(endorsement_text: str) -> dict:
    """
    Detect whether the endorsement text mentions anything about 'Deductible'.

    Returns:
        {
            "Deductible": {
                "Deductible Applicable?": "Yes" or "No",
                "Deductible": ""
            }
        }
    """

    prompt = f"""
You are an expert in interpreting insurance endorsement clauses.
Your task is to determine whether the following endorsement text discusses
anything related to **Deductible**.

Rules:
1. Mark "Deductible Applicable?" as "Yes" if there is any mention of deductible,
   such as phrases like "deductible", "policy deductible", "excess amount", or
   "subject to a deductible of Rs.", etc.
2. If there is no mention of deductible or anything conceptually similar,
   mark it as "No".
3. Do not fill any numeric or descriptive value for the deductible.
4. Return only valid JSON in this exact format:
{{
  "Deductible": {{
      "Deductible Applicable?": "",
      "Deductible": ""
  }}
}}

Endorsement text:
{endorsement_text}
"""

    try:
        # Query the LLM (replace this with your actual LLM call, e.g., chat.invoke)
        raw_response = chat.invoke(prompt)
        if hasattr(raw_response, "content"):
            raw_response = raw_response.content

        # Try to extract JSON from the LLM response
        match = re.search(r"\{[\s\S]*\}", raw_response)
        if match:
            result = json.loads(match.group())
            val = (
                result.get("Deductible", {})
                .get("Deductible Applicable?", "No")
                .strip()
                .lower()
            )
            return {
                "Deductible": {
                    "Deductible Applicable?": "Yes" if val == "yes" else "No",
                    "Deductible": ""
                }
            }

        # 🔁 Regex fallback
        pattern = r"\b(deductible|excess\s*(amount)?|policy\s*deductible|subject\s*to\s*a\s*deductible)\b"
        if re.search(pattern, endorsement_text, re.IGNORECASE):
            return {"Deductible": {"Deductible Applicable?": "Yes", "Deductible": ""}}
        else:
            return {"Deductible": {"Deductible Applicable?": "No", "Deductible": ""}}

    except Exception:
        # 🔁 Fallback if LLM or JSON parsing fails
        pattern = r"\b(deductible|excess\s*(amount)?|policy\s*deductible|subject\s*to\s*a\s*deductible)\b"
        if re.search(pattern, endorsement_text, re.IGNORECASE):
            return {"Deductible": {"Deductible Applicable?": "Yes", "Deductible": ""}}
        return {"Deductible": {"Deductible Applicable?": "No", "Deductible": ""}}


def run_deductible(pdf_path):
    end_spl_cond=get_endorsements_and_spl_cond(pdf_path,desired_endorsements=['8'])
    result=detect_deductible(end_spl_cond)

    return result

def run_copay_and_deductible(pdf_path,si_list):
    copay_output=run_copay(pdf_path,si_list)
    deductible_output=run_deductible(pdf_path)

    merged_output = {}

    for d in copay_output:
        if not isinstance(d, dict):
            continue

        for k, v in d.items():
            # If key not present, just add it
            if k not in merged_output:
                merged_output[k] = v
            else:
                # Special handling for Ailment list overwrite case
                if k == "Ailment":
                    # If existing is list, and new is dict — keep list
                    if isinstance(merged_output[k], list) and isinstance(v, dict):
                        continue
                    # If existing is dict, and new is list — replace dict with list
                    elif isinstance(merged_output[k], dict) and isinstance(v, list):
                        merged_output[k] = v
                else:
                    # Normal merge for other keys
                    merged_output[k] = v


    return copay_output,deductible_output


def get_copay_ordered_dict(result):

    inner_order = [
        'Relationship', 'Claim Frequency', 'Ailment',
        'Type of Provider', 'Relationship and Ailment',
        'Age Based', 'Claim Amount', 'Capped Ailment'
    ]


    first_dict, second_dict = result

    # Build ordered first dict
    ordered_first = OrderedDict()
    ordered_first['Co-Pay Applicable?'] = first_dict.get('Co-Pay Applicable?', '')
    ordered_first['Co-Pay Based On'] = first_dict.get('Co-Pay Based On', '')

    # Add inner sections in desired order
    for key in inner_order:
        if key in first_dict:
            ordered_first[key] = first_dict[key]

    # Create final tuple
    ordered_data = (ordered_first, second_dict)

    return ordered_data


import pandas as pd
from collections import OrderedDict

def ordered_dict_to_dfs(data_tuple):
    """
    Convert a co-pay OrderedDict into multiple logical DataFrames as per instructions:
    1. Co-Pay Applicable? and Co-Pay Based On → df1
    2. Relationship → Claim Frequency → df2
    3. Ailment → df3
    4. Type of Provider → Capped Ailment_Applicability → df4
    5. Deductible → df5
    Finally, combine all dfs column-wise and return.
    """
    main_dict, ded_dict = data_tuple
    ded_dict = ded_dict.get("Deductible", {})

    # -------------------------------
    # 1️⃣ Co-Pay Applicable? & Co-Pay Based On
    # -------------------------------
    cp_df = pd.DataFrame({
        'Co-Pay Applicable?': [main_dict.get('Co-Pay Applicable?', '')],
        'Co-Pay Based On': [main_dict.get('Co-Pay Based On', '')]
    })

    # -------------------------------
    # 2️⃣ Relationship → Claim Frequency
    # -------------------------------
    rel_dict = main_dict.get('Relationship', {})
    cf_dict = main_dict.get('Claim Frequency', {})

    rel_cf_data = {}
    # Relationship dict
    for k, v in rel_dict.items():
        rel_cf_data[f"Relationship_{k}"] = [v]
    # Claim Frequency dict
    for k, v in cf_dict.items():
        rel_cf_data[f"Claim Frequency_{k}"] = [v]

    rel_cf_df = pd.DataFrame(rel_cf_data)

    # -------------------------------
    # 3️⃣ Ailment dict
    # -------------------------------
    ailment_dict = main_dict.get('Ailment', {})
    if ailment_dict:
        # make sure all lists are same length
        max_len = max((len(v) for v in ailment_dict.values()), default=1)
        ail_norm = {f"Ailment_{k}": (v + [""]*(max_len - len(v)) if isinstance(v, list) else [v]*max_len)
                    for k, v in ailment_dict.items()}
        ailment_df = pd.DataFrame(ail_norm)
    else:
        ailment_df = pd.DataFrame(columns=[
            'Ailment_Ailment Name','Ailment_Limit Applicable On','Ailment_% Limit',
            'Ailment_Limit','Ailment_Applicability'
        ])

    # -------------------------------
    # 4️⃣ Type of Provider → Capped Ailment_Applicability
    # -------------------------------
    type_provider_dict = main_dict.get('Type of Provider', {})
    rel_ail_dict = main_dict.get('Relationship and Ailment', {})
    age_based_dict = main_dict.get('Age Based', {})
    claim_amount_dict = main_dict.get('Claim Amount', {})
    capped_ail_dict = main_dict.get('Capped Ailment', {})

    combined_dict = {}
    # Type of Provider
    for tp_key, tp_val in type_provider_dict.items():
        for k, v in tp_val.items():
            combined_dict[f"{tp_key}_{k}"] = [v]
    # Relationship and Ailment
    for k, v in rel_ail_dict.items():
        combined_dict[f"Relationship and Ailment_{k}"] = [v]
    # Age Based
    for k, v in age_based_dict.items():
        combined_dict[f"Age Based_{k}"] = [v]
    # Claim Amount
    for k, v in claim_amount_dict.items():
        combined_dict[f"Claim Amount_{k}"] = [v]
    # Capped Ailment
    for k, v in capped_ail_dict.items():
        combined_dict[f"Capped Ailment_{k}"] = [v]

    type_capped_df = pd.DataFrame(combined_dict)

    # -------------------------------
    # 5️⃣ Deductible dict
    # -------------------------------
    ded_df = pd.DataFrame({
        'Deductible_Deductible Applicable?': [ded_dict.get('Deductible Applicable?', '')],
        'Deductible_Deductible': [ded_dict.get('Deductible', '')]
    })

    # -------------------------------
    # Combine all dfs column-wise
    # -------------------------------
    final_df = pd.concat([cp_df, rel_cf_df, ailment_df, type_capped_df, ded_df], axis=1)

    return final_df


def get_copay_and_deductible(pdf_path,source_dir):
    si_list=load_sum_insured_from_json(pdf_path)
    result=run_copay_and_deductible(pdf_path,si_list)
    ordered_data=get_copay_ordered_dict(result)
    df=ordered_dict_to_dfs(ordered_data)
    output_folder = source_dir  # define your working folder
    os.makedirs(output_folder, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(pdf_path))[0]  
    excel_filename = os.path.join(output_folder, f"{base_name}_copay_and_deductible.xlsx")
    

    # Save to Excel
    df.to_excel(excel_filename, index=False)

    
    
    

if __name__ =="__main__":
    pdf_path=r""
    si_list=""
    source_dir=r""
    get_copay_and_deductible(pdf_path,source_dir)
    
    