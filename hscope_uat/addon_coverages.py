# Ambulance Cover
# Convalescence Benefit
# Daily/Hospital Cash Benefit
# Doctor & Nurse Home Visit Cover
# Out Patient Cover
# Critical Illness Benefit



# %%
import pandas as pd
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



def get_endorsements_and_spl_cond(pdf_path,desired_endorsements = ['16']
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



def fill_data(data):
    sum_insured = [float(x) for x in data['Sum Insured']]
    limit_amounts = data['Limit Amount']
    limit_percents = data['Limit Percentage']
   
    new_limit_amount = []
    new_limit_percent = []

    for i in range(len(sum_insured)):
        insured = sum_insured[i]
        amount = limit_amounts[i]
        percent = limit_percents[i]

        if percent in ('', None) and amount not in ('', None):
            calc_percent = round((float(amount) / insured) * 100, 2)
            new_limit_percent.append(str(calc_percent))
            new_limit_amount.append(amount)

        elif amount in ('', None) and percent not in ('', None):
            calc_amount = round((float(percent) / 100) * insured, 2)
            new_limit_amount.append(str(calc_amount))
            new_limit_percent.append(percent)

        else:
            new_limit_amount.append(amount)
            new_limit_percent.append(percent)

    data['Limit Amount'] = new_limit_amount
    data['Limit Percentage'] = new_limit_percent
    return data

def prefix_inner_keys(d):
    """
    Prefixes all inner dict keys with their outer key name,
    and returns the modified nested dictionary.

    Example:
        {'A': {'x': 1, 'y': 2}} 
        → {'A': {'A_x': 1, 'A_y': 2}}
    """
    new_dict = {}
    for outer_key, inner_dict in d.items():
        if isinstance(inner_dict, dict):
            prefixed = {
                f"{outer_key}_{inner_key}": value
                for inner_key, value in inner_dict.items()
            }
            new_dict[outer_key] = prefixed
        else:
            new_dict[outer_key] = inner_dict
    return new_dict




# Ambulance cover
def query_ambulance(context,sum_insured):
    prompt = f"""
You are given a context describing Ambulance Cover insurance details. Please extract the following fields:
ambulance_cover_schema ={{
    "Ambulance Cover": 
        "Number Of Trips": "",
            "Sum Insured": "",                  # e.g., "1,00,000"
            "% Limit Applicable On": "",        # e.g., "Sum Insured"  (default)
            "Limit Percentage": "",             # e.g., "100%" or "50%"
            "Limit Amount": "",                 # e.g., "5,000"
            "Applicability": None               # e.g.,"LOWER"
}}

Sum Insured = {sum_insured}  #given
Limit Applicable On = "Sum Insured" 
Applicability": "LOWER"

Find:
Number Of Trips
Limit Percentage
Limit Amount

give the output in the ambulance_cover_schema format

number of values per key = len(sum) for all

If a value is not present, leave it as an empty string .

Format your answer as a JSON list with values in exactly this order
Return the output strictly as JSON — starting and ending with braces {{ }}.
   Do NOT include any explanation or code block formatting.



example output (same format):
{{            
            "Number Of Trips": ["10", "10", "10"],
            "Sum Insured": ["1", "2", "3"],
            "% Limit Applicable On": ["Sum Insured", "Sum Insured", "Sum Insured"],
            "Limit Percentage": ["","",""],
            "Limit Amount": ["1500", "1500", "1500"],
            "Applicability": ["lower", "lower", "lower"]
}}

{context}
"""
    response = chat.invoke(prompt).content.strip()

    import json
    try:
        # Expecting JSON list like: ["3", "1,00,000", "Sum Insured", "100%", "5,000", "LOWER"]
        values_list = json.loads(response)
    except json.JSONDecodeError:
        # If plain text returned, parse manually or handle error (fallback)
        values_list = response.splitlines()  # naive fallback

    return values_list





# Helper to convert string with commas to int or float safely
def to_number(x):
    try:
        x = str(x).replace(',', '').strip()
        return float(x) if '.' in x else int(x)
    except ValueError:
        return x  # return original if conversion fails
    

def get_ambulance_cover(pdf_path):
    sum_insured=load_sum_insured_from_json(pdf_path)
    #sum_insured=[100000,200000]
    desired_endorsements=['16']
    endnt_and_spl_cond_context=get_endorsements_and_spl_cond(pdf_path,desired_endorsements)
    
    data = query_ambulance(endnt_and_spl_cond_context,sum_insured)
    
    data['Sum Insured'] = [to_number(x) for x in data['Sum Insured']]
    data['Limit Amount'] = [to_number(x) for x in data['Limit Amount']]
    data['Limit Percentage']=[to_number(x) for x in data['Limit Percentage']]
    data=fill_data(data)
    
    result={'Ambulance Cover': data}
    result=prefix_inner_keys(result)
    return result


# Doctor_Nurse Cover
import json

def dn_visit(context: str, sum_insured: list):
    """
    Extracts Home Nursing Allowance from a given context using an LLM prompt.
    
    Parameters:
        context (str): Text describing the  details.
        sum_insured (list): List of Sum Insured values (e.g. ["1,00,000", "2,00,000"]).

    Returns:
        dict: Parsed JSON response with extracted details.
    """

    prompt = f"""
You are given a context describing Home Nursing Allowance insurance details. 
Please extract the following fields in the exact format below.

Dn_schema = {{
    "Applicability of Doctor's Home Visit & Nursing Charges": {{
        "No. of days Allowed": "",
        "Sum Insured": "",                  
        "% Limit Applicable On": "",        
        "Limit Percentage": "",             
        "Limit Amount": "",                 
        "Applicability": ""                 
    }}
}}

Given:
- Sum Insured = {sum_insured}
- % Limit Applicable On = "Sum Insured"
- Applicability = "LOWER"

Your task:
1. Find and fill these fields:
   - "No. of days Allowed"
   - "Limit Percentage"
   - "Limit per Claim" (map this to "Limit Amount")
2. The number of values per key must equal len(Sum Insured).
3. If a value is missing, leave it as an empty string "".
4. Return the output strictly as JSON — starting and ending with braces {{ }}.
   Do NOT include any explanation or code block formatting.

Expected output example (structure only):
{{
    "No of days Allowed": ["10", "10", "10"],
    "Sum Insured": ["1,00,000", "2,00,000", "3,00,000"],
    "% Limit Applicable On": ["Sum Insured", "Sum Insured", "Sum Insured"],
    "Limit Percentage": ["", "", ""],
    "Limit Amount": ["1500", "1500", "1500"],
    "Applicability": ["lower", "lower", "lower"]
}}

Context:
{context}
"""

    # Call the LLM (assuming `chat.invoke()` is your model interface)
    response = chat.invoke(prompt).content.strip()

    # Try parsing JSON safely
    try:
        result = json.loads(response)
    except json.JSONDecodeError:
        print("⚠️ Warning: LLM response was not valid JSON. Returning raw response.")
        result = {"raw_response": response}

    return result



# Helper to convert string with commas to int or float safely
def to_number(x):
    try:
        x = str(x).replace(',', '').strip()
        return float(x) if '.' in x else int(x)
    except ValueError:
        return x  # return original if conversion fails
    

# conver to int or float
from collections import defaultdict

def merge_dicts_to_list(data_list):
    """
    Merge a list of dicts into one dict where each key
    maps to a list of its values across all dicts.
    """
    merged = defaultdict(list)
    for d in data_list:
        for k, v in d.items():
            merged[k].append(v)
    return dict(merged)



def get_doctor_and_nurse_cover(pdf_path):
    sum_insured=load_sum_insured_from_json(pdf_path)
    #sum_insured = [100000, 200000]
    desired_endorsements = ['17']
    endnt_and_spl_cond_context=get_endorsements_and_spl_cond(pdf_path,desired_endorsements)

    data = dn_visit(endnt_and_spl_cond_context, sum_insured)
    # cover_key = "Applicability of Doctor's Home Visit & Nursing Charges"

    data['Sum Insured'] = [to_number(x) for x in data['Sum Insured']]
    data['Limit Amount'] = [to_number(x) for x in data['Limit Amount']]
    data['Limit Percentage'] = [to_number(x) for x in data['Limit Percentage']]

    cover_data = fill_data(data)
    expense_dict = {
        "Applicable On?": "",
        "Is Doctor & Nursing Charges Combined?": "",
        "Type of Expenses": "",
        "% Limit Applicable On": "",
        "Limit Percentage": "",
        "Limit Amount": "",
        "Applicability": "",
        "No of days Allowed": ""
    }
    df = pd.DataFrame([expense_dict])

    # Define keys
    exclude_keys = ["Sum Insured"]
    update_keys = [key for key in cover_data if key not in exclude_keys]
    types = ["doctor", "nursing"]
    num_items = len(next(iter(cover_data.values())))

    # ✅ Extract "No of days Allowed" value upfront
    no_days_value = data.get("No of days Allowed", [""])
    no_days_str = str(no_days_value[0]) if isinstance(no_days_value, list) and len(no_days_value) > 0 else ""

    # Loop through to populate values
    row_cursor = 0
    for t_idx, t in enumerate(types):
        for i in range(num_items):
            df.at[row_cursor, "Type of Expenses"] = t

            # Fill first row only for these fields
            if row_cursor == 0:
                df.at[row_cursor, "Applicable On?"] = "Post Hospitalization"
                df.at[row_cursor, "Is Doctor & Nursing Charges Combined?"] = "No"
            else:
                df.at[row_cursor, "Applicable On?"] = ""
                df.at[row_cursor, "Is Doctor & Nursing Charges Combined?"] = ""

            # Fill all other fields
            for key in update_keys:
                if key in df.columns and key != "No of days Allowed":
                    df.at[row_cursor, key] = data[key][i]

            # ✅ Now fill "No of days Allowed" for every row
            df.at[row_cursor, "No of days Allowed"] = no_days_str

            row_cursor += 1

    # ✅ Replace NaN with empty string to avoid blanks
    df = df.fillna("")

    # Add prefix
    prefix = "Applicability of Doctor's Home Visit & Nursing Charges"
    df.columns = [f"{prefix}_{col}" for col in df.columns]

    data_dict = df.to_dict(orient="records")
    result = merge_dicts_to_list(data_dict)

    return result




import re
import json

def extract_convalescence_benefit(snippet: str, chat, sum_insured_list: list) -> dict:
    """
    Extract Convalescence Benefit details from an insurance endorsement text using LLM + fallback.

    Output JSON format:
    {
      "Convalescence Benefit": {
        "Minimum LOS in days": "",
        "Applicable From": "",
        "Sum Insured": [],
        "Benefit Amount": []
      }
    }

    Rules:
    - 'Applicable From' is always 'Date of Admission'
    - 'Sum Insured' is repeated n times (length of sum_insured_list)
    - Extract 'Minimum LOS in days' (the number of days hospitalization must exceed)
    - Extract 'Benefit Amount' (any mentioned rupee amount)
    """

    prompt = f"""
You are an expert health insurance policy analyst.
Extract the required fields for **Convalescence Benefit / Special Care / Recovery Benefit**
from the text below.

Target JSON format:
{{
  "Convalescence Benefit": {{
    "Minimum LOS in days": "",
    "Applicable From": "Date of Admission",
    "Sum Insured": {sum_insured_list},
    "Benefit Amount": []
  }}
}}

Extraction Rules:
1. 'Minimum LOS in days' → Extract the number of days after which this benefit applies
   (e.g., from phrases like "if hospitalization exceeds 7 days").
2. 'Benefit Amount' → Extract any lump sum rupee amount (e.g., "Rs. 50,000" → 50000)
   and repeat it for all Sum Insured values.
3. 'Applicable From' → Always "Date of Admission".
4. 'Sum Insured' → Use the provided list, repeated in full.
5.  Return the output strictly as JSON — starting and ending with braces {{ }}.
   Do NOT include any explanation or code block formatting.
6. If any value not found, leave as an empty string "".

Text Snippet:
\"\"\"{snippet}\"\"\"
"""

    try:
        # ✅ Call LLM
        raw_response = chat.invoke(prompt)
        if hasattr(raw_response, "content"):
            raw_response = raw_response.content

        # ✅ Extract JSON safely
        json_match = re.search(r"\{(?:[^{}]|(?R))*\}", raw_response, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            conv_data = data.get("Convalescence Benefit", {})

            # Extract values
            min_los = conv_data.get("Minimum LOS in days", "")
            benefit_list = conv_data.get("Benefit Amount", [])

            # Ensure list format for benefit amount
            if not isinstance(benefit_list, list):
                benefit_list = [benefit_list] * len(sum_insured_list)
            elif len(benefit_list) != len(sum_insured_list):
                benefit_list = [benefit_list[0] if benefit_list else ""] * len(sum_insured_list)

            return {
                "Convalescence Benefit": {
                    "Minimum LOS in days": min_los,
                    "Applicable From": "Date of Admission",
                    "Sum Insured": sum_insured_list,
                    "Benefit Amount": benefit_list
                }
            }

        # ❌ Fallback to regex-based extraction
        raise ValueError("No valid JSON detected")

    except Exception:
        # Fallback extraction (regex)
        days_match = re.search(r"exceeds?\s*(\d+)\s*day", snippet, re.IGNORECASE)
        amt_match = re.search(r"Rs\.?\s*([\d,]+)", snippet, re.IGNORECASE)

        min_los = days_match.group(1) if days_match else ""
        benefit_amt = amt_match.group(1).replace(",", "") if amt_match else ""

        # Repeat the benefit amount for each sum insured
        benefit_list = [benefit_amt] * len(sum_insured_list) if benefit_amt else [""] * len(sum_insured_list)

        return {
            "Convalescence Benefit": {
                "Minimum LOS in days": min_los,
                "Applicable From": "Date of Admission",
                "Sum Insured": sum_insured_list,
                "Benefit Amount": benefit_list
            }
        }




def get_convalescence_benefit(pdf_path):
    si_list=load_sum_insured_from_json(pdf_path)
    desired_endorsements=['15']
    endnt_and_spl_cond_context=get_endorsements_and_spl_cond(pdf_path,desired_endorsements)
    
    result=extract_convalescence_benefit(endnt_and_spl_cond_context,chat,si_list)
    data=prefix_inner_keys(result)
    return data






import json
import re

def extract_json_from_llm(raw_response: str) -> dict:
    """
    Extract JSON from LLM output safely without using unsupported recursive regex.
    """
    try:
        # Find first '{' and last '}' and extract in-between
        start = raw_response.find("{")
        end = raw_response.rfind("}")
        if start != -1 and end != -1:
            json_str = raw_response[start:end+1]
            return json.loads(json_str)
    except Exception:
        pass
    raise ValueError("No valid JSON detected in LLM output")


def extract_hospital_cash_allowance_llm(snippet: str, chat, sum_insured: list) -> dict:
    prompt = f"""
You are an expert health insurance policy analyst.
Extract the **Hospital Cash Allowance** details from the text below and fill them into the exact JSON format.

Rules:
- 'Minimum LOS in days' → extract from phrases like "more than 3 days", "exceeds 2 days".
- 'Max Days Per Policy year' → extract maximum days benefit is available ("maximum days of 4 per person per event or per policy").
- 'Max Days Per Illness' → same as 'Max Days Per Policy year'.
- 'Fixed limit' → always "Per Day".
- 'Sum Insured' & 'Threshold' → use provided list.
- 'Limit Amount' → any daily rupee amount mentioned.
- 'Limit Percentage' → if % mentioned, extract it; else "".
- 'Applicability' → list of "Lower" repeated for each sum insured.
- Leave missing fields as "".

Target JSON format:
{{
  "Hospital Cash Allowance": {{
    "Minimum LOS in days": "",
    "Over And Above Policy Sum Insured?": "Yes",
    "Max Days Per Policy year": "",
    "Max Days Per Illness": "",
    "Fixed limit": "Per Day",
    "Sum Insured": {sum_insured},
    "Threshold": {sum_insured},
    "% Limit Applicable On": "Sum Insured",
    "Limit Percentage": "",
    "Limit Amount": [],
    "Applicability": [],
    "Open range": "",
    "Threshold SI": "",
    "Daily Limit Range From": "",
    "Daily Limit Range To": ""
  }}
}}

Text Snippet:
\"\"\"{snippet}\"\"\"

Return the output strictly as JSON — starting and ending with braces {{ }}.
   Do NOT include any explanation or code block formatting.
"""
    raw_response = chat.invoke(prompt)
    if hasattr(raw_response, "content"):
        raw_response = raw_response.content

    data = extract_json_from_llm(raw_response)
    hosp_data = data.get("Hospital Cash Allowance", {})
    limit_amount=hosp_data['Limit Amount']

    # Ensure consistent fields
    hosp_data["Sum Insured"] = sum_insured
    hosp_data["Threshold"] = sum_insured
    hosp_data["Applicability"] = ["Lower"] * len(sum_insured)
    hosp_data['Limit Amount']=limit_amount* len(sum_insured)
    hosp_data["Fixed limit"] = "Per Day"
    hosp_data["Over And Above Policy Sum Insured?"] = "Yes"
    hosp_data["Max Days Per Illness"] = hosp_data.get("Max Days Per Policy year", "")

    return {"Daily Cash": hosp_data}


def fill_data_daily_cash(data):
    sum_insured = [float(x) for x in data.get('Sum Insured', [])]
    limit_amounts = data.get('Limit Amount', [])
    limit_percents = data.get('Limit Percentage', [])

    # --- Normalize all to lists ---
    # Handle scalar or empty string cases
    if not isinstance(limit_amounts, list):
        limit_amounts = [limit_amounts] if limit_amounts not in ('', None) else [''] * len(sum_insured)
    if not isinstance(limit_percents, list):
        limit_percents = [limit_percents] if limit_percents not in ('', None) else [''] * len(sum_insured)

    # If lists shorter than sum_insured, extend with blanks
    if len(limit_amounts) < len(sum_insured):
        limit_amounts += [''] * (len(sum_insured) - len(limit_amounts))
    if len(limit_percents) < len(sum_insured):
        limit_percents += [''] * (len(sum_insured) - len(limit_percents))

    new_limit_amount = []
    new_limit_percent = []

    # --- Core calculation ---
    for i in range(len(sum_insured)):
        insured = sum_insured[i]
        amount = limit_amounts[i]
        percent = limit_percents[i]

        # Handle missing percent
        if percent in ('', None) and amount not in ('', None, ''):
            calc_percent = round((float(amount) / insured) * 100, 2)
            new_limit_percent.append(str(calc_percent))
            new_limit_amount.append(str(amount))

        # Handle missing amount
        elif amount in ('', None, '') and percent not in ('', None, ''):
            calc_amount = round((float(percent) / 100) * insured, 2)
            new_limit_amount.append(str(calc_amount))
            new_limit_percent.append(str(percent))

        # Both present
        else:
            new_limit_amount.append(str(amount))
            new_limit_percent.append(str(percent))

    # --- Assign results back ---
    data['Limit Amount'] = new_limit_amount
    data['Limit Percentage'] = new_limit_percent
    return data



def get_daily_cash_allowance(pdf_path):
        desired_endorsements=['14']
        sum_insured=load_sum_insured_from_json(pdf_path)
        #sum_insured=[100000,200000]
        endnt_and_spl_cond_context=get_endorsements_and_spl_cond(pdf_path,desired_endorsements)
   
        result=extract_hospital_cash_allowance_llm(endnt_and_spl_cond_context,chat,sum_insured)
        ans=fill_data_daily_cash(result['Daily Cash'])
        edited_ans={"Daily Cash":ans}        
        result=prefix_inner_keys(edited_ans)
        return result





# critical illness

def extract_critical_illness(snippet: str, chat, sum_insured: list) -> dict:
    """
    Extracts 'Critical Illness' benefit details from a given text snippet using LLM rules.
    """

    prompt = f"""
You are an expert health insurance policy analyst.
Extract the **Critical Illness** details from the text below and fill them into the exact JSON format.

Rules:
1. **Over And Above Policy Sum Insured?**
   - If the text mentions that this benefit is "over and above" or "in addition to" the Sum Insured → "Yes".
   - Otherwise default → "No".

2. **Survival Period Applicable?**
   - If the text mentions a 'Survival Period' → "Yes".
   - Otherwise → "No".

3. **Survival Period Applicable From**
   - If Survival Period Applicable is "Yes" → default = "Date of Admission".
   - If "No" → leave as "".

4. **Number of Days**
   - Extract the numeric number of days mentioned for survival period, e.g. "30 days", "15 days", etc.
   - If not mentioned → leave blank ("").

5. The following fields should remain blank (""):
   "Relationship", "Age", "Metro", "Provider Type", "Hospital Type",
   "Age From", "Age To", "City", "Limit Applicable On", "% Applicable",
   "Amount Applicable", "Applicable Limit".

Target JSON format:
{{
  "Critical Illness": {{
    "Over And Above Policy Sum Insured?": "",
    "Survival Period Applicable?": "",
    "Survival Period Applicable From": "",
    "Number of Days": "",
    "Relationship": "",
    "Age": "",
    "Metro": "",
    "Provider Type": "",
    "Hospital Type": "",
    "Age From": "",
    "Age To": "",
    "City": "",
    "Limit Applicable On": "",
    "% Applicable": "",
    "Amount Applicable": "",
    "Applicable Limit": ""
  }}
}}

Text Snippet:
\"\"\"{snippet}\"\"\"

Return the output strictly as valid JSON — starting and ending with braces {{ }}.
Do NOT include any explanations, code blocks, or commentary.
"""
    # Get LLM response
    raw_response = chat.invoke(prompt)
    if hasattr(raw_response, "content"):
        raw_response = raw_response.content

    # Convert to dict
    data = extract_json_from_llm(raw_response)
    crit_data = data.get("Critical Illness", {})

    # Apply logical post-processing
    survival_applicable = crit_data.get("Survival Period Applicable?", "").strip().lower()

    if survival_applicable == "yes":
        crit_data["Survival Period Applicable From"] = "Date of Admission"
    else:
        crit_data["Survival Period Applicable From"] = ""
        crit_data["Number of Days"] = ""

    # Ensure default "No" if field missing
    crit_data["Over And Above Policy Sum Insured?"] = (
        crit_data.get("Over And Above Policy Sum Insured?", "").title() or "No"
    )

    # Fill missing optional fields as ""
    optional_fields = [
        "Relationship", "Age", "Metro", "Provider Type", "Hospital Type",
        "Age From", "Age To", "City", "Limit Applicable On", "% Applicable",
        "Amount Applicable", "Applicable Limit"
    ]
    for f in optional_fields:
        crit_data[f] = crit_data.get(f, "")

    return {"Critical Illness": crit_data}


def get_critical_illness(pdf_path):
        desired_endorsements=['20']
        sum_insured=load_sum_insured_from_json(pdf_path)
        endnt_and_spl_cond_context=get_endorsements_and_spl_cond(pdf_path,desired_endorsements)
        
        context = endnt_and_spl_cond_context
        crit_values = extract_critical_illness(context,sum_insured)

        result=prefix_inner_keys(crit_values)
        return result




def fill_data_opc(data):
    sum_insured = [float(x) for x in data.get('Sum Insured', [])]
    limit_amounts = data.get('Limit Amount', [])
    limit_percents = data.get('Limit Percentage', [])

    # --- Normalize all to lists ---
    if not isinstance(limit_amounts, list):
        limit_amounts = [limit_amounts] * len(sum_insured)
    if not isinstance(limit_percents, list):
        limit_percents = [limit_percents] * len(sum_insured)

    # If list length mismatch, pad to equal length
    if len(limit_amounts) < len(sum_insured):
        limit_amounts += [''] * (len(sum_insured) - len(limit_amounts))
    if len(limit_percents) < len(sum_insured):
        limit_percents += [''] * (len(sum_insured) - len(limit_percents))

    new_limit_amount = []
    new_limit_percent = []

    for i in range(len(sum_insured)):
        insured = sum_insured[i]
        amount = limit_amounts[i]
        percent = limit_percents[i]

        if percent in ('', None) and amount not in ('', None):
            calc_percent = round((float(amount) / insured) * 100, 2)
            new_limit_percent.append(str(calc_percent))
            new_limit_amount.append(str(amount))

        elif amount in ('', None) and percent not in ('', None):
            calc_amount = round((float(percent) / 100) * insured, 2)
            new_limit_amount.append(str(calc_amount))
            new_limit_percent.append(str(percent))

        else:
            new_limit_amount.append(str(amount))
            new_limit_percent.append(str(percent))

    data['Limit Amount'] = new_limit_amount
    data['Limit Percentage'] = new_limit_percent
    return data




def extract_outpatient_cover(snippet: str, chat, sum_insured: list) -> dict:
    """
    Extracts 'Out Patient Configuration' (OPC) details from the given text snippet using LLM rules.
    """

    prompt = f"""
You are an expert health insurance policy analyst.
Extract the **Out Patient Configuration (OPC)** details from the text below and fill them into the exact JSON format.

Rules:
1. **Over And Above Policy Sum Insured?**
   - Always default to "No".

2. **Aggregate Limit**
   - If the text mentions an aggregate limit or cap (e.g. "aggregate limit of ₹5,000 per policy year") → extract the numeric or rupee value.
   - If not mentioned → leave as "".

3. **Per Person / Family Detection**
   - If text explicitly mentions **"per person"**, **"per member"**, **"individual"**, then:
     - OP Treatment Limit → "Person"
   - If text explicitly mentions **"per family"**, **"family floater"**, etc., then:
     - OP Treatment Limit → "Family"
   - If both "per person/family" or both terms are present → **give priority to "Family"**.
   - If none mentioned → leave as "".

4. **Limit Amount**
   - If a rupee or amount is mentioned along with "per person", "per family", or "per person/family", extract that value.
   - If not mentioned → leave as "".

5. **Sum Insured, % Limit Applicable On, Limit Percentage**
   - These are filled **only when** there is mention of "per person", "per family", or "per person/family" (i.e. when OP Treatment Limit is non-empty).
   - Sum Insured → {sum_insured}
   - % Limit Applicable On → "Sum Insured"
   - Limit Percentage → extract percentage value if any, else "".

6. **Applicability**
   - Default = "Lower" (repeat for each Sum Insured value when applicable).

7. **Action**
   - Leave blank ("") by default.

Target JSON format:
{{
  "Out Patient Configuration": {{
    "Over And Above Policy Sum Insured?": "No",
    "Aggregate Limit": "",
    "Sum Insured": [],
    "% Limit Applicable On": [],
    "Limit Percentage": "",
    "Limit Amount": [],
    "OP Treatment Limit": "",
    "Applicability": [],
    "Action": ""
  }}
}}

Text Snippet:
\"\"\"{snippet}\"\"\"


Return the output strictly as valid JSON — starting and ending with braces {{ }}.
Do NOT include explanations, markdown, or commentary.
"""

    # --- Call the LLM ---
    raw_response = chat.invoke(prompt)
    if hasattr(raw_response, "content"):
        raw_response = raw_response.content

    # --- Convert JSON safely ---
    data = extract_json_from_llm(raw_response)
    opc_data = data.get("Out Patient Configuration", {})

    # --- Logical defaults ---
    opc_data["Over And Above Policy Sum Insured?"] = "No"

    # Handle OP Treatment Limit prioritization
    op_limit_type = opc_data.get("OP Treatment Limit", "").strip().lower()
    if "person" in op_limit_type and "family" in op_limit_type:
        opc_data["OP Treatment Limit"] = "Family"
    elif "family" in op_limit_type:
        opc_data["OP Treatment Limit"] = "Family"
    elif any(term in op_limit_type for term in ["person", "member", "individual"]):
        opc_data["OP Treatment Limit"] = "Person"
    else:
        opc_data["OP Treatment Limit"] = ""

    # --- Fill dependent fields when OP Treatment Limit is present ---
    if opc_data["OP Treatment Limit"]:
        opc_data["Sum Insured"] = sum_insured

        # Repeat dependent values for each Sum Insured
        opc_data["% Limit Applicable On"] = ["Sum Insured"] * len(sum_insured)
        opc_data["Applicability"] = ["Lower"] * len(sum_insured)
        opc_data["OP Treatment Limit"] = [opc_data["OP Treatment Limit"]] * len(sum_insured)

        # --- Ensure Limit Amount repeats correctly ---
        limit_amount = opc_data.get("Limit Amount", "")

        if isinstance(limit_amount, list):
            if len(limit_amount) == 1:
                opc_data["Limit Amount"] = limit_amount * len(sum_insured)
            elif len(limit_amount) != len(sum_insured):
                opc_data["Limit Amount"] = (limit_amount * len(sum_insured))[:len(sum_insured)]
        elif isinstance(limit_amount, str) and limit_amount:
            opc_data["Limit Amount"] = [limit_amount] * len(sum_insured)
        else:
            opc_data["Limit Amount"] = ["" for _ in sum_insured]
    else:
        # No OP Treatment Limit → blank dependent fields
        opc_data["Sum Insured"] = []
        opc_data["% Limit Applicable On"] = []
        opc_data["Applicability"] = []
        opc_data["Limit Amount"] = []
        opc_data["OP Treatment Limit"] = []

    # --- Ensure required keys exist ---
    for key in [
        "Aggregate Limit", "Limit Percentage", "Limit Amount",
        "OP Treatment Limit", "Action"
    ]:
        opc_data[key] = opc_data.get(key, "")

    return {"Out Patient Configuration": opc_data}



def flatten_dict_opc(d, parent_key='', sep='_'):
    """
    Recursively flattens a nested dictionary into a single-level dictionary.
    Example:
        {'A': {'B': 1, 'C': {'D': 2}}}
        -> {'A_B': 1, 'A_C_D': 2}
    """
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            # Recursively flatten sub-dictionaries
            items.update(flatten_dict_opc(v, new_key, sep=sep))
        else:
            # For lists or other values, just assign
            items[new_key] = '' if v is None else v
    return items


def flatten_out_patient_config(input_dict):
    """
    Flattens the 'Out Patient Configuration' block as per required structure.
    """
    flattened = flatten_dict_opc(input_dict, parent_key='Out Patient Configuration')

    # ensure all expected keys exist (initialize missing as empty string)
    expected_keys = [
        "Over And Above Policy Sum Insured?",
        "Aggregate Limit",
        "Sum Insured",
        "% Limit Applicable On",
        "Limit Percentage",
        "Limit Amount",
        "OP Treatment Limit",
        "Applicability",
        "Action",
        "Is Inclusive of Dental & Optical",
        "Individual_SI Amount",
        "Individual_Sum Insured",
        "Individual_Benefit Applicable For",
        "Individual_% Limit Applicable On",
        "Individual_Limit Percentage",
        "Individual_Limit Amount",
        "Individual_Applicability",
        "Individual_Action",
        "Combined_Sum Insured",
        "Combined_% Limit Applicable On",
        "Combined_Limit Percentage",
        "Combined_Limit Amount",
        "Combined_Applicability",
        "Combined_Action",
        "Relationship",
        "Age",
        "Metro",
        "Provider Type",
        "Hospital Type",
        "Age From",
        "Age To",
        "City",
        "Limit Applicable On",
        "% Applicable",
        "Amount Applicable",
        "Applicable Limit",
    ]

    final_output = {}
    for key in expected_keys:
        full_key = f"Out Patient Configuration_{key}"
        final_output[full_key] = flattened.get(full_key, '')

    return {"Out Patient Configuration": final_output}



def get_opc_cover(pdf_path):
        desired_endorsements=['7(i)']
        sum_insured=load_sum_insured_from_json(pdf_path)
        endnt_and_spl_cond_context=get_endorsements_and_spl_cond(pdf_path,desired_endorsements)
        
        context = endnt_and_spl_cond_context
        result = extract_outpatient_cover(context,chat,sum_insured)
        result=fill_data_opc(result['Out Patient Configuration'])
        reamin_opc_dict={'Is Inclusive of Dental & Optical': None,
 'Individual': {'SI Amount': None,
  'Sum Insured': None,
  'Benefit Applicable For': None,
  '% Limit Applicable On': None,
  'Limit Percentage': None,
  'Limit Amount': None,
  'Applicability': None,
  'Action': None},
 'Combined': {'Sum Insured': None,
  '% Limit Applicable On': None,
  'Limit Percentage': None,
  'Limit Amount': None,
  'Applicability': None,
  'Action': None},
 'Relationship': None,
 'Age': None,
 'Metro': None,
 'Provider Type': None,
 'Hospital Type': None,
 'Age From': None,
 'Age To': None,
 'City': None,
 'Limit Applicable On': None,
 '% Applicable': None,
 'Amount Applicable': None,
 'Applicable Limit': None}
        final_dict={}
        final_dict.update(**result,**reamin_opc_dict)
        

        result = flatten_out_patient_config(final_dict)
        
        return result


############## Sheet Logic 

import json
import os

def load_json_to_dict(file_path):
    """
    Load a JSON file and return its contents as a Python dictionary or list.

    Args:
        file_path (str): Path to the JSON file.

    Returns:
        dict | list: Parsed JSON content.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ JSON file not found: {file_path}")
    
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    return data

import json

def extract_yes_keys_from_dict(addon_covers_dict):
    

    yes_keys = []
    for record in addon_covers_dict:
        for key, value in record.items():
            if str(value).strip().lower() == "yes":
                yes_keys.append(key)
    
    return yes_keys

def flatten_benefit_schema(schema: dict) -> dict:
    """
    Flattens each nested benefit dict by prefixing its keys with the benefit name.
    Maintains the outer structure.
    
    Example:
        {'Convalescence Benefit': {'Minimum LOS in days': None}}
        → {'Convalescence Benefit': {'ConvalescenceBenefit_Minimum LOS in days': ''}}
    """
    result = {}
    for benefit_name, fields in schema.items():
        # Remove spaces and special chars from benefit name for key prefix
        prefix = benefit_name.replace("'", "").replace("/", "").replace("&", "And")
        
        flat_fields = {}
        for k, v in fields.items():
            if isinstance(v, dict):
                # flatten nested dict recursively (one level deep)
                for inner_k in v.keys():
                    flat_fields[f"{prefix}_{k}_{inner_k}"] = ""
            else:
                flat_fields[f"{prefix}_{k}"] = ""
        
        result[benefit_name] = flat_fields
    return result



benefit_schema = {
    "Ambulance Cover": {
        "Number Of Trips": None,
        "Sum Insured": None,
        "% Limit Applicable On": None,
        "Limit Percentage": None,
        "Limit Amount": None,
        "Applicability": None
    },
    "Anyone Illness": {
        "Valid from last consultation": None,
        "Consultation days": None,
        "Valid from date of discharge": None,
        "Discharge days": None,
        "Relationship": None,
        "Age": None,
        "Metro": None,
        "Provider Type": None,
        "Hospital Type": None,
        "Age From": None,
        "Age To": None,
        "City": None,
        "Limit Applicable On": None,
        "% Applicable": None,
        "Amount Applicable": None,
        "Applicable Limit": None
    },
    "Attendant Care": {
        "Over And Above Policy Sum Insured?": None,
        "Relationship": None,
        "Age": None,
        "Metro": None,
        "Provider Type": None,
        "Hospital Type": None,
        "Age From": None,
        "Age To": None,
        "City": None,
        "Limit Applicable On": None,
        "% Applicable": None,
        "Amount Applicable": None,
        "Applicable Limit": None,
        "Max Days Per Illness": None,
        "Min Days Per Illness": None
    },
    "Cancer Cover": {
        "Over And Above Policy Sum Insured?": None
    },
    "Convalescence Benefit": {
        "Minimum LOS in days": None,
        "Applicable From": None,
        "Sum Insured": None,
        "Benefit Amount": None
    },
    "Critical Illness": {
        "Over And Above Policy Sum Insured?": None,
        "Survival Period Applicable?": None,
        "Survival Period Applicable From": None,
        "Number of Days": None,
        "Relationship": None,
        "Age": None,
        "Metro": None,
        "Provider Type": None,
        "Hospital Type": None,
        "Age From": None,
        "Age To": None,
        "City": None,
        "Limit Applicable On": None,
        "% Applicable": None,
        "Amount Applicable": None,
        "Applicable Limit": None
    },
    "Daily Cash": {
        "Minimum LOS in days": None,
        "Over And Above Policy Sum Insured?": None,
        "Max Days Per Policy year": None,
        "Max Days Per Illness": None,
        "Fixed limt": None,
        "Sum Insured": None,
        "Threshold": None,
        "% Limit Applicable On": None,
        "Limit Percentage": None,
        "Limit Amount": None,
        "Applicability": None,
        "Open range": None,
        "Threshold SI": None,
        "Daily Limit Range From": None,
        "Daily Limit Range To": None
    },
    "Dental Care": {
        "Over And Above Policy Sum Insured?": None,
        "Sum Insured": None,
        "% Limit Applicable On": None,
        "Limit Percentage": None,
        "Limit Amount": None,
        "Applicability": None,
        "Relationship": None,
        "Age": None,
        "Metro": None,
        "Provider Type": None,
        "Hospital Type": None,
        "Age From": None,
        "Age To": None,
        "City": None,
        "Limit Applicable On": None,
        "% Applicable": None,
        "Amount Applicable": None,
        "Applicable Limit": None
    },
    "Diabetic Cover": {
        "Over And Above Policy Sum Insured?": None
    },
    "Applicability of Doctor's Home Visit & Nursing Charges": {
        "Applicable On?": None,
        "Is Doctor & Nursing Charges Combined?": None,
        "% Limit Applicable On": None,
        "Limit Percentage": None,
        "Limit Amount": None,
        "Applicability": None,
        "No of days Allowed": None
    },
    "Education Fund": {
        "Over And Above Sum Insured?": None,
        "Sum Insured": None,
        "% Limit Applicable On": None,
        "Limit Percentage": None,
        "Limit Amount": None,
        "Limit per child": None,
        "Applicability": None
    },
    "Funeral Expenses": {
        "Over And Above Sum Insured?": None,
        "Relationship": None,
        "Age": None,
        "Metro": None,
        "Provider Type": None,
        "Hospital Type": None,
        "Age From": None,
        "Age To": None,
        "City": None,
        "Limit Applicable On": None,
        "% Applicable": None,
        "Amount Applicable": None,
        "Applicable Limit": None
    },
    "Get Well Benefit": {
        "Over And Above Sum Insured?": None,
        "Relationship": None,
        "Age": None,
        "Metro": None,
        "Provider Type": None,
        "Hospital Type": None,
        "Age From": None,
        "Age To": None,
        "City": None,
        "Limit Applicable On": None,
        "% Applicable": None,
        "Amount Applicable": None,
        "Applicable Limit": None
    },
    "Hardship Critical Illness Cover": {
        "Over And Above Policy Sum Insured?": None,
        "Select Type": None,
        "% Limit Applicable On": None,
        "Limit Percentage": None,
        "Limit Amount": None,
        "Applicability": None,
        "Relationship": None,
        "Age": None,
        "Metro": None,
        "Provider Type": None,
        "Hospital Type": None,
        "Age From": None,
        "Age To": None,
        "City": None,
        "Limit Applicable On": None,
        "% Applicable": None,
        "Amount Applicable": None,
        "Applicable Limit": None
    },
    "Health Check-up": {
        "Claims Free Option Applicable": None,
        "Over And Above Policy Sum Insured?": None,
        "Benefit Applicability after policy years": None,
        "Frequency Of Health Check-up": None,
        "Relationship": None,
        "Age": None,
        "Metro": None,
        "Provider Type": None,
        "Hospital Type": None,
        "Age From": None,
        "Age To": None,
        "City": None,
        "Limit Applicable On": None,
        "% Applicable": None,
        "Amount Applicable": None,
        "Applicable Limit": None
    },
    "Hypertension Cover": {
        "Over And Above Policy Sum Insured?": None
    },
    "Intensive Care Benefit": {
        "Minimum LOS in days": None,
        "No. of Hospital Beds": None,
        "SI Amount": None,
        "Minimum No. Days": None,
        "Maximum No. Days": None,
        "Sum Insured": None,
        "Benefit Amount": None,
        "Action": None
    },
    "Loss Of Pay Cover": {
        "Over And Above Policy Sum Insured?": None,
        "Time Access(In Days)": None,
        "Relationship": None,
        "Age": None,
        "Metro": None,
        "Provider Type": None,
        "Hospital Type": None,
        "Age From": None,
        "Age To": None,
        "City": None,
        "Limit Applicable On": None,
        "% Applicable": None,
        "Amount Applicable": None,
        "Applicable Limit": None
    },
    "Medical Evacuation": {
        "Over And Above Policy Sum Insured?": None,
        "Sum Insured": None,
        "% Limit Applicable On": None,
        "Limit Percentage": None,
        "Limit Amount": None,
        "Applicability": None,
        "Action": None
    },
    "Medical Second Opinion": {
        "Family Level Limits": {
            "No of opinion Allowed": None,
            "Sum Insured": None,
            "% Limit Applicable On": None,
            "Limit Percentage": None,
            "Limit Amount": None,
            "Applicability": None,
            "Action": None
        },
        "Member Level Limits": {
            "No of opinion Allowed": None,
            "Sum Insured": None,
            "% Limit Applicable On": None,
            "Limit Percentage": None,
            "Limit Amount": None,
            "Applicability": None,
            "Action": None
        },
        "Illness Applicability?": None
    },
    "Non-Medical Expenses Cover": {
        "Over And Above Policy Sum Insured?": None,
        "Relationship": None,
        "Age": None,
        "Metro": None,
        "Provider Type": None,
        "Hospital Type": None,
        "Age From": None,
        "Age To": None,
        "City": None,
        "Limit Applicable On": None,
        "% Applicable": None,
        "Amount Applicable": None,
        "Applicable Limit": None,
        "Action": None
    },
    "Out Patient Configuration": {
        "Over And Above Policy Sum Insured?": None,
        "Aggregate Limit": None,
        "Sum Insured": None,
        "% Limit Applicable On": None,
        "Limit Percentage": None,
        "Limit Amount": None,
        "OP Treatment Limit": None,
        "Applicability": None,
        "Action": None,
        "Is Inclusive of Dental & Optical": None,
        "Individual": {
            "SI Amount": None,
            "Sum Insured": None,
            "Benefit Applicable For": None,
            "% Limit Applicable On": None,
            "Limit Percentage": None,
            "Limit Amount": None,
            "Applicability": None,
            "Action": None
        },
        "Combined": {
            "Sum Insured": None,
            "% Limit Applicable On": None,
            "Limit Percentage": None,
            "Limit Amount": None,
            "Applicability": None,
            "Action": None
        },
        "Relationship": None,
        "Age": None,
        "Metro": None,
        "Provider Type": None,
        "Hospital Type": None,
        "Age From": None,
        "Age To": None,
        "City": None,
        "Limit Applicable On": None,
        "% Applicable": None,
        "Amount Applicable": None,
        "Applicable Limit": None,
        "Action": None
    },
    "Optical Limit": {
        "Is Inclusive of Implant(Glass/Lens)": None,
        "Over And Above Policy Sum Insured?": None,
        "Select Type": None,
        "Lens": {
            "Sum Insured": None,
            "% Limit Applicable On": None,
            "Limit Percentage": None,
            "Limit Amount": None,
            "Implant Applicable": None,
            "Implant Amount": None,
            "Applicability": None,
            "Action": None
        },
        "Glass": {
            "Sum Insured": None,
            "% Limit Applicable On": None,
            "Limit Percentage": None,
            "Limit Amount": None,
            "Implant Applicable": None,
            "Implant Amount": None,
            "Applicability": None,
            "Action": None
        },
        "Sum Insured": None,
        "% Limit Applicable On": None,
        "Limit Percentage": None,
        "Limit Amount": None,
        "Applicability": None,
        "Action": None
    },
    "Organ Donor Medical Expenses": {
        "Over And Above Policy Sum Insured?": None,
        "Relationship": None,
        "Age": None,
        "Metro": None,
        "Provider Type": None,
        "Hospital Type": None,
        "Age From": None,
        "Age To": None,
        "City": None,
        "Limit Applicable On": None,
        "% Applicable": None,
        "Amount Applicable": None,
        "Applicable Limit": None,
        "Action": None
    },
    "Personal Accident Cover": {
        "Over And Above Policy Sum Insured?": None,
        "Sum Insured": None,
        "PA Sum Insured": None,
        "Action": None
    },
    "Pre Existing Disease Benefit": {
        "Member Waiting Period": None,
        "Family Waiting Period": None,
        "Policy Waiting Period": None
    },
    "Psychiatric Care": {
        "Over And Above Policy Sum Insured?": None,
        "Sum Insured": None,
        "% Limit Applicable On": None,
        "Limit Percentage": None,
        "Limit Amount": None,
        "Applicability": None,
        "Action": None,
        "Relationship": None,
        "Age": None,
        "Metro": None,
        "Provider Type": None,
        "Hospital Type": None,
        "Age From": None,
        "Age To": None,
        "City": None,
        "Limit Applicable On": None,
        "% Applicable": None,
        "Amount Applicable": None,
        "Applicable Limit": None,
        "Action": None
    },
    "Recovery Benefit": {
        "Recovery Period": None,
        "Over And Above Policy Sum Insured?": None,
        "Applicable From": None,
        "Relationship": None,
        "Age": None,
        "Metro": None,
        "Provider Type": None,
        "Hospital Type": None,
        "Age From": None,
        "Age To": None,
        "City": None,
        "Limit Applicable On": None,
        "% Applicable": None,
        "Amount Applicable": None,
        "Applicable Limit": None,
        "Action": None
    },
    "Referral Hospital Care": {
        "Relationship": None,
        "Age": None,
        "Metro": None,
        "Provider Type": None,
        "Hospital Type": None,
        "Age From": None,
        "Age To": None,
        "City": None,
        "Limit Applicable On": None,
        "% Applicable": None,
        "Amount Applicable": None,
        "Applicable Limit": None,
        "Action": None
    },
    "Surgical Benefit": {
        "Over And Above Policy Sum Insured?": None,
        "Surgeries Covered": None,
        "Relationship": None,
        "Age": None,
        "Metro": None,
        "Provider Type": None,
        "Hospital Type": None,
        "Age From": None,
        "Age To": None,
        "City": None,
        "Limit Applicable On": None,
        "% Applicable": None,
        "Amount Applicable": None,
        "Applicable Limit": None,
        "Action": None
    },
    "Top Up Cover": {
        "Over And Above Policy Sum Insured?": None,
        "Sum Insured": None,
        "Top Up Sum Insured": None,
        "Action": None
    },
    "Vaccination/Immunization Cover": {
        "Over And Above Policy Sum Insured?": None,
        "Relationship": None,
        "Age": None,
        "Metro": None,
        "Provider Type": None,
        "Hospital Type": None,
        "Age From": None,
        "Age To": None,
        "City": None,
        "Limit Applicable On": None,
        "% Applicable": None,
        "Amount Applicable": None,
        "Applicable Limit": None,
        "Action": None
    }
}


def flatten_big_dict(big_dict):
    """
    Flattens a nested dictionary of the form:
        { outer_key: { inner_key: value, ... }, ... }
    into a single flat dictionary { inner_key: value, ... }.
    """
    common_dict = {}
    for outer_key, inner_dict in big_dict.items():
        if isinstance(inner_dict, dict):
            for inner_key, value in inner_dict.items():
                common_dict[inner_key] = value
    return common_dict


import pandas as pd
import numpy as np

def dict_to_excel_rowwise(flat_dict, output_path):
    """
    Converts a flat dictionary to a properly aligned Excel table.
    - Expands list values into rows.
    - Scalar values are repeated or left blank appropriately.
    - NaN/None are replaced with "".
    """

    # 1️⃣ Find the max list length (to know how many rows we need)
    max_len = 1
    for v in flat_dict.values():
        if isinstance(v, list):
            max_len = max(max_len, len(v))

    # 2️⃣ Build a DataFrame with aligned rows
    data = {}
    for col, val in flat_dict.items():
        if isinstance(val, list):
            # Pad shorter lists with blanks
            filled = [(x if x not in [None, np.nan] else "") for x in val]
            filled += [""] * (max_len - len(filled))
            data[col] = filled
        else:
            # Repeat scalar for all rows
            v = "" if val in [None, np.nan] else val
            data[col] = [v] * max_len

    df = pd.DataFrame(data)

    # 3️⃣ Write to Excel
    df.to_excel(output_path, index=False)
    print(f"✅ Excel created successfully at: {output_path}")


def run_addon_coverage(pdf_path,source_dir):
    
    output_folder = source_dir  # define your working folder
    os.makedirs(output_folder, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(pdf_path))[0] 
    json_filename=os.path.join(source_dir, f"{base_name}_addon_covers.json")
    addon_covers_dict=load_json_to_dict(json_filename)
    addon_present =extract_yes_keys_from_dict(addon_covers_dict)

    addon_coverage_func_dict={
            "Ambulance Cover": get_ambulance_cover,
            "Convalescence Benefit":get_convalescence_benefit ,
            "Daily/Hospital Cash Benefit":get_daily_cash_allowance ,
            "Doctor & Nurse Home Visit Cover": get_doctor_and_nurse_cover,
            "Out Patient Cover": get_opc_cover,
            "Critical Illness Benefit": get_critical_illness
}
    results=[]
    for i in addon_present:
        result=addon_coverage_func_dict[i](pdf_path)
        results.append(result)
    big_dict= flatten_benefit_schema(benefit_schema)

    for res in results:
        if not isinstance(res, dict):
            continue
        
        # Each result should have a single key like "Ambulance Cover"
        for addon_name, addon_data in res.items():
            if addon_name not in big_dict:
                print(f"⚠️ Skipping unknown add-on: {addon_name}")
                continue
            
            # Go field by field
            for field_key, field_val in addon_data.items():
                if field_key in big_dict[addon_name]:
                    big_dict[addon_name][field_key] = field_val
                else:
                    # Optional: handle partial matches or suffix matches
                    matched_key = next(
                        (k for k in big_dict[addon_name].keys() if k.lower().endswith(field_key.lower())),
                        None
                    )
                    if matched_key:
                        big_dict[addon_name][matched_key] = field_val
                    else:
                        print(f"⚠️ Field '{field_key}' not found in template for '{addon_name}'")

    
    ans=flatten_big_dict(big_dict)
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]  
    output_path = os.path.join(output_folder, f"{base_name}_addon_coverages.xlsx")

    dict_to_excel_rowwise(ans,output_path)



if __name__ =="__main__":
    pdf_path=r"/home/ubuntu/THINESH_WS/royal_sunadaram_hscope/samples/HG Cases/HG00000006000124.pdf"
    source_dir=r""

    run_addon_coverage(pdf_path,source_dir)

        
