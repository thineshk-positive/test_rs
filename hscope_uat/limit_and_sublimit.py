# %%
import pymupdf4llm
import re
from collections import OrderedDict
import subprocess
import json
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass
from langchain_ollama import ChatOllama
import json
import re
import logging
import os
from openpyxl import Workbook 
from typing import Dict, Any, List
from hscope_uat.UAT_config import *

from hscope_uat.helper.get_si import load_sum_insured_from_json

# ----------------------------
# Initialize Chat
# ----------------------------
chat = ChatOllama(
    model="gpt-oss:20b",
    temperature=0.0,
    top_p=1.0,
    max_tokens=1024,
    verbose=False
)


import pandas as pd

# =========================
# Configure logging
# =========================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =========================
# PDF to Markdown
# =========================
def pdf_to_md(pdf_path: str) -> str:
    return pymupdf4llm.to_markdown(pdf_path)

# =========================
# Normalize endorsement IDs
# =========================
def normalize_id(eid: str) -> str:
    m = re.match(r"^(\d+)(?:\(?([a-zivx]+)\)?)?$", eid, re.IGNORECASE)
    if not m:
        return eid
    num, suffix = m.groups()
    return f"{num}({suffix.lower()})" if suffix else num

# =========================
# Extract endorsements
# =========================
def extract_endorsements(md_text: str, debug: bool = False) -> OrderedDict:
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
        re.compile(pat, re.IGNORECASE) for pat in [
            r"^group health policy", r"^uin[:\s]", r"^irda", r"^policy number",
            r"^name of the insured", r"^period of insurance", r"endorsements attached",
            r"Group Health Policy – Endorsements", r"^Page\s+\*\*\d+\*\*\s+of\s+\*\*\d+\*\*$",
            r"^\*\*Policy Number.*\*\*$", r"^\*\*Name of the Insured.*\*\*$",
            r"^\*\*Period of Insurance.*\*\*$", r"^//\s*\d+\s*//$", r"^royal sundaram.*",
            r"^regd office.*", r"^corporate office.*", r"^email[:\s].*", r"^website[:\s].*",
            r"^ph[:\s].*", r"^.*irda regn.*", r"^.*cin[-:\s].*", r".*\bchennai\s*\d{3}\s*\d{3}.*"
        ]
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
                print(f"LINE {i}: ✅ {line} -> {canonical_id}")

            if current_id and buffer:
                endorsements[current_id] = "\n".join(clean_buffer(buffer)).strip()

            current_id = canonical_id
            buffer = [line]
        elif current_id:
            buffer.append(line)

    if current_id and buffer:
        endorsements[current_id] = "\n".join(clean_buffer(buffer)).strip()

    return endorsements

# =========================
# Analysis Result Dataclass
# =========================
@dataclass
class AnalysisResult:
    room_rent_member_limit_applicable: str
    room_rent_member_limit_percentage: Any
    room_rent_combined_limit: str
    room_rent_claim_level_limit_applicable: str
    room_rent_daily_level_limit_applicable: str
    raw_response: Optional[str] = None
    error: Optional[str] = None

# =========================
# Insurance Policy Analyzer
# =========================
class InsurancePolicyAnalyzer:
    def __init__(self, model: str = "gpt-oss:20b"):
        self.model = model
        self.prompt_template = self._build_prompt_template()

    # ----------------------------
    # Prompt template for float %
    # ----------------------------
    def _build_prompt_template(self) -> str:
        return """
You are an expert insurance policy analyst. Analyze the given text for endorsement no. 5 information and extract specific data points.

=== ANALYSIS RULES ===

[ROOM RENT SECTION]

1. "Room_rent_Member_Level_Limit_Applicable":
   - "Yes" = If relationship-wise member limit IS applicable
   - "No" = If relationship-wise member limit is NOT applicable
   - "Not Mentioned" = If info not found

1a. "Room_rent_Member_Level_Limit_Percentage":
   - Extract numeric percentage (float) if "Room_rent_Member_Level_Limit_Applicable" is "Yes"
   - Example: 50.0
   - If not applicable, set as empty string ""

2. "Room_rent_Combined_Limit_For_Room_And_Nursing_Charges":
   - Always "No"

3. "Room_rent_Daily_Level_Limit_Applicable":
   - "Yes" = If "Per Day wise limit" applicable
   - "No" = If not
   - "Not Mentioned" = If info not found

3a. "Room_rent_Claim_Level_Limit_Applicable":
   - "Yes" = If daily level limit is "No"
   - "No" = If daily level limit is "Yes"
   - "Not Mentioned" = If parent is "Not Mentioned"

=== OUTPUT FORMAT ===
Respond ONLY with JSON:
{{
    "Room_rent_Member_Level_Limit_Applicable": "",
    "Room_rent_Member_Level_Limit_Percentage": "",
    "Room_rent_Combined_Limit_For_Room_And_Nursing_Charges": "",
    "Room_rent_Daily_Level_Limit_Applicable": "",
    "Room_rent_Claim_Level_Limit_Applicable": ""
}}

=== TEXT TO ANALYZE ===
\"\"\"{chunk_text}\"\"\""""

    # ----------------------------
    # Apply business logic with float %
    # ----------------------------
    def _apply_business_logic(self, data: Dict[str, Any]) -> Dict[str, Any]:
        validated_data = data.copy()
        member_limit_applicable = validated_data.get("Room_rent_Member_Level_Limit_Applicable", "")

        if member_limit_applicable != "Yes":
            validated_data["Room_rent_Member_Level_Limit_Percentage"] = ""
        else:
            val = validated_data.get("Room_rent_Member_Level_Limit_Percentage", "")
            try:
                if isinstance(val, str):
                    val = val.strip().replace('%', '')
                validated_data["Room_rent_Member_Level_Limit_Percentage"] = float(val)
            except:
                validated_data["Room_rent_Member_Level_Limit_Percentage"] = ""

        validated_data["Room_rent_Combined_Limit_For_Room_And_Nursing_Charges"] = "No"

        daily_limit_applicable = validated_data.get("Room_rent_Daily_Level_Limit_Applicable", "")
        if daily_limit_applicable == "Yes":
            validated_data["Room_rent_Claim_Level_Limit_Applicable"] = "No"
        elif daily_limit_applicable == "No":
            validated_data["Room_rent_Claim_Level_Limit_Applicable"] = "Yes"
        elif daily_limit_applicable == "Not Mentioned":
            validated_data["Room_rent_Claim_Level_Limit_Applicable"] = "Not Mentioned"

        return validated_data

    # ----------------------------
    # Query Ollama
    # ----------------------------
    def query_ollama(self, prompt: str) -> str:
        try:
            result = subprocess.run(
                ["ollama", "run", self.model],
                input=prompt.encode("utf-8"),
                capture_output=True,
                timeout=120
            )
            if result.returncode != 0:
                raise RuntimeError(result.stderr.decode("utf-8"))
            return result.stdout.decode("utf-8").strip()
        except Exception as e:
            raise RuntimeError(f"Ollama query failed: {str(e)}")

    # ----------------------------
    # Extract and validate JSON
    # ----------------------------
    def extract_and_validate_json(self, response: str) -> Dict[str, Any]:
        """Extract, validate, and clean JSON from model response safely."""
        try:
            import re, json

            # Find first { ... } block (non-greedy)
            match = re.search(r"\{.*?\}\s*(?=\n|$)", response, re.DOTALL)
            if not match:
                return {"error": "No JSON detected", "raw_response": response}

            json_text = match.group(0)
            parsed_data = json.loads(json_text)

            # Validate required fields
            required_fields = [
                "Room_rent_Member_Level_Limit_Applicable",
                "Room_rent_Member_Level_Limit_Percentage",
                "Room_rent_Combined_Limit_For_Room_And_Nursing_Charges",
                "Room_rent_Daily_Level_Limit_Applicable",
                "Room_rent_Claim_Level_Limit_Applicable",
            ]
            missing_fields = [f for f in required_fields if f not in parsed_data]
            if missing_fields:
                return {"error": f"Missing required fields: {missing_fields}", "raw_response": response}

            # Apply business logic
            return self._apply_business_logic(parsed_data)

        except json.JSONDecodeError as e:
            return {"error": f"Invalid JSON: {str(e)}", "raw_response": response}
        except Exception as e:
            return {"error": f"Processing error: {str(e)}", "raw_response": response}



    # ----------------------------
    # Analyze a chunk
    # ----------------------------
    def analyze_document_chunk(self, chunk_text: str) -> AnalysisResult:
        try:
            prompt = self.prompt_template.format(chunk_text=chunk_text)
            raw_response = self.query_ollama(prompt)
            json_result = self.extract_and_validate_json(raw_response)

            if "error" in json_result:
                return AnalysisResult(
                    room_rent_member_limit_applicable="Error",
                    room_rent_member_limit_percentage="Error",
                    room_rent_combined_limit="Error",
                    room_rent_claim_level_limit_applicable="Error",
                    room_rent_daily_level_limit_applicable="Error",
                    raw_response=raw_response,
                    error=json_result["error"]
                )

            return AnalysisResult(
                room_rent_member_limit_applicable=json_result["Room_rent_Member_Level_Limit_Applicable"],
                room_rent_member_limit_percentage=json_result["Room_rent_Member_Level_Limit_Percentage"],
                room_rent_combined_limit=json_result["Room_rent_Combined_Limit_For_Room_And_Nursing_Charges"],
                room_rent_claim_level_limit_applicable=json_result["Room_rent_Claim_Level_Limit_Applicable"],
                room_rent_daily_level_limit_applicable=json_result["Room_rent_Daily_Level_Limit_Applicable"],
                raw_response=raw_response
            )

        except Exception as e:
            return AnalysisResult(
                room_rent_member_limit_applicable="Error",
                room_rent_member_limit_percentage="Error",
                room_rent_combined_limit="Error",
                room_rent_claim_level_limit_applicable="Error",
                room_rent_daily_level_limit_applicable="Error",
                error=str(e)
            )

# =========================
# Utility to get endorsements content
# =========================
def get_endorsements_content(pdf_path: str,desired_endorsements=["5(i)", "5a", "5b"]) -> str:
    md_text = pdf_to_md(pdf_path)
    endorsements = extract_endorsements(md_text)
    desired_endorsements = [normalize_id(eid) for eid in desired_endorsements]
    matching = {eid: endorsements[eid] for eid in desired_endorsements if eid in endorsements}
    return "\n\n".join(matching.values())

# =========================
# Main function
# =========================
def main(pdf_path: str):
    analyzer = InsurancePolicyAnalyzer(model="gpt-oss:20b")
    sample_text = get_endorsements_content(pdf_path)

    print("=" * 60)
    print("INSURANCE POLICY ANALYSIS")
    print("=" * 60)

    result = analyzer.analyze_document_chunk(sample_text)

    print(f"Room Rent Member Level Limit Applicable: {result.room_rent_member_limit_applicable}")
    print(f"Room Rent Member Level Limit Percentage: {result.room_rent_member_limit_percentage}")
    print(f"Room Rent Combined Limit: {result.room_rent_combined_limit}")
    print(f"Room Rent Daily Level Limit Applicable: {result.room_rent_daily_level_limit_applicable}")
    print(f"Room Rent Claim Level Limit Applicable: {result.room_rent_claim_level_limit_applicable}")

    if result.error:
        print(f"\nError: {result.error}")

    print("\n" + "=" * 60)
    print("RAW LLM RESPONSE:")
    print("=" * 60)
    print(result.raw_response)

    return result


# Updated utility function with "Not Mentioned" -> "No" logic
def extract_room_rent_fields(result: AnalysisResult) -> dict:
    """
    Receives an AnalysisResult object and returns only the Room Rent fields as a dict.
    Treats 'Not Mentioned' as 'No' for all fields.
    """
    # Initial extraction
    room_rent_dict = {
        "Room_rent_Member_Level_Limit_Applicable": result.room_rent_member_limit_applicable,
        "Room_rent_Member_Level_Limit_Percentage": result.room_rent_member_limit_percentage,
        "Room_rent_Combined_Limit": result.room_rent_combined_limit,
        "Room_rent_Claim_Level_Limit_Applicable": result.room_rent_claim_level_limit_applicable,
        "Room_rent_Daily_Level_Limit_Applicable": result.room_rent_daily_level_limit_applicable
    }

    # Overlay: convert "Not Mentioned" -> "No"
    for key, value in room_rent_dict.items():
        if value == "Not Mentioned":
            room_rent_dict[key] = "No"

    return room_rent_dict


def get_room_limits(pdf_path):
    result=main(pdf_path)
    extracted_room_feilds=extract_room_rent_fields(result)
    return extracted_room_feilds


def claim_level_limits(extracted_room_feilds):
    if extracted_room_feilds['Room_rent_Claim_Level_Limit_Applicable']=='No':
        return  {"claim_level": {
        "Room Charge": {
            "Room_charges": {
                "Sum Insured": "",
                "% Limit Applicable On": "",
                "% Limit": "",
                "Limit": "",
                "Applicability": ""
            },
            "ICU_charges": {
                "Sum Insured": "",
                "% Limit Applicable On": "",
                "% Limit": "",
                "Limit": "",
                "Applicability": ""
            }
        },
        "Nursing Charges": {
            "Room_charges": {
                "Sum Insured": "",
                "% Limit Applicable On": "",
                "% Limit": "",
                "Limit": "",
                "Applicability": ""
            },
            "ICU_charges": {
                "Sum Insured": "",
                "% Limit Applicable On": "",
                "% Limit": "",
                "Limit": "",
                "Applicability": ""
            }
        }
    }
}


def overaly_endt_context(pdf_path):
    endnt_context=get_endorsements_content(pdf_path,desired_endorsements=['5a','5b'])
    if endnt_context =="":
        endnt_context="There is no 5a or 5b available, so you can cosnider No for removal of room limit"
    return endnt_context


def extract_removal_of_limit(endorsement_text: str) -> dict:
    """
    Extracts only Endt. No. 5A or 5B and uses LLM to determine if 
    Room/Boarding Expenses limitation exists.

    Returns:
      {"Removal of Limit Applicable": "Yes"} or {"Removal of Limit Applicable": "No"}
    """

    # --- STRICT REGEX for only 5A or 5B ---
    endt_5ab_pattern = r"(Endt\.\s*No\.\s*5\s*[ab][\s\S]*?)(?=Endt\.|$)"
    matches = re.findall(endt_5ab_pattern, endorsement_text, flags=re.IGNORECASE)

    if not matches:
        return {"Removal of Limit Applicable": "No"}

    relevant_text = "\n".join(matches).strip()

    # ---- LLM prompt ----
    prompt = f"""
You are an expert insurance policy analyst.
Analyze the following endorsement text (only Endt. No. 5A or 5B):

Decide if it explicitly or implicitly mentions a restriction such as:
"Room, Boarding Expenses as provided by the Hospital/Nursing Home is subject to a limit of ... per day"
or any similar wording indicating a cap or limit on Room Rent or Boarding Expenses.

If such a limitation exists, respond:
{{"Removal of Limit Applicable": "Yes"}}

If it does not, respond:
{{"Removal of Limit Applicable": "No"}}

Respond ONLY with JSON.

--- Endorsement Text ---
{relevant_text}
"""

    try:
        # Replace with your actual LLM call
        response = chat.invoke(prompt)  # Example: ChatOllama, OpenAI, etc.
        if hasattr(response, "content"):
            response = response.content

        match = re.search(r"\{[\s\S]*\}", response)
        if not match:
            return {"Removal of Limit Applicable": "No"}

        data = json.loads(match.group())
        val = str(data.get("Removal of Limit Applicable", "No")).strip().lower()
        return {"Removal of Limit Applicable": "Yes" if val == "yes" else "No"}

    except Exception:
        return {"Removal of Limit Applicable": "No"}



def generate_actuals_for_room_icu_dict(si_list):
    """
    Generate dictionary structure with Room and ICU charges for given SI list,
    with 100% for both Percentage and corresponding Amounts.

    Args:
        si_list (list): List of Sum Insured strings, e.g., ['1,00,000', '2,00,000']

    Returns:
        dict: Structured dictionary with 100% Room & ICU charges
    """
    def parse_si(si_str):
        return float(si_str.replace(",", ""))
    
    print('3')

    #room_amounts = [int(parse_si(si)) for si in si_list]  # 100% of SI
    #icu_amounts = [int(parse_si(si)) for si in si_list]   # 100% of SI

    
    room_amounts = [si for si in si_list]  # 100% of SI
    icu_amounts = [si for si in si_list] 

    return {
        "Sum Insured": si_list,
        "Room Charge": {
            "Percentage": [100.0] * len(si_list),
            "Amount": room_amounts
        },
        "ICU Charge": {
            "Percentage": [100.0] * len(si_list),
            "Amount": icu_amounts
        },
        "Sum Insured Excluded": []
    }


def calculate_missing_room_icu(data: dict) -> dict:
    """
    Fill missing Room/ICU % or Amount using the corresponding Sum Insured.
    Assumes 'Sum Insured' are strings like '1,00,000'.
    """
    def parse_si(si_str):
        return float(si_str.replace(",", ""))
    
    print('1')

    #si_values = [parse_si(s) for s in data["Sum Insured"]]
    si_values = [s for s in data["Sum Insured"]]


    for section in ["Room Charge", "ICU Charge"]:
        perc_list = data[section]["Percentage"]
        amt_list = data[section]["Amount"]

        for i in range(len(si_values)):
            si = si_values[i]
            perc = perc_list[i]
            amt = amt_list[i]

            # If Percentage is missing but Amount is present
            if perc is None and amt is not None:
                perc_list[i] = round((amt / si) * 100, 2)

            # If Amount is missing but Percentage is present
            if amt is None and perc is not None:
                amt_list[i] = int(round(si * perc / 100, 0))

            print("calc_missing_room_icu")

    return data



import json
import re
from langchain_ollama import ChatOllama

import json
import re
from langchain_ollama import ChatOllama

#chat = ChatOllama(model="llama3.1")

def extract_room_icu_limits(snippet: str, si_list: list) -> dict:
    """
    Extract Room Rent & ICU limits and Sum Insured exclusions using only LLM.
    Handles % or ₹ values, exclusion patterns like 'up to', 'not applicable', etc.
    Ensures stable structure and fills missing data safely.
    """
    prompt = f"""
You are an expert insurance policy analyst.

Read the endorsement text below and extract:
1. Room Rent charge limit (either % or ₹ per day)
2. ICU charge limit (either % or ₹ per day)
3. Which Sum Insured (SI) amounts are excluded.

---
### Strict JSON Output Format:
{{
  "Sum Insured": [],
  "Room Charge": {{"Percentage": [], "Amount": []}},
  "ICU Charge": {{"Percentage": [], "Amount": []}},
  "Sum Insured Excluded": []
}}
---

### Extraction & Interpretation Rules:

#### 1. Reference Context
- Use this reference Sum Insured list: {si_list}
- Extract only **numeric values** (e.g., 2 for 2%, 2500 for ₹2,500).
- If a value is not mentioned, use `null`.

#### 2. Handling of “Actuals”, “Unlimited”, and “No Limit”
- If the text contains **"as per actuals"**, **"at actuals"**, **"actuals"**, **"no limit"**, **"without limit"**, **"unlimited"**, or similar phrases for Room Rent or ICU:
  - Interpret that as **100% coverage** for that charge.
  - Example: 
    - “ICU – at actuals” → ICU Charge Percentage = [100]
    - “No limit for ICU” → ICU Charge Percentage = [100]
    - “Unlimited Room Rent” → Room Charge Percentage = [100]

#### 3. Metro vs Non-Metro
- If both **Metro** and **Non-Metro** (or similar) values are mentioned:
  - **Always extract only the Metro city value** (ignore Non-Metro, Other cities, Rest of India, etc.).
  - Example: “Metro – ₹8,000/day, Non-Metro – ₹7,000/day” → use ₹8,000/day.
  - If only one is mentioned, take that one.
  - If text specifies “for Metro cities” within a percentage (e.g., “2% for Metro”), extract that percentage.

#### 4. Sum Insured Exclusion Logic
- “Up to ₹3,00,000” → exclude all SIs ≤ 3,00,000.
- “Not applicable for ₹4,00,000 and ₹5,00,000” → exclude those SIs specifically.
- Apply extracted Room & ICU limits to all remaining (included) SIs.

#### 5. Formatting Rules
- For each key, always provide lists even if only one value exists:
  - Example: `"Percentage": [2]`
- If both % and ₹ are present, record both in their respective fields.
- Do **not** include any textual explanation or reasoning — respond **strictly in JSON only**.

---
### Endorsement Text:
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

    # Guarantee the nested keys exist
    for section in ["Room Charge", "ICU Charge"]:
        data.setdefault(section, {})
        for field in ["Percentage", "Amount"]:
            vals = data[section].get(field)
            if not isinstance(vals, list):
                vals = [] if vals is None else [vals]
            data[section][field] = vals

    # ---- Normalize and clean numbers ----
    def clean_num(val):
        if val in [None, "", "null"]:
            return None
        try:
            print("2")
            v = re.search(r"(\d+(\.\d+)?)", str(val).replace(",", ""))
            return float(v.group(1)) if v else None
        except Exception:
            return None

    for section in ["Room Charge", "ICU Charge"]:
        for field in ["Percentage", "Amount"]:
            vals = data[section][field]
            cleaned = [clean_num(v) for v in vals]
            data[section][field] = cleaned

    n = len(data["Sum Insured"])
    # Pad lists to match Sum Insured count
    for section in ["Room Charge", "ICU Charge"]:
        for field in ["Percentage", "Amount"]:
            vals = data[section][field]
            if len(vals) == 0:
                vals = [None] * n
            elif len(vals) == 1:
                vals = vals * n
            elif len(vals) < n:
                vals += [vals[-1]] * (n - len(vals))
            data[section][field] = vals

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

    for section in ["Room Charge", "ICU Charge"]:
        for field in ["Percentage", "Amount"]:
            data[section][field] = fill_forward(data[section][field])

    return data




def extract_daily_room_level_limit(end_context,si_list):
    result=extract_room_icu_limits(end_context,si_list)
    filled_data = calculate_missing_room_icu(result)

    return filled_data

def map_room_icu_limits_to_schema(extracted_result):
    """
    Maps the extracted room/ICU limit details into the standard nested schema format.

    Parameters:
        extracted_result (dict): The dictionary output from daily_level_limit function.

    Returns:
        dict: Mapped schema with 'Daily level' containing Room Charge and Nursing Charges.
    """

    # Extract base data
    si_list = extracted_result.get('Sum Insured', [])
    room_data = extracted_result.get('Room Charge', {})
    icu_data = extracted_result.get('ICU Charge', {})

    # Helper to build charge sections
    def build_charge_section(percentages, amounts):
        return {
            "Sum Insured": si_list,
            "% Limit Applicable On": ["Sum Insured"] * len(si_list),
            "% Limit": percentages,
            "Limit": amounts,
            "Applicability": ["Lower"] * len(si_list)
        }

    # Build each charge type
    room_charges = build_charge_section(
        room_data.get('Percentage', []),
        room_data.get('Amount', [])
    )

    icu_charges = build_charge_section(
        icu_data.get('Percentage', []),
        icu_data.get('Amount', [])
    )

    # Final schema nested under "Daily level"
    schema_mapped_result = {
        "Daily level": {
            "Room Charge": {
                "Room_charges": room_charges,
                "ICU_charges": icu_charges
            },
            "Nursing Charges": {
                "Room_charges": room_charges,
                "ICU_charges": icu_charges
            }
        }
    }

    return schema_mapped_result



def daily_level_limit(extracted_room_feilds,si_list,pdf_path):
    if extracted_room_feilds['Room_rent_Daily_Level_Limit_Applicable']=='Yes':
        endnt_context=overaly_endt_context(pdf_path)
        removal_result=extract_removal_of_limit(endnt_context)

        if removal_result['Removal of Limit Applicable'] =="No":
            endnt_context=get_endorsements_content(pdf_path,desired_endorsements=['5(i)'])
            result=extract_daily_room_level_limit(endnt_context,si_list)
            value=map_room_icu_limits_to_schema(result)

            return value

        else: 
            result = generate_actuals_for_room_icu_dict(si_list)
            value=map_room_icu_limits_to_schema(result)
            return value

    else:
        return {"Daily level": {
        "Room Charge": {
            "Room_charges": {
                "Sum Insured": "",
                "% Limit Applicable On": "",
                "% Limit": "",
                "Limit": "",
                "Applicability": ""
            },
            "ICU_charges": {
                "Sum Insured": "",
                "% Limit Applicable On": "",
                "% Limit": "",
                "Limit": "",
                "Applicability": ""
            }
        },
        "Nursing Charges": {
            "Room_charges": {
                "Sum Insured": "",
                "% Limit Applicable On": "",
                "% Limit": "",
                "Limit": "",
                "Applicability": ""
            },
            "ICU_charges": {
                "Sum Insured": "",
                "% Limit Applicable On": "",
                "% Limit": "",
                "Limit": "",
                "Applicability": ""
            }
        }
    }
}

import re
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
        spl_cond_context= """There is no special conditon is provided, so you can mention member Level Limit Applicable? = "NO"
            "Member Level Limit(%)" = ""
            "Claim Level Limit Applicable?" = "NO"
            """
        endnt_context['special condition']= spl_cond_context
        return endnt_context
    return endnt_context




# ----------------------------
# Function to extract Professional Fees fields
# ----------------------------
def extract_professional_fees() -> dict:
    return {'pro_fees': {'Member Level Limit Applicable?': 'No',
  'Member Level Limit(%)': '',
  'Claim Level Limit Applicable': ''}}



# def extract_professional_fees(snippet: str) -> dict:
#     prompt = f"""
# You are an expert insurance policy analyst. Extract the following fields
# from the given text snippet:

# "Professional Fees": {{
#     "Member Level Limit Applicable?": "",
#     "Member Level Limit(%)": "",
#     "Claim Level Limit Applicable": ""
# }}

# Rules:
# 1. "Member Level Limit Applicable?" → Yes or No
# 2. "Member Level Limit(%)" → Only extract numerical % if member level applicable is Yes, else return ""
# 3. "Claim Level Limit Applicable" → Yes or No

# Text Snippet:
# \"\"\"{snippet}\"\"\" 

# Respond ONLY with JSON like this:
# {{
#     "Member Level Limit Applicable?": "",
#     "Member Level Limit(%)": "",
#     "Claim Level Limit Applicable": ""
# }}
# """

#     try:
#         # Query ChatOllama
#         raw_response = chat.invoke(prompt)
#         if hasattr(raw_response, "content"):
#             raw_response = raw_response.content  # <-- Convert AIMessage to string

#         # Extract JSON from response
#         match = re.search(r"\{[\s\S]*\}", raw_response)
#         if not match:
#             logging.warning("No JSON detected in LLM response. Returning defaults.")
#             return {
#                 "Member Level Limit Applicable?": "No",
#                 "Member Level Limit(%)": "",
#                 "Claim Level Limit Applicable": "No"
#             }

#         data = json.loads(match.group())

#         # Ensure defaults
#         member_applicable = data.get("Member Level Limit Applicable?", "No")
#         if member_applicable not in ["Yes", "No"]:
#             member_applicable = "No"

#         claim_applicable = data.get("Claim Level Limit Applicable", "No")
#         if claim_applicable not in ["Yes", "No"]:
#             claim_applicable = "No"

#         member_percent = data.get("Member Level Limit(%)", "")
#         if member_applicable != "Yes":
#             member_percent = ""

#         return {'pro_fees':{
#             "Member Level Limit Applicable?": member_applicable,
#             "Member Level Limit(%)": member_percent,
#             "Claim Level Limit Applicable": claim_applicable
#         }}

#     except Exception as e:
#         logging.error(f"Failed to extract professional fees: {e}")
#         return {'pro_fees':{
#             "Member Level Limit Applicable?": "No",
#             "Member Level Limit(%)": "",
#             "Claim Level Limit Applicable": "No"
#         }}

#The above version can be used to extract from the text, now it is hardcoded.


def pro_fees_claim_level_limit(extract_pro_fees):
    if extract_pro_fees['pro_fees']['Claim Level Limit Applicable']=='':
        return{"pro_fees_claim_level_limt":
               {"Select Type": "",             # (LLM json → type of fee, e.g., Surgeon/Consultant)
        "Sum Insured": "",             # (from suminsured function)
        "% Limit Applicable On": "",   # (default = sum insured or sublimit base)
        "% Limit": "",                 # (% value from LLM json)
        "Limit": "",                   # (calculated value)
        "Applicability": ""}}


def return_others_default():
    return {"Others": {
        # ----------------- Member Level -----------------
        "Member Level Limit Applicable?": "",  
        "Member Level Limit(%)": "",           

        # ----------------- Claim Level -----------------
        "Claim Level Limit Applicable": "",    

        # ----------------- Charges -----------------
        "Select Type": "",             
        "Sum Insured": "",             
        "% Limit Applicable On": "",   
        "% Limit": "",                 
        "Limit": "",                   
        "Applicability": ""            
    }}


def extract_limit_and_subLimits(pdf_path,si_list):
    extracted_room_feilds=get_room_limits(pdf_path)
    extracted_claim_level_limits=claim_level_limits(extracted_room_feilds)
    extracted_daily_level_limits=daily_level_limit(extracted_room_feilds,si_list,pdf_path)
    print("extracted daily level limits")
    md_text=pdf_to_md(pdf_path)
    spl_cond=overaly_specail_condtion(md_text)
    extracted_pro_fees = extract_professional_fees()
    pro_fees_claim_limit=pro_fees_claim_level_limit(extracted_pro_fees)
    print(extracted_pro_fees,pro_fees_claim_limit)
    pro_fees = {**extracted_pro_fees, **pro_fees_claim_limit}
    others=return_others_default()

    ordered_room_feilds = {k: v for k, v in extracted_room_feilds.items() if k != 'Room_rent_Daily_Level_Limit_Applicable'}
    result={**ordered_room_feilds,**extracted_claim_level_limits,
     'Room_rent_Daily_Level_Limit_Applicable':extracted_room_feilds['Room_rent_Daily_Level_Limit_Applicable'],
     **extracted_daily_level_limits,**pro_fees, **others
     }
    return result

def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)




def run_limit_and_sublimit(pdf_path,source_dir):
    
    si_list=load_sum_insured_from_json(pdf_path)


    daily_cols = [
        'Daily level_Room Charge_Room_charges_Sum Insured', 
        'Daily level_Room Charge_Room_charges_% Limit Applicable On', 
        'Daily level_Room Charge_Room_charges_% Limit',
        'Daily level_Room Charge_Room_charges_Limit', 
        'Daily level_Room Charge_Room_charges_Applicability', 
        'Daily level_Room Charge_ICU_charges_Sum Insured', 
        'Daily level_Room Charge_ICU_charges_% Limit Applicable On', 
        'Daily level_Room Charge_ICU_charges_% Limit', 
        'Daily level_Room Charge_ICU_charges_Limit', 
        'Daily level_Room Charge_ICU_charges_Applicability', 
        'Daily level_Nursing Charges_Room_charges_Sum Insured', 
        'Daily level_Nursing Charges_Room_charges_% Limit Applicable On', 
        'Daily level_Nursing Charges_Room_charges_% Limit', 
        'Daily level_Nursing Charges_Room_charges_Limit', 
        'Daily level_Nursing Charges_Room_charges_Applicability', 
        'Daily level_Nursing Charges_ICU_charges_Sum Insured', 
        'Daily level_Nursing Charges_ICU_charges_% Limit Applicable On', 
        'Daily level_Nursing Charges_ICU_charges_% Limit', 
        'Daily level_Nursing Charges_ICU_charges_Limit', 
        'Daily level_Nursing Charges_ICU_charges_Applicability'
    ]
     



    #si_list=[100000,200000,300000]
    result=extract_limit_and_subLimits(pdf_path,si_list)

    result_dict=flatten_dict(result)
    df = pd.DataFrame({k: [v] for k, v in result_dict.items()})

    print("1")
    print(result_dict)
    #df=pd.DataFrame(result_dict)

    #print(df)

    scalar_cols = [c for c in df.columns if c not in daily_cols]

    # keep only the first row of scalar columns
    scalar_first_row = df[scalar_cols].iloc[[0]]

    # keep all rows of daily columns
    daily_df = df[daily_cols]

    # create a DataFrame where scalar columns only have the first row
    # we set the extra rows to NaN
    scalar_expanded = pd.concat(
        [scalar_first_row] + [pd.DataFrame({c:[pd.NA] for c in scalar_cols}) for _ in range(len(daily_df)-1)],
        ignore_index=True
    )

    # combine scalar columns with daily columns
    final_df = pd.concat([scalar_expanded.reset_index(drop=True),
                        daily_df.reset_index(drop=True)], axis=1)

    # keep original column order
    final_df = final_df[df.columns]



    col_order=final_df.columns
    col_list=col_order.to_list()

    df_sub = final_df[daily_cols].copy()

    # Step 2: Explode all list columns simultaneously (aligned explosion)
    # This works only if all list columns have the same length per row
    df_exploded = pd.DataFrame({
        col: df_sub[col].explode().values
        for col in df_sub.columns
    })

    # Step 3: Reset index
    df_exploded.reset_index(drop=True, inplace=True)

    # Optional: If you want to keep other columns from final_df (non-list ones)
    # you can merge them back based on the original index if needed


    non_list_cols = [c for c in final_df.columns if c not in daily_cols]

    df_exploded = (
        final_df
        .explode(daily_cols[0])  # explode one col first
        .reset_index(drop=True)
    )

    # For remaining list columns, expand each
    for col in daily_cols[1:]:
        df_exploded[col] = final_df[col].explode().values

    # Non-list columns will repeat automatically if needed
    df_exploded = pd.concat([final_df[non_list_cols], df_exploded[daily_cols]], axis=1)



    # Reorder the DataFrame
    df_exploded = df_exploded.reindex(columns=col_list)

    # Optional: reset index if needed
    df_exploded.reset_index(drop=True, inplace=True)

    df_exploded.fillna("")
    
    output_folder = source_dir  # define your working folder
    os.makedirs(output_folder, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(pdf_path))[0]  
    excel_filename = os.path.join(output_folder, f"{base_name}_limit_and_sublimit.xlsx")

    df_exploded.to_excel(excel_filename, index=False)
    print(f"Excel saved at: {excel_filename}")



    return df_exploded






if __name__=="__main__":
    pdf_path=''
    source_dir=r""
    run_limit_and_sublimit(pdf_path,source_dir)