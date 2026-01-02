from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate
import pymupdf4llm
import re
from collections import OrderedDict
import json
import os
import pandas as pd
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


# ----------------------------
# PDF → Markdown
# ----------------------------
def pdf_to_md(pdf_path):
    md_text = pymupdf4llm.to_markdown(pdf_path)
    return md_text


# ----------------------------
# Normalize IDs
# ----------------------------
def normalize_id(eid: str) -> str:
    m = re.match(r"^(\d+)(?:\(?([a-zivx]+)\)?)?$", eid, re.IGNORECASE)
    if not m:
        return eid
    num, suffix = m.groups()
    if suffix:
        return f"{num}({suffix.lower()})"
    return num


# ----------------------------
# Extract Endorsements (with cleanup)
# ----------------------------
def extract_endorsements(md_text: str, debug: bool = False):
    lines = md_text.splitlines()

    heading_pattern = re.compile(
        r"^\s*\*{0,2}\s*(?:Endorsement|Endt\.?)\.?\s*No\.?\s*"
        r"(\d+)"
        r"(?:\s*\(\s*([A-Za-z0-9ivxIVX]{1,3})\s*\)"
        r"|-(?:([A-Za-z0-9ivxIVX]{1,3}))"
        r"|([A-Za-z]{1,3})(?=\s|[-–]|$)"
        r")?",
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
        re.compile(r".*\bchennai\s*\d{3}\s*\d{3}.*", re.IGNORECASE)
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

            canonical_id = num
            if suffix and len(suffix) <= 3:
                canonical_id = f"{num}({suffix.lower()})"

            if debug:
                print(f"LINE {i}: {line}")
                print(f"       -> canonical id: {canonical_id}")

            if current_id and buffer:
                endorsements[current_id] = "\n".join(clean_buffer(buffer)).strip()

            current_id = canonical_id
            buffer = [line]

        elif current_id:
            buffer.append(line)

    if current_id and buffer:
        endorsements[current_id] = "\n".join(clean_buffer(buffer)).strip()

    return endorsements


# ----------------------------
# Helper: Number conversion
# ----------------------------
def _convert_to_number(val):
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return val
    if isinstance(val, str):
        try:
            if val.isdigit():
                return int(val)
            return float(val.replace("%", "").replace(",", "").strip())
        except Exception:
            return val
    return val


# ----------------------------
# Extract Domiciliary Limits (via LLM)
# ----------------------------
def domiciliary_limit_extractor(sample_text: str, chat) -> Dict[str, Any]:
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an expert insurance policy analyzer. "
         "Task:\n"
         "1. Check for endorsement no.19 in the given context.\n"
         "2. If endorsement 19 is available, check if any limit percentage is mentioned.\n"
         "3. If limit percentage is present, then:\n"
         "   - Set \"Domiciliary Hospital Limit Applicable\" = \"Yes\".\n"
         "   - Extract percentage and amount values.\n"
         "4. If limit percentage is not present, set it to \"No\" and leave percentage/amount as null.\n"
         "Always return strictly valid JSON."
        ),
        ("user",
         "Here is the endorsement context:\n\n{context}\n\n"
         "Return JSON like this:\n"
         "{{\n"
         "  \"Domiciliary\": {{\n"
         "    \"Domiciliary Hospital Limit Applicable\": \"Yes\" or \"No\"\n"
         "  }},\n"
         "  \"limit_percentage\": {{\"percent\": float or null, \"amount\": float or null}},\n"
         "  \"limit_amount\": {{\"percent\": float or null, \"amount\": float or null}},\n"
         "  \"sum_insured_exclusion\": float or null\n"
         "}}"
        )
    ])

    chain = prompt | chat
    result = chain.invoke({"context": sample_text})

    # Print raw LLM content for debugging
    print("RAW LLM OUTPUT:", result.content)

    text = (result.content or "").strip()

    # Remove common wrappers/code fences
    try:
        # Remove any leading/trailing code fences
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE).rstrip("`\n\r ")
            text = re.sub(r"\s*```$", "", text).strip()

        # Extract substring between the first { and last }
        first = text.find("{")
        last = text.rfind("}")
        if first != -1 and last != -1 and last > first:
            json_text = text[first:last + 1]
        else:
            json_text = text

        extracted = json.loads(json_text)
    except Exception as e:
        print("Failed to parse LLM JSON output; falling back. Parse error:", str(e))
        extracted = {
            "Domiciliary": {"Domiciliary Hospital Limit Applicable": "No"},
            "limit_percentage": {"percent": None, "amount": None},
            "limit_amount": {"percent": None, "amount": None},
            "sum_insured_exclusion": None,
        }

    # Normalise/ensure structure
    extracted.setdefault("Domiciliary", {})
    dh = extracted["Domiciliary"]

    # Normalize the Domiciliary Hospital Limit Applicable flag
    raw_flag = dh.get("Domiciliary Hospital Limit Applicable")
    if isinstance(raw_flag, str):
        flag_normalized = raw_flag.strip().lower()
        dh["Domiciliary Hospital Limit Applicable"] = "Yes" if flag_normalized in ("yes", "y", "true", "applicable") else "No"
    elif isinstance(raw_flag, bool):
        dh["Domiciliary Hospital Limit Applicable"] = "Yes" if raw_flag else "No"
    else:
        dh["Domiciliary Hospital Limit Applicable"] = "No"

    # Ensure limit structures exist and convert numeric strings
    extracted.setdefault("limit_percentage", {"percent": None, "amount": None})
    extracted.setdefault("limit_amount", {"percent": None, "amount": None})

    for key in ("limit_percentage", "limit_amount"):
        obj = extracted.get(key, {})
        obj["percent"] = _convert_to_number(obj.get("percent"))
        obj["amount"] = _convert_to_number(obj.get("amount"))
        extracted[key] = obj

    extracted["sum_insured_exclusion"] = _convert_to_number(extracted.get("sum_insured_exclusion"))

    return extracted






# ----------------------------
# Build Dom_schema
# ----------------------------
def build_dom_schema(extracted: Dict[str, Any], si_list: List[float] = None) -> Dict[str, Any]:
    """
    Build a consistent Domiciliary Limit schema from the normalized `extracted` dict.
    """
    raw_flag = extracted.get("Domiciliary", {}).get("Domiciliary Hospital Limit Applicable", "No")
    applicable = "Yes" if str(raw_flag).strip().lower() == "yes" else "No"

    if applicable == "Yes":
        # Get sum insured list
        if si_list is None:
            si_list = get_dummy_sum_insured()

        # Prefer percent from limit_percentage, else from limit_amount
        percent = extracted.get("limit_percentage", {}).get("percent")
        if percent is None:
            percent = extracted.get("limit_amount", {}).get("percent")

        # Prefer amount from limit_percentage, else from limit_amount
        absolute_amount = extracted.get("limit_percentage", {}).get("amount")
        if absolute_amount is None:
            absolute_amount = extracted.get("limit_amount", {}).get("amount")

        # Compute limit amounts
        limit_amounts = []
        if percent is not None:
            try:
                p = float(percent)
            except Exception:
                p = None
            if p is not None:
                for si in si_list:
                    if isinstance(si, (int, float)):
                        limit_amounts.append(round(si * p / 100.0, 2))
                    else:
                        limit_amounts.append(None)
            else:
                limit_amounts = [None for _ in si_list]
        elif absolute_amount is not None:
            try:
                amt = float(absolute_amount)
            except Exception:
                amt = None
            limit_amounts = [amt for _ in si_list]
        else:
            limit_amounts = [None for _ in si_list]

        schema = {
            "Domiciliary Limit": {
                "Domiciliary Hospital Limit Applicable": "Yes",
                "Sum insured": si_list,
                "% Limit Applicable On": "Sum insured",
                "Limit Percentage": percent,
                "Limit Amount": limit_amounts,
                "Applicability": "Lower"
            }
        }
    else:
        # EXACT shape requested when not applicable
        schema = {
            "Domiciliary Limit": {
                "Domiciliary Hospital Limit Applicable": "No",
                "Sum insured": None,
                "% Limit Applicable On": None,
                "Limit Percentage": None,
                "Limit Amount": None,
                "Applicability": None
            }
        }

    return schema


# ----------------------------
# Fallback extractor
# ----------------------------
def extract_domiciliary_limit(text: str) -> dict:
    return {
        "Domiciliary Limit": {
            "Domiciliary Hospital Limit Applicable": "NO",#Not Applicable
            "Sum insured": None,
            "% Limit Applicable On": None,
            "Limit Percentage": None,
            "Limit Amount": None,
            "Applicability": None
        }
    }


# ----------------------------
# Save JSON to Excel
# ----------------------------
def save_to_excel(json_data: dict, pdf_path: str,source_dir):
    """Save nested JSON to Excel in expanded row format, Excel name same as PDF name."""
    try:
        rows = []
        for category, details in json_data.items():
            sum_insured = details.get("Sum insured", [])
            limit_amounts = details.get("Limit Amount", [])

            if isinstance(sum_insured, list) and isinstance(limit_amounts, list):
                for si, la in zip(sum_insured, limit_amounts):
                    row = {
                        "Domiciliary Hospital Limit Applicable": details.get("Domiciliary Hospital Limit Applicable"),
                        "Sum insured": si,
                        "% Limit Applicable On": details.get("% Limit Applicable On"),
                        "Limit Percentage": details.get("Limit Percentage"),
                        "Limit Amount": la,
                        "Applicability": details.get("Applicability")
                    }
                    rows.append(row)
            else:
                row = {
                    "Domiciliary Hospital Limit Applicable": details.get("Domiciliary Hospital Limit Applicable"),
                    "Sum insured": sum_insured,
                    "% Limit Applicable On": details.get("% Limit Applicable On"),
                    "Limit Percentage": details.get("Limit Percentage"),
                    "Limit Amount": details.get("Limit Amount"),
                    "Applicability": details.get("Applicability")
                }
                rows.append(row)

        df = pd.DataFrame(rows)

        # Remove duplicates ONLY for "Domiciliary Hospital Limit Applicable"
        col = "Domiciliary Hospital Limit Applicable"
        df[col] = df[col].mask(df[col] == df[col].shift(), "")

        
    # --- Create Excel filename same as PDF ---
    
        output_folder = source_dir  # define your working folder
        os.makedirs(output_folder, exist_ok=True)

        base_name = os.path.splitext(os.path.basename(pdf_path))[0]  
        excel_filename = os.path.join(output_folder, f"{base_name}_dom_limt.xlsx")

        df.to_excel(excel_filename, index=False, engine="openpyxl")
        print(f"Excel saved at: {excel_filename}")

        return excel_filename

    except Exception as e:
        print(f"Failed to save Excel: {str(e)}")
        return None


# ----------------------------
# Driver
# ----------------------------
def generate_dom_limits(pdf_path: str, chat,source_dir):
    md_text = pdf_to_md(pdf_path)
    endorsements = extract_endorsements(md_text)

    # Try to extract sum insured from policy (or use dummy values)
    output_folder = source_dir  # define your working folder
    os.makedirs(output_folder, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]  
    #json_filename = os.path.join(output_folder, f"{base_name}_sum_insured.json")
    si_list = load_sum_insured_from_json(pdf_path)

    sample_key = next((eid for eid in endorsements if eid.startswith("19")), None)

    if not sample_key:
        final_output = extract_domiciliary_limit(md_text)
        print("\nFinal Dom_schema Output (No Endorsement 19):")
        print(json.dumps(final_output, indent=2))
    else:
        sample_text = endorsements[sample_key]
        extracted = domiciliary_limit_extractor(sample_text, chat)
        final_output = build_dom_schema(extracted, si_list)
        print("\nFinal Dom_schema Output:")
        print(json.dumps(final_output, indent=2))

    save_to_excel(final_output, pdf_path,source_dir)
    return final_output


# ----------------------------
# Run
# ----------------------------
if __name__ == "__main__":
    pdf_path = r"/home/ubuntu/rspdf/Domiciliary_sheet/Domicilary_input/HG00007358000100.pdf"
    source_dir=r""
    generate_dom_limits(pdf_path, chat,source_dir)