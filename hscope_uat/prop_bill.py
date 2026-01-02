from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate
import pymupdf4llm
import re
from collections import OrderedDict
import json
import logging
import os
import pandas as pd
from openpyxl import Workbook
from typing import Dict, Any, Optional
from dataclasses import dataclass  
from hscope_uat.UAT_config import *

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

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

@dataclass
class AnalysisResult:
    """Data class for structured Proportionate Billing results."""
    proportionate_billing_applicable: str
    irda_level_1_description: str
    irda_level_2_description: str
    as_per_product_config: str
    as_per_soc: str
    remarks: str
    endorsement_5d_found: bool
    extracted_endorsements: Dict[str, str]  # Added field
    raw_response: Optional[str] = None
    error: Optional[str] = None

# ----------------------------
# PDF → Markdown
# ----------------------------
def pdf_to_md(pdf_path):
    """Convert PDF to markdown using pymupdf4llm."""
    md_text = pymupdf4llm.to_markdown(pdf_path)
    return md_text

# ----------------------------
# Normalize IDs
# ----------------------------
def normalize_id(eid: str) -> str:
    """Convert endorsement IDs to canonical form (5d -> 5(d))."""
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
    """Extract endorsements from markdown text."""
    lines = md_text.splitlines()

    # FIXED: Added \* to the lookahead to handle markdown bold markers
    heading_pattern = re.compile(
        r"^\s*\*{0,2}\s*(?:Endorsement|Endt\.?)\.?\s*"
        r"No\.?\s*(\d+)"
        r"(?:\s*\(\s*([A-Za-z0-9ivxIVX]{1,3})\s*\)|-(?:([A-Za-z0-9ivxIVX]{1,3}))|([A-Za-z]{1,3})(?=\s|[-–]|\*|$))?",
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
        """Remove unwanted header/footer lines."""
        return [
            line for line in buf
            if not any(p.search(line) for p in junk_patterns)
        ]

    for i, line in enumerate(lines, 1):
        m = heading_pattern.search(line)
        if m:
            num = m.group(1)
            suffix = m.group(2) or m.group(3) or m.group(4)
            canonical_id = num
            if suffix and len(suffix) <= 3:
                canonical_id = f"{num}({suffix.lower()})"
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

# ----------------------------
# Check if 5D exists and Get Content
# ----------------------------
def get_endorsements_content(pdf_path):
    """Extract endorsements and check if 5D exists."""
    md_text = pdf_to_md(pdf_path)
    endorsements = extract_endorsements(md_text)
   
    # Check if 5D exists
    desired_endorsements = ["5d", "5D", "5(d)"]
    desired_endorsements = [normalize_id(eid) for eid in desired_endorsements]
   
    matching_endorsements = {eid: endorsements[eid] for eid in desired_endorsements if eid in endorsements}
   
    endorsement_5d_found = len(matching_endorsements) > 0
   
    # Log what we found
    if endorsement_5d_found:
        logger.info(f"✓ Endorsement 5D FOUND in PDF")
    else:
        logger.info(f"✗ Endorsement 5D NOT FOUND in PDF")
   
    # Get content for analysis
    matched_endorsements = list(matching_endorsements.values())
   
    if not matched_endorsements:
        logger.info("Using all endorsements for analysis")
        matched_endorsements = list(endorsements.values())
   
    endorsements_context = ""
    for endnt in matched_endorsements:
        endorsements_context += endnt + "\n\n"
   
    return endorsements_context, endorsement_5d_found, endorsements

# ----------------------------
# Proportionate Billing Analyzer
# ----------------------------
class InsurancePolicyAnalyzer:
    """Proportionate Billing analyzer using langchain_ollama."""

    def __init__(self, chat_model):
        self.chat = chat_model
        self.prompt_template = self._build_prompt_template()

    def _build_prompt_template(self):
        """Build the ChatPromptTemplate for analysis."""
        return ChatPromptTemplate.from_messages([
            ("system",
             "You are an expert insurance policy analyst. "
             "Analyze the given text to determine if endorsement 5D is present.\n\n"
             "CRITICAL RULES:\n"
             "1. Search ONLY for 'Endorsement 5D' or 'Endorsement No. 5D' or 'Endt. 5D'\n"
             "2. If you find endorsement 5D → 'Is_5D_Present': 'Yes'\n"
             "3. If you DO NOT find endorsement 5D → 'Is_5D_Present': 'No'\n\n"
             "Return ONLY valid JSON. No explanations or additional text."
            ),
            ("user",
             "Here is the endorsement context:\n\n{context}\n\n"
             "Analyze carefully and return JSON:\n\n"
             "{{\n"
             "    \"Is_5D_Present\": \"Yes\" or \"No\"\n"
             "}}\n\n"
             "IMPORTANT: Output ONLY the JSON object, nothing else."
            )
        ])

    def analyze_document_chunk(self, chunk_text: str, endorsement_5d_found: bool, all_endorsements: Dict[str, str]) -> AnalysisResult:
        """Analyze document chunk for proportionate billing information."""
        try:
            logger.info("Analyzing document with LLM")
           
            # Use the actual detection result from extraction
            is_5d_present = endorsement_5d_found
           
            # Determine Proportionate Billing Applicable
            # CORRECT LOGIC: If 5D is present → Proportionate Billing = No
            #                If 5D is NOT present → Proportionate Billing = Yes
            if is_5d_present:
                proportionate_billing = "No"
                irda_level_1 = ""
                irda_level_2 = ""
            else:
                proportionate_billing = "Yes"
                irda_level_1 = "Room & Nursing Charges"
                irda_level_2 = "Room & Nursing Charges"
           
            # Create response JSON
            response_json = {
                "Proportionate_Billing_Applicable": proportionate_billing,
                "IRDA_Level_1_Description": irda_level_1,
                "IRDA_Level_2_Description": irda_level_2,
                "As_per_Product_Config": "",
                "As_per_SOC": "",
                "Remarks": ""
            }
           
            raw_response = json.dumps(response_json, indent=4)
           
            result = AnalysisResult(
                proportionate_billing_applicable=proportionate_billing,
                irda_level_1_description=irda_level_1,
                irda_level_2_description=irda_level_2,
                as_per_product_config="",
                as_per_soc="",
                remarks="",
                endorsement_5d_found=is_5d_present,
                extracted_endorsements=all_endorsements,
                raw_response=raw_response
            )
            return result
           
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            return AnalysisResult(
                proportionate_billing_applicable="Error",
                irda_level_1_description="Error",
                irda_level_2_description="Error",
                as_per_product_config="Error",
                as_per_soc="Error",
                remarks="Error",
                endorsement_5d_found=False,
                extracted_endorsements={},
                error=str(e)
            )

def save_to_excel(json_data: dict, output_path: str):
    """Save JSON data to an Excel file."""
    try:
        df = pd.DataFrame([json_data])
        df.to_excel(output_path, index=False, engine="openpyxl")
        logger.info(f"Excel saved at: {output_path}")
    except Exception as e:
        logger.error(f"Failed to save Excel: {str(e)}")

# ----------------------------
# Main Driver
# ----------------------------
def generate_prop_billing(pdf_path: str,source_dir):
    """Main function to analyze PDF for proportionate billing."""
    analyzer = InsurancePolicyAnalyzer(chat)
    endorsements_content, endorsement_5d_found, all_endorsements = get_endorsements_content(pdf_path)

    print("=" * 60)
    print("PROPORTIONATE BILLING ANALYSIS")
    print("=" * 60)

    if not endorsements_content.strip():
        print("No endorsements found in the PDF")
        return

    # Print all extracted endorsements
    print("\n" + "=" * 60)
    print("EXTRACTED ENDORSEMENTS")
    print("=" * 60)
    if all_endorsements:
        for endorsement_id, content in all_endorsements.items():
            print(f"\n--- Endorsement {endorsement_id} ---")
            # Print first 200 characters of content
            preview = content[:200] + "..." if len(content) > 200 else content
            print(preview)
            print()
    else:
        print("No endorsements extracted from PDF")
    print("=" * 60)


    # Show detection result
    print(f"\nEndorsement 5D Detection: {'FOUND' if endorsement_5d_found else 'NOT FOUND'}")
    print("-" * 60)

    result = analyzer.analyze_document_chunk(endorsements_content, endorsement_5d_found, all_endorsements)

    # Display results based on scenario
    print(f"\nProportionate Billing Applicable: {result.proportionate_billing_applicable}")
   
    if result.proportionate_billing_applicable == "Yes":
        # Scenario: 5D NOT present
        print(f"IRDA Level 1 Description: {result.irda_level_1_description}")
        print(f"IRDA Level 2 Description: {result.irda_level_2_description}")
    elif result.proportionate_billing_applicable == "No":
        # Scenario: 5D present
        print("IRDA Level 1 Description: (Empty)")
        print("IRDA Level 2 Description: (Empty)")
   
    # Only show if there's an error
    if result.error:
        print(f"\nError: {result.error}")

    print("=" * 60)

    # Create Excel filename same as PDF
    output_folder = source_dir  # define your working folder
    os.makedirs(output_folder, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(pdf_path))[0]  
    excel_filename = os.path.join(output_folder, f"{base_name}_prop_bill.xlsx")


    # Convert to JSON and save to Excel
    try:
        json_data = json.loads(result.raw_response)
        save_to_excel(json_data, excel_filename)
        print(f"\n✓ Results saved to: {excel_filename}")
    except Exception as e:
        logger.error(f"Failed to save Excel: {str(e)}")
        print(f"\n✗ Could not save Excel file")

    return result



if __name__ == "__main__":
    pdf_file ="/home/ubuntu/rspdf/Domiciliary_sheet/proportionate_input/HG00007706000100.pdf"
    source_dir=r""
    generate_prop_billing(pdf_file,source_dir)