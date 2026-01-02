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
            any(word in header_text for word in ["condition", "conditions", "clause", "clauses", "coverage"," Endorsement", " Endorsements"])):
            
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



def get_remark(pdf_path,source_dir):
    

    
    output_folder = source_dir  # define your working folder
    os.makedirs(output_folder, exist_ok=True)
    md_text=pdf_to_md(pdf_path)
    result=overaly_specail_condtion(md_text)

    condiotnal_check = """There is no special conditon is provided, so you can consider No context from specail conditons section.
            """

    if result['special condition'] == condiotnal_check:
        data = {'special condition': ""}
    else:
        data = result

    text=data['special condition']
    lines = [line.strip() for line in text.split('\n') if line.strip()]  # removes empty lines

    # Step 2: Create a DataFrame
    df = pd.DataFrame({'Content': lines})

    
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]  
    output_path = os.path.join(output_folder, f"{base_name}_remark_sheet.xlsx")

    # Step 3: Save to Excel
    df.to_excel(output_path, index=False)

        
        