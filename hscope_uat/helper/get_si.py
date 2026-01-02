from langchain_ollama import ChatOllama
import pandas as pd
import re
from bs4 import BeautifulSoup
from io import StringIO
import os
from pathlib import Path
import json
from hscope_uat.helper.get_mineru import parse_doc

try:
    from langchain_ollama import ChatOllama
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

LANGCHAIN_AVAILABLE=True

# ============================================================
# CLEANING UTILITIES (ADDED FROM CODE 2)
# ============================================================

def clean_sum_insured_value(val):
    """Clean and standardize sum insured values"""
    if pd.isna(val) or val == '' or str(val).lower() in ['nan', 'none', 'null']:
        return None

    val = str(val).strip()
    val = (
        val.replace("Rs.", "")
           .replace("rs.", "")
           .replace("Rs", "")
           .replace("rs", "")
           .replace("/-", "")
           .strip()
    )

    dot_count = val.count('.')
    comma_count = val.count(',')

    if dot_count > 0 or comma_count > 0:
        if re.search(r'\d[.,]\d{2}[.,]\d{3}', val):
            val = val.replace('.', '').replace(',', '')
        elif dot_count >= 2:
            val = val.replace('.', '')
        elif comma_count >= 2:
            val = val.replace(',', '')
        elif dot_count == 1 and comma_count == 1:
            val = val.replace('.', '').replace(',', '')
        elif dot_count == 1:
            parts = val.split('.')
            if len(parts) == 2 and len(parts[1]) >= 2:
                val = val.replace('.', '')
        elif comma_count == 1:
            val = val.replace(',', '')

    return val.strip()

def preprocess_html_tables(html_text):
    """Preprocess HTML tables to standardize number formats"""
    soup = BeautifulSoup(html_text, "html.parser")

    for cell in soup.find_all(['td', 'th']):
        text = cell.get_text(strip=True)

        if re.search(r'\d+[.,]+\d+', text) or re.search(r'Rs\.?\s*\d+', text, re.IGNORECASE):
            cleaned = re.sub(r'Rs\.?\s*', '', text, flags=re.IGNORECASE)
            cleaned = re.sub(r'[.,]', '', cleaned)
            cleaned = re.sub(r'[^\d]', '', cleaned)

            if cleaned.isdigit() and len(cleaned) >= 5:
                cell.string = cleaned

    return str(soup)

# ============================================================
# ORIGINAL CODE WITH CLEANING INTEGRATED
# ============================================================

# Initialize the ChatOllama model
chat = ChatOllama(
    model="gpt-oss:20b",       # specify GPT-20B model
    temperature=0.0,        # deterministic output
    top_p=1.0,              # deterministic sampling
    max_tokens=1024,        # max tokens per response
    verbose=True
)


def extract_si_primary(md_file_path, chat=None):
    """
    Primary SI extraction function - Enhanced version with AI column detection
    Returns None if lives mismatch, otherwise returns unique SI values
    """
    with open(md_file_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    # ADDED: Preprocess HTML tables before parsing
    md_text = preprocess_html_tables(md_text)

    soup = BeautifulSoup(md_text, "html.parser")
    tables = soup.find_all("table")
    dfs = [pd.read_html(StringIO(str(table)))[0] for table in tables]

    # Extract number of lives covered
    match = re.search(r"\bno\s*of\s*lives\s*covered\b\s*[:\-=]?\s*(\d+)", md_text, re.IGNORECASE)
    if not match:
        raise ValueError("Could not extract number of lives covered from markdown text!")
    lives_covered_count = int(match.group(1))
    print(f"? Lives covered: {lives_covered_count}")

    keywords = ["name", "gender", "relationship", "dob", "date of birth", "emp", "employee", "si.no", "sr.no"]
    all_member_dfs = []
    sr_col_name = None
    sum_col_name = None
    saved_columns = None

    print(f"Total tables found: {len(dfs)}")
    
    for i, df in enumerate(dfs):
        print(f"\n--- Processing Table {i+1} ---")
        print(f"Table shape: {df.shape}")
        
        if df.shape[0] < 2 or df.shape[1] < 5:
            print("Table too small, skipping")
            continue
        
        # Handle duplicate columns first
        cols = df.columns.tolist()
        seen = {}
        new_cols = []
        for col in cols:
            if col in seen:
                seen[col] += 1
                new_cols.append(f"{col}_{seen[col]}")
            else:
                seen[col] = 0
                new_cols.append(col)
        df.columns = new_cols
        
        # Identify header row
        header_row_idx = None
        for idx in range(min(10, len(df))):
            row_text = " ".join(df.iloc[idx].astype(str).str.lower().tolist())
            if any(k in row_text for k in keywords):
                header_row_idx = idx
                print(f"Header found at row {idx}")
                break

        if header_row_idx is not None:
            # Promote header
            df.columns = df.iloc[header_row_idx].astype(str).str.strip()
            df = df.drop(index=range(0, header_row_idx + 1)).reset_index(drop=True)
            df.columns = [" ".join(str(c).split()).strip() for c in df.columns]
            
            # Handle duplicate column names again after promoting header
            cols = df.columns.tolist()
            seen = {}
            new_cols = []
            for col in cols:
                if col in seen:
                    seen[col] += 1
                    new_cols.append(f"{col}_{seen[col]}")
                else:
                    seen[col] = 0
                    new_cols.append(col)
            df.columns = new_cols
            
            saved_columns = df.columns
            print(f"Cleaned columns: {df.columns.tolist()}")
        else:
            if saved_columns is not None and df.shape[1] == len(saved_columns):
                df.columns = saved_columns
                print("? Using saved header from previous table")
            else:
                print("No header found, skipping this table")
                continue

        # Validate with keywords
        col_text = " ".join([str(c).lower() for c in df.columns])
        if not any(k in col_text for k in keywords):
            print("No keywords found in columns, skipping this table")
            continue

        # Detect Serial Number & Sum Insured (hybrid approach with AI)
        if sr_col_name is None or sum_col_name is None:
            if chat is not None:
                try:
                    # Serial Number detection with AI
                    prompt_sr = f"""
You are an insurance expert.
Columns: {df.columns.tolist()}
Identify the column that represents the serial number (Sr No) of the members.
Ignore names, gender, relationship, sum insured, employee numbers, dates of birth.
Reply ONLY with the column name.
"""
                    response_sr = chat.invoke(input=prompt_sr)
                    sr_col_name_ai = response_sr.content.strip()

                    # Sum Insured detection with AI
                    prompt_si = f"""
You are an insurance expert.
Columns: {df.columns.tolist()}
Identify the column that most likely represents the Sum Insured of each member.
It can be called things like 'Floater SI', 'Family SI', 'Family Sum Insured', 'Individual SI', 'Individual Sum Insured', etc.
Ignore serial numbers, employee numbers, names, gender, relationship, dates of birth.
Reply ONLY with the column name that contains the sum insured.
"""
                    response_si = chat.invoke(input=prompt_si)
                    sum_col_name_ai = response_si.content.strip()
                    
                    if sr_col_name_ai in df.columns and sr_col_name_ai.lower() not in ['nan', 'none', 'null']:
                        sr_col_name = sr_col_name_ai
                    
                    if sum_col_name_ai in df.columns and sum_col_name_ai.lower() not in ['nan', 'none', 'null']:
                        sum_col_name = sum_col_name_ai
                    
                    print(f"AI detected SR column: {sr_col_name}")
                    print(f"AI detected SI column: {sum_col_name}")
                    
                except Exception as e:
                    print(f"AI detection failed: {e}")
                    chat = None
            
            # Pattern-based detection (fallback)
            if sr_col_name is None:
                for col in df.columns:
                    col_lower = col.lower()
                    if any(term in col_lower for term in ['si.no', 'sr.no', 'serial', 's.no', 'sno']):
                        sr_col_name = col
                        break
            
            if sum_col_name is None:
                for col in df.columns:
                    col_lower = col.lower()
                    if any(term in col_lower for term in ['si.no', 'sr.no', 'serial', 's.no', 'sno']):
                        continue
                    if any(term in col_lower for term in ['sum insured', 'floater', 'insured', 'amount']) and 'sum' in col_lower:
                        test_vals = df[col].dropna().astype(str).str.replace(',', '').str.replace('rs.', '').str.replace('rs', '').str.strip()
                        numeric_vals = []
                        for val in test_vals.head(10):
                            if val and val.replace('.', '').isdigit():
                                numeric_vals.append(float(val))
                        
                        if numeric_vals and any(val > 1000 for val in numeric_vals):
                            sum_col_name = col
                            break

            print(f"Final detected SR column: {sr_col_name}")
            print(f"Final detected SI column: {sum_col_name}")

        # Process valid tables
        if sr_col_name and sum_col_name and sr_col_name in df.columns and sum_col_name in df.columns:
            print(f"? Table has required columns: {sr_col_name}, {sum_col_name}")

            si_series = df[sum_col_name]
            if isinstance(si_series, pd.DataFrame):
                si_series = si_series.iloc[:, 0]
                print("?? Sum Insured column had duplicates, using first occurrence")
            
            # MODIFIED: Use the new cleaning function
            df[sum_col_name] = si_series.apply(clean_sum_insured_value)
            df[sum_col_name] = pd.to_numeric(df[sum_col_name], errors="coerce").astype("Int64")

            sr_series = df[sr_col_name]
            if isinstance(sr_series, pd.DataFrame):
                sr_series = sr_series.iloc[:, 0]
                print("?? Serial Number column had duplicates, using first occurrence")
            
            df[sr_col_name] = pd.to_numeric(sr_series, errors="coerce")

            df_before = len(df)
            df = df.dropna(subset=[sr_col_name])
            df = df[df[sr_col_name] > 0]
            df = df.dropna(subset=[sum_col_name])
            print(f"Rows after cleaning: {len(df)} (removed {df_before - len(df)} invalid rows)")

            if len(df) > 0:
                all_member_dfs.append(df)
                print(f"? Added table {i+1} to member tables (total: {len(all_member_dfs)})")
        else:
            print(f"? Missing required columns in table {i+1}")
            print(f"Available columns: {df.columns.tolist()}")

    if not all_member_dfs:
        raise ValueError("Could not find any member table with expected columns!")

    # Combine all member tables
    member_df = pd.concat(all_member_dfs, ignore_index=True, sort=False)

    # Trim to lives covered & remove duplicates
    member_df = member_df.sort_values(sr_col_name).reset_index(drop=True)
    member_df = member_df.drop_duplicates(subset=[sr_col_name], keep="first")
    member_df = member_df[member_df[sr_col_name] <= lives_covered_count]

    # Unique Sum Insured
    si_series = member_df[sum_col_name]
    if isinstance(si_series, pd.DataFrame):
        si_series = si_series.iloc[:, 0]
    
    unique_sum_insured = si_series.dropna().unique()

    print("Final Members DF (head):")
    print(member_df.head(10))
    print(f"Shape: {member_df.shape}")
    print(f"Serial Number column: {sr_col_name}")
    print(f"Sum Insured column: {sum_col_name}")
    print(f"Unique Sum Insured values: {unique_sum_insured}")
    print(f"Expected lives: {lives_covered_count}, Found lives: {len(member_df)}")

    if lives_covered_count != len(member_df):
        print(f"? Lives mismatch: Expected {lives_covered_count} but found {len(member_df)}. Returning None.")
        return None
    else:
        print("? Member count matches expected lives exactly")
        return sorted(unique_sum_insured.tolist())

def extract_si_fallback(md_file_path, chat=None):
    """
    Fallback SI extraction function - More flexible approach
    """
    print("\n?? Running FALLBACK function")
    
    with open(md_file_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    # ADDED: Preprocess HTML tables before parsing
    md_text = preprocess_html_tables(md_text)

    soup = BeautifulSoup(md_text, "html.parser")
    tables = soup.find_all("table")
    dfs = [pd.read_html(StringIO(str(table)))[0] for table in tables]

    match = re.search(r"\bno\s*of\s*lives\s*covered\b\s*[:\-=]?\s*(\d+)", md_text, re.IGNORECASE)
    if not match:
        raise ValueError("Could not extract number of lives covered from markdown text!")
    lives_covered_count = int(match.group(1))
    print(f"? Lives covered: {lives_covered_count}")

    keywords = ["name", "gender", "relationship", "dob", "date of birth", "emp", "employee", "si.no", "sr.no"]
    all_member_dfs = []
    sr_col_name = None
    sum_col_name = None
    saved_columns = None

    print(f"Total tables found: {len(dfs)}")
    
    def clean_column_name(col):
        """Clean column names to standardize them"""
        col_str = str(col).strip()
        col_lower = col_str.lower()
        
        if 'sr.no' in col_lower or 'si.no' in col_lower or 'serial' in col_lower:
            return 'Sr.No'
        
        if 'sum insured' in col_lower:
            return 'Sum Insured'
        
        return col_str
    
    for i, df in enumerate(dfs):
        print(f"\n--- Processing Table {i+1} ---")
        print(f"Table shape: {df.shape}")
        
        if df.shape[0] < 2 or df.shape[1] < 5:
            print("Table too small, skipping")
            continue
        
        header_row_idx = None
        for idx in range(min(10, len(df))):
            row_text = " ".join(df.iloc[idx].astype(str).str.lower().tolist())
            if any(k in row_text for k in keywords):
                header_row_idx = idx
                print(f"Header found at row {idx}")
                break

        if header_row_idx is not None:
            df.columns = df.iloc[header_row_idx].astype(str).str.strip()
            df = df.drop(index=range(0, header_row_idx + 1)).reset_index(drop=True)
            df.columns = [" ".join(str(c).split()).strip() for c in df.columns]
            
            df.columns = [clean_column_name(col) for col in df.columns]
            saved_columns = df.columns
            print(f"Cleaned columns: {df.columns.tolist()}")
        else:
            if saved_columns is not None and df.shape[1] == len(saved_columns):
                df.columns = saved_columns
                print("? Using saved header from previous table")
            else:
                print("No header found, skipping this table")
                continue

        col_text = " ".join([str(c).lower() for c in df.columns])
        if not any(k in col_text for k in keywords):
            print("No keywords found in columns, skipping this table")
            continue

        current_sr_col = None
        current_sum_col = None
        
        if chat is not None:
            try:
                prompt_sr = f"""
You are an insurance expert.
Columns: {df.columns.tolist()}
Identify the column that represents the serial number (Sr No) of the members.
Ignore names, gender, relationship, sum insured, employee numbers, dates of birth.
Reply ONLY with the column name.
"""
                response_sr = chat.invoke(input=prompt_sr)
                sr_col_name_ai = response_sr.content.strip()

                prompt_si = f"""
You are an insurance expert.
Columns: {df.columns.tolist()}
Identify the column that most likely represents the Sum Insured of each member.
It can be called things like 'Floater SI', 'Family SI', 'Family Sum Insured', 'Individual SI', 'Individual Sum Insured', etc.
Ignore serial numbers, employee numbers, names, gender, relationship, dates of birth.
Reply ONLY with the column name that contains the sum insured.
"""
                response_si = chat.invoke(input=prompt_si)
                sum_col_name_ai = response_si.content.strip()
                
                if sr_col_name_ai in df.columns and sr_col_name_ai.lower() not in ['nan', 'none', 'null']:
                    current_sr_col = sr_col_name_ai
                    if sr_col_name is None:
                        sr_col_name = sr_col_name_ai
                
                if sum_col_name_ai in df.columns and sum_col_name_ai.lower() not in ['nan', 'none', 'null']:
                    current_sum_col = sum_col_name_ai
                    if sum_col_name is None:
                        sum_col_name = sum_col_name_ai
                
                print(f"AI detected SR column: {current_sr_col}")
                print(f"AI detected SI column: {current_sum_col}")
                
            except Exception as e:
                print(f"AI detection failed: {e}")
        
        if current_sr_col is None:
            for col in df.columns:
                col_lower = col.lower()
                if any(term in col_lower for term in ['si.no', 'sr.no', 'serial', 's.no', 'sno']):
                    current_sr_col = col
                    if sr_col_name is None:
                        sr_col_name = col
                    break
        
        if current_sum_col is None:
            for col in df.columns:
                col_lower = col.lower()
                if any(term in col_lower for term in ['si.no', 'sr.no', 'serial', 's.no', 'sno']):
                    continue
                if 'sum insured' in col_lower or 'sum_insured' in col_lower:
                    test_vals = df[col].dropna().astype(str).str.replace(',', '').str.replace('rs.', '').str.replace('rs', '').str.strip()
                    numeric_vals = []
                    for val in test_vals.head(10):
                        if val and val.replace('.', '').isdigit():
                            numeric_vals.append(float(val))
                    
                    if numeric_vals and any(val > 1000 for val in numeric_vals):
                        current_sum_col = col
                        if sum_col_name is None:
                            sum_col_name = col
                        break

        print(f"Final detected SR column for this table: {current_sr_col}")
        print(f"Final detected SI column for this table: {current_sum_col}")

        if current_sr_col and current_sum_col and current_sr_col in df.columns and current_sum_col in df.columns:
            print(f"? Table has required columns: {current_sr_col}, {current_sum_col}")

            # MODIFIED: Use the new cleaning function
            df[current_sum_col] = df[current_sum_col].apply(clean_sum_insured_value)
            df[current_sum_col] = pd.to_numeric(df[current_sum_col], errors="coerce").astype("Int64")

            df[current_sr_col] = pd.to_numeric(df[current_sr_col], errors="coerce")

            df_before = len(df)
            df = df.dropna(subset=[current_sr_col])
            df = df[df[current_sr_col] > 0]
            print(f"Rows after cleaning: {len(df)} (removed {df_before - len(df)} invalid rows)")

            if len(df) > 0:
                df = df.rename(columns={current_sr_col: 'Sr.No', current_sum_col: 'Sum Insured'})
                all_member_dfs.append(df)
                print(f"? Added table {i+1} to member tables (total: {len(all_member_dfs)})")
        else:
            print(f"? Missing required columns in table {i+1}")
            print(f"Available columns: {df.columns.tolist()}")

    if not all_member_dfs:
        raise ValueError("Could not find any member table with expected columns!")

    member_df = pd.concat(all_member_dfs, ignore_index=True, sort=False)

    member_df = member_df.sort_values('Sr.No').reset_index(drop=True)
    member_df = member_df.drop_duplicates(subset=['Sr.No'], keep="first")
    member_df = member_df[member_df['Sr.No'] <= lives_covered_count]

    unique_sum_insured = member_df['Sum Insured'].dropna().unique()

    print("Final Members DF (head):")
    print(member_df.head(10))
    print(f"Shape: {member_df.shape}")
    print(f"Serial Number column: Sr.No")
    print(f"Sum Insured column: Sum Insured")
    print(f"Unique Sum Insured values: {unique_sum_insured}")
    print(f"Expected lives: {lives_covered_count}, Found lives: {len(member_df)}")

    if lives_covered_count != len(member_df):
        print(f"? Lives mismatch: Expected {lives_covered_count} but found {len(member_df)}. Returning None.")
        return None
    else:
        print("? Member count matches expected lives exactly")
        return sorted(unique_sum_insured.tolist())

def extract_si_specific(md_file_path):
    """
    Specific version for PDF format that directly targets the member table
    """
    with open(md_file_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    # ADDED: Preprocess HTML tables before parsing
    md_text = preprocess_html_tables(md_text)

    # Extract lives covered
    match = re.search(r"\bno\s*of\s*lives\s*covered\b\s*[:\-=]?\s*(\d+)", md_text, re.IGNORECASE)
    if not match:
        raise ValueError("Could not extract number of lives covered!")
    lives_covered = int(match.group(1))
    print(f"Lives covered: {lives_covered}")

    # Parse tables
    soup = BeautifulSoup(md_text, "html.parser")
    tables = soup.find_all("table")
    dfs = [pd.read_html(StringIO(str(table)))[0] for table in tables]

    member_tables = []
    
    for i, df in enumerate(dfs):
        print(f"\nTable {i+1}: Shape {df.shape}")
        
        # Look for tables with member data characteristics
        if df.shape[0] > 10 and df.shape[1] >= 8:  # Large tables likely to contain member data
            print(f"Analyzing table {i+1} columns:")
            print(df.columns.tolist())
            
            # Check if this looks like a member table
            col_text = " ".join([str(c).lower() for c in df.columns])
            if any(term in col_text for term in ['name', 'relationship', 'gender', 'emp', 'si.no', 'insured']):
                print(f"Found potential member table {i+1}")
                
                # Find the right columns
                sr_col = None
                si_col = None
                
                for col in df.columns:
                    col_lower = str(col).lower()
                    if 'si.no' in col_lower or 'sr.no' in col_lower or 'serial' in col_lower:
                        sr_col = col
                    elif 'sum insured' in col_lower or 'floater' in col_lower:
                        si_col = col
                
                if sr_col and si_col:
                    print(f"Found SR: {sr_col}, SI: {si_col}")
                    
                    # Clean the data
                    df_clean = df.copy()
                    
                    # MODIFIED: Use the new cleaning function
                    df_clean[si_col] = df_clean[si_col].apply(clean_sum_insured_value)
                    df_clean[si_col] = pd.to_numeric(df_clean[si_col], errors='coerce')
                    
                    # Clean serial number
                    df_clean[sr_col] = pd.to_numeric(df_clean[sr_col], errors='coerce')
                    
                    # Remove invalid rows
                    df_clean = df_clean.dropna(subset=[sr_col, si_col])
                    df_clean = df_clean[df_clean[sr_col] > 0]
                    
                    if len(df_clean) > 0:
                        member_tables.append(df_clean)
                        print(f"Added {len(df_clean)} valid members from table {i+1}")

    if not member_tables:
        print("No member tables found. Let me show you all table structures:")
        for i, df in enumerate(dfs):
            print(f"\nTable {i+1} (Shape: {df.shape}):")
            print(df.head(2))
        return None

    # Combine all member data
    all_members = pd.concat(member_tables, ignore_index=True)
    
    # Check if lives match exactly
    if lives_covered != len(all_members):
        print(f"? Lives mismatch: Expected {lives_covered} but found {len(all_members)}. Returning None.")
        return None
    
    # Get unique sum insured values
    si_columns = [col for table in member_tables for col in table.columns 
                  if 'sum insured' in str(col).lower() or 'floater' in str(col).lower()]
    
    if si_columns:
        si_col = si_columns[0]  # Take the first SI column found
        unique_si = all_members[si_col].dropna().unique()
        print(f"Unique Sum Insured values: {unique_si}")
        return sorted(unique_si.tolist())
    return None

def extract_si(md_file_path, chat=None):
    """
    Main function that tries primary first, then falls back if needed
    """
    print("?? Starting SI extraction with PRIMARY method...")
    
    try:
        result = extract_si_primary(md_file_path, chat)
        
        if result is not None:
            print("? PRIMARY method succeeded!")
            return result
        else:
            print("? PRIMARY method returned None (lives mismatch)")
            print("?? Trying FALLBACK method...")
            
            result = extract_si_fallback(md_file_path, chat)
            
            if result is not None:
                print("? FALLBACK method succeeded!")
                return result
            else:
                print("? Both methods failed - no unique SI found")
                return None
                
    except Exception as e:
        print(f"? PRIMARY method failed with error: {e}")
        print("?? Trying FALLBACK method...")
        
        try:
            result = extract_si_fallback(md_file_path, chat)
            
            if result is not None:
                print("? FALLBACK method succeeded!")
                return result
            else:
                print("? FALLBACK method also returned None")
                return None
                
        except Exception as e2:
            print(f"? FALLBACK method also failed with error: {e2}")
            print("?? Trying SPECIFIC method as last resort...")
            
            try:
                result = extract_si_specific(md_file_path)
                
                if result is not None:
                    print("? SPECIFIC method succeeded!")
                    return result
                else:
                    print("? All methods failed")
                    return None
                    
            except Exception as e3:
                print(f"? All methods failed. Final error: {e3}")
                return None

def get_sum_insured_from_mineru(pdf_path: str, method_used: str):
    """
    Extract Sum Insured values from a policy PDF using MinerU pipeline
    """
    try:
        pdf_path_obj = Path(pdf_path)
        file_base_name = pdf_path_obj.stem
        temp_output_dir = pdf_path_obj.parent / "mineru_temp"
        temp_output_dir.mkdir(exist_ok=True)

        print(f"[get_sum_insured_from_mineru] Processing: {pdf_path}")
        print(f"[get_sum_insured_from_mineru] Using method: {method_used}")

        # Run MinerU for all pages (third sheet needs full PDF)
        parse_doc([pdf_path_obj], str(temp_output_dir), backend="pipeline")

        # Locate markdown - use method-specific naming to avoid conflicts with first sheet
        md_file_path = temp_output_dir / file_base_name / "auto" / f"{file_base_name}.md"
        if not md_file_path.exists():
            print(f"[get_sum_insured_from_mineru] Markdown file not found: {md_file_path}")
            return []

        # Initialize ChatOllama if available
        chat = None
        if LANGCHAIN_AVAILABLE:
            try:
                chat = ChatOllama(
                    model="gpt-oss:20b",
                    temperature=0.0,
                    top_p=1.0,
                    max_tokens=1024,
                    verbose=False
                )
            except Exception as e:
                print(f"[get_sum_insured_from_mineru] ChatOllama init failed: {e}")

        # Extract sum insured
        try:
            sum_insured_values = extract_si(str(md_file_path), chat)
            if sum_insured_values:
                sum_insured_values = list(map(int, sum_insured_values))
        except Exception as e:
            print(f"[get_sum_insured_from_mineru] extract_si failed: {e}")
            sum_insured_values = []

        if not isinstance(sum_insured_values, list):
            print(f"[get_sum_insured_from_mineru] Warning: extract_si returned non-list -> {sum_insured_values}")
            sum_insured_values = []

        print(f"[get_sum_insured_from_mineru] Extracted SI values: {sum_insured_values}")

        # Only write JSON if list is not empty
        if sum_insured_values:
            json_path = pdf_path_obj.parent / f"{file_base_name}_si.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump({"sum_insured_values": sum_insured_values}, f, indent=4, ensure_ascii=False)
            print(f"[get_sum_insured_from_mineru] Saved JSON: {json_path}")
        else:
            print(f"[get_sum_insured_from_mineru] No SI values extracted; JSON not written.")

        return sum_insured_values

    except Exception as e:
        print(f"[get_sum_insured_from_mineru] Error: {e}")
        return []

def load_sum_insured_from_json(pdf_path: str, source_dir: str = None):
    """
    Load Sum Insured values from the JSON file.
    Expects filename <pdf_basename>_si.json in source_dir (or same directory as PDF if not provided).
    """
    try:
        pdf_path_obj = Path(pdf_path)
        
        # If source_dir is provided, use it; otherwise use PDF's directory
        if source_dir:
            json_path = Path(source_dir) / f"{pdf_path_obj.stem}_si.json"
        else:
            json_path = pdf_path_obj.parent / f"{pdf_path_obj.stem}_si.json"
        
        if not json_path.exists():
            print(f"[load_sum_insured_from_json] File not found: {json_path}")
            return []
        
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        si_values = data.get("sum_insured_values", [])
        print(f"[load_sum_insured_from_json] Loaded SI values: {si_values}")
        return si_values
    except Exception as e:
        print(f"[load_sum_insured_from_json] Error: {e}")
        return []


# Example usage and test
if __name__ == "__main__":
    # Test the extraction
    md_file_path = "/home/ubuntu/rspdf/newtest/Dinesh_22_09_S3_Thineshcode_UAT/output/HG00007537000100/auto/HG00007537000100.md"
    
    print("=" * 80)
    print("STARTING SI EXTRACTION PROCESS")
    print("=" * 80)
    
    # Run the main extraction function
    unique_si = extract_si(md_file_path, chat)
    
    print("=" * 80)
    print("FINAL RESULT:")
    if unique_si is not None:
        print(f"? Unique SI found: {unique_si}")
    else:
        print("? No unique SI could be extracted")
    print("=" * 80)