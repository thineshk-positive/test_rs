import os
import shutil
import pandas as pd
from zipfile import ZipFile
from langchain_ollama import ChatOllama

from hscope_uat.UAT_config import *
from hscope_uat.org_sheet import run_org
from hscope_uat.plan_details import plan_details_run
from hscope_uat.product_details import product_setup_run
from hscope_uat.dom_limit import generate_dom_limits
from hscope_uat.prop_bill import generate_prop_billing
from hscope_uat.limit_and_sublimit import run_limit_and_sublimit
from hscope_uat.claims_and_conditions import run_and_save_claim_conditons
from hscope_uat.eligibility import *
from hscope_uat.addon_covers import run_addon_covers
from hscope_uat.copay import get_copay_and_deductible
from hscope_uat.primary_cover import run_primary_cover
from hscope_uat.ICD_3 import run_and_save_ICD
from hscope_uat.addon_coverages import run_addon_coverage
from hscope_uat.remark_sheet import get_remark

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


def combine_excels(pdf_path: str, src_dir: str, dest_dir: str):
    """
    Combines Excel files:
      - {base_name}_org_sheet.xlsx
      - {base_name}_product_sheet.xlsx
      - {base_name}_plan_details_sheet.xlsx
    into a single master Excel in fixed order.
    """
    os.makedirs(dest_dir, exist_ok=True)

    # Extract base name from PDF
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    master_filename = f"{base_name}.xlsx"
    master_path = os.path.join(dest_dir, master_filename)

    if os.path.exists(master_path):
        os.remove(master_path)

    # Expected input → Output sheet mapping (fixed order)
    suffix_to_sheet = [
        ("_org_sheet.xlsx", f"{base_name}_orgsheet"),
        ("_product_setup.xlsx", "product_setup"),
        ("_plan_details.xlsx", "plan_details"),
        ("_eligibility_sheet.xlsx","eligibility"),
        ("_primay_cover.xlsx",'primary_cover'),
        ('_addon_covers.xlsx',"addon_covers"),
        ("_addon_coverages.xlsx","addon_coverages"),
        ("_limit_and_sublimit.xlsx","limit_and_sublimit"),
        ("_copay_and_deductible.xlsx",'copay_and_deductible'),
        ("_claim_and_conditons.xlsx","claim_and_conditons"),
        ("_dom_limt.xlsx","dom_limit"),
        ("_prop_bill.xlsx","prop_bill"),
        ("_remark_sheet.xlsx","_remarks"),
        ("_ICD_3_sheet.xlsx","ICD_3")     

    ]

    with pd.ExcelWriter(master_path, engine="openpyxl") as writer:
        for suffix, sheet_name in suffix_to_sheet:
            file_name = f"{base_name}{suffix}"
            file_path = os.path.join(src_dir, file_name)

            if os.path.exists(file_path):
                df = pd.read_excel(file_path)
                df.to_excel(writer, sheet_name=sheet_name[:31], index=False)
                print(f"✅ Added {file_name} → sheet '{sheet_name}'")
            else:
                print(f"⚠️ Missing expected file: {file_name}")

    # Delete source files after combining
    for file in os.listdir(src_dir):
        if file.startswith(base_name) and file.endswith(".xlsx"):
            os.remove(os.path.join(src_dir, file))
            print(f"🗑️ Deleted source file: {file}")

    print(f"\n📂 Master Excel saved at: {master_path}")


def delete_folder(folder_path: str):
    """
    Delete a folder and all its contents.

    Args:
        folder_path (str): Path of the folder to delete
    """
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        shutil.rmtree(folder_path)
        print(f"✅ Folder deleted: {folder_path}")
    else:
        print(f"⚠️ Folder not found: {folder_path}")


def run_all_sheets(pdf_path: str, source_dir: str, dest_dir: str):
    """
    Run all processing steps for a given PDF and combine outputs into a master Excel.
    """
    #process_pdf_to_orgschema(pdf_path,work_dir, save_raw_llm=True)
    run_org(pdf_path,source_dir=source_dir)
    plan_details_run(pdf_path,source_dir=source_dir)
    product_setup_run(pdf_path,source_dir=source_dir)
    run_limit_and_sublimit(pdf_path,source_dir=source_dir)
    generate_dom_limits(pdf_path, chat,source_dir=source_dir)
    generate_prop_billing(pdf_path,source_dir=source_dir)
    run_and_save_claim_conditons(pdf_path,chat,source_dir=source_dir)
    #generate_eligibility(pdf_path,chat,source_dir=source_dir)
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    md_path = generate_md_path_from_pdf(pdf_path)
    
    if not os.path.exists(md_path):
         print(f"?? Warning: Markdown not found at {md_path}")
         print("Skipping eligibility sheet generation")
    else:
        generate_eligibility(md_path, chat, source_dir=source_dir)
    run_addon_covers(pdf_path,source_dir=source_dir)
    get_copay_and_deductible(pdf_path,source_dir=source_dir)
    run_primary_cover(pdf_path,chat,source_dir=source_dir)
    run_and_save_ICD(pdf_path,source_dir=source_dir)
    run_addon_coverage(pdf_path,source_dir)

    get_remark(pdf_path,source_dir=source_dir)

    combine_excels(pdf_path, source_dir, dest_dir)

    # Clean up temporary folder
    #delete_folder("/home/ubuntu/THINESH_WS/royal_sunadaram_hscope/UAT/work_dir/mineru_temp")


if __name__ == "__main__":
    src_folder = work_dir
    dest_folder = output_dir
    pdf_path = r"/home/ubuntu/THINESH_WS/royal_sunadaram_hscope/UAT/work_dir/sample.pdf"

    run_all_sheets(pdf_path, src_folder, dest_folder)
