from __future__ import annotations

import os
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import pdfplumber
import pypdf
from openai import OpenAI

API_KEY = "sk-hc-v1-fef245f99c594bb4b20c70bbae781c12bec2d0143f0d4f98b24bf21835624b6a"
YEARS_TO_CLEAN = ("2020", "2021", "2022")
ALL_YEARS = ("2019", "2020", "2021", "2022", "2023", "2024")
MAX_WORKERS = 4

SCRIPT_DIR = Path(__file__).resolve().parent
PDFS_DIR = SCRIPT_DIR / "pdfs"
CSV_DIR = SCRIPT_DIR / "csvs"
OUTPUT_DIR = SCRIPT_DIR / "output"

client = OpenAI(api_key=API_KEY, base_url="https://ai.hackclub.com/proxy/v1")


def find_appendix_page(pdf: pypdf.PdfReader, num_pages: int) -> int | None:
    for page_num in range(num_pages - 1, -1, -1):
        page = pdf.pages[page_num]
        text = page.extract_text() or ""
        if (
            "Appendix" in text
            and "visits, spending, and economic contributions to local economies"
            in text.lower()
        ):
            return page_num
    return None


def table_signature(df: pd.DataFrame) -> str:
    columns = df.columns.tolist()
    if not columns:
        raise ValueError("DataFrame has no columns; cannot build signature")
    return " | ".join(str(col) for col in columns)


def extract_tables_from_pdf(
    pdf_path: Path, start_page: int, end_page: int
) -> List[pd.DataFrame]:
    tables = []

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num in range(start_page, min(end_page, len(pdf.pages))):
                page = pdf.pages[page_num]
                page_tables = []

                try:
                    extracted = page.extract_tables()
                    if extracted:
                        page_tables.extend(extracted)
                except Exception:
                    pass

                if not page_tables:
                    try:
                        extracted = page.extract_tables(
                            table_settings={
                                "vertical_strategy": "text",
                                "horizontal_strategy": "text",
                                "intersection_tolerance": 3,
                            }
                        )
                        if extracted:
                            page_tables.extend(extracted)
                    except Exception:
                        pass

                for table_data in page_tables:
                    if table_data and len(table_data) > 1:
                        try:
                            clean_data = []
                            for row in table_data:
                                if row and any(
                                    cell and str(cell).strip() for cell in row
                                ):
                                    clean_row = [
                                        str(cell).strip() if cell else ""
                                        for cell in row
                                    ]
                                    clean_data.append(clean_row)

                            if len(clean_data) >= 2:
                                df = pd.DataFrame(clean_data[1:], columns=clean_data[0])
                                if len(df.columns) >= 3 and len(df) >= 1:
                                    tables.append(df)
                        except Exception:
                            continue

    except Exception as e:
        print(f"Error extracting tables from {pdf_path.name}: {e}")

    return tables


def extract_all_pdfs() -> None:
    print("\n" + "=" * 60)
    print("STEP 1: Extracting tables from PDFs")
    print("=" * 60)

    CSV_DIR.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(PDFS_DIR.glob("NPS_*_Visitor_Spending_Effects.pdf"))

    if not pdf_files:
        print(f"No PDF files found in {PDFS_DIR}")
        return

    for pdf_path in pdf_files:
        print(f"\nProcessing: {pdf_path.name}")

        try:
            pdf_reader = pypdf.PdfReader(pdf_path)
            num_pages = len(pdf_reader.pages)
            appendix_page = find_appendix_page(pdf_reader, num_pages)

            if appendix_page is None:
                print(f"  Warning: Could not find appendix in {pdf_path.name}")
                continue

            print(
                f"  Total pages: {num_pages}, Appendix starts at page: {appendix_page + 1}"
            )

            tables = extract_tables_from_pdf(pdf_path, appendix_page, num_pages)
            print(f"  Tables extracted: {len(tables)}")

            table_groups: Dict[str, pd.DataFrame] = {}
            for table in tables:
                try:
                    sig = table_signature(table)
                    if sig not in table_groups:
                        table_groups[sig] = table
                    else:
                        table_groups[sig] = pd.concat(
                            [table_groups[sig], table], ignore_index=True
                        )
                except ValueError:
                    continue

            base_name = pdf_path.stem
            for idx, (sig, table) in enumerate(table_groups.items()):
                csv_path = CSV_DIR / f"{base_name}_table_{idx}.csv"
                table.to_csv(csv_path, index=False)

            print(f"  Saved {len(table_groups)} unique table(s) to CSVs")

        except Exception as e:
            print(f"  Error processing {pdf_path.name}: {e}")


def normalize_ai_content(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(item.get("text", ""))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(content)


def clean_csv_with_ai(path: Path) -> Tuple[str, bool, str | None]:
    try:
        raw_csv = path.read_text()
        response = client.chat.completions.create(
            model="google/gemini-2.5-flash",
            messages=[
                {
                    "role": "user",
                    "content": f"Clean up this CSV data and return only the cleaned CSV:\n\n{raw_csv}",
                }
            ],
        )
        cleaned_csv = (
            normalize_ai_content(response.choices[0].message.content)
            .strip("```")
            .strip("csv")
            .strip()
        )
        path.write_text(cleaned_csv)
        return (path.name, True, None)
    except Exception as exc:
        return (path.name, False, str(exc))


def clean_erroneous_csvs() -> None:
    print("\n" + "=" * 60)
    print("STEP 2: Cleaning erroneous CSVs with AI")
    print("=" * 60)

    target_files = [
        path
        for path in CSV_DIR.iterdir()
        if path.is_file()
        and path.suffix == ".csv"
        and any(year in path.name for year in YEARS_TO_CLEAN)
    ]

    if not target_files:
        print("No target CSV files found for cleaning.")
        return

    print(f"Found {len(target_files)} files to clean")

    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_map = {
            executor.submit(clean_csv_with_ai, path): path for path in target_files
        }
        for future in as_completed(future_map):
            results.append(future.result())

    successes = [name for name, ok, _ in results if ok]
    failures = [(name, err) for name, ok, err in results if not ok]

    if successes:
        print(f"Successfully cleaned {len(successes)} file(s)")
    if failures:
        print("Failures:")
        for name, err in failures:
            print(f"  - {name}: {err}")


def slugify(columns) -> str:
    parts = []
    for column in columns:
        fragment = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(column))
        fragment = fragment.strip("_")
        if fragment:
            parts.append(fragment)
        if len(parts) == 3:
            break
    base = "_".join(parts) or "table"
    while "__" in base:
        base = base.replace("__", "_")
    return base[:31]


def extract_year_from_filename(filename: str) -> str | None:
    match = re.search(r"NPS_(\d{4})_", filename)
    return match.group(1) if match else None


def coalesce_to_excel() -> None:
    print("\n" + "=" * 60)
    print("STEP 3: Coalescing CSVs into Excel files by year")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    csvs_by_year: Dict[str, List[Tuple[Path, pd.DataFrame]]] = defaultdict(list)

    for csv_path in sorted(CSV_DIR.glob("*.csv")):
        year = extract_year_from_filename(csv_path.name)
        if not year:
            print(f"  Skipping {csv_path.name}: could not determine year")
            continue

        try:
            df = pd.read_csv(csv_path)
            if df.empty or df.columns.empty:
                print(f"  Skipping {csv_path.name}: empty or headerless")
                continue
            csvs_by_year[year].append((csv_path, df))
        except Exception as exc:
            print(f"  Skipping {csv_path.name}: {exc}")

    if not csvs_by_year:
        print("No valid CSV files found to coalesce.")
        return

    for year in sorted(csvs_by_year.keys()):
        entries = csvs_by_year[year]
        excel_path = OUTPUT_DIR / f"NPS_{year}_Visitor_Spending_Effects.xlsx"

        print(f"\nCreating {excel_path.name} with {len(entries)} table(s)")

        tables_by_sig: Dict[str, List[pd.DataFrame]] = defaultdict(list)
        for path, df in entries:
            try:
                sig = table_signature(df)
                tables_by_sig[sig].append(df)
            except ValueError:
                continue

        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            sheet_names_used = set()

            for idx, (sig, dfs) in enumerate(tables_by_sig.items(), start=1):
                combined = pd.concat(dfs, ignore_index=True)

                base_sheet_name = slugify(combined.columns)
                sheet_name = base_sheet_name
                counter = 1
                while sheet_name in sheet_names_used:
                    suffix = f"_{counter}"
                    sheet_name = base_sheet_name[: 31 - len(suffix)] + suffix
                    counter += 1
                sheet_names_used.add(sheet_name)

                combined.to_excel(writer, sheet_name=sheet_name, index=False)
                print(f"  Sheet '{sheet_name}': {len(combined)} rows")

    print(f"\nExcel files saved to: {OUTPUT_DIR}")


def main() -> None:
    print("=" * 60)
    print("NPS Economic Effects Data Processing Pipeline")
    print("=" * 60)

    extract_all_pdfs()
    clean_erroneous_csvs()
    coalesce_to_excel()

    print("\n" + "=" * 60)
    print("Processing complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
