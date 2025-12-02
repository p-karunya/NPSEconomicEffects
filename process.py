from __future__ import annotations

import base64
import io
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


def parse_text_table_row(text_line: str, expect_state: bool = False) -> List[str]:
    parts = text_line.split()
    if len(parts) < 6:
        return []

    numeric_indices = []
    for i, part in enumerate(parts):
        clean = (
            part.replace(",", "")
            .replace("$", "")
            .replace("%", "")
            .replace("(", "")
            .replace(")", "")
        )
        if clean.replace(".", "").replace("-", "").isdigit() or clean == "0":
            try:
                float(clean) if clean else None
                numeric_indices.append(i)
            except ValueError:
                pass

    if len(numeric_indices) >= 6:
        first_num_idx = numeric_indices[0]
        name = " ".join(parts[:first_num_idx])
        numbers = parts[first_num_idx : first_num_idx + 6]
        return [name] + numbers

    return []


def extract_park_table_from_text(page_text: str) -> List[List[str]]:
    lines = page_text.strip().split("\n")
    rows = []

    skip_patterns = [
        "Appendix",
        "Table A-",
        "Park Unit",
        "Total Visitor",
        "Total Recreation",
        "Economic",
        "Labor Income",
        "Value Added",
        "NPS Visitor Spending Effects",
        "Spending",
        "Recreation Visits",
        "a For these parks",
        "b Trip characteristic",
        "c Areas that were",
        "profiles or best",
        "definitions were updated",
    ]

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if any(pat in line for pat in skip_patterns):
            continue

        if "$" not in line and "," not in line:
            continue

        row = parse_text_table_row(line)
        if row and len(row) == 7 and row[0]:
            rows.append(row)

    return rows


def extract_state_table_from_text(page_text: str) -> List[List[str]]:
    lines = page_text.strip().split("\n")
    rows = []

    skip_patterns = [
        "Appendix",
        "Table A-",
        "State",
        "Total Visitor",
        "Total Recreation",
        "Economic",
        "Labor Income",
        "Value Added",
        "NPS Visitor Spending Effects",
        "Spending",
        "Recreation Visits",
        "Visits",
        "Jobs",
        "($Millions",
    ]

    us_states = [
        "Alabama",
        "Alaska",
        "American Samoa",
        "Arizona",
        "Arkansas",
        "California",
        "Colorado",
        "Connecticut",
        "Delaware",
        "District of Columbia",
        "Florida",
        "Georgia",
        "Guam",
        "Hawaii",
        "Idaho",
        "Illinois",
        "Indiana",
        "Iowa",
        "Kansas",
        "Kentucky",
        "Louisiana",
        "Maine",
        "Maryland",
        "Massachusetts",
        "Michigan",
        "Minnesota",
        "Mississippi",
        "Missouri",
        "Montana",
        "Nebraska",
        "Nevada",
        "New Hampshire",
        "New Jersey",
        "New Mexico",
        "New York",
        "North Carolina",
        "North Dakota",
        "Northern Mariana",
        "Ohio",
        "Oklahoma",
        "Oregon",
        "Pennsylvania",
        "Puerto Rico",
        "Rhode Island",
        "South Carolina",
        "South Dakota",
        "Tennessee",
        "Texas",
        "U.S. Virgin Islands",
        "Utah",
        "Vermont",
        "Virginia",
        "Washington",
        "West Virginia",
        "Wisconsin",
        "Wyoming",
        "Total",
    ]

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if any(pat in line for pat in skip_patterns):
            continue

        if "$" not in line and "," not in line:
            continue

        is_state_line = any(line.startswith(state) for state in us_states)
        if not is_state_line:
            continue

        row = parse_text_table_row(line)
        if row and len(row) == 7 and row[0]:
            rows.append(row)

    return rows


def extract_tables_from_pdf(
    pdf_path: Path, start_page: int, end_page: int
) -> List[pd.DataFrame]:
    tables = []
    year = None
    for y in ALL_YEARS:
        if y in pdf_path.name:
            year = y
            break

    is_problematic_year = year in YEARS_TO_CLEAN

    try:
        with pdfplumber.open(pdf_path) as pdf:
            park_unit_rows = []
            state_summary_rows = []
            other_tables = []

            for page_num in range(start_page, min(end_page, len(pdf.pages))):
                page = pdf.pages[page_num]
                page_text = page.extract_text() or ""

                if is_problematic_year:
                    if "Park Unit" in page_text and (
                        "Recreation" in page_text or "Visits" in page_text
                    ):
                        text_rows = extract_park_table_from_text(page_text)
                        park_unit_rows.extend(text_rows)
                    elif "state economies" in page_text.lower() or (
                        "Table A-3" in page_text and "State" in page_text
                    ):
                        text_rows = extract_state_table_from_text(page_text)
                        state_summary_rows.extend(text_rows)

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
                                    other_tables.append(df)
                        except Exception:
                            continue

            if park_unit_rows:
                df = pd.DataFrame(park_unit_rows, columns=PARK_UNIT_REFERENCE_HEADERS)
                tables.append(df)

            if state_summary_rows:
                df = pd.DataFrame(
                    state_summary_rows, columns=STATE_SUMMARY_REFERENCE_HEADERS
                )
                tables.append(df)

            tables.extend(other_tables)

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


EXPECTED_TABLE_TYPES = [
    {
        "name": "park_unit_data",
        "description": "Park-level visitor spending and economic impacts",
        "headers": [
            "Park Unit",
            "Total Recreation Visits",
            "Total Visitor Spending ($000s)",
            "Jobs",
            "Labor Income ($000s)",
            "Value Added ($000s)",
            "Economic Output ($000s)",
        ],
    },
    {
        "name": "state_summary",
        "description": "State-level summary of visitor spending and economic impacts",
        "headers": [
            "State",
            "Total Recreation Visits",
            "Total Visitor Spending ($Millions)",
            "Jobs",
            "Labor Income ($Millions)",
            "Value Added ($Millions)",
            "Economic Output ($Millions)",
        ],
    },
    {
        "name": "multi_state_parks",
        "description": "Parks that span multiple states with percentage shares",
        "headers": ["Park Unit", "State", "Share"],
    },
    {
        "name": "spending_sectors",
        "description": "Spending categories mapped to economic sectors",
        "headers": ["Spending Group", "Sector Name", "IMPLAN Sector", "Weight"],
    },
    {
        "name": "visitor_spending_categories",
        "description": "Visitor spending breakdown by category",
        "headers": ["Spending Category", "Amount"],
    },
]

PARK_UNIT_REFERENCE_HEADERS = [
    "Park Unit",
    "Total Recreation Visits",
    "Total Visitor Spending ($000s)",
    "Jobs",
    "Labor Income ($000s)",
    "Value Added ($000s)",
    "Economic Output ($000s)",
]

STATE_SUMMARY_REFERENCE_HEADERS = [
    "State",
    "Total Recreation Visits",
    "Total Visitor Spending ($Millions)",
    "Jobs",
    "Labor Income ($Millions)",
    "Value Added ($Millions)",
    "Economic Output ($Millions)",
]


def get_all_appendix_page_images(year: str) -> List[str]:
    pdf_path = PDFS_DIR / f"NPS_{year}_Visitor_Spending_Effects.pdf"
    if not pdf_path.exists():
        return []
    images = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            pdf_reader = pypdf.PdfReader(pdf_path)
            num_pages = len(pdf_reader.pages)
            appendix_page = find_appendix_page(pdf_reader, num_pages)
            if appendix_page is None:
                return []
            for page_num in range(appendix_page, min(appendix_page + 20, num_pages)):
                page = pdf.pages[page_num]
                img = page.to_image(resolution=150)
                img_buffer = io.BytesIO()
                img.save(img_buffer, format="PNG")
                img_buffer.seek(0)
                images.append(base64.b64encode(img_buffer.read()).decode("utf-8"))
    except Exception:
        pass
    return images


def get_table_image_from_pdf(year: str) -> str | None:
    pdf_path = PDFS_DIR / f"NPS_{year}_Visitor_Spending_Effects.pdf"
    if not pdf_path.exists():
        return None
    try:
        with pdfplumber.open(pdf_path) as pdf:
            pdf_reader = pypdf.PdfReader(pdf_path)
            num_pages = len(pdf_reader.pages)
            appendix_page = find_appendix_page(pdf_reader, num_pages)
            if appendix_page is None:
                return None
            page = pdf.pages[appendix_page]
            img = page.to_image(resolution=150)
            img_buffer = io.BytesIO()
            img.save(img_buffer, format="PNG")
            img_buffer.seek(0)
            return base64.b64encode(img_buffer.read()).decode("utf-8")
    except Exception:
        return None


def clean_single_csv_with_ai(
    path: Path, year: str, image_b64: str | None
) -> Tuple[str, str | None, str | None]:
    try:
        raw_csv = path.read_text()

        example_park_data = """Park Unit,Total Recreation Visits,Total Visitor Spending ($000s),Jobs,Labor Income ($000s),Value Added ($000s),Economic Output ($000s)
Abraham Lincoln Birthplace NHP,239950,16070,242,8000,12523,23391
Acadia NP,3879890,475175,6603,229830,390600,685376
Adams NHP,25229,1691,21,1017,1586,2569"""

        example_state_data = """State,Total Recreation Visits,Total Visitor Spending ($Millions),Jobs,Labor Income ($Millions),Value Added ($Millions),Economic Output ($Millions)
Alabama,1287291,88.9,1338,39.2,63.6,121.9
Alaska,3254809,1504.9,21274,838.7,1292.4,2307.6
Arizona,10809520,1225.0,17319,684.2,1156.7,1997.0"""

        prompt = f"""This CSV was extracted from a PDF and has errors. Fix it to match the correct format.

REFERENCE FORMAT FOR PARK UNIT DATA (7 columns exactly):
{example_park_data}

REFERENCE FORMAT FOR STATE SUMMARY DATA (7 columns exactly):
{example_state_data}

RULES:
1. The table must have EXACTLY 7 columns
2. First column is either "Park Unit" (park names) or "State" (state names)
3. Remove any columns with .1, .2, .3 suffixes - these are duplicates
4. Remove any trailing empty columns
5. Remove any rows that look like headers mixed into data
6. Each data row should have: name, visits number, spending amount, jobs number, labor income, value added, economic output
7. Numbers can have commas or $ signs but must be in correct columns

RAW CSV TO FIX:
{raw_csv}

Return ONLY the corrected CSV. No explanation."""

        if image_b64:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
        else:
            messages = [{"role": "user", "content": prompt}]

        response = client.chat.completions.create(
            model="google/gemini-2.5-flash",
            messages=messages,
        )
        cleaned_csv = (
            normalize_ai_content(response.choices[0].message.content)
            .strip("```")
            .strip("csv")
            .strip()
        )
        return (path.name, cleaned_csv, None)
    except Exception as exc:
        return (path.name, None, str(exc))


def validate_and_fix_csv(csv_text: str, table_type: str | None) -> str:
    try:
        df = pd.read_csv(io.StringIO(csv_text))

        cols_to_drop = []
        for c in df.columns:
            col_str = str(c)
            if (
                col_str.startswith("Unnamed")
                or ".1" in col_str
                or ".2" in col_str
                or ".3" in col_str
                or col_str.strip() == ""
            ):
                cols_to_drop.append(c)

        if cols_to_drop:
            df = df.drop(columns=cols_to_drop, errors="ignore")

        df = df.dropna(axis=1, how="all")

        df = df.loc[:, ~df.columns.duplicated()]

        if table_type == "park_unit_data":
            if len(df.columns) > 7:
                df = df.iloc[:, :7]
            if len(df.columns) == 7:
                df.columns = PARK_UNIT_REFERENCE_HEADERS
        elif table_type == "state_summary":
            if len(df.columns) > 7:
                df = df.iloc[:, :7]
            if len(df.columns) == 7:
                df.columns = STATE_SUMMARY_REFERENCE_HEADERS

        return df.to_csv(index=False)
    except Exception:
        return csv_text


def identify_table_type(df: pd.DataFrame) -> str | None:
    columns_lower = [str(c).lower() for c in df.columns]
    cols_str = " ".join(columns_lower)

    if "park unit" in cols_str and "state" in cols_str and "share" in cols_str:
        return "multi_state_parks"
    if (
        "spending group" in cols_str
        or "sector name" in cols_str
        or "implan" in cols_str
    ):
        return "spending_sectors"
    if "spending category" in cols_str or (
        "category" in cols_str and "amount" in cols_str
    ):
        return "visitor_spending_categories"
    if (
        "state" in cols_str
        and "park" not in cols_str
        and ("visits" in cols_str or "jobs" in cols_str)
    ):
        return "state_summary"
    if ("park" in cols_str or "unit" in cols_str) and (
        "visits" in cols_str or "jobs" in cols_str
    ):
        return "park_unit_data"

    return None


def consolidate_tables_by_type(year: str) -> None:
    target_files = [
        path
        for path in CSV_DIR.iterdir()
        if path.is_file() and path.suffix == ".csv" and f"NPS_{year}_" in path.name
    ]

    if not target_files:
        return

    tables_by_type: Dict[str, List[Tuple[Path, pd.DataFrame]]] = defaultdict(list)

    for path in target_files:
        try:
            df = pd.read_csv(path)
            if df.empty or df.columns.empty:
                continue

            cols_to_drop = [
                c
                for c in df.columns
                if str(c).startswith("Unnamed") or ".1" in str(c) or ".2" in str(c)
            ]
            if cols_to_drop:
                df = df.drop(columns=cols_to_drop, errors="ignore")

            table_type = identify_table_type(df)
            if table_type:
                tables_by_type[table_type].append((path, df))
        except Exception:
            continue

    print(f"    Classified into {len(tables_by_type)} table type(s)")
    for table_type, entries in tables_by_type.items():
        print(f"      - {table_type}: {len(entries)} file(s)")

    for old_path in target_files:
        try:
            old_path.unlink()
        except Exception:
            pass

    for idx, (table_type, entries) in enumerate(tables_by_type.items()):
        fixed_dfs = []
        for _, df in entries:
            if table_type == "park_unit_data":
                df = df.iloc[:, :7]
                if len(df.columns) == 7:
                    df.columns = PARK_UNIT_REFERENCE_HEADERS
                    fixed_dfs.append(df)
            elif table_type == "state_summary":
                df = df.iloc[:, :7]
                if len(df.columns) == 7:
                    df.columns = STATE_SUMMARY_REFERENCE_HEADERS
                    fixed_dfs.append(df)
            else:
                fixed_dfs.append(df)

        if not fixed_dfs:
            continue

        combined = pd.concat(fixed_dfs, ignore_index=True)
        combined = combined.drop_duplicates()

        new_path = CSV_DIR / f"NPS_{year}_Visitor_Spending_Effects_table_{idx}.csv"
        combined.to_csv(new_path, index=False)
        print(f"      Saved {table_type} ({len(combined)} rows) to {new_path.name}")


def group_and_consolidate_with_ai(year: str) -> None:
    target_files = sorted(
        [
            path
            for path in CSV_DIR.iterdir()
            if path.is_file() and path.suffix == ".csv" and f"NPS_{year}_" in path.name
        ]
    )

    if len(target_files) <= 5:
        print(
            f"    Year {year} already has {len(target_files)} tables, skipping consolidation"
        )
        return

    tables_info = []
    for path in target_files:
        try:
            df = pd.read_csv(path)
            if df.empty:
                continue
            preview = df.head(3).to_csv(index=False)
            tables_info.append(
                {
                    "filename": path.name,
                    "columns": list(df.columns),
                    "row_count": len(df),
                    "preview": preview,
                }
            )
        except Exception:
            continue

    if not tables_info:
        return

    tables_desc = "\n\n".join(
        [
            f"File: {t['filename']}\nColumns: {t['columns']}\nRows: {t['row_count']}\nPreview:\n{t['preview']}"
            for t in tables_info
        ]
    )

    prompt = f"""I have {len(tables_info)} CSV files from the same year that need to be grouped into ~5 logical table types.

Expected table types:
1. park_unit_data: Park-level visitor spending and economic impacts (Park Unit, Recreation Visits, Spending, Jobs, etc.)
2. state_summary: State-level summary (State, Recreation Visits, Spending, Jobs, etc.)
3. multi_state_parks: Parks spanning multiple states (Park Unit, State, Share percentage)
4. spending_sectors: Spending categories to economic sectors mapping (Spending Group, Sector Name, IMPLAN Sector)
5. visitor_spending_categories: Spending breakdown by category

Here are the tables:
{tables_desc}

Return a JSON object mapping each filename to its table type. Example:
{{"NPS_2020_table_0.csv": "park_unit_data", "NPS_2020_table_1.csv": "state_summary", ...}}

Return ONLY the JSON, nothing else."""

    try:
        response = client.chat.completions.create(
            model="google/gemini-2.5-flash",
            messages=[{"role": "user", "content": prompt}],
        )

        result_text = normalize_ai_content(response.choices[0].message.content).strip()
        result_text = result_text.strip("```").strip("json").strip()

        import json

        groupings = json.loads(result_text)

        tables_by_type: Dict[str, List[pd.DataFrame]] = defaultdict(list)

        for path in target_files:
            if path.name in groupings:
                table_type = groupings[path.name]
                try:
                    df = pd.read_csv(path)
                    tables_by_type[table_type].append(df)
                except Exception:
                    pass
            path.unlink()

        for idx, (table_type, dfs) in enumerate(tables_by_type.items()):
            if not dfs:
                continue

            fixed_dfs = []
            for df in dfs:
                if table_type == "park_unit_data" and len(df.columns) >= 7:
                    df = df.iloc[:, :7]
                    df.columns = PARK_UNIT_REFERENCE_HEADERS
                elif table_type == "state_summary" and len(df.columns) >= 7:
                    df = df.iloc[:, :7]
                    df.columns = STATE_SUMMARY_REFERENCE_HEADERS
                fixed_dfs.append(df)

            combined = pd.concat(fixed_dfs, ignore_index=True)
            combined = combined.drop_duplicates()
            new_path = CSV_DIR / f"NPS_{year}_Visitor_Spending_Effects_table_{idx}.csv"
            combined.to_csv(new_path, index=False)
            print(
                f"      Consolidated {table_type}: {len(combined)} rows -> {new_path.name}"
            )

    except Exception as exc:
        print(f"    Error in AI grouping: {exc}")
        consolidate_tables_by_type(year)


def clean_erroneous_csvs() -> None:
    print("\n" + "=" * 60)
    print("STEP 2: Consolidating and cleaning CSVs")
    print("=" * 60)

    for year in YEARS_TO_CLEAN:
        print(f"\nProcessing year {year}...")

        target_files = [
            path
            for path in CSV_DIR.iterdir()
            if path.is_file() and path.suffix == ".csv" and f"NPS_{year}_" in path.name
        ]

        if not target_files:
            print(f"  No CSV files found for {year}")
            continue

        print(f"  Found {len(target_files)} CSV files")

        print("  Grouping and consolidating tables by type...")
        consolidate_tables_by_type(year)

        final_count = len(
            [
                p
                for p in CSV_DIR.iterdir()
                if p.is_file() and p.suffix == ".csv" and f"NPS_{year}_" in p.name
            ]
        )
        print(f"  Final table count for {year}: {final_count}")

    print("\nConsolidation complete")


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
