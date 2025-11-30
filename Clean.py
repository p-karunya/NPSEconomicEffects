import pdfplumber
import pypdf
import os
import pandas as pd


def findAppendixPage(pdf, num_pages):
    for page_num in range(num_pages - 1, -1, -1):
        page = pdf.pages[page_num]
        text = page.extract_text()
        if (
            "Appendix" in text
            and "visits, spending, and economic contributions to local economies"
            in text.lower()
        ):
            return page_num
    return None


def id(table):
    res = str(table.columns.tolist()[0])
    for col in table.columns:
        res += " | " + str(col)
    return res


def extract_tables_with_pdfplumber(pdf_path, start_page, end_page):
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
                except:
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
                    except:
                        pass

                for table_data in page_tables:
                    if table_data and len(table_data) > 1:
                        try:
                            clean_data = []
                            for row in table_data:
                                if row and any(
                                    cell and str(cell).strip()
                                    for cell in row  # if cell is not None and not empty
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
                        except Exception as e:
                            continue

    except Exception as e:
        print(f"Error extracting tables: {e}")

    return tables


# Main execution
script_dir = os.path.dirname(os.path.abspath(__file__))
pdfs_dir = os.path.join(script_dir, "pdfs")

Pdf_files = [
    os.path.join(pdfs_dir, "NPS_2023_Visitor_Spending_Effects.pdf"),
    os.path.join(pdfs_dir, "NPS_2024_Visitor_Spending_Effects.pdf"),
    os.path.join(pdfs_dir, "NPS_2022_Visitor_Spending_Effects.pdf"),
    os.path.join(pdfs_dir, "NPS_2021_Visitor_Spending_Effects.pdf"),
    os.path.join(pdfs_dir, "NPS_2020_Visitor_Spending_Effects.pdf"),
    os.path.join(pdfs_dir, "NPS_2019_Visitor_Spending_Effects.pdf"),
]

for file in Pdf_files:
    PDF = pypdf.PdfReader(file)
    num_pages = len(PDF.pages)
    appendix_page = findAppendixPage(PDF, num_pages)

    print(
        f"File: {file}, Total pages: {num_pages}, Appendix starts at page: {appendix_page + 1}"
    )

    tables = extract_tables_with_pdfplumber(file, appendix_page, num_pages)

    print(f"Total tables extracted: {len(tables)} in {file.rsplit(os.sep)[-1]}")

    table_ids = {}

    for i, table in enumerate(tables):
        table_id = id(table)
        if table_id not in table_ids.keys():
            table_ids[table_id] = table
        else:
            table_ids[table_id] = pd.concat([table_ids[table_id], table])

    x = 0

    for tid, table in table_ids.items():
        with open(
            f"csvs/{file.rsplit(os.sep)[-1].rsplit('.', 1)[0]}_table_{x}.csv", "w"
        ) as f:
            f.write(table.to_csv(index=False))
        x += 1
