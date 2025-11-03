import warnings
warnings.filterwarnings("ignore")

import io
import yaml
import pdfplumber
from typing import List, Dict, Any
import json

from utils.call_gemini import GeminiAPIClient

from dotenv import load_dotenv
load_dotenv()

class PdfplumberLoader:
    """
    Extracts structured text and tables with pdfplumber.
    Falls back to Gemini OCR and table reconstruction when needed.
    """

    def __init__(self, pdf_path: str, ocr_threshold: int = 30):
        self.pdf_path = pdf_path
        self.ocr_threshold = ocr_threshold
        self.gemini = GeminiAPIClient()

    def load(self) -> Dict[str, Any]:
        structured = {"metadata": {"source": self.pdf_path}, "elements": []}
        ocr_pages: List[int] = []

        with pdfplumber.open(self.pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                text = (page.extract_text() or "").strip()

                #  OCR fallback if text missing
                if len(text) < self.ocr_threshold:
                    img = page.to_image(resolution=300).original
                    buf = io.BytesIO()
                    img.save(buf, format="PNG")

                    ocr_resp = self.gemini.generate_content(
                        "Extract all visible text and numbers from this document image.",
                        image_bytes=buf.getvalue()
                    )
                    text = ocr_resp.text
                    ocr_pages.append(page_num)

                # Extract tables
                tables = page.extract_tables()
                table_md_blocks = []
                for t in tables or []:
                    if not t:
                        continue
                    try:
                        header, *rows = t
                        # convert manually to markdown if needed
                        if not any("|" in (cell or "") for row in rows for cell in row):
                            table_md = self._to_markdown(t)
                        else:
                            table_md = "\n".join(["|".join(r or "") for r in t])
                        # ensure valid markdown
                        if "|" not in table_md:
                            table_prompt = (
                                "Convert the following messy or OCRed text into a valid Markdown table "
                                "preserving numeric precision:\n\n" + table_md
                            )
                            table_md = self.gemini.generate_content(table_prompt).text
                        table_md_blocks.append(table_md)
                    except Exception as e:
                        print(f"[WARN] Table parsing failed on page {page_num}: {e}")
                        continue

                structured["elements"].append({
                    "page": page_num,
                    "content_markdown": text,
                })


        print(f"[INFO] Gemini OCR triggered on pages: {ocr_pages or 'None'}")
        return structured

    def _to_markdown(self, table: List[List[str]]) -> str:
        """Convert list-of-lists table to Markdown."""
        md = []
        for i, row in enumerate(table):
            md.append("| " + " | ".join(cell.strip() if cell else "" for cell in row) + " |")
            if i == 0:
                md.append("|" + "|".join(["---"] * len(row)) + "|")
        return "\n".join(md)
    
def main():
    # Load config file
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    pdf_fp = config.get("pdf_fp")
    if not pdf_fp:
        raise ValueError(" Missing 'pdf_fp' in config.yaml.")

    loader = PdfplumberLoader(pdf_path=pdf_fp)
    structured_output = loader.load()

    output_path = config["extracted_text_path"]
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(structured_output, f, ensure_ascii=False, indent=2)

    print(f"Extraction complete. Output saved to: {output_path}")


if __name__ == "__main__":
    main()