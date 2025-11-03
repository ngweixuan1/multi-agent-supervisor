import warnings
warnings.filterwarnings("ignore")

import os
import json
import yaml
from typing import Any, Dict, List
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel


from dotenv import load_dotenv
load_dotenv()

from utils.prompts import FIELD_EXTRACTION_PROMPT
from utils.model import FinancialFields


class FieldExtractionChain:
    """
    Uses Gemini (LangChain wrapper) to extract structured key metrics
    from selected pages in a parsed Budget document.
    """

    def __init__(self, model: str = "gemini-2.5-flash"):
        self.model = ChatGoogleGenerativeAI(
            model=model,
            temperature=0,
            convert_system_message_to_human=True,
        ).with_structured_output(FinancialFields)

    def run(
        self,
        structured_text: Dict[str, Any],
        target_pages: List[int],
        prompt_template: str,
    ) -> Dict[str, Any]:
        """
        Iterate through selected pages, extract structured data per page,
        and merge into a single dictionary.
        """
        elements = structured_text.get("elements", [])
        results = {
            "corporate_income_tax_2024_billion": None,
            "corporate_income_tax_yoy_percent": None,
            "total_topups_2024_billion": None,
            "operating_revenue_taxes_list": [],
            "latest_actual_fiscal_position_billion": None,
        }
        for page in target_pages:
            page_text = "\n\n".join(
                el["content_markdown"] for el in elements if el["page"] == page
            ).strip()
            if not page_text:
                print(f"[INFO] Skipping empty page {page}")
                continue

            prompt = prompt_template.format(text_block=page_text)

            structured_resp = self.model.invoke(prompt)
            page_data = structured_resp.model_dump()
            results = self._merge_results(results, page_data)
            print(f"[INFO] Structured extraction successful for page {page}")

        return results

    def _merge_results(self, base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
        """Merge structured page-level results."""
        for k, v in update.items():
            if v is None:
                continue
            if isinstance(v, list):
                base[k].extend([x for x in v if x not in base[k]])
            else:
                base[k] = v
        return base


def main():
    # Load config
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    structured_json_fp = config.get("extracted_text_path")
    target_pages = config.get("target_pages_part_1", [])
    gemini_model = config.get("gemini_model", "gemini-2.5-flash")

    if not structured_json_fp or not os.path.exists(structured_json_fp):
        raise FileNotFoundError("extracted_text_path not found in config.yaml or file missing.")

    with open(structured_json_fp, "r", encoding="utf-8") as f:
        structured_text = json.load(f)

    extractor = FieldExtractionChain(model=gemini_model)
    results = extractor.run(structured_text, target_pages, FIELD_EXTRACTION_PROMPT)

    # Save results to JSON
    output_fp = config["extracted_field_path"]
    with open(output_fp, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Field extraction complete. Results saved to: {output_fp}")

if __name__ == "__main__":
    main()