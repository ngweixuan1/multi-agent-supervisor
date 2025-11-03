import warnings
warnings.filterwarnings("ignore")

import os
import json
import yaml
from typing import List, Dict, Any

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent

from utils.prompts import REASONING_NORMALIZED_DATE_PROMPT, NORMALIZED_DATE_AGENT_PROMPT
from utils.model import ExtractedTextModel, Part2AnswerSchema, ConfigModel
from mcp_client.mcp_client import MCPClient



class BudgetDatePipeline:
    def __init__(self, config: ConfigModel, extracted: ExtractedTextModel):
        self.config = config
        self.extracted = extracted
        
        # Initialize MCP client
        self.mcp_client = MCPClient("mcp_server/normalize_date_server.py")

        # Register tool
        @tool("normalize_date", return_direct=True)
        def normalize_date(date_string: str) -> str:
            """Normalize budget-style dates to ISO (YYYY-MM-DD)."""
            return self.mcp_client.call(date_string)

        # LLM setup
        self.model = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0,
            convert_system_message_to_human=True,
        )

        self.strucutured_model = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0,
            convert_system_message_to_human=True,
        ).with_structured_output(Part2AnswerSchema)

        
        self.agent = create_react_agent(model=self.model, tools=[normalize_date], prompt=NORMALIZED_DATE_AGENT_PROMPT)

    def process_pages(self) -> Dict[int, List[Dict[str, Any]]]:
        page_results = []

        for page in self.config.target_pages_part_2:
            page_elems = [e for e in self.extracted.elements if e["page"] == page]
            

            for elem in page_elems:
                text = elem.get("content_markdown", "")
                if not text.strip():
                    continue

                # Part 1
                norm_response = self.agent.invoke({"messages": [{"role": "user", "content": "Text: " + text}]})
                normalized_output = norm_response["messages"][-1].content
                # Part 2
                summary_response = self.strucutured_model.invoke(REASONING_NORMALIZED_DATE_PROMPT.format(normalized_output, text))
                
                page_results.append(summary_response.model_dump())
                print(summary_response.model_dump(), page_results)
        return page_results

if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    if not os.getenv("GOOGLE_API_KEY"):
        raise EnvironmentError("Missing GOOGLE_API_KEY in .env")

    # Load YAML config
    with open("config.yaml", "r") as f:
        cfg_dict = yaml.safe_load(f)
    config = ConfigModel(**cfg_dict)

    # Load extracted JSON
    with open(config.extracted_text_path, "r") as f:
        extracted_json = json.load(f)
    
    extracted = ExtractedTextModel(**extracted_json)

    # Initialize pipeline
    pipeline = BudgetDatePipeline(config=config, extracted=extracted)

    # Run pipeline 
    results = pipeline.process_pages()
    print(results)

    # Save final combined output
    os.makedirs(config.output_dir, exist_ok=True)
    output_path = os.path.join(config.output_dir, "normalized_part2.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n Full normalization + summarization results saved to: {output_path}")