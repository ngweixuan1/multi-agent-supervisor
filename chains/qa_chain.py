import warnings
warnings.filterwarnings("ignore")

import os
import json
import yaml
import argparse
from typing import List, Dict, Any

from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langsmith import traceable

from utils.model import RevenueOutput, ExpenditureOutput, FinalAnswer, Router, BudgetState, Part3ConfigModel
from utils.prompts import REVENUE_AGENT_PROMPT, EXPENDITURE_AGENT_PROMPT, SUPERVISOR_SYSTEM_PROMPT, REVIEWER_SYSTEM_PROMPT, ROUTER_PROMPT
from mcp_client.mcp_client import MCPClient

from langsmith import traceable

from dotenv import load_dotenv
load_dotenv()

def load_config(path: str = "config.yaml") -> Part3ConfigModel:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return Part3ConfigModel(**cfg)


class BudgetSupervisorPipeline:
    def __init__(self, config: Part3ConfigModel):
        self.config = config
        self.llm = ChatGoogleGenerativeAI(
            model=self.config.model_name,
            temperature=0,
        )

        self.RevenueParser = self.llm.with_structured_output(RevenueOutput)
        self.ExpenditureParser = self.llm.with_structured_output(ExpenditureOutput)
        self.Reviewer = self.llm.with_structured_output(FinalAnswer)

        self.mcp_client = MCPClient(server_path="mcp_server/search_budget_server.py")
        self._init_tools()
        self._init_agents()
        self._init_graph()

        


    def _init_tools(self):
        mcp_ref = self.mcp_client
        json_path = self.config.extracted_text_path

        @tool("search_budget_text", return_direct=True)
        def search_budget_text(keyword: str) -> List[Dict[str, Any]]:
            """Call the MCP budget text search server to find text containing the keyword."""
            # Send both keyword and file path to the MCP server
            return mcp_ref.call(
                "search_budget_text",
                arguments={"keyword": keyword, "structured_json_path": json_path},
            )

        self.search_budget_text = search_budget_text

    def _init_agents(self):
        self.RevenueAgent = create_agent(
            model=self.llm,
            tools=[self.search_budget_text],
            system_prompt=REVENUE_AGENT_PROMPT,
        )

        self.ExpenditureAgent = create_agent(
            model=self.llm,
            tools=[self.search_budget_text],
            system_prompt=EXPENDITURE_AGENT_PROMPT,
        )

    @traceable(name="SupervisorNode")
    def supervisor_node(self, state: Dict[str, Any]) -> Command:
        user_query = state.get("query", "")
        loop_count = state.get("loop_count", 0)
        last_node = state.get("last_node")
        revenue = state.get("revenue", "")
        expenditure = state.get("expenditure", "")
        cur_reasoning = state.get("cur_reasoning", "")

        print(f"\n[Supervisor Loop {loop_count}] - Deciding next worker...")
        print(f"last_node={last_node}")

        if loop_count >= self.config.max_loop:
            print("Max loop count reached. Ending process.")
            goto = "FINISH"
            reasoning = "Stopped after maximum allowed loops."
        else:
            members_dict = {
                "revenue_node": "Handles revenue/tax/income-related queries.",
                "expenditure_node": "Handles expenditure/fund/budget-related queries.",
            }
            worker_info = "\n\n".join(
                [f"WORKER: {k}\nDESCRIPTION: {v}" for k, v in members_dict.items()]
            ) + "\n\nWORKER: FINISH\nDESCRIPTION: Stop when query fully answered."

            system_prompt = SUPERVISOR_SYSTEM_PROMPT.format(worker_info= worker_info)

            router_llm = self.llm.with_structured_output(Router)
            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": ROUTER_PROMPT.format(
                        user_query= user_query,
                        last_node = last_node,
                        loop_count=loop_count,
                        cur_reasoning= cur_reasoning,
                        revenue= revenue or '<empty>',
                        expenditure=expenditure  or '<empty>'
                    ),
                },
            ]
            response = router_llm.invoke(messages)
            goto = response["next"]
            reasoning = response["reasoning"]

        print(f"Supervisor routed to: {goto}")
        print(f"Reasoning: {reasoning}")

        if last_node == goto and goto != "FINISH":
            print("Same route repeated → forcing FINISH.")
            goto = "FINISH"
            reasoning += " (Stopped because same node repeated.)"

        if goto == "FINISH":
            combined = self.Reviewer.invoke(REVIEWER_SYSTEM_PROMPT.format(revenue=revenue, expenditure= expenditure, user_query= user_query))
            print("Supervisor completed summary.\n")
            return Command(
                goto=END,
                update={"final_output": combined, "cur_reasoning": reasoning},
            )

        return Command(
            goto=goto,
            update={
                "query": user_query,
                "cur_reasoning": reasoning,
                "loop_count": loop_count + 1,
                "last_node": goto,
            },
        )

    @traceable(name="RevenueAgentNode")
    def node_revenue(self, state: Dict[str, Any]) -> Dict[str, Any]:
        print("Running Revenue Agent...")
        resp = self.RevenueAgent.invoke({"messages": [{"role": "user", "content": "Past actions: " + state["query"]}]})
        raw = resp["messages"][-1].content
        try:
            structured = self.RevenueParser.invoke(
                f"Convert to JSON with field 'revenue_streams': {raw}"
            )
            revenue_value = getattr(structured, "revenue_streams", raw)
        except Exception as e:
            print(f"RevenueParser failed: {e}")
            revenue_value = raw
        return {"revenue": revenue_value, "expenditure": state.get("expenditure"),  "Past actions: " +  "query": state["query"]}

    @traceable(name="ExpenditureAgentNode")
    def node_expenditure(self, state: Dict[str, Any]) -> Dict[str, Any]:
        print("Running Expenditure Agent...")
        resp = self.ExpenditureAgent.invoke({"messages": [{"role": "user", "content": "Past actions: " + state["query"]}]})
        raw = resp["messages"][-1].content
        try:
            structured = self.ExpenditureParser.invoke(
                f"Convert to JSON with field 'expenditure_streams': {raw}"
            )
            expenditure_value = getattr(structured, "expenditure_streams", raw)
        except Exception as e:
            print(f"ExpenditureParser failed: {e}")
            expenditure_value = raw
        return {"expenditure": expenditure_value, "revenue": state.get("revenue"), "query": "Past actions: " + state["query"]}

    def _init_graph(self):
        graph = StateGraph(BudgetState)
        graph.add_node("supervisor", self.supervisor_node)
        graph.add_node("revenue_node", self.node_revenue)
        graph.add_node("expenditure_node", self.node_expenditure)
        graph.add_edge(START, "supervisor")
        graph.add_edge("revenue_node", "supervisor")
        graph.add_edge("expenditure_node", "supervisor")
        graph.add_edge("supervisor", END)
        self.app = graph.compile()

    def run(self, user_query: str):
        print("\nSTARTING GRAPH EXECUTION\n")
        result = self.app.invoke({"query": user_query, "loop_count": 0, "last_node": None})        
        return result["final_output"].model_dump()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Government Budget Supervisor Pipeline")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML config file.")
    parser.add_argument("--query", type=str, required=True, help="User query to analyze.")
    args = parser.parse_args()

    config = load_config(args.config)
    pipeline = BudgetSupervisorPipeline(config)
    result = pipeline.run(args.query)
    print("Final Result: ", result["direct_answer"])