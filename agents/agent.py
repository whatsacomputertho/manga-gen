from langchain_ollama import ChatOllama
from pydantic import BaseModel

class Agent:
    def __init__(
            self,
            name: str,
            llm: ChatOllama,
            think_schema: BaseModel,
            act_schema: BaseModel
        ):
        """
        Constructor for the base Agent class

        Args:
            name (str): The name of the agent
            llm (ChatOllama): The LLM powering the agent
            think_schema (BaseModel): The pydantic model for structuring the agent think functino
            act_schema (BaseModel): The pydantic model for structuring the agent act function

        Returns:
            Agent: The agent instance
        """
        self.name = name
        self.llm = llm
        self.think_schema = think_schema
        self.act_schema = act_schema

    def think(self, prompt: str):
        """
        The think function where the agent decides its next action

        Args:
            prompt (str): The prompt for the agent think function
        
        Returns:
            dict: The structured output from the model following the think schema
        """
        print(f"[{self.name}] Think")
        structured = self.llm.with_structured_output(self.think_schema)
        res = None
        for chunk in structured.stream(prompt):
            print(
                str(chunk)[len(str(res)) - 1 if res is not None else 0:]
                    .replace("'", "")
                    .replace('"', ""),
                end="",
                flush=True
            )
            res = chunk
        print("")
        return res.model_dump()
    
    def act(self, prompt: str):
        print("[AnimeSummaryAgent] Act")
        structured = self.llm.with_structured_output(self.act_schema)
        res = None
        for chunk in structured.stream(prompt):
            print(
                str(chunk)[len(str(res)) - 1 if res is not None else 0:]
                    .replace("'", "")
                    .replace('"', ""),
                end="",
                flush=True
            )
            res = chunk
        print("")
        return res.model_dump()
