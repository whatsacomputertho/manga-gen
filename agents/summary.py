from agents.agent import Agent
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

# The manga summary generator LLM
SUMMARY_LLM = ChatOllama(model="summary")

# Response schema for think action
class ThinkResponseSchema(BaseModel):
    action: str = Field(description="The action to take on the summary")
    reason: str = Field(description="Your detailed reasoning for deciding to take that action")

# Prompt template for think function
THINK_PROMPT_TEMPLATE = PromptTemplate(
    template="""User prompt: {prompt}

    {format_instructions}

    Choose your next action to take in developing your manga summary and explain why you chose that action.

    Review your manga summary, if it is sufficient then choose as your action: 'done'
    
    So far, your summary is: {summary}""",
    input_variables=["prompt", "summary"],
    partial_variables={"format_instructions": ""},
)

# Response schema for act function
class ActResponseSchema(BaseModel):
    thought: str = Field(description="Your internal monologue while you update the manga summary")
    summary: str = Field(description="Your updated manga summary")

# Prompt template for act function
ACT_PROMPT_TEMPLATE = PromptTemplate(
    template="""User prompt: {prompt}

    {format_instructions}

    Now take action to develop your manga summary.
    
    You should iteratively add to your summary and try not to remove details from it as you develop it.

    You chose to take action: {action}

    Your reasoning for doing so was: {reason}

    So far, your summary is: {summary}""",
    input_variables=["prompt", "action", "reason", "summary"],
    partial_variables={"format_instructions": ""},
)

class MangaSummaryAgent(Agent):
    def __init__(self):
        """
        Constructor for the MangaSummaryAgent class

        Returns:
            MangaSummaryAgent: The agent for developing manga summaries
        """
        super().__init__(
            "MangaSummaryAgent",
            SUMMARY_LLM,
            ThinkResponseSchema,
            ActResponseSchema
        )

    def run(self, prompt: str, iterations: int=3):
        summary = ""
        for _ in range(iterations):
            thought = self.think(
                THINK_PROMPT_TEMPLATE.format(
                    prompt=prompt,
                    summary=summary
                )
            )
            if thought["action"] == "done":
                break
            action = self.act(
                ACT_PROMPT_TEMPLATE.format(
                    prompt=prompt,
                    action=thought["action"],
                    reason=thought["reason"],
                    summary=summary
                )
            )
            summary = action["summary"]
        return summary
