from agents.agent import Agent
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

# The manga character generator LLM
CHARACTER_LLM = ChatOllama(model="character")

# Response schema for think action
class ThinkResponseSchema(BaseModel):
    action: str = Field(description="The action to take on the character designs")
    reason: str = Field(description="Your detailed reasoning for deciding to take that action")

# Prompt template for think function
THINK_PROMPT_TEMPLATE = PromptTemplate(
    template="""Prompt: Given this manga summary, develop extremely detailed character descriptions for it: {summary}

    {format_instructions}

    Choose your next action to take in developing your manga character designs and explain why you chose that action.

    If you are done developing your manga character designs, then as your action you may choose 'done'
    
    So far, your character designs are as follows: {characters}""",
    input_variables=["prompt", "summary", "characters"],
    partial_variables={"format_instructions": ""},
)

# Response schema for act function
class ActResponseSchema(BaseModel):
    thought: str = Field(description="Your thoughts while taking the action")
    characters: str = Field(description="The updated manga character designs")

# Prompt template for act function
ACT_PROMPT_TEMPLATE = PromptTemplate(
    template="""Prompt: Given this manga summary, develop extremely detailed character descriptions for it: {summary}

    {format_instructions}

    Now take action to develop your character designs.
    
    You should iteratively revise your character designs and try not to remove details from them as you develop them.

    You chose to take action: {action}

    Your reasoning for doing so was: {reason}

    So far, your character designs are: {characters}""",
    input_variables=["prompt", "action", "reason", "summary"],
    partial_variables={"format_instructions": ""},
)

class MangaCharacterAgent(Agent):
    def __init__(self):
        """
        Constructor for the MangaCharacterAgent class

        Returns:
            MangaCharacterAgent: The agent for developing manga character designs
        """
        super().__init__(
            "MangaCharacterAgent",
            CHARACTER_LLM,
            ThinkResponseSchema,
            ActResponseSchema
        )

    def run(self, summary: str, iterations: int=3):
        characters = ""
        for _ in range(iterations):
            thought = self.think(
                THINK_PROMPT_TEMPLATE.format(
                    summary=summary,
                    characters=characters
                )
            )
            if thought["action"] == "done":
                break
            action = self.act(
                ACT_PROMPT_TEMPLATE.format(
                    summary=summary,
                    action=thought["action"],
                    reason=thought["reason"],
                    characters=characters
                )
            )
            characters = action["characters"]
        return characters
