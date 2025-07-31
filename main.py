import os
from agents.character import MangaCharacterAgent
from agents.summary import MangaSummaryAgent
from datetime import datetime
from diffusers import DiffusionPipeline

TIMESTAMP = datetime.now().strftime("%Y%m%d%H%M%S")
os.makedirs(f"./out/{TIMESTAMP}", exist_ok=True)

# Generate an manga summary
summary_agent = MangaSummaryAgent()
summary = summary_agent.run("Write me an manga about a car crash", iterations=1)
with open(f"./out/{TIMESTAMP}/summary.txt", 'w') as summary_file:
    summary_file.write(summary)

# Develop concept art for the manga
pipe = DiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5"
)
image = pipe(
    f"Draw concept art for this manga in the style of a manga artist\n\n{summary}"
).images[0]
image.save(f"./out/{TIMESTAMP}/concept.png")

# Generate manga character descriptions
character_agent = MangaCharacterAgent()
characters = character_agent.run(summary, iterations=1)
with open(f"./out/{TIMESTAMP}/characters.txt", 'w') as characters_file:
    characters_file.write(characters)
