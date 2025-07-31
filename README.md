# Manga Gen

Uses LangChain, Ollama, and Stable Diffusion to automatically generate manga.  This is still a work-in-progress.

> [!WARNING]
> All manga generated using this tool, if published, MUST disclose that it was AI generated.
> 
> More generally, please keep in mind that AI does not bring with it the artistic process, which is what makes art enjoyable and relatable (and, well... art).  Note that artwork produced by this tool is shallow in nature, albeit fun to explore and play with.

## Usage

After installing the required dependencies, run
```sh
python3 main.py
```

from the root of this repository.  The model output will be stored in `./out/<timestamp>`, including all summaries, character designs, and imagery.

### Dependencies

Install the required dependencies
1. Install [GNU Make](https://www.gnu.org/software/make/), [`python`](https://www.python.org/), [`pip`](https://pypi.org/project/pip/), and [`ollama`](https://ollama.com/)
2. Run the `make dependencies` recipe
```
make dependencies
```
