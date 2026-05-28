# REPRODUCE.md - TCM-SCI-Patent-Forge

## Prerequisites

- **Python**: 3.8+
- **OS**: Linux / macOS / Windows
- **GPU**: Not required
- **API Key**: DeepSeek API key required

## Install

```bash
cd TCM-SCI-Patent-Forge
pip install -r requirements.txt
```

Dependencies: httpx (core), tqdm (optional)

## Run

```bash
# Single generation
python paper_generator.py --api_key YOUR_DEEPSEEK_KEY
python patent_generator.py --api_key YOUR_DEEPSEEK_KEY

# Batch generation
python generate_batch.py --api_key YOUR_DEEPSEEK_KEY
```

## Expected Outputs

- SCI paper drafts for 100 TCM AI research directions
- Patent technical disclosure documents
- Output format: Markdown documents

## Known Issues

- **HARDCODED PLACEHOLDER**: `paper_generator.py` and `patent_generator.py` contain `DEEPSEEK_API_KEY = "sk-..."` placeholder
  - Must pass real key via `--api_key` argument
  - Or set environment variable
- Requires active DeepSeek API access (paid)
- No tests directory
- No hardcoded absolute paths in core code
