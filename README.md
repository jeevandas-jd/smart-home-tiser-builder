Here is a comprehensive, production-grade `README.md` tailored specifically for your project. It outlines your multi-agent architecture, details the project file structure, explains the resume/fallback mechanics, and provides setup instructions.

You can copy and paste this directly into your newly cloned repository.

---

```markdown
# Smart Home TISER Dataset Builder 🤖🏡

An agentic data engineering pipeline designed to convert raw, structured smart home simulator state logs (JSON) into a high-quality dataset featuring multi-stage temporal and structural reasoning traces. [cite_start]This pipeline leverages **Microsoft AutoGen** to implement the dataset construction mechanics introduced in the **TISER** (*Timeline Self-Reflection*) framework[cite: 12].

[cite_start]The resulting dataset is optimized for Supervised Fine-Tuning (SFT) [cite: 137] [cite_start]to equip smaller open-source language models with test-time compute scaling and autonomous self-reflection behaviors[cite: 13, 28].

---

## 🏗️ Core Architecture & TISER Alignment

[cite_start]Following the paper's data generation design, this repository uses **Ground-Truth Forced Reasoning** to reverse-engineer flawless internal reasoning trails[cite: 130]. [cite_start]It processes your smart home configurations through a multi-agent assembly line that populates the four critical TISER stages[cite: 43]:


```

```
  Raw Simulator JSON
          │
          ▼

```

┌──────────────────────┐
│ 1. Context Shaper    │ ───► Injects user query & maps target room attributes
└──────────┬───────────┘
▼
┌──────────────────────┐
│ 2. TISER Synthesizer │ ───► Generates  traces & explicit structure 
└──────────┬───────────┘
▼
┌──────────────────────┐
│ 3. Validator (QC)    │ ───► Parses tags, runs regex & matches against ground-truth 
└──────────┬───────────┘
├──────────────────────────────┐
▼ (Passes Consistency Match)    ▼ (Fails / Hallucinates)
┌──────────────────────┐       ┌──────────────────────┐
│ 4. Archivist Agent   │       │ Auto-Retry Loop      │ ───► Maximum 3 attempts
│ Saves to /tiser_ready│       │ (Rotates Backup Keys)│      before marking fail
└──────────────────────┘       └──────────────────────┘

```

### Supported Token Mappings
[cite_start]Output files are systematically compiled using the strict XML format specified by the TISER framework[cite: 139]:
* [cite_start]`<reasoning>`: Step-by-step structural breakdown of room status lookups[cite: 139].
* [cite_start]`<timeline>`: An ordered list of isolated status truths (e.g., room illuminance values) acting as the structured snapshot[cite: 139].
* [cite_start]`<reflection>`: A self-verification run matching extracted facts against natural constraints[cite: 139].
* [cite_start]`<answer>`: The final, concise query resolution matching your simulator's true execution target[cite: 139].

---

## 📂 Repository Blueprint

```text
smart-home-tiser-builder/
│
├── config/
│   ├── oai_config.json          # Multi-account API configuration & key matrices
[cite_start]│   └── agent_prompts.py         # System prompt blueprints outlining strict formatting rules [cite: 139]
│
├── data/
│   ├── raw_episodes/            # Input folder containing your 600 raw JSON simulator outputs
│   └── tiser_ready/             # Output folder saving compiled, fully verified samples (.json / .jsonl)
│
├── src/
│   ├── __init__.py
│   ├── ledger.py                # SQLite database coordinator handling processing checkpoints
│   ├── tools.py                 # Core validation logic, regex checkers, and parsing filters
│   └── agents.py                # Microsoft AutoGen agent setups and conversation configurations
│
├── state.db                     # Dynamic tracking ledger generated at runtime
├── main.py                      # Main entry script executing the batch control loops
├── requirements.txt             # Project Python dependencies
└── README.md                    # Project documentation

```

---

## ⚡ Key Engineering Features

* **Resilient Key Rotation & Failover:** Powered by AutoGen's `config_list`. If your primary API engine flags a rate-limit error ($429$) or server fault ($5\times\times$), the synthesizer dynamically drops back to alternating API accounts or switches down to a locally deployed instance (e.g., Qwen2.5 / Mistral via Ollama/vLLM).


* **Transactional Progress Checkpointing:** All progress is backed by a local SQLite instance (`state.db`). If the program loses network access midway through processing your 600 files, restarting `main.py` seamlessly picks up from the exact file ID it was executing at execution death.
* 
**TISER Filtering (Algorithm 2):** Any generation where the synthetic thinking path outputs structural configurations or metrics that drift from the exact ground-truth values embedded in the file's `eval.goals` block is instantly rejected and scheduled for regeneration.



---

## 🚀 Getting Started

### 1. Prerequisites & Installation

Ensure you have Python 3.10+ installed on your system.

```bash
# Clone the repository (you are already inside the folder)
git status

# Create and activate a clean virtual environment
python3 -m venv venv
source venv/bin/activate

# Install required dependencies
pip install -r requirements.txt

```

### 2. Configure API Credentials

Set up your model keys inside `config/oai_config.json`. You can list multiple accounts, platforms, or local endpoints for fallback security:

```json
[
    {
        "model": "gpt-4o",
        "api_key": "sk-proj-PRIMARY_OPENAI_KEY"
    },
    {
        "model": "deepseek-chat",
        "api_key": "sk-BACKUP_DEEPSEEK_KEY"
    },
    {
        "model": "qwen2.5-7b",
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama"
    }
]

```

### 3. Populating Inputs

Dump your 600 raw simulation files into the raw processing path:

```bash
mkdir -p data/raw_episodes data/tiser_ready
# Move your simulator files here, e.g., episode_1.json, episode_2.json ...

```

### 4. Running the Pipeline

Execute the primary orchestrator loop. The script automatically reads raw objects, instantiates your AutoGen workforce, verifies tag alignment, and commits finalized profiles to storage:

```bash
python3 main.py

```

---

## 📊 Dataset Properties & SFT Objectives

The dataset created by this framework maps structural context data directly into long-context reasoning tracks. During fine-tuning, the training targets guide models to learn structural logic explicitly:

```text
Prompt -> Context: Smart Home Device/Room JSON Attributes + User Query
Target -> <reasoning> Structural analysis </reasoning> <answer> Clean output </answer>

```

---

## 📝 References

* 
**TISER Framework:** *Learning to Reason Over Time: Timeline Self-Reflection for Improved Temporal Reasoning in Language Models* (Bazaga et al., arXiv:2504.05258).



```

```
