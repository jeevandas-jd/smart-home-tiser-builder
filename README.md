# Smart Home TISER Dataset Builder

A research-oriented, multi-agent dataset construction pipeline for generating high-quality temporal reasoning traces from structured smart-home simulation environments.

This project implements a scalable agentic data engineering workflow inspired by the **TISER (Timeline Self-Reflection)** framework for training language models to perform structured temporal reasoning, reflection, and verification over environment states.

The pipeline converts raw simulator episode logs into supervised fine-tuning (SFT) samples containing:

* Explicit reasoning traces
* Structured temporal representations
* Self-reflection and verification steps
* Ground-truth aligned answers

The resulting dataset is designed for training and evaluating open-source language models on long-context reasoning, temporal consistency, and self-corrective inference.

---

# Research Motivation

Large Language Models often struggle with:

* Multi-step temporal reasoning
* State tracking across evolving environments
* Reflection and self-correction
* Maintaining consistency over structured contexts

This repository explores an agentic data-generation approach where multiple collaborating LLM agents synthesize reasoning trajectories directly from simulator ground truth.

Unlike conventional synthetic dataset generation pipelines, this framework performs:

* Structured reasoning synthesis
* Reflection generation
* Automatic validation against ground truth
* Regeneration upon hallucination or inconsistency
* Persistent checkpoint recovery
* Multi-provider API failover

The generated outputs are intended for:

* Supervised Fine-Tuning (SFT)
* Temporal reasoning benchmarks
* Chain-of-thought research
* Reflection-based alignment studies
* Test-time scaling experiments
* Autonomous reasoning system evaluation

---

# System Architecture

The repository follows a multi-agent orchestration architecture inspired by the TISER data construction methodology.

```text
Raw Simulator Episodes (JSON)
            │
            ▼
┌──────────────────────────────┐
│ 1. Context Shaper Agent      │
│ • Parses simulator states    │
│ • Extracts environment facts │
│ • Maps query targets         │
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│ 2. TISER Synthesizer Agent   │
│ • Generates reasoning traces │
│ • Builds timeline structure  │
│ • Produces reflection steps  │
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│ 3. Validator / QC Agent      │
│ • Parses XML tags            │
│ • Verifies consistency       │
│ • Matches ground truth       │
└──────────────┬───────────────┘
        ┌──────┴──────┐
        ▼             ▼
   Validation Pass    Validation Fail
        ▼             ▼
┌────────────────┐   ┌─────────────────────┐
│ Archivist      │   │ Auto Retry Engine   │
│ Saves samples  │   │ Regenerates traces  │
└────────────────┘   └─────────────────────┘
```

---

# TISER Output Structure

Each generated training sample follows a strict XML-style schema.

```xml
<reasoning>
Step-by-step reasoning trace
</reasoning>

<timeline>
Structured temporal facts
</timeline>

<reflection>
Self-verification and correction
</reflection>

<answer>
Final grounded answer
</answer>
```

## Component Descriptions

### `<reasoning>`

Contains the synthesized chain-of-thought explaining how the agent arrived at the answer.

### `<timeline>`

Stores structured temporal or environmental state facts extracted from the simulator.

### `<reflection>`

Performs self-consistency checks against generated reasoning and known constraints.

### `<answer>`

Contains the concise, ground-truth-aligned final response.

---

# Repository Structure

```text
smart-home-tiser-builder/
│
├── config/
│   ├── oai_config.json
│   └── agent_prompts.py
│
├── data/
│   ├── raw_episodes/
│   └── tiser_ready/
│
├── src/
│   ├── __init__.py
│   ├── agents.py
│   ├── tools.py
│   ├── ledger.py
│   └── validators.py
│
├── state.db
├── main.py
├── requirements.txt
└── README.md
```

---

# Core Engineering Features

## Multi-Agent Dataset Construction

Uses Microsoft AutoGen to coordinate specialized agents for:

* Context extraction
* Reasoning synthesis
* Reflection generation
* Verification and filtering
* Persistent archival

---

## Ground-Truth Forced Reasoning

Generated reasoning traces are validated against the simulator’s exact ground-truth state.

Any mismatch between generated outputs and simulator facts triggers:

* Automatic rejection
* Retry scheduling
* Regeneration loops

This significantly reduces hallucinated reasoning trajectories.

---

## Persistent Checkpoint Recovery

The pipeline uses SQLite-based state tracking (`state.db`) to maintain execution progress.

If execution is interrupted due to:

* Network failure
* API rate limits
* System crashes
* GPU interruptions

processing resumes automatically from the last successful episode.

---

## API Failover and Key Rotation

Supports resilient multi-provider inference routing.

Features include:

* OpenAI failover
* DeepSeek fallback
* Local model inference
* Automatic retry policies
* Rate-limit handling
* Dynamic provider switching

Compatible with:

* OpenAI APIs
* Ollama
* vLLM
* OpenRouter
* Local inference endpoints

---

## Automatic Quality Control

Validation modules perform:

* XML tag parsing
* Structural verification
* Regex validation
* Consistency scoring
* Ground-truth alignment checks
* Reflection completeness verification

---

# Installation

## Prerequisites

* Python 3.10+
* Linux/macOS/WSL recommended
* GPU optional but recommended for local inference

---

## Clone the Repository

```bash
git clone <repository-url>
cd smart-home-tiser-builder
```

---

## Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Configuration

## API Configuration

Edit:

```text
config/oai_config.json
```

Example configuration:

```json
[
    {
        "model": "gpt-4o",
        "api_key": "YOUR_OPENAI_KEY"
    },
    {
        "model": "deepseek-chat",
        "api_key": "YOUR_DEEPSEEK_KEY"
    },
    {
        "model": "qwen2.5-7b",
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama"
    }
]
```

---

# Input Data Format

Place raw simulator episode files inside:

```text
data/raw_episodes/
```

Each episode should contain:

* Environment states
* Device attributes
* Room metadata
* Temporal transitions
* Evaluation targets
* Ground-truth goals

Example:

```json
{
  "episode_id": 1,
  "rooms": [...],
  "devices": [...],
  "eval": {
    "goal": "Find the room with active lights"
  }
}
```

---

# Running the Pipeline

Execute the main orchestration loop:

```bash
python3 main.py
```

The system will:

1. Load raw simulator episodes
2. Initialize AutoGen agents
3. Generate reasoning traces
4. Validate outputs
5. Retry failed generations
6. Save verified samples

Generated outputs are stored in:

```text
data/tiser_ready/
```

---

# Dataset Objectives

The generated dataset is intended to improve:

* Temporal reasoning
* Long-context understanding
* Structured planning
* Reflection-based correction
* Simulator state tracking
* Multi-hop inference

Example training mapping:

```text
Input  -> Simulator State + User Query
Output -> Reasoning + Timeline + Reflection + Answer
```

---

# Research Applications

Potential research directions enabled by this framework:

* Temporal reasoning benchmarks
* Reflective language models
* Agentic workflow research
* Test-time compute scaling
* Structured reasoning supervision
* Hallucination reduction
* Long-context adaptation
* Environment-grounded reasoning systems

---

# Recommended Model Targets

The generated datasets are suitable for fine-tuning:

* Qwen2.5
* Llama 3
* Mistral
* Gemma
* Phi
* DeepSeek models

Using:

* LoRA
* QLoRA
* Full SFT
* DPO-style reflection alignment

---

# Future Work

Planned improvements include:

* Multi-modal environment support
* Real-time simulator streaming
* Distributed generation workers
* Reinforcement learning integration
* Automatic curriculum generation
* Reflection quality scoring
* Benchmark suite integration

---

# Citation

If you use this repository in academic work, please cite both the repository and the original TISER paper.

```bibtex
@article{tiser2025,
  title={Learning to Reason Over Time: Timeline Self-Reflection for Improved Temporal Reasoning in Language Models},
  author={Bazaga et al.},
  year={2025},
  journal={arXiv preprint arXiv:2504.05258}
}
```

---

# Acknowledgements

This project builds upon ideas from:

* Microsoft AutoGen
* Temporal reasoning research
* Reflection-based alignment methods
* Chain-of-thought supervision literature
* Structured synthetic dataset generation systems

---

# License

Specify your preferred license here.

Example:

```text
MIT License
```

---

# Contact

Maintained by Jeevandas M S.

For collaborations, research discussions, or contributions, feel free to open an issue or submit a pull request.
