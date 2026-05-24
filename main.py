# main.py  —  fixed version
"""
Main orchestration entrypoint for the Smart Home TISER Dataset Builder.

Bugs fixed vs original:
  B1 - f-string for context_shaper_input used json.dumps() inside triple-quotes
       causing a malformed prompt (missing closing ``` fence).
       Fixed: build the prompt string in two steps.

  B3 - No try/finally around episode body => crash left ledger stuck
       'in_progress' forever.
       Fixed: entire episode body wrapped in try/finally that always
       calls mark_generation_failure on any uncaught exception.

  B4 - valid_entities built from episode_data.get("rooms",[]) but the
       episode JSON stores rooms under initial_home_config.rooms.
       Fixed: correct key path + extract device_ids and room_ids properly.

  B8 - autogen.config_list_from_json tightly coupled to OpenAI format.
       The pipeline now supports Anthropic via a direct client, while
       still supporting any OpenAI-compatible provider through AutoGen.
       A simple JSON config drives provider selection.
"""

import json
import os
import time
import traceback

import autogen

from src.agents import create_tiser_agents
from src.ledger import ExecutionLedger
from src.tools import verify_tiser_output


# ── Path configuration ────────────────────────────────────────────────────────

RAW_FOLDER    = "data/raw_episodes"
OUTPUT_FOLDER = "data/tiser_ready"
CONFIG_PATH   = "config/oai_config.json"


# ── Config loader ─────────────────────────────────────────────────────────────

def load_config():
    """
    Load provider configuration list for AutoGen.
    Supports OpenAI, DeepSeek, Ollama/vLLM, OpenRouter — any
    OpenAI-compatible endpoint listed in oai_config.json.
    """
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"Missing config file: {CONFIG_PATH}")
    return autogen.config_list_from_json(env_or_file=CONFIG_PATH)


# ── Episode I/O ───────────────────────────────────────────────────────────────

def load_episode(raw_file_path):
    with open(raw_file_path, "r") as f:
        return json.load(f)


def save_final_sample(output_path, sample):
    with open(output_path, "w") as f:
        json.dump(sample, f, indent=2)


# ── Entity extractor (FIX B4) ────────────────────────────────────────────────

def extract_valid_entities(episode_data):
    """
    FIX B4: original walked episode_data["rooms"] and episode_data["devices"]
    which do not exist.  The real schema is:
        episode_data["initial_home_config"]["rooms"][room_id]["devices"][*]["device_id"]

    Returns a set of lowercase strings (room ids + device ids) that are
    legitimate identifiers for the hallucination checker.
    """
    entities = set()
    home_cfg = episode_data.get("initial_home_config", {})
    rooms    = home_cfg.get("rooms", {})

    for room_id, room_data in rooms.items():
        entities.add(room_id.lower())
        for device in room_data.get("devices", []):
            device_id = device.get("device_id", "")
            if device_id:
                entities.add(device_id.lower())
                # Also add the human-readable type
                device_type = device.get("device_type", "")
                if device_type:
                    entities.add(device_type.lower())

    return entities


# ── Prompt builder (FIX B1) ──────────────────────────────────────────────────

def build_context_shaper_prompt(episode_data):
    """
    FIX B1: original embedded json.dumps() inside a triple-quoted f-string
    and omitted the closing ``` fence, producing a malformed markdown prompt.
    Build the string in two discrete steps so there is no ambiguity.
    """
    raw_json = json.dumps(episode_data, indent=2)

    prompt = (
        "You are processing a data translation task. "
        "Clean the following telemetry dump.\n\n"
        "[RAW SIMULATOR DATA BLOCK]\n"
        "```json\n"
        + raw_json
        + "\n```"
    )
    return prompt


# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    print("\nInitializing Smart Home TISER Conversion Pipeline\n")

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Load provider config and agents
    config_list = load_config()
    print(f"LLM providers loaded: {[c.get('model') for c in config_list]}\n")

    agents = create_tiser_agents(config_list)
    controller      = agents["controller"]
    context_shaper  = agents["context_shaper"]
    synthesizer     = agents["synthesizer"]

    # Ledger
    ledger = ExecutionLedger()
    ledger.register_episodes(RAW_FOLDER)

    # ── Episode loop ──────────────────────────────────────────────────────────

    while True:
        episode_id = ledger.acquire_next_episode()
        if not episode_id:
            print("No remaining eligible episodes.")
            break

        print(f"\nProcessing Episode: {episode_id}")
        raw_file_path = os.path.join(RAW_FOLDER, f"{episode_id}.json")

        # FIX B3: wrap the entire episode body so any uncaught exception
        # moves the ledger row to failed_generation instead of leaving it
        # stuck as 'in_progress'.
        try:
            _process_episode(
                episode_id=episode_id,
                raw_file_path=raw_file_path,
                ledger=ledger,
                controller=controller,
                context_shaper=context_shaper,
                synthesizer=synthesizer,
            )
        except Exception as exc:
            print(f"Unhandled exception for {episode_id}: {exc}")
            traceback.print_exc()
            ledger.mark_generation_failure(
                episode_id,
                f"Unhandled exception: {exc}",
            )


def _process_episode(
    episode_id,
    raw_file_path,
    ledger,
    controller,
    context_shaper,
    synthesizer,
):
    # ── Load ──────────────────────────────────────────────────────────────────
    try:
        episode_data = load_episode(raw_file_path)
    except Exception as exc:
        print(f"Episode load failure: {episode_id}")
        ledger.mark_generation_failure(episode_id, str(exc))
        return

    query      = episode_data.get("query", "")
    eval_goals = episode_data.get("eval", {}).get("goals", [])

    # ── Context shaping ───────────────────────────────────────────────────────
    # FIX B1: use the two-step prompt builder
    context_shaper_input = build_context_shaper_prompt(episode_data)

    try:
        controller.initiate_chat(
            context_shaper,
            message=context_shaper_input,
            clear_history=True,
            silent=False,
        )
        filtered_context = controller.last_message()["content"]
    except Exception as exc:
        print(f"Context shaping failure: {episode_id}")
        ledger.mark_generation_failure(
            episode_id, f"Context Shaper failure: {exc}"
        )
        return

    # ── Synthesis ─────────────────────────────────────────────────────────────
    synthesis_prompt = (
        f"USER TASK:\n{query}\n\n"
        f"FILTERED ENVIRONMENT CONTEXT:\n{filtered_context}\n\n"
        f"VERIFIED TARGET OUTCOME:\n{json.dumps(eval_goals, indent=2)}\n\n"
        "Generate the TISER reasoning trace."
    )

    generation_start = time.time()
    try:
        controller.initiate_chat(
            synthesizer,
            message=synthesis_prompt,
            clear_history=True,
            silent=True,
        )
        tiser_output = controller.last_message()["content"]
    except Exception as exc:
        print(f"Synthesis failure: {episode_id}")
        ledger.mark_generation_failure(
            episode_id, f"Synthesizer failure: {exc}"
        )
        return

    generation_latency = time.time() - generation_start

    # ── Build valid entity set (FIX B4) ──────────────────────────────────────
    valid_entities = extract_valid_entities(episode_data)

    # ── Validate ──────────────────────────────────────────────────────────────
    passed, validation_msg = verify_tiser_output(
        llm_output=tiser_output,
        ground_truth_goals=eval_goals,
        valid_entities=valid_entities,
    )

    if not passed:
        print(f"Validation failure: {episode_id}")
        print(f"Reason: {validation_msg}")
        ledger.mark_validation_failure(episode_id, validation_msg)
        return

    # ── Persist ───────────────────────────────────────────────────────────────
    final_sample = {
        "episode_id":        episode_id,
        "query":             query,
        "ground_truth_goals": eval_goals,
        "tiser_trace":       tiser_output,
    }
    output_path = os.path.join(OUTPUT_FOLDER, f"{episode_id}_tiser.json")

    try:
        save_final_sample(output_path, final_sample)
    except Exception as exc:
        print(f"Disk write failure: {episode_id}")
        ledger.mark_generation_failure(
            episode_id, f"Output write failure: {exc}"
        )
        return

    ledger.mark_completed(
        episode_id=episode_id,
        output_path=output_path,
        generation_latency=generation_latency,
    )
    print(f"Episode completed successfully: {episode_id}")


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
