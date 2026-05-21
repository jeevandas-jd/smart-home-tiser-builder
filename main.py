# main.py

"""
Main orchestration entrypoint for the Smart Home TISER Dataset Builder.

This pipeline converts structured smart-home simulator episodes into
high-quality TISER reasoning traces using a multi-agent synthesis workflow.

Pipeline Stages:
    1. Episode Acquisition
    2. Context Shaping
    3. TISER Reasoning Synthesis
    4. Deterministic Validation
    5. Archival + Ledger Tracking

The execution engine is fully resumable through the SQLite execution ledger.
If interrupted, restarting the pipeline resumes from the last unresolved episode.
"""

import json
import os
import time

import autogen

from src.agents import create_tiser_agents
from src.ledger import ExecutionLedger
from src.tools import verify_tiser_output


# -------------------------------------------------------------------
# Path Configuration
# -------------------------------------------------------------------

RAW_FOLDER = "data/raw_episodes"
OUTPUT_FOLDER = "data/tiser_ready"

CONFIG_PATH = "config/oai_config.json"


# -------------------------------------------------------------------
# Configuration Loader
# -------------------------------------------------------------------

def load_config():

    """
    Load provider configuration list used by AutoGen.

    Supports:
        - OpenAI
        - DeepSeek
        - Anthropic
        - Local vLLM/Ollama endpoints
    """

    if not os.path.exists(CONFIG_PATH):

        raise FileNotFoundError(
            f"Missing configuration file: {CONFIG_PATH}"
        )

    return autogen.config_list_from_json(
        env_or_file=CONFIG_PATH
    )


# -------------------------------------------------------------------
# Episode Loader
# -------------------------------------------------------------------

def load_episode(raw_file_path):

    """
    Load and parse a raw simulator episode.
    """

    with open(raw_file_path, "r") as f:
        return json.load(f)


# -------------------------------------------------------------------
# Output Writer
# -------------------------------------------------------------------

def save_final_sample(
    output_path,
    sample
):

    """
    Persist validated TISER samples to disk.
    """

    with open(output_path, "w") as out_f:

        json.dump(
            sample,
            out_f,
            indent=2
        )


# -------------------------------------------------------------------
# Main Orchestration Loop
# -------------------------------------------------------------------

def main():

    print(
        "\nInitializing Smart Home TISER Conversion Pipeline\n"
    )

    # --------------------------------------------------------------
    # Ensure Output Directory Exists
    # --------------------------------------------------------------

    os.makedirs(
        OUTPUT_FOLDER,
        exist_ok=True
    )

    # --------------------------------------------------------------
    # Load LLM Provider Mesh
    # --------------------------------------------------------------

    config_list = load_config()
    print(
        f"LLM Provider configuration loaded successfully.\n{config_list}"
    )
    # --------------------------------------------------------------
    # Initialize Execution Ledger
    # --------------------------------------------------------------

    ledger = ExecutionLedger()

    ledger.register_episodes(
        RAW_FOLDER
    )

    # --------------------------------------------------------------
    # Initialize Multi-Agent Workforce
    # --------------------------------------------------------------

    agents = create_tiser_agents(
        config_list
    )

    controller = agents["controller"]

    context_shaper = agents["context_shaper"]

    synthesizer = agents["synthesizer"]

    # --------------------------------------------------------------
    # Main Processing Loop
    # --------------------------------------------------------------

    while True:

        # ----------------------------------------------------------
        # Acquire Next Available Episode
        # ----------------------------------------------------------

        episode_id = ledger.acquire_next_episode()

        if not episode_id:

            print(
                "No remaining eligible episodes detected."
            )

            break

        print(f"\nProcessing Episode: {episode_id}")

        raw_file_path = os.path.join(
            RAW_FOLDER,
            f"{episode_id}.json"
        )

        # ----------------------------------------------------------
        # Load Episode
        # ----------------------------------------------------------

        try:

            episode_data = load_episode(
                raw_file_path
            )

        except Exception as e:

            print(
                f"Episode loading failure: {episode_id}"
            )

            ledger.mark_generation_failure(
                episode_id,
                str(e)
            )

            continue

        # ----------------------------------------------------------
        # Extract Core Metadata
        # ----------------------------------------------------------

        query = episode_data.get(
            "query",
            ""
        )

        eval_goals = (
            episode_data
            .get("eval", {})
            .get("goals", [])
        )

        # ----------------------------------------------------------
        # Build Context Shaper Input
        # ----------------------------------------------------------

        context_shaper_input = f"""You are processing a data translation task. Clean the following telemetry dump.

[RAW SIMULATOR DATA BLOCK]
```json
{json.dumps(episode_data)}"""

        # ----------------------------------------------------------
        # Context Shaping Stage
        # ----------------------------------------------------------

        try:

            controller.initiate_chat(
                context_shaper,
                message=context_shaper_input,
                clear_history=True,
                silent=False
            )

            filtered_context = (
                controller
                .last_message()["content"]
            )

        except Exception as e:

            print(
                f"Context shaping failure: {episode_id}"
            )

            ledger.mark_generation_failure(
                episode_id,
                f"Context Shaper Failure: {str(e)}"
            )

            continue

        # ----------------------------------------------------------
        # Build Synthesis Prompt
        # ----------------------------------------------------------

        synthesis_prompt = f"""
USER TASK:
{query}

FILTERED ENVIRONMENT CONTEXT:
{filtered_context}

VERIFIED TARGET OUTCOME:
{json.dumps(eval_goals, indent=2)}

Generate the TISER reasoning trace.
"""

        # ----------------------------------------------------------
        # TISER Synthesis Stage
        # ----------------------------------------------------------

        generation_start = time.time()

        try:

            controller.initiate_chat(
                synthesizer,
                message=synthesis_prompt,
                clear_history=True,
                silent=True
            )

            tiser_output = (
                controller
                .last_message()["content"]
            )

        except Exception as e:

            print(
                f"Synthesis failure: {episode_id}"
            )

            ledger.mark_generation_failure(
                episode_id,
                f"Synthesizer Failure: {str(e)}"
            )

            continue

        generation_latency = (
            time.time() - generation_start
        )

        # ----------------------------------------------------------
        # Build Valid Entity Set
        # ----------------------------------------------------------

        valid_entities = set()

        for room in episode_data.get("rooms", []):

            if "name" in room:

                valid_entities.add(
                    room["name"].lower()
                )

        for device in episode_data.get("devices", []):

            if "name" in device:

                valid_entities.add(
                    device["name"].lower()
                )

        # ----------------------------------------------------------
        # Deterministic Validation Stage
        # ----------------------------------------------------------

        passed, validation_msg = verify_tiser_output(
            llm_output=tiser_output,
            ground_truth_goals=eval_goals,
            valid_entities=valid_entities
        )

        # ----------------------------------------------------------
        # Validation Failure Handling
        # ----------------------------------------------------------

        if not passed:

            print(
                f"Validation failure: {episode_id}"
            )

            print(
                f"Reason: {validation_msg}"
            )

            ledger.mark_validation_failure(
                episode_id,
                validation_msg
            )

            continue

        # ----------------------------------------------------------
        # Construct Final Training Record
        # ----------------------------------------------------------

        final_sample = {
            "episode_id": episode_id,
            "query": query,
            "ground_truth_goals": eval_goals,
            "tiser_trace": tiser_output
        }

        output_path = os.path.join(
            OUTPUT_FOLDER,
            f"{episode_id}_tiser.json"
        )

        # ----------------------------------------------------------
        # Persist Final Sample
        # ----------------------------------------------------------

        try:

            save_final_sample(
                output_path,
                final_sample
            )

        except Exception as e:

            print(
                f"Disk write failure: {episode_id}"
            )

            ledger.mark_generation_failure(
                episode_id,
                f"Output Write Failure: {str(e)}"
            )

            continue

        # ----------------------------------------------------------
        # Mark Episode as Completed
        # ----------------------------------------------------------

        ledger.mark_completed(
            episode_id=episode_id,
            output_path=output_path,
            generation_latency=generation_latency
        )

        print(
            f"Episode completed successfully: "
            f"{episode_id}"
        )


# -------------------------------------------------------------------
# Entrypoint
# -------------------------------------------------------------------

if __name__ == "__main__":

    main()
