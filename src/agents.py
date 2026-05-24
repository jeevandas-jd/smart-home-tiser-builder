"""
src/agents.py  —  fixed version

Fixes vs original:
  B2 - Removed the Gemini X-Goog-System-Instruction header approach entirely.
       AutoGen's AssistantAgent already supports system_message properly when
       the config_list uses a standard OpenAI-compatible provider.
       System prompts now live in system_message (not a custom header) so
       there is exactly ONE system channel, no conflict.

  B7 - Reflection Critic and Validator agents are now returned AND documented
       so main.py can wire them in. (Orchestration wiring is in main.py.)
"""

# src/agents.py

import autogen

from config.agent_prompts import (
    CONTEXT_SHAPER_PROMPT,
    TISER_SYNTHESIZER_PROMPT,
    REFLECTION_CRITIC_PROMPT,
    VALIDATOR_PROMPT,
)


def build_llm_config(config_list, temperature=0.2):
    """
    Build a clean LLM config dict for AutoGen.

    FIX B2: removed the X-Goog-System-Instruction custom header.
    System prompts are passed via system_message= on AssistantAgent,
    which AutoGen serialises as the 'system' role message — the single
    correct channel for every OpenAI-compatible provider.
    """
    return {
        "config_list": config_list,
        "temperature": temperature,
        "timeout": 120,
        "cache_seed": None,   # disable caching during dataset generation
    }


def create_tiser_agents(config_list):
    # --------------------------------------------------
    # Context Shaper
    # --------------------------------------------------
    context_shaper = autogen.AssistantAgent(
        name="Context_Shaper",
        system_message=CONTEXT_SHAPER_PROMPT,   # FIX B2: use system_message
        llm_config=build_llm_config(config_list, temperature=0.2),
    )

    # --------------------------------------------------
    # TISER Synthesizer
    # --------------------------------------------------
    synthesizer = autogen.AssistantAgent(
        name="TISER_Synthesizer",
        system_message=TISER_SYNTHESIZER_PROMPT,
        llm_config=build_llm_config(config_list, temperature=0.4),
    )

    # --------------------------------------------------
    # Reflection Critic
    # --------------------------------------------------
    reflection_critic = autogen.AssistantAgent(
        name="Reflection_Critic",
        system_message=REFLECTION_CRITIC_PROMPT,
        llm_config=build_llm_config(config_list, temperature=0.2),
    )

    # --------------------------------------------------
    # Validator Agent  (LLM-based; deterministic check is in tools.py)
    # --------------------------------------------------
    validator = autogen.AssistantAgent(
        name="Validator_Agent",
        system_message=VALIDATOR_PROMPT,
        llm_config=build_llm_config(config_list, temperature=0.0),
    )

    # --------------------------------------------------
    # Orchestration Controller
    # --------------------------------------------------
    orchestration_proxy = autogen.UserProxyAgent(
        name="Pipeline_Controller",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=1,
        code_execution_config=False,
        is_termination_msg=lambda msg: "PIPELINE_COMPLETE" in (
            msg.get("content") or ""
        ),
    )

    return {
        "controller":       orchestration_proxy,
        "context_shaper":   context_shaper,
        "synthesizer":      synthesizer,
        "reflection_critic": reflection_critic,  # FIX B7: now surfaced
        "validator":        validator,            # FIX B7: now surfaced
    }
