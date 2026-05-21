# src/agents.py
import autogen

from config.agent_prompts import (
    CONTEXT_SHAPER_PROMPT,
    TISER_SYNTHESIZER_PROMPT,
    REFLECTION_CRITIC_PROMPT,
    VALIDATOR_PROMPT
)

def build_llm_config(config_list, system_instruction=None):
    config = {
        "config_list": config_list,
        "temperature": 0.2,
        "timeout": 120,
        "cache_seed": 42,
    }
    
    # Injecting instructions cleanly through headers for Gemini's OpenAI compatibility layer
    if system_instruction:
        config["default_headers"] = {
            "X-Goog-System-Instruction": system_instruction
        }
    return config

def create_tiser_agents(config_list):
    # --------------------------------------------------
    # Context Shaper
    # --------------------------------------------------
    context_shaper = autogen.AssistantAgent(
        name="Context_Shaper",
        llm_config=build_llm_config(config_list, CONTEXT_SHAPER_PROMPT),
        system_message="", # Clear this out so AutoGen doesn't send a legacy system role message
    )

    # --------------------------------------------------
    # Main Reasoning Synthesizer
    # --------------------------------------------------
    synth_config = build_llm_config(config_list, TISER_SYNTHESIZER_PROMPT)
    synth_config["temperature"] = 0.4
    
    synthesizer = autogen.AssistantAgent(
        name="TISER_Synthesizer",
        llm_config=synth_config,
        system_message="", 
    )

    # --------------------------------------------------
    # Reflection Critic
    # --------------------------------------------------
    reflection_critic = autogen.AssistantAgent(
        name="Reflection_Critic",
        llm_config=build_llm_config(config_list, REFLECTION_CRITIC_PROMPT),
        system_message="",
    )

    # --------------------------------------------------
    # Validator Agent
    # --------------------------------------------------
    validator = autogen.AssistantAgent(
        name="Validator_Agent",
        llm_config=build_llm_config(config_list, VALIDATOR_PROMPT),
        system_message="",
    )

    # --------------------------------------------------
    # Orchestration Controller
    # --------------------------------------------------
    orchestration_proxy = autogen.UserProxyAgent(
        name="Pipeline_Controller",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=1,
        code_execution_config=False,
        is_termination_msg=lambda msg: "PIPELINE_COMPLETE" in (msg.get("content") or "")
    )

    return {
        "controller": orchestration_proxy,
        "context_shaper": context_shaper,
        "synthesizer": synthesizer,
        "reflection_critic": reflection_critic,
        "validator": validator,
    }