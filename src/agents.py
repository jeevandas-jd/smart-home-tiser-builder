import autogen

from config.agent_prompts import (
    CONTEXT_SHAPER_PROMPT,
    TISER_SYNTHESIZER_PROMPT,
    REFLECTION_CRITIC_PROMPT,
    VALIDATOR_PROMPT
)


def build_llm_config(config_list):

    return {
        "config_list": config_list,
        "temperature": 0.2,
        "timeout": 120,
        "cache_seed": 42,
    }


def create_tiser_agents(config_list):

    llm_config = build_llm_config(config_list)

    # --------------------------------------------------
    # Context Shaper
    # --------------------------------------------------

    context_shaper = autogen.AssistantAgent(
        name="Context_Shaper",
        llm_config=llm_config,
        system_message=CONTEXT_SHAPER_PROMPT,
    )

    # --------------------------------------------------
    # Main Reasoning Synthesizer
    # --------------------------------------------------

    synthesizer = autogen.AssistantAgent(
        name="TISER_Synthesizer",
        llm_config={
            **llm_config,
            "temperature": 0.4
        },
        system_message=TISER_SYNTHESIZER_PROMPT,
    )

    # --------------------------------------------------
    # Reflection Critic
    # --------------------------------------------------

    reflection_critic = autogen.AssistantAgent(
        name="Reflection_Critic",
        llm_config=llm_config,
        system_message=REFLECTION_CRITIC_PROMPT,
    )

    # --------------------------------------------------
    # Validator Agent
    # --------------------------------------------------

    validator = autogen.AssistantAgent(
        name="Validator_Agent",
        llm_config=llm_config,
        system_message=VALIDATOR_PROMPT,
    )

    # --------------------------------------------------
    # Orchestration Controller
    # --------------------------------------------------

    orchestration_proxy = autogen.UserProxyAgent(
        name="Pipeline_Controller",

        human_input_mode="NEVER",

        max_consecutive_auto_reply=1,

        code_execution_config=False,

        is_termination_msg=lambda msg:
            "PIPELINE_COMPLETE" in msg["content"]
    )

    return {
        "controller": orchestration_proxy,
        "context_shaper": context_shaper,
        "synthesizer": synthesizer,
        "reflection_critic": reflection_critic,
        "validator": validator,
    }
