from typing import Union
from google.genai.types import GenerateContentConfig, GenerationConfig, ThinkingConfig
from vertexai._genai.types.common import Prompt, SchemaPromptSpecPromptMessage
from google.adk.planners import BasePlanner, BuiltInPlanner, built_in_planner


def prompt_to_generate_content_config(prompt: Prompt) -> GenerateContentConfig:
    prompt_generation_config: GenerationConfig = prompt.prompt_data.generation_config
    prompt_data = prompt.prompt_data
    if prompt_data is None:
        prompt_data = SchemaPromptSpecPromptMessage()
    generate_content_config = GenerateContentConfig(
        system_instruction=None, # System instruction should be set via instruction argument.
        temperature=prompt_generation_config.temperature,
        top_p=prompt_generation_config.top_p,
        top_k=prompt_generation_config.top_k,
        candidate_count=prompt_generation_config.candidate_count,
        max_output_tokens=prompt_generation_config.max_output_tokens,
        stop_sequences=prompt_generation_config.stop_sequences,
        response_logprobs=prompt_generation_config.response_logprobs,
        logprobs=prompt_generation_config.logprobs,
        presence_penalty=prompt_generation_config.presence_penalty,
        frequency_penalty=prompt_generation_config.frequency_penalty,
        seed=prompt_generation_config.seed,
        response_mime_type=prompt_generation_config.response_mime_type,
        response_schema=prompt_generation_config.response_schema,
        response_json_schema=prompt_generation_config.response_json_schema,
        routing_config=prompt_generation_config.routing_config,
        model_selection_config=prompt_generation_config.model_selection_config,
        safety_settings=prompt_data.safety_settings,
        tools=prompt_data.tools,
        tool_config=prompt_data.tool_config,
        audio_timestamp=prompt_generation_config.audio_timestamp,
        thinking_config=None, # Thinking config should be set via planner feature.
        labels=None,
        cached_content=None,
        response_modalities=None,
        media_resolution=None,
        automatic_function_calling=None,
        speech_config=None,
        image_config=None
    )
    return generate_content_config

def prompt_to_model(prompt: Prompt) -> str:
    return prompt.prompt_data.model


def prompt_to_instruction(prompt: Prompt) -> str:
    return prompt.prompt_data.system_instruction.parts[0].text

def prompt_to_planner(prompt: Prompt) -> BasePlanner | None:
    prompt_generation_config: GenerationConfig = prompt.prompt_data.generation_config
    if prompt_generation_config.thinking_config is None:
        return None
    thinking_config = prompt_generation_config.thinking_config
    if not thinking_config.include_thoughts:
        return None
    planner = BuiltInPlanner(thinking_config=prompt_generation_config.thinking_config)
    return planner

def prompt_to_llmagent_kwargs(prompt: Prompt) -> dict[str, Union[GenerateContentConfig, str]]:
    generate_content_config = prompt_to_generate_content_config(prompt)
    model = prompt_to_model(prompt)
    instruction = prompt_to_instruction(prompt)
    planner = prompt_to_planner(prompt)
    return {
        "model": model,
        "generate_content_config": generate_content_config,
        "instruction": instruction,
        "planner": planner
    }
