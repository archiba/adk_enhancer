import os
from typing import Optional
from vertexai import types as _
from vertexai._genai.types.common import Prompt, CreatePromptConfig
from vertexai._genai.prompts import Prompts

from adk_enhancer.settings_enhancer import VERTEXAI_CLIENT


class PromptEnhancer:
    def __init__(self, prompt_id_var_prefix: str):
        self.local_prompts = {}
        self.prompt_id_var_prefix = prompt_id_var_prefix
        self.gcp_prompt_ids = self.load_prompt_ids_from_env()
    
    def get_prompt_envvar_name(self, agent_name: str) -> str:
        return f"{self.prompt_id_var_prefix}{agent_name.upper()}"
    
    def load_prompt_ids_from_env(self):
        prompt_ids = {}
        env_keys = os.environ.keys()
        for env_key in env_keys:
            if env_key.startswith(self.prompt_id_var_prefix):
                agent_name = env_key[len(self.prompt_id_var_prefix):].lower()
                prompt_ids[agent_name] = os.environ[env_key]
        return prompt_ids
    
    def add_prompt(self, agent_name: str, prompt: Prompt):
        self.local_prompts[agent_name] = prompt
    
    def get_prompt_from_vertex_ai(self, agent_name: str) -> Optional[Prompt]:
        if agent_name not in self.gcp_prompt_ids.keys():
            return None
        prompt_id = self.gcp_prompt_ids[agent_name]
        prompt_manager_client: Prompts = VERTEXAI_CLIENT.prompts
        try:
            prompt_manager_client.get(prompt_id=prompt_id)
        except:
            return None

    def get_prompt_from_local(self, agent_name: str) -> Optional[Prompt]:
        prompts_container = self.local_prompts

        if agent_name not in prompts_container.keys():
            return None
        return prompts_container[agent_name]

    def get_prompt(self, agent_name: str) -> Prompt:
        prompt = self.get_prompt_from_vertex_ai(agent_name)
        if prompt is not None:
            print("GCP Prompt used:", agent_name)
            return prompt
        prompt = self.get_prompt_from_local(agent_name)
        if prompt is not None:
            print("Local Prompt used:", agent_name)
            return prompt
        raise KeyError(agent_name)

    def save_prompts(self, adk_project_name: str = None):
        prompt_manager_client: Prompts = VERTEXAI_CLIENT.prompts
        envvars_to_export = {}
        for agent_name, agent_definition in self.local_prompts.items():
            if agent_name in self.gcp_prompt_ids.keys():
                created_prompt = prompt_manager_client.create_version(
                    prompt_id=self.gcp_prompt_ids[agent_name],
                    prompt=agent_definition
                )
            else:
                config = None
                if adk_project_name is not None:
                    config = CreatePromptConfig(
                        prompt_display_name=f"{adk_project_name}__{agent_name}"
                    )
                created_prompt: Prompt = prompt_manager_client.create(
                    prompt=agent_definition,
                    config=config
                )
                envvar_name = self.get_prompt_envvar_name(agent_name)
                envvars_to_export[envvar_name] = created_prompt.prompt_id
        print("下記の環境変数を設定してください。")
        for envvar_name, prompt_id in envvars_to_export.items():
            print(f"{envvar_name}={prompt_id}")
