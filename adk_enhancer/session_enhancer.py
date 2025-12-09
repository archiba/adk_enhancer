import json
from datetime import datetime, timedelta, timezone
from pydantic import BaseModel, Field

from google.genai.pagers import Pager

from vertexai._genai.agent_engines import AgentEngines
from vertexai._genai.sessions import Sessions
from vertexai._genai.session_events import SessionEvents
from google.genai.types import Content, Part
from vertexai._genai.types.common import (
    Session,
    SessionEvent,
    CreateAgentEngineSessionConfig,
    AppendAgentEngineSessionEventConfig,
    AgentEngineSessionOperation,
    AppendAgentEngineSessionEventResponse,
)
from adk_enhancer.settings_enhancer import VERTEXAI_CLIENT


class AsyncStreamQueryContent(BaseModel):
    message: str | list[Part]
    # Multimodal examples
    # message=[Part.from_text(text="text_part"), Part.from_uri(file_uri="gs://bucket_name/file_name", mime_type="image/png")]
    user_id: str
    session_id: str | None = Field(default=None)


agent_engine_manager: AgentEngines = VERTEXAI_CLIENT.agent_engines


class SessionEnhancer:
    def __init__(self, project_id: str, gcp_region_name: str, agent_engine_id: str):
        self.project_id = project_id
        self.gcp_region_name = gcp_region_name
        self.agent_engine_id = agent_engine_id
    
    @property
    def agent_name(self):
        return f"projects/{self.project_id}/locations/{self.gcp_region_name}/reasoningEngines/{self.agent_engine_id}"

    def session_name(self, session_id: str):
        return f"{self.agent_name}/sessions/{session_id}"
    
    def create_new_session(
        self, 
        user_id: str,
        session_name: str | None = None,
        initial_states: str | dict[str, str] | None = None,
        expired_in_n_days: int | None = None
    ) -> Session:
        expire_time: datetime | None = None
        if expired_in_n_days is not None:
            current_datetime = datetime.now(tz=timezone.utc)
            expire_time = current_datetime + timedelta(days=expired_in_n_days)        
        if initial_states is None:
            initial_states = {}
        elif isinstance(initial_states, str):
            initial_states = json.loads(initial_states)
        session_manager: Sessions = agent_engine_manager.sessions
        session_config = CreateAgentEngineSessionConfig(
            display_name=session_name,
            session_state=initial_states,
            expire_time=expire_time
        )
        new_session_op: AgentEngineSessionOperation = session_manager.create(
            name=self.agent_name,
            user_id=user_id,
            config=session_config
        )
        if not new_session_op.done:
            raise ValueError(new_session_op.error)
        return new_session_op.response

    def get_session(self, session_id: str) -> Session:
        session_manager: Sessions = agent_engine_manager.sessions
        session = session_manager.get(name=self.session_name(session_id), )
        return session

    def get_list_of_sessions(self) -> Pager[Session]:
        session_manager: Sessions = agent_engine_manager.sessions
        sessions_page: Pager[Session] = session_manager.list(name=self.agent_name)
        return sessions_page

    def get_list_of_session_events(self, session_id: str) -> Pager[SessionEvent]:
        session_manager: Sessions = agent_engine_manager.sessions
        session_event_manager: SessionEvents = session_manager.events
        session_events: Pager[SessionEvent] = session_event_manager.list(
            name=self.session_name(session_id)
        )
        return session_events

    def send_user_message(
        self, 
        user_id: str,
        session_id: str, 
        content: Content | str
    ):
        agent_engine = agent_engine_manager.get(name=self.agent_name)
        # https://github.com/google/adk-docs/issues/930
        # Currently, string message is the only supported way to send message.
        if False:
            if isinstance(content, str):
                input_parts = [Part.from_text(text=content)]
            elif isinstance(content, Content):
                input_parts = content.parts
            else:
                raise ValueError("content must be str or Content type.")
        else:
            if isinstance(content, str):
                input_parts = content
            else:
                raise ValueError("content must be str type.")
        input_data = AsyncStreamQueryContent(
            message=input_parts,
            user_id="chiba",
            session_id=session_id
        )
        async_events = agent_engine.async_stream_query(
            **input_data.model_dump()
        )
        return async_events


async def main():
    enh = SessionEnhancer(project_id="MY_PROJECT", gcp_region_name="us-central1", agent_engine_id="5760169894104530911")
    session = enh.get_session(session_id="815460220443557855")
    print(list(enh.get_list_of_session_events(session_id="815460220443557855")))
    async_events = enh.send_user_message(user_id="chiba", session_id="815460220443557855", content="シニアデータサイエンティストの主な業務：\n大規模データ分散処理・データサイエンティスト・BI・データ可視化")
    async for event in async_events:
        print(type(event))
        print(event)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
