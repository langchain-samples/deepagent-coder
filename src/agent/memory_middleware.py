"""
Middleware for loading agent-specific long-term memory into the system prompt.
Uses LangSmith Deployment's long-term persistence
"""
import json
from collections.abc import Awaitable, Callable
from typing import NotRequired, TypedDict, cast

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
)
from langgraph.runtime import Runtime

from agent.settings import settings


class AgentMemoryState(AgentState):
    """State for the agent memory middleware."""

    user_memory: NotRequired[str]
    """User preferences from /memories/{assistant}/ (applies everywhere)."""


class AgentMemoryStateUpdate(TypedDict):
    """A state update for the agent memory middleware."""

    user_memory: NotRequired[str]
    """User preferences from /memories/{assistant}/ (applies everywhere)."""


LONGTERM_MEMORY_SYSTEM_PROMPT = """

## Long-term Memory

Your long-term memory is stored in files on the filesystem and persists across sessions.

**User Memory Location**: `{agent_dir_absolute}`

**When to CHECK/READ memories (CRITICAL - do this FIRST):**
- **At the start of ANY new session**: Check memories
  - `ls {agent_dir_absolute}`
- **BEFORE answering questions**: If asked "what do you know about X?" or "how do I do Y?", check user memories FIRST
- **When user asks you to do something**: Check if you have specific guides or examples
- **When user references past work**: Search memory files for related context

**Memory-first response pattern:**
1. User asks a question → Check user memory directory first: `ls {agent_dir_absolute}`
2. If relevant files exist → Read them with `read_file '{agent_dir_absolute}/[filename]'`
3. Base your answer on saved knowledge supplemented by general knowledge

**When to update memories:**
- **IMMEDIATELY when the user describes your role or how you should behave**
- **IMMEDIATELY when the user gives feedback on your work** - Update memories to capture what was wrong and how to do it better
- When the user explicitly asks you to remember something
- When patterns or preferences emerge (coding styles, conventions, workflows)
- After significant work where context would help in future sessions

**Learning from feedback:**
- When user says something is better/worse, capture WHY and encode it as a pattern
- Each correction is a chance to improve permanently - don't just fix the immediate issue, update your instructions
- When user says "you should remember X" or "be careful about Y", treat this as HIGH PRIORITY - update memories IMMEDIATELY
- Look for the underlying principle behind corrections, not just the specific mistake

## Deciding to Store Memory

When writing or updating agent memory, decide whether each fact, configuration, or behavior belongs in:

### User Agent File: `{agent_dir_absolute}/agent.json`
→ Describes the agent's **personality, style, and universal behavior** across all projects.

**Store here:**
- Your general tone and communication style
- Universal coding preferences (formatting, comment style, etc.)
- General workflows and methodologies you follow
- Tool usage patterns that apply everywhere
- Personal preferences that don't change per-project

**Examples:**
- "Be concise and direct in responses"
- "Always use type hints in Python"
- "Prefer functional programming patterns"

DO NOT store task-specific information unless you are confident it will be helpful across tasks.

### File Operations:

**User memory:**
```
ls {agent_dir_absolute}                                # List user memory files
write_file '{agent_dir_absolute}/agent.json'           # Create user preference files if it does not yet exist 
read_file '{agent_dir_absolute}/agent.json'            # Read user preferences
edit_file '{agent_dir_absolute}/agent.json' ...        # Update user preferences
``
"""


DEFAULT_MEMORY_SNIPPET = """<user_memory>
{user_memory}
</user_memory>
"""


class AgentMemoryMiddleware(AgentMiddleware):
    """Middleware for loading agent-specific long-term memory.

    This middleware loads the agent's long-term memory from a file (agent.md)
    and injects it into the system prompt. The memory is loaded once at the
    start of the conversation and stored in state.
    """

    state_schema = AgentMemoryState

    def __init__(
        self,
        assistant_id: str,
    ) -> None:
        """Initialize the agent memory middleware.

        Args:
            assistant_id: The agent identifier.
        """
        # Configure memory path
        self.assistant_id = assistant_id
        self.agent_dir_absolute = settings.memories_base_path_template.format(assistant_id=assistant_id)
        self.system_prompt_template = DEFAULT_MEMORY_SNIPPET


    def before_agent(
        self,
        state: AgentMemoryState,
        runtime: Runtime,
    ) -> AgentMemoryStateUpdate:
        """Load agent memory from file before agent execution.

        Dynamically checks for file existence on every call to catch updates.

        Args:
            state: Current agent state.
            runtime: Runtime context.

        Returns:
            Updated state with user_memory and project_memory populated.
        """
        result: AgentMemoryStateUpdate = {}

        # Load user memory if not already in state
        if "user_memory" not in state:
            memories = runtime.store.get(("memories", f"{self.assistant_id}"), "agent.json")
            result["user_memory"] = json.dumps(memories, default=str)

        return result

    async def abefore_agent(
        self,
        state: AgentMemoryState,
        runtime: Runtime,
    ) -> AgentMemoryStateUpdate:
        """Load agent memory from file before agent execution.

        Dynamically checks for file existence on every call to catch updates.

        Args:
            state: Current agent state.
            runtime: Runtime context.

        Returns:
            Updated state with user_memory and project_memory populated.
        """
        result: AgentMemoryStateUpdate = {}

        # Load user memory if not already in state
        if "user_memory" not in state:
            memories = await runtime.store.aget(("memories", f"{self.assistant_id}"), "agent.json")
            result["user_memory"] = json.dumps(memories, default=str)

        return result

    def _build_system_prompt(self, request: ModelRequest) -> str:
        """Build the complete system prompt with memory sections.

        Args:
            request: The model request containing state and base system prompt.

        Returns:
            Complete system prompt with memory sections injected.
        """
        # Extract memory from state
        state = cast("AgentMemoryState", request.state)
        user_memory = state.get("user_memory")


        # Format memory section
        memory_section = self.system_prompt_template.format(
            user_memory=user_memory if user_memory else "(No user agent.md)",
        )

        system_prompt = memory_section

        base_system_prompt = request.system_prompt
        if base_system_prompt:
            system_prompt += "\n\n" + base_system_prompt

        system_prompt += "\n\n" + LONGTERM_MEMORY_SYSTEM_PROMPT.format(
            agent_dir_absolute=self.agent_dir_absolute,
        )

        return system_prompt

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Inject agent memory into the system prompt.

        Args:
            request: The model request being processed.
            handler: The handler function to call with the modified request.

        Returns:
            The model response from the handler.
        """
        system_prompt = self._build_system_prompt(request)
        return handler(request.override(system_prompt=system_prompt))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """(async) Inject agent memory into the system prompt.

        Args:
            request: The model request being processed.
            handler: The handler function to call with the modified request.

        Returns:
            The model response from the handler.
        """
        system_prompt = self._build_system_prompt(request)
        return await handler(request.override(system_prompt=system_prompt))
