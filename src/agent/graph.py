"""LangGraph single-node graph template.

Returns a predefined response. Replace logic and configuration as needed.
"""
from contextlib import asynccontextmanager
from dataclasses import dataclass

from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, StoreBackend
from langchain.chat_models import init_chat_model
from langchain_core.runnables import RunnableConfig

from agent.memory_middleware import AgentMemoryMiddleware
from agent.sandbox import create_daytona_sandbox
from agent.skills_middleware import SkillsMiddleware
from agent.tools import http_request, fetch_url, web_search
from agent.settings import settings

# You may want to replace this with a more powerful model
model = init_chat_model('openai:gpt-5-mini')

def get_system_prompt() -> str:
    """Get the base system prompt for the agent.
    Returns:
        The system prompt string (without agent.md content)
    """

    # Can modify this if you are using a provider other than Daytona
    working_dir_section = f"""### Current Working Directory

    You are operating in an ephemeral sandbox environment. Your base dir is {settings.daytona_base_path}
    You can use the local filesystem to write code and test it,
    but the end user cannot see what you create. When you are done, draft a message containing the final code
    you have generated.
   
    You can execute code to test it as much as you would like. If you need a Python package, use pip to install it.
    You do not need to ask permission before installing a package.  
    """
    return (
        working_dir_section
        + f"""
        ### Human-in-the-Loop Tool Approval
        
        Some tool calls require user approval before execution. When a tool call is rejected by the user:
        1. Accept their decision immediately - do NOT retry the same command
        2. Explain that you understand they rejected the action
        3. Suggest an alternative approach or ask for clarification
        4. Never attempt the exact same rejected command again
        
        Respect the user's decisions and work with them collaboratively.
        
        ### Web Search Tool Usage
        
        When you use the web_search tool:
        1. The tool will return search results with titles, URLs, and content excerpts
        2. You MUST read and process these results, then respond naturally to the user
        3. NEVER show raw JSON or tool results directly to the user
        4. Synthesize the information from multiple sources into a coherent answer
        5. Cite your sources by mentioning page titles or URLs when relevant
        6. If the search doesn't find what you need, explain what you found and ask clarifying questions
        
        The user only sees your text responses - not tool results. Always provide a complete, natural language answer after using web_search.
        
        ### Todo List Management
        
        When using the write_todos tool:
        1. Keep the todo list MINIMAL - aim for 3-6 items maximum
        2. Only create todos for complex, multi-step tasks that truly need tracking
        3. Break down work into clear, actionable items without over-fragmenting
        4. For simple tasks (1-2 steps), just do them directly without creating todos
        5. When first creating a todo list for a task, ALWAYS ask the user if the plan looks good before starting work
           - Create the todos, let them render, then ask: "Does this plan look good?" or similar
           - Wait for the user's response before marking the first todo as in_progress
           - If they want changes, adjust the plan accordingly
        6. Update todo status promptly as you complete each item
        
        The todo list is a planning tool - use it judiciously to avoid overwhelming the user with excessive task tracking.
    
    ### Final output
    Return your final code string as a message to the end user. Format any code inside a markdown-style code block.
    """
        )



agent_tools = [http_request, fetch_url, web_search]


@dataclass
class ContextSchema:
    # We only need the assistant ID that is automatically included
    pass


@asynccontextmanager
async def agent(config: RunnableConfig):
    assistant_id = config['configurable'].get('assistant_id')
    memory_middleware = AgentMemoryMiddleware(assistant_id)
    skills_middleware = SkillsMiddleware()
    async with create_daytona_sandbox() as sandbox_backend:
        composite_backend = lambda rt: CompositeBackend(
            default=sandbox_backend,
            routes={"/memories/": StoreBackend(rt)}
        )
        agent = create_deep_agent(
            model=model,
            system_prompt=get_system_prompt(),
            middleware=[memory_middleware, skills_middleware],
            tools=agent_tools,
            backend=composite_backend,
        ).with_config({"recursion_limit": 1000})
        yield agent
