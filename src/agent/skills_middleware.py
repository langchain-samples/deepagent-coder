"""Middleware for loading and exposing agent skills to the system prompt.

This middleware implements Anthropic's "Agent Skills" pattern with progressive disclosure:
1. Parse YAML frontmatter from SKILL.md files at session start
2. Inject skills metadata (name + description) into system prompt
3. Agent reads full SKILL.md content when relevant to a task

Skills directory structure:
skills/
├── web-research/
│   ├── SKILL.md        # Required: YAML frontmatter + instructions
│   └── helper.py       # Optional: supporting files
├── code-review/
│   ├── SKILL.md
│   └── checklist.md
"""
import os
import re
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import NotRequired, TypedDict, cast

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
)

from agent.settings import settings

class SkillMetadata(TypedDict):
    """Metadata for a skill."""

    name: str
    """Name of the skill."""

    description: str
    """Description of what the skill does."""

    path: str
    """Path to the SKILL.md file."""


def _parse_skill_metadata(skill_md_path: Path) -> SkillMetadata | None:
    """Parse YAML frontmatter from a SKILL.md file.

    Args:
        skill_md_path: Path to the SKILL.md file.

    Returns:
        SkillMetadata with name, description, and path, or None if parsing fails.
    """
    try:
        # Security: Check file size to prevent DoS attacks
        file_size = skill_md_path.stat().st_size
        if file_size > settings.max_skill_file_size:
            # Silently skip files that are too large
            return None

        content = skill_md_path.read_text(encoding="utf-8")

        # Match YAML frontmatter between --- delimiters
        frontmatter_pattern = r"^---\s*\n(.*?)\n---\s*\n"
        match = re.match(frontmatter_pattern, content, re.DOTALL)

        if not match:
            return None

        frontmatter = match.group(1)

        # Parse key-value pairs from YAML (simple parsing, no nested structures)
        metadata: dict[str, str] = {}
        for line in frontmatter.split("\n"):
            # Match "key: value" pattern
            kv_match = re.match(r"^(\w+):\s*(.+)$", line.strip())
            if kv_match:
                key, value = kv_match.groups()
                metadata[key] = value.strip()

        # Validate required fields
        if "name" not in metadata or "description" not in metadata:
            return None

        return SkillMetadata(
            name=metadata["name"],
            description=metadata["description"],
            path=str(skill_md_path),
        )

    except (OSError, UnicodeDecodeError):
        # Silently skip malformed or inaccessible files
        return None


def list_skills() -> list[SkillMetadata]:
    """List all skills from the skills directory.

    Scans the skills directory for subdirectories containing SKILL.md files,
    parses YAML frontmatter, and returns skill metadata.

    Skills are organized as:
    skills/
    ├── skill-name/
    │   ├── SKILL.md        # Required: instructions with YAML frontmatter
    │   ├── script.py       # Optional: supporting files
    │   └── config.json     # Optional: supporting files

    Returns:
        List of skill metadata dictionaries with name, description, and path.

    Example:
        ```python
        skills = list_skills(skills_dir)
        for skill in skills:
            print(f"{skill['name']}: {skill['description']}")
        ```
    """
    # Check if skills directory exists
    skills_dir = Path(os.path.join(Path(__file__).parent, "skills"))
    print(skills_dir)

    skills: list[SkillMetadata] = []
    print(os.listdir(skills_dir))
    # Iterate through subdirectories
    for skill_dir in skills_dir.iterdir():
        # Look for SKILL.md file
        skill_md_path = skill_dir / "SKILL.md"
        if not skill_md_path.exists():
            continue

        # Parse metadata
        metadata = _parse_skill_metadata(skill_md_path)
        if metadata:
            skills.append(metadata)

    return skills


class SkillsState(AgentState):
    """State for the skills middleware."""

    skills_metadata: NotRequired[list[SkillMetadata]]
    """List of loaded skill metadata (name, description, path)."""


class SkillsStateUpdate(TypedDict):
    """State update for the skills middleware."""

    skills_metadata: list[SkillMetadata]
    """List of loaded skill metadata (name, description, path)."""


# Skills System Documentation
SKILLS_SYSTEM_PROMPT = """

## Skills System

You have access to a skills library that provides specialized capabilities and domain knowledge.

**Skills Location**: `{skills_dir_absolute}` 

**Available Skills:**

{skills_list}

**How to Use Skills (Progressive Disclosure):**

Skills follow a **progressive disclosure** pattern - you know they exist (name + description above), but you only read the full instructions when needed:

1. **Recognize when a skill applies**: Check if the user's task matches any skill's description
2. **Read the skill's full instructions**: The skill list above shows the exact path to use with read_file
3. **Follow the skill's instructions**: SKILL.md contains step-by-step workflows, best practices, and examples
4. **Access supporting files**: Skills may include Python scripts, configs, or reference docs - use absolute paths

**When to Use Skills:**
- When the user's request matches a skill's domain (e.g., "research X" → web-research skill)
- When you need specialized knowledge or structured workflows
- When a skill provides proven patterns for complex tasks

**Skills are Self-Documenting:**
- Each SKILL.md tells you exactly what the skill does and how to use it
- You can explore available skills with `ls {skills_dir_absolute}`
- You can read any skill's directory with `ls {skills_dir_absolute}/[skill-name]`

**Executing Skill Scripts:**
Skills may contain Python scripts or other executable files. Always use absolute paths:
Example: `bash python {skills_dir_absolute}/web-research/fetch_data.py`

**Example Workflow:**

User: "Can you research the latest developments in quantum computing?"

1. Check available skills above → See "web-research" skill with its full path
2. Read the skill using the path shown: `read_file '{skills_dir_absolute}/web-research/SKILL.md'`
3. Follow the skill's research workflow (search → organize → synthesize)
4. Use any helper scripts with absolute paths: `bash python {skills_dir_absolute}/web-research/script.py`

Remember: Skills are tools to make you more capable and consistent. When in doubt, check if a skill exists for the task!
"""


class SkillsMiddleware(AgentMiddleware):
    """Middleware for loading and exposing agent skills.

    This middleware implements Anthropic's agent skills pattern:
    - Loads skills metadata (name, description) from YAML frontmatter at session start
    - Injects skills list into system prompt for discoverability
    - Agent reads full SKILL.md content when a skill is relevant (progressive disclosure)
    """

    state_schema = SkillsState

    def __init__(
        self,
    ) -> None:
        """Initialize the skills middleware.

        Args:
            assistant_id: The agent identifier.
        """
        self.system_prompt_template = SKILLS_SYSTEM_PROMPT

    def _format_skills_list(self, skills: list[SkillMetadata]) -> str:
        """Format skills metadata for display in system prompt."""
        if not skills:
            return f"(No skills available yet. You can create skills in the skills dir)"

        lines = []
        for skill in skills:
            skill_dir = Path(skill["path"]).parent.name
            lines.append(f"- **{skill['name']}**: {skill['description']}")
            lines.append(
                f"  → Read `{skill_dir}/SKILL.md` for full instructions"
            )

        return "\n".join(lines)

    def before_agent(
        self,
        state: SkillsState,
        runtime,
    ) -> SkillsStateUpdate | None:
        """Load skills metadata before agent execution.

        This runs once at session start to discover available skills.

        Args:
            state: Current agent state.
            runtime: Runtime context.

        Returns:
            Updated state with skills_metadata populated.
        """
        # We re-load skills on every new interaction with the agent to capture
        # any changes in the skills directory.
        skills = list_skills()
        return SkillsStateUpdate(skills_metadata=skills)

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Inject skills documentation into the system prompt.

        This runs on every model call to ensure skills info is always available.

        Args:
            request: The model request being processed.
            handler: The handler function to call with the modified request.

        Returns:
            The model response from the handler.
        """
        # Get skills metadata from state
        skills_metadata = request.state.get("skills_metadata", [])

        # Format skills list
        skills_list = self._format_skills_list(skills_metadata)

        # Format the skills documentation
        skills_section = self.system_prompt_template.format(
            skills_list=skills_list,
            skills_dir_absolute=str(settings.skills_base_path),
        )

        if request.system_prompt:
            system_prompt = request.system_prompt + "\n\n" + skills_section
        else:
            system_prompt = skills_section

        return handler(request.override(system_prompt=system_prompt))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """(async) Inject skills documentation into the system prompt.

        Args:
            request: The model request being processed.
            handler: The handler function to call with the modified request.

        Returns:
            The model response from the handler.
        """
        # The state is guaranteed to be SkillsState due to state_schema
        state = cast("SkillsState", request.state)
        skills_metadata = state.get("skills_metadata", [])

        # Format skills list
        skills_list = self._format_skills_list(skills_metadata)

        # Format the skills documentation
        skills_section = self.system_prompt_template.format(
            skills_list=skills_list,
            skills_dir_absolute=str(settings.skills_base_path),
        )

        # Inject into system prompt
        if request.system_prompt:
            system_prompt = request.system_prompt + "\n\n" + skills_section
        else:
            system_prompt = skills_section

        return await handler(request.override(system_prompt=system_prompt))
