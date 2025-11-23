import os
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    tavily_api_key: Optional[str] = Field(default=None, validation_alias="TAVILY_API_KEY")
    daytona_api_key: Optional[str] = Field(default=None, validation_alias="DAYTONA_API_KEY")
    daytona_base_path: Path = "/home/daytona"
    skills_base_path: Path = os.path.join(daytona_base_path, "skills")
    max_skill_file_size: int = 10 * 1024 * 1024 # 10 MB

settings = Settings()
