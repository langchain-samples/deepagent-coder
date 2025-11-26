import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from daytona import DaytonaConfig, AsyncDaytona, AsyncSandbox, FileUpload
from deepagents.backends.protocol import ExecuteResponse, FileDownloadResponse, FileUploadResponse
from deepagents.backends.sandbox import BaseSandbox

from agent.settings import settings


class DaytonaBackend(BaseSandbox):
    """Daytona backend implementation conforming to SandboxBackendProtocol.

    This implementation inherits all file operation methods from BaseSandbox
    and only implements the execute() method using Daytona's API.

    Deepagents currently expects "execute" to not be an async function. Since we are using an async Daytona backend,
    we keep the event loop of our agent server as an instance method here, so, within the sync method, we can
    still have asyncio call the Daytona backend.
    """

    def __init__(self, sandbox: AsyncSandbox, event_loop) -> None:
        """Initialize the DaytonaBackend with a Daytona sandbox client.

        Args:
            sandbox: Daytona sandbox instance
        """
        self._sandbox = sandbox
        self._loop = event_loop

    @property
    def id(self) -> str:
        """Unique identifier for the sandbox backend."""
        return self._sandbox.id

    def execute(
        self,
        command: str,
    ) -> ExecuteResponse:
        """Execute a command in the sandbox and return ExecuteResponse.

        Args:
            command: Full shell command string to execute.

        Returns:
            ExecuteResponse with combined output, exit code, optional signal, and truncation flag.
        """
        async def execute():
            return await self._sandbox.process.exec(command)
        future = asyncio.run_coroutine_threadsafe(execute(), self._loop)
        result = future.result()
        return ExecuteResponse(
            output=result.result,  # Daytona combines stdout/stderr
            exit_code=result.exit_code,
            truncated=False,
        )

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Download multiple files from the Daytona sandbox.

        Leverages Daytona's native batch download API for efficiency.
        Supports partial success - individual downloads may fail without
        affecting others.

        Args:
            paths: List of file paths to download.

        Returns:
            List of FileDownloadResponse objects, one per input path.
            Response order matches input order.

        TODO: Map Daytona API error strings to standardized FileOperationError codes.
        Currently only implements happy path.
        """
        from daytona import FileDownloadRequest

        # Create batch download request using Daytona's native batch API
        download_requests = [FileDownloadRequest(source=path) for path in paths]
        daytona_responses = self._sandbox.fs.download_files(download_requests)

        # Convert Daytona results to our response format
        # TODO: Map resp.error to standardized error codes when available
        return [
            FileDownloadResponse(
                path=resp.source,
                content=resp.result,
                error=None,  # TODO: map resp.error to FileOperationError
            )
            for resp in daytona_responses
        ]

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Upload multiple files to the Daytona sandbox.

        Leverages Daytona's native batch upload API for efficiency.
        Supports partial success - individual uploads may fail without
        affecting others.

        Args:
            files: List of (path, content) tuples to upload.

        Returns:
            List of FileUploadResponse objects, one per input file.
            Response order matches input order.

        TODO: Map Daytona API error strings to standardized FileOperationError codes.
        Currently only implements happy path.
        """
        from daytona import FileUpload

        # Create batch upload request using Daytona's native batch API
        upload_requests = [FileUpload(source=content, destination=path) for path, content in files]
        self._sandbox.fs.upload_files(upload_requests)

        # TODO: Check if Daytona returns error info and map to FileOperationError codes
        return [FileUploadResponse(path=path, error=None) for path, _ in files]


async def upload_skills(sandbox: AsyncSandbox) -> None:
    """
    Upload the skills directory to the sandbox so we can read and execute files, including any code within the skills.
    """
    def get_files():
        files_to_upload = []
        skills_dir = Path(__file__).parent / "skills"
        for file_path in skills_dir.rglob("*"):
            if file_path.is_file():
                rel_path = file_path.relative_to(skills_dir)
                with open(file_path, "rb") as f:
                    files_to_upload.append(
                        FileUpload(
                            source=f.read(),
                            destination=f"/home/daytona/skills/{rel_path.as_posix()}",
                        )
                    )
        return files_to_upload
    files_to_upload = await asyncio.to_thread(get_files)
    try:
        await sandbox.fs.upload_files(files_to_upload)
        pass
    except Exception as e:
        print(e)


@asynccontextmanager
async def create_daytona_sandbox() -> AsyncGenerator:
    """Create Daytona sandbox.

    Yields: DaytonaBackend
    """

    print("Starting Daytona sandbox...")

    # Creating the client involves some synchronous reads to the local filesystem, hence the call to to_thread
    daytona = await asyncio.to_thread(lambda: AsyncDaytona(DaytonaConfig(api_key=settings.daytona_api_key)))
    sandbox = await daytona.create()
    sandbox_id = sandbox.id

    # Poll until running (Daytona requires this)
    for _ in range(90):  # 180s timeout (90 * 2s)
        # Check if sandbox is ready by attempting a simple command
        try:
            result = await sandbox.process.exec("echo ready", timeout=5)
            if result.exit_code == 0:
                await upload_skills(sandbox)
                break
        except Exception as e:
            print("Error creating sandbox: retrying", e)
        await asyncio.sleep(2)
    else:
        try:
            # Clean up if possible
            await sandbox.delete()
        finally:
            raise RuntimeError("Daytona sandbox failed to start within 180 seconds")

    loop = asyncio.get_running_loop()
    backend = DaytonaBackend(sandbox, loop)
    print(f"✓ Daytona sandbox ready: {backend.id}")
    try:
        yield backend
    finally:
        print(f"Deleting Daytona sandbox {sandbox_id}...")
        try:
            await sandbox.delete()
            print(f"✓ Daytona sandbox {sandbox_id} terminated")
        except Exception as e:
            print(f"⚠ Cleanup failed: {e}")
