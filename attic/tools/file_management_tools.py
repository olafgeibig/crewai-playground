from tempfile import TemporaryDirectory
from langchain.tools import Tool
from langchain_community.agent_toolkits import FileManagementToolkit
from langchain_community.agent_toolkits import ReadFileTool

# We'll make a temporary directory to avoid clutter

class FileTooFactory():
    working_directory: str
    toolkit: FileManagementToolkit

    def __init__(self, file_name: str) -> None:
        self.file_name = file_name
        self.working_directory = TemporaryDirectory()
        self.toolkit = FileManagementToolkit(
            root_dir=self.working_directory.name,
            selected_tools=["read_file", "write_file"]
        )

    def get_read_file_tool(self) -> ReadFileTool:
        read_file, write_file = self.toolkit.get_tools()
        tool = Tool(
            name=read_file.name
            description=read_file.description

        )
        return self.toolkit.get_tools()

    def get_read_file_tool(self) -> ReadFileTool:
        return self.toolkit.get_tools()
