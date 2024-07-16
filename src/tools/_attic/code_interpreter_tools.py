from langchain_community.tools.e2b_data_analysis.tool import E2BDataAnalysisTool
from langchain.tools import tool
from langchain.tools import BaseTool
from dotenv import load_dotenv
import os

class CodeInterpreterTool(E2BDataAnalysisTool):

    def __init__(self):
        load_dotenv()
        self.api_key=os.getenv('E2B_API_KEY')
        # self.e2b = E2BDataAnalysisTool(
        #     on_stdout=lambda stdout: print("stdout:", stdout),
        #     on_stderr=lambda stderr: print("stderr:", stderr),
        #     # on_artifact=save_artifact,
        #     api_key=os.getenv('E2B_API_KEY')
        # )

    # @tool("Run python code in a sandbox")
    # def run_python(code: str) -> str:
    #     """Executes pure python code. Do not wrap the code in markdown or quotes. just pure python. 
    #      The whole code must be in one file."""
    #     result = "foo"#self.e2b._run(code)
    #     try:
    #         result = e2b._run(code)
    #     except Exception:
    #         result = "An error occored"
    #     return result

    # def get_run_tool(self):
    #     print("XXX Tool Exec")
    #     return Tool(
    #         description="""Executes pure python code. Do not wrap the code in markdown or quotes. just pure python. 
    #         The whole code must be in one file.""",
    #         name="runPython",
    #         func=runP
    #     )

    # def runPython(code: str) -> str:
    #     e2b = get_e2b()

    #     # result = self.e2b._run(code)
    #     # try:
    #     #     result = self.e2b._run(code)
    #     # except SyntaxError:
    #     #     result = "An error occored"
    #     # except:
    #     #     result = "Unexpected error"
    #     return result

    # def get_e2b() -> E2BDataAnalysisTool:
    #     if CodeInterpreterTool.e2b != None:
    #         CodeInterpreterTool.e2b = E2BDataAnalysisTool(
    #             on_stdout=lambda stdout: print("stdout:", stdout),
    #             on_stderr=lambda stderr: print("stderr:", stderr),
    #             # on_artifact=save_artifact,
    #             api_key=os.getenv('E2B_API_KEY')
    #         )             
    #     return CodeInterpreterTool.e2b
    
#     description="""evaluates python code in a sandbox environment. \
# The environment is long running and exists across multiple executions. \
# You must send the whole script every time and print your outputs. \
# Script should be pure python code that can be evaluated. \
# It should be in python format NOT markdown. \
# The code should NOT be wrapped in backticks. \
# Do not use markdown or KITTENS WILL DIE""",

