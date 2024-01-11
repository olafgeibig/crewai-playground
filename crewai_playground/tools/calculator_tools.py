from langchain.tools import tool


class CalculatorTools():

  @tool("Make calculation")
  def calculate(operation):
    """Useful to perform any mathematical calculations, 
    like sum, minus, mutiplcation, division, etc.
    The input to this tool should be a mathematical 
    experission, a couple examples are `200*7` or `5000/2*10`
    """
    try:
        return eval(operation)
    except Exception as e:
        return f"Error: {str(e)}"