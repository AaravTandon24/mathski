from langchain_core.tools import tool
from sympy import Symbol, sympify, Integral
from langchain_community.tools.tavily_search import TavilySearchResults


@tool
def multiply(a: float, b: float) -> float:
    """
    Multiplies two floating-point numbers and returns the result.

    Args:
        a (float): The first number to multiply.
        b (float): The second number to multiply.

    Returns:
        float: The product of the two numbers.

    Example:
        result = multiply(4.5, 2.0)
        print(result)  # Output: 9.0
    """
    return a * b


@tool
def subtract(a: float, b: float) -> float:
    """
    Subtracts the second number from the first and returns the result.

    Args:
        a (float): The number from which to subtract.
        b (float): The number to subtract.

    Returns:
        float: The difference of the two numbers.

    Example:
        result = subtract(10.0, 5.5)
        print(result)  # Output: 4.5
    """
    return a - b


@tool
def divide(a: float, b: float) -> float:
    """
    Divides the first number by the second and returns the result.

    Args:
        a (float): The dividend.
        b (float): The divisor.

    Returns:
        float: The quotient of the two numbers.

    Example:
        result = divide(8.0, 2.0)
        print(result)  # Output: 4.0
    """
    return a / b


@tool
def add(a: float, b: float) -> float:
    """
    Adds two floating-point numbers and returns the sum.

    Args:
        a (float): The first number to add.
        b (float): The second number to add.

    Returns:
        float: The sum of the two numbers.

    Example:
        result = add(3.5, 2.5)
        print(result)  # Output: 6.0
    """
    return a + b


@tool
def exponent(a: float, b: float) -> float:
    """
    Raises the first number to the power of the second number and returns the result.

    Args:
        a (float): The base number.
        b (float): The exponent.

    Returns:
        float: The result of raising a to the power of b.

    Example:
        result = exponent(2.0, 3.0)
        print(result)  # Output: 8.0
    """
    return a**b


@tool
def symbolic_derivative(expression: str, variable: str = "x") -> str:
    """
    Calculates the symbolic derivative of a given expression with respect to a variable.

    Args:
        expression (str): The symbolic expression as a string.
        variable (str): The variable with respect to which the derivative is calculated (default: 'x').

    Returns:
        str: The derivative of the expression as a string.

    Example:
        expression = "5*x - 3*x**2"
        derivative = symbolic_derivative(expression)
        print(derivative)  # Output: "5 - 6*x"
    """
    var = Symbol(variable)
    func = sympify(expression)
    derivative = func.diff(var)

    return str(derivative)


@tool
def symbolic_integral(expression: str, variable: str = "x") -> str:
    """
    Calculates the symbolic indefinite integral of a given expression with respect to a variable.

    Args:
        expression (str): The symbolic expression as a string.
        variable (str): The variable with respect to which the integration is performed (default: 'x').

    Returns:
        str: The indefinite integral of the expression as a string.

    Example:
        expression = "5*x - 3*x**2"
        integral = symbolic_integration(expression)
        print(integral)  # Output: "5*x**2/2 - x**3"
    """
    var = Symbol(variable)
    expr = sympify(expression)
    integral = Integral(expr, var).doit()

    return str(integral)


@tool
def definite_integral(
    expression: str, variable: str, lower_limit: float, upper_limit: float
) -> str:
    """
    Calculates the symbolic definite integral of a given expression with respect to a variable,
    within the specified limits.

    Args:
        expression (str): The symbolic expression as a string.
        variable (str): The variable with respect to which the integration is performed.
        lower_limit (float): The lower limit of the definite integral.
        upper_limit (float): The upper limit of the definite integral.

    Returns:
        str: The definite integral of the expression as a string.

    Example:
        expression = "5*x - 3*x**2"
        variable = 'x'
        lower_limit = 0
        upper_limit = 1
        definite_integral = symbolic_definite_integral(expression, variable, lower_limit, upper_limit)
        print(definite_integral)  # Output: "-11/6"
    """
    var = Symbol(variable)
    expr = sympify(expression)
    definite_integral = Integral(expr, (var, lower_limit, upper_limit)).doit()

    return str(definite_integral)

@tool
def web_searcher(query: str):
    """
    Used to solve questions about topics that require external knowledge. For example: Questions requiring use of laplace transforms, fourier transforms etc. Use the web searcher tool to get information on the required topic, and solve the question using the respective tools required.
    
    Args:
        query (str): The search query string.
    
    Returns:
        str: The search results.
    """
    tavily_tool = TavilySearchResults()
    return tavily_tool.search(query)