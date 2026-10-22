"""A deliberately tiny, safe interpreter for editable numeric functions.

RippleTest 2.0 lets a developer paste two versions of a small numeric Python
function. The application must run the *edited* text, but it must never execute
arbitrary Python on this machine. There is no ``exec``, no ``eval``, no import,
and no call into the host interpreter.

Instead, this module parses the pasted text with the standard ``ast`` module and
interprets a strict allowlist of expression nodes itself:

    parameters, finite int/float constants, + - * / // % **, unary +/-,
    comparisons, ``and``/``or``, and conditional (ternary) expressions.

Everything else — function calls, attributes, indexing, imports, assignments,
loops, f-strings, comprehensions — is rejected during parsing with a precise,
human-readable explanation. Depth, node count, and exponent size are capped so
that hostile input cannot stall the machine.
"""

from __future__ import annotations

import ast
import math
from dataclasses import dataclass
from typing import Callable, Mapping

from .scenarios import Parameter, Scenario, scenario_matching_sources

# ---------------------------------------------------------------------------
# Documented limits (also surfaced in the UI and the README)
# ---------------------------------------------------------------------------

MAX_SOURCE_CHARS = 4_000
MAX_PARAMETERS = 6
MAX_NODES = 400
MAX_DEPTH = 30
MAX_ABS_EXPONENT = 32.0
CUSTOM_PARAMETER_LOW = -10_000.0
CUSTOM_PARAMETER_HIGH = 10_000.0

ENGINE_TRUSTED = "trusted-bundled"
ENGINE_INTERPRETER = "restricted-ast-interpreter"

ENGINE_LABELS = {
    ENGINE_TRUSTED: "Trusted bundled function",
    ENGINE_INTERPRETER: "Restricted AST interpreter",
}

ENGINE_NOTES = {
    ENGINE_TRUSTED: "Executed by importing the reviewed function from the application itself.",
    ENGINE_INTERPRETER: (
        "Executed by interpreting an allowlisted expression tree. "
        "No exec(), no eval(), no imports, no function calls."
    ),
}

CUSTOM_PAIR_ID = "custom"
CUSTOM_PAIR_LABEL = "Custom code"

SUPPORTED_SUBSET_HINT = (
    "Supported: exactly one function, numeric parameters, one return statement, and an "
    "expression built from parameters, finite numbers, + - * / // % **, comparisons, "
    "and if/else ternaries. Not supported: calls, imports, attributes, indexing, "
    "assignments, loops, f-strings, decorators, or defaults."
)


class EditError(Exception):
    """A pasted function that falls outside the supported, safe subset."""

    def __init__(self, message: str, line: int | None = None, hint: str = SUPPORTED_SUBSET_HINT):
        super().__init__(message)
        self.message = message
        self.line = line
        self.hint = hint

    def __str__(self) -> str:
        location = f"Line {self.line}: " if self.line else ""
        return f"{location}{self.message}"


@dataclass(frozen=True)
class ParsedFunction:
    """One parsed function: names, the return expression node, and raw text."""

    name: str
    parameters: tuple[str, ...]
    return_node: ast.expr
    source: str


@dataclass(frozen=True)
class FunctionPair:
    """Two validated versions of the same numeric function, ready to execute."""

    function_name: str
    parameters: tuple[Parameter, ...]
    original: Callable[..., object]
    modified: Callable[..., object]
    original_source: str
    modified_source: str
    original_name: str
    modified_name: str

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return tuple(parameter.name for parameter in self.parameters)

    # --- Duck-compatible surface with Scenario, so the differential engine,
    # validators, and Ollama client work unchanged on editable code. ---

    @property
    def id(self) -> str:
        return CUSTOM_PAIR_ID

    @property
    def label(self) -> str:
        return CUSTOM_PAIR_LABEL

    @property
    def contract(self) -> str:
        """Honest contract for edited code: no stated spec, so search for divergence."""
        names = ", ".join(self.parameter_names)
        return (
            f"No stated contract was supplied for {self.function_name}({names}). "
            "Find inputs where the two versions plausibly differ — ordinary values, "
            "zero, extremes, and exact boundaries."
        )


# ---------------------------------------------------------------------------
# Allowed AST nodes
# ---------------------------------------------------------------------------

_ALLOWED_EXPR_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.BoolOp,
    ast.Compare,
    ast.IfExp,
    ast.Constant,
    ast.Name,
    ast.Load,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
    ast.USub,
    ast.UAdd,
    ast.And,
    ast.Or,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
)

_REJECTED_HINTS: dict[type, str] = {
    ast.Call: "Function calls are not supported.",
    ast.Attribute: "Attribute access (for example math.floor or obj.x) is not supported.",
    ast.Subscript: "Indexing and slicing are not supported.",
    ast.Lambda: "Lambda functions are not supported.",
    ast.ListComp: "List comprehensions are not supported.",
    ast.SetComp: "Set comprehensions are not supported.",
    ast.DictComp: "Dict comprehensions are not supported.",
    ast.GeneratorExp: "Generator expressions are not supported.",
    ast.Await: "await is not supported.",
    ast.Yield: "yield is not supported.",
    ast.Starred: "Star-args are not supported.",
    ast.JoinedStr: "f-strings are not supported.",
    ast.FormattedValue: "f-strings are not supported.",
    ast.List: "List literals are not supported.",
    ast.Tuple: "Tuple literals are not supported.",
    ast.Set: "Set literals are not supported.",
    ast.Dict: "Dict literals are not supported.",
    ast.NamedExpr: "Walrus assignments (:=) are not supported.",
}


def _reject(node: ast.AST, message: str, hint: str = SUPPORTED_SUBSET_HINT) -> EditError:
    return EditError(message, line=getattr(node, "lineno", None), hint=hint)


# ---------------------------------------------------------------------------
# Parsing and validation
# ---------------------------------------------------------------------------


def parse_function(source: str) -> ParsedFunction:
    """Parse and validate one pasted function. Raises EditError when unsafe."""
    if not isinstance(source, str) or not source.strip():
        raise EditError("This panel is empty. Paste one numeric function to compare.")
    if len(source) > MAX_SOURCE_CHARS:
        raise EditError(
            f"The pasted text is {len(source)} characters; the limit is {MAX_SOURCE_CHARS}."
        )

    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise EditError(
            f"The pasted text is not valid Python: {exc.msg}.", line=exc.lineno
        ) from exc

    if len(tree.body) != 1:
        raise EditError(
            "Expected exactly one function definition and nothing else "
            "(no imports, assignments, or extra statements)."
        )
    node = tree.body[0]
    if isinstance(node, ast.AsyncFunctionDef):
        raise EditError("Async functions are not supported.")
    if not isinstance(node, ast.FunctionDef):
        raise EditError("Expected a single top-level function definition.")
    if node.decorator_list:
        raise EditError("Decorators are not supported.", line=node.lineno)

    arguments = node.args
    for label, group in (
        ("positional-only parameters", arguments.posonlyargs),
        ("keyword-only parameters", arguments.kwonlyargs),
    ):
        if group:
            raise EditError(f"{label} are not supported.", line=node.lineno)
    if arguments.vararg or arguments.kwarg:
        raise EditError("*args and **kwargs are not supported.", line=node.lineno)
    if arguments.defaults or any(arguments.kw_defaults):
        raise EditError("Default parameter values are not supported.", line=node.lineno)

    parameters = tuple(argument.arg for argument in arguments.args)
    if not parameters:
        raise EditError("The function needs at least one numeric parameter.", line=node.lineno)
    if len(parameters) > MAX_PARAMETERS:
        raise EditError(
            f"At most {MAX_PARAMETERS} parameters are supported.", line=node.lineno
        )
    if len(set(parameters)) != len(parameters):
        raise EditError("Parameter names must be unique.", line=node.lineno)

    if len(node.body) != 1:
        raise EditError(
            "The function body must be a single return statement — "
            "no statements, loops, assignments, or imports.",
            line=node.lineno,
        )
    statement = node.body[0]
    if not isinstance(statement, ast.Return):
        raise EditError(
            "The function body must be a single return statement.", line=statement.lineno
        )
    if statement.value is None:
        raise EditError("The return statement needs a value.", line=statement.lineno)

    expression = statement.value
    _validate_expression(expression, parameters)
    return ParsedFunction(
        name=node.name, parameters=parameters, return_node=expression, source=source
    )


def _validate_expression(node: ast.expr, parameters: tuple[str, ...]) -> None:
    """Reject anything outside the numeric allowlist, with precise messages."""
    node_count = 0
    stack: list[ast.AST] = [node]
    while stack:
        current = stack.pop()
        node_count += 1
        if node_count > MAX_NODES:
            raise EditError(f"The expression has more than {MAX_NODES} nodes.")

        if type(current) not in _ALLOWED_EXPR_NODES:
            hint = _REJECTED_HINTS.get(type(current))
            if hint:
                raise _reject(current, hint)
            raise _reject(
                current,
                f"Statements of type {type(current).__name__} are not supported.",
            )

        if isinstance(current, ast.Constant):
            value = current.value
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise _reject(
                    current,
                    "Only finite numeric constants are supported "
                    "(no text, None, or booleans as values).",
                )
            if not math.isfinite(float(value)):
                raise _reject(current, "Only finite numeric constants are supported.")

        if isinstance(current, ast.Name):
            if current.id not in parameters:
                raise _reject(
                    current,
                    f"'{current.id}' is not a parameter of this function. "
                    f"Parameters: {', '.join(parameters)}.",
                )

        stack.extend(ast.iter_child_nodes(current))

    if _expression_depth(node) > MAX_DEPTH:
        raise EditError(f"Expressions nested deeper than {MAX_DEPTH} levels are not supported.")


def _expression_depth(node: ast.AST) -> int:
    children = list(ast.iter_child_nodes(node))
    if not children:
        return 1
    return 1 + max(_expression_depth(child) for child in children)


def build_callable(parsed: ParsedFunction) -> Callable[..., object]:
    """Turn a validated function into a pure callable evaluated by our interpreter.

    The returned callable accepts keyword arguments only, mirroring how the
    trusted scenarios are invoked, and raises normal Python exceptions.
    """
    parameters = parsed.parameters
    return_node = parsed.return_node
    function_name = parsed.name

    def _call(**arguments: object) -> object:
        missing = [name for name in parameters if name not in arguments]
        if missing:
            raise TypeError(f"Missing argument(s): {', '.join(missing)}.")
        environment = {name: arguments[name] for name in parameters}
        for name, value in environment.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a number.")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite.")
        result = _interpret(return_node, environment, depth=1)
        if isinstance(result, bool):
            raise TypeError(
                "This function returned a boolean; the supported subset returns a number."
            )
        return _as_number(result, "Return value")

    _call.__name__ = function_name
    _call.__qualname__ = function_name
    return _call


# ---------------------------------------------------------------------------
# The interpreter itself
# ---------------------------------------------------------------------------


def _as_number(value: object, context: str) -> "int | float":
    """Validate a value is a finite number, preserving int/float as real Python would."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{context} produced a non-numeric value.")
    if not math.isfinite(float(value)):
        raise ValueError(f"{context} produced a non-finite value.")
    return value


def _interpret(node: ast.expr, environment: Mapping[str, object], depth: int) -> object:
    """Evaluate one allowlisted node. Raises real Python errors on bad input."""
    if depth > MAX_DEPTH:
        raise ValueError("The expression is nested too deeply to evaluate safely.")

    if isinstance(node, ast.Constant):
        return node.value

    if isinstance(node, ast.Name):
        return environment[node.id]

    if isinstance(node, ast.UnaryOp):
        value = _as_number(_interpret(node.operand, environment, depth + 1), "Unary operator")
        if isinstance(node.op, ast.USub):
            return -value
        if isinstance(node.op, ast.UAdd):
            return +value
        raise TypeError("Unsupported unary operator.")

    if isinstance(node, ast.BinOp):
        left = _as_number(_interpret(node.left, environment, depth + 1), "Left operand")
        right = _as_number(_interpret(node.right, environment, depth + 1), "Right operand")
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.FloorDiv):
            return left // right
        if isinstance(node.op, ast.Mod):
            return left % right
        if isinstance(node.op, ast.Pow):
            if abs(right) > MAX_ABS_EXPONENT:
                raise ValueError(
                    f"Exponent {right:g} exceeds the supported magnitude "
                    f"of {MAX_ABS_EXPONENT:g}."
                )
            return left**right
        raise TypeError("Unsupported arithmetic operator.")

    if isinstance(node, ast.Compare):
        left = _interpret(node.left, environment, depth + 1)
        for operator, comparator_node in zip(node.ops, node.comparators):
            right = _interpret(comparator_node, environment, depth + 1)
            if not _compare(operator, left, right):
                return False
            left = right
        return True

    if isinstance(node, ast.BoolOp):
        # Mirror Python's short-circuit behaviour on numeric/boolean operands.
        if isinstance(node.op, ast.And):
            result: object = True
            for child in node.values:
                result = _interpret(child, environment, depth + 1)
                _as_truthy(result, "and/or operand")
                if not result:
                    return result
            return result
        for child in node.values:
            result = _interpret(child, environment, depth + 1)
            _as_truthy(result, "and/or operand")
            if result:
                return result
        return result

    if isinstance(node, ast.IfExp):
        condition = _interpret(node.test, environment, depth + 1)
        _as_truthy(condition, "Condition")
        branch = node.body if condition else node.orelse
        return _interpret(branch, environment, depth + 1)

    raise TypeError(f"Node of type {type(node).__name__} cannot be evaluated.")


def _as_truthy(value: object, context: str) -> object:
    """Conditions may be booleans (from comparisons) or plain numbers."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value
    raise TypeError(f"{context} produced a value that cannot be tested as a condition.")


def _compare(operator: ast.cmpop, left: object, right: object) -> bool:
    a = _as_number(left, "Comparison")
    b = _as_number(right, "Comparison")
    if isinstance(operator, ast.Eq):
        return a == b
    if isinstance(operator, ast.NotEq):
        return a != b
    if isinstance(operator, ast.Lt):
        return a < b
    if isinstance(operator, ast.LtE):
        return a <= b
    if isinstance(operator, ast.Gt):
        return a > b
    if isinstance(operator, ast.GtE):
        return a >= b
    raise TypeError("Unsupported comparison operator.")


# ---------------------------------------------------------------------------
# Pair construction
# ---------------------------------------------------------------------------


def default_parameters(names: tuple[str, ...]) -> tuple[Parameter, ...]:
    """Bounds used when a developer supplies their own function.

    Without a stated contract, the model may propose values across a symmetric,
    finite range; anything outside it is rejected before execution.
    """
    return tuple(
        Parameter(
            name=name,
            low=CUSTOM_PARAMETER_LOW,
            high=CUSTOM_PARAMETER_HIGH,
            description=f"Numeric argument `{name}`.",
        )
        for name in names
    )


def parse_pair(original_source: str, modified_source: str) -> tuple[ParsedFunction, ParsedFunction]:
    """Parse both panels and require identical parameter signatures."""
    original = parse_function(original_source)
    modified = parse_function(modified_source)
    if original.parameters != modified.parameters:
        raise EditError(
            "The two versions must take the same parameters in the same order: "
            f"original has ({', '.join(original.parameters)}), "
            f"modified has ({', '.join(modified.parameters)})."
        )
    return original, modified


def build_pair(original_source: str, modified_source: str) -> FunctionPair:
    """Validate both panels and build pure callables for the differential engine."""
    original, modified = parse_pair(original_source, modified_source)
    parameters = default_parameters(original.parameters)
    return FunctionPair(
        function_name=modified.name,
        parameters=parameters,
        original=build_callable(original),
        modified=build_callable(modified),
        original_source=original.source,
        modified_source=modified.source,
        original_name=original.name,
        modified_name=modified.name,
    )


def resolve_execution_target(
    original_source: str, modified_source: str
) -> "Scenario | FunctionPair":
    """Choose how to execute a pasted pair, honestly and without surprises.

    1. If the pasted text matches a bundled example exactly, run that example's
       reviewed, imported functions (the reliable trusted path).
    2. Otherwise interpret the pasted text with the restricted AST interpreter.

    Editing the text can only ever route it through the interpreter, so what is
    displayed is always what runs. Raises EditError for unsupported syntax.
    """
    scenario = scenario_matching_sources(original_source, modified_source)
    if scenario is not None:
        return scenario
    return build_pair(original_source, modified_source)


def engine_for(target: "Scenario | FunctionPair") -> str:
    return ENGINE_TRUSTED if isinstance(target, Scenario) else ENGINE_INTERPRETER
