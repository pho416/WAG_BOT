from __future__ import annotations

import ast
import re
import unicodedata
from decimal import Decimal, localcontext


def normalized(text):
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", text)
    return " ".join(text.split()).casefold()


def quote_supported(quote, text):
    needle = normalized(quote)
    return len(needle) >= 12 and needle in normalized(text)


def decimal_calculate(expression, operands):
    """Evaluate arithmetic only. No Python eval/exec, imports, calls or attributes."""
    if len(expression) > 500 or len(operands) > 32:
        raise ValueError("Calculation too large")
    tree = ast.parse(expression, mode="eval")
    if len(list(ast.walk(tree))) > 150:
        raise ValueError("Calculation too complex")
    variables = {name: Decimal(str(value)) for name, value in operands.items()}
    if any(not value.is_finite() or abs(value.adjusted()) > 100 for value in variables.values()):
        raise ValueError("Invalid numeric operand")

    def visit(node):
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Name) and node.id in variables:
            return variables[node.id]
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            # Source segment preserves the literal's decimal precision.
            return Decimal(ast.get_source_segment(expression, node))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            result = visit(node.operand)
            return -result if isinstance(node.op, ast.USub) else result
        if isinstance(node, ast.BinOp):
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Add):
                result = left + right
            elif isinstance(node.op, ast.Sub):
                result = left - right
            elif isinstance(node.op, ast.Mult):
                result = left * right
            elif isinstance(node.op, ast.Div):
                result = left / right
            elif isinstance(node.op, ast.Pow) and right == right.to_integral_value() and abs(right) <= 12:
                result = left ** int(right)
            else:
                raise ValueError("Unsupported arithmetic operator")
            if not result.is_finite() or (result and abs(result.adjusted()) > 100):
                raise ValueError("Result exceeds arithmetic bounds")
            return result
        raise ValueError("Only numeric literals, declared operands and + - * / ** are allowed")

    with localcontext() as context:
        context.prec = 40
        result = visit(tree)
        if not result.is_finite():
            raise ValueError("Nonfinite result")
        return format(result, "f")


def number_in_quote(value, quote):
    try:
        target = Decimal(str(value))
    except Exception:
        return False
    candidates = re.findall(r"(?<![\w.])[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][+-]?\d+)?(?![\w.])", quote)
    return any(Decimal(candidate.replace(",", "")) == target for candidate in candidates)
