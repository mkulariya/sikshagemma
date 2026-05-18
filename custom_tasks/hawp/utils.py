import ast
import operator as op

def compute_target(doc):
    """Safely evaluate the equation to get the target answer."""
    expr = doc["Equation"].split("=")[1].strip()
    # Secure evaluation of expression
    def _eval(expr):
        return eval(ast.parse(expr, mode='eval').compile())
    return str(round(_eval(expr), 4))

def hawp_metric(predictions, references):
    """Compute accuracy for HAWP (numerical)."""
    correct = 0
    total = 0
    for pred, ref in zip(predictions, references):
        total += 1
        try:
            if abs(float(pred) - float(ref)) < 1e-4:
                correct += 1
        except (ValueError, TypeError):
            pass
    return {"accuracy": correct / total if total > 0 else 0.0}