import torch
from src.counterfactual.solve import solve


class AbsoluteConstraint(torch.autograd.Function):
    """
    Computes the feasibility of the explanation constraint for absolute explanations.
    """
    @staticmethod
    def forward(ctx, cp, w_alt, opt_model, task):

        ctx.task = task

        w_opt, _ = solve(cp, opt_model, task)

        ctx.save_for_backward(w_opt, w_alt)

        if task == "knapsack":
             return torch.dot(cp, w_opt-w_alt) # Opposite constraint as the optimization problem is formalized as a Maximization problem.

        return torch.dot(cp, w_alt-w_opt)

    @staticmethod
    def backward(ctx, grad_output):

        w_opt, w_alt = ctx.saved_tensors

        if ctx.task == "knapsack":
             return grad_output*(w_opt-w_alt), None, None, None  # Opposite gradient as the optimization problem is formalized as a Maximization problem.
        
        return grad_output*(w_alt-w_opt), None, None, None
