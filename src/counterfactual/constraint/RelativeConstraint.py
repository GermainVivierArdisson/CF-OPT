import torch

class RelativeConstraint(torch.autograd.Function):
    """
    Computes the feasibility of the explanation constraint for relative explanations.
    """

    @staticmethod
    def forward(ctx, cp, w_init, w_alt, task):
        ctx.save_for_backward(w_init, w_alt)
        ctx.task = task
        if task == "knapsack":
            return torch.dot(cp, w_init-w_alt) # Opposite constraint as the optimization problem is formalized as a Maximization problem.
        return torch.dot(cp, w_alt-w_init)

    @staticmethod
    def backward(ctx, grad_output):
        w_init, w_alt = ctx.saved_tensors
        if ctx.task == "knapsack":
            return grad_output*(w_init-w_alt), None, None, None # Opposite gradient as the optimization problem is formalized as a Maximization problem.
        return grad_output*(w_alt-w_init), None, None, None
