import datetime
import mdmm
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import gurobipy as gp
from src.optimization.KnapSackModel import knapsackModel
from src.counterfactual.constraint.RelativeConstraint import RelativeConstraint
from src.counterfactual.constraint.AbsoluteConstraint import AbsoluteConstraint
from src.counterfactual.constraint.EpsilonConstraint import EpsilonConstraint
from src.counterfactual.solve import solve
from src.plot.plot_explanation import plot_explanation_knapsack


def get_counterfactual_knapsack(m, weights, caps, pred_model, dataset, initial_index,
                       alternative_index, vae=None, explanation_type="relative",
                       epsilon=0.1, objective="feature",
                       max_countdown=100, max_iter=6000, step_size=0.003,
                       update_tol=0.9, plausibility_reg = "sphere", beta = 10,
                       plot_result=True, write_tb=False,
                       log_directory='logs/', return_sol = False, return_value_feasibility_and_nsteps = False):
    """
    Returns the result of the counterfactual optimization.
    We use the package found at "https://github.com/crowsonkb/mdmm/blob/master/" for simplicity.

    Args:
      m (int): number of items.
      weights (array of floats): weights associated to the knapsack task.
      caps (tuple of int): capacity of the knapsack.
      pred_model (nn.module): prediction model of the structured prediction pipeline to be explained.
      dataset (PyTorch dataset): dataset from which the initial context and alternative solution will be taken from.
      initial_index (int): index of the test dataset decision to explain.
      alternative_index (int): index of the test dataset sample from which
                               the decision is the alternative decision in
                               the counterfactual explanation setting.
      vae (nn.module): VAE model used to generate the explanation.
      explanation_type (str): either 'relative', 'absolute', or 'epsilon'.
                              Indicates if the counterfactual
                              explanation must be relative, absolute, or an epsilon-explanation.
      epsilon (float): epsilon in epsilon-explanation.
      objective (str): proximity metric used. 
                       Can be either 'latent', or 'feature'.
      max_countdown (int): maximum number of non-improving iterations.
      max_iter (int): maximum number of iterations.
      step_size (float): step size used for MDMM algorithm.
      update_tol (float): ratio used to define improving iterations (must be between 0 and 1).
      plausibility_reg (float): plausibility regularization of the optimization objective. 
                                Can be either 'sphere' for latent hypersphere regularization, 
                                'center' for log-likelihood regularization, 
                                'none' for no regularization.
      beta (float): weight of the plausibility regularization.
      plot_result (bool): indicates if the result must be plotted.
      write_tb (bool): indicates if optimization variables are written into tensorboard.
      log_directory (str): logging directory used for tensorboard.
      return_sol (bool): indicates if the result must be returned.
      return_value_feasibility_and_nsteps (bool): indicates if the number of iterations, loss and feasibility of the best solution must be returned.
    """

    task = "knapsack"

    torch.set_num_threads(1)

    if explanation_type not in ["relative", "absolute", "epsilon"]:
        raise ValueError('The Explanation type must be either'
                         ' \'relative\', \'absolute\', or \'epsilon\'.')
    
    if plausibility_reg not in ["sphere", "center", "none"]:
        raise ValueError('The plausibility regularization must be either'
                         ' \'sphere\', \'center\', or \'none\'.')
    
    if vae==None:
        use_vae = False
        plausibility_reg = "none"

    else:
        use_vae = True
    
    with gp.Env(empty=True) as env:
        env.setParam('OutputFlag', 0)
        env.start()

        with knapsackModel(weights = weights, capacity = caps, env=env) as knapsack_model:

            if use_vae: vae.eval()
            pred_model.eval()
            if use_vae: vae = vae.to("cpu")
            pred_model = pred_model.to("cpu")

            if objective=="latent" and not use_vae:
                raise ValueError("Input a VAE model in order to use a latent objective")

            if write_tb:
                current_datetime = datetime.datetime.now().strftime("%m-%d_%H-%M")
                writer = SummaryWriter(log_dir=log_directory
                                    + explanation_type + "_" + current_datetime)

            x_init = dataset[initial_index][0]
            with torch.no_grad():
                c_init = pred_model(x_init)
            w_init, _ = solve(c_init, knapsack_model, task)

            x_init = x_init.to("cpu")
            w_init = w_init.to("cpu")

            if use_vae: latent_x_init = vae.encoder(x_init.unsqueeze(0))[0].squeeze()

            if explanation_type in ["relative", "absolute"]:
                x_alt = dataset[alternative_index][0]
                with torch.no_grad():
                    c_alt = pred_model(x_alt)
                w_alt, _ = solve(c_alt, knapsack_model, task)
                
                x_alt = x_alt.to("cpu")
                w_alt = w_alt.to("cpu")
                c_alt = c_alt.to("cpu")

            relative_optimality_criterion = RelativeConstraint.apply
            absolute_optimality_criterion = AbsoluteConstraint.apply
            epsilon_optimality_criterion = EpsilonConstraint.apply

            opt_crit = 0

            # Defining dummy function in order to facilitate the use of mdmm package
            def fn():
                return opt_crit

            # Initializing the relative explanation at the starting point
            if use_vae: new_latent_x = torch.clone(latent_x_init.detach()).requires_grad_().to("cpu")
            else: new_x = torch.clone(x_init.unsqueeze(0).detach()).requires_grad_().to("cpu")

            # Plug the constraint into the adequate object and define the optimizer
            constraints = [mdmm.MaxConstraint(fn, 0)]

            mdmm_module = mdmm.MDMM(constraints)
            if use_vae: opt = mdmm_module.make_optimizer([new_latent_x],
                                            lr=step_size,
                                            optimizer=torch.optim.SGD)
            else: opt = mdmm_module.make_optimizer([new_x],
                                            lr=step_size,
                                            optimizer=torch.optim.SGD)

            # Defining the coherence bound
            if use_vae: latent_radius = vae.latent_dim**0.5

            # Optimization
            distance_from_x = []
            optimalities = []
            optimal_indexes = []
            lambdas = []
            best_loss = float('inf')
            checkpoint_loss = float('inf')
            best_feasibility = float('inf')
            best_feasibility_loss = float('inf')
            best_sol = None
            best_index = 0
            countdown = max_countdown
            counting_down = False
            first_feasible = True
            first_feasible_index = 0

            for n_steps in range(max_iter):

                if countdown == 0:
                    break
                if counting_down:
                    countdown -= 1

                lambdas.append(-constraints[0].lmbda.item())

                if use_vae: new_x = vae.decoder(new_latent_x)
                new_cp = pred_model(new_x)

                if objective == "latent":
                    latent_distance = torch.norm(new_latent_x-latent_x_init)**2
                    distance_from_x.append(latent_distance.item())
                    if write_tb:
                        writer.add_scalar("Latent Space Distance",
                                        latent_distance.item(), n_steps)
                        writer.add_scalar("Latent Space Norm",
                                        torch.norm(new_latent_x).item(), n_steps)
                    if plausibility_reg == "sphere": loss = latent_distance + beta*(torch.norm(new_latent_x)-latent_radius)**2
                    elif plausibility_reg == "center": loss = latent_distance + beta*(torch.norm(new_latent_x))**2
                    elif plausibility_reg == "none": loss = latent_distance


                elif objective == "feature":
                    distance = nn.MSELoss(reduction="sum")(new_x.squeeze(), x_init)
                    distance_from_x.append(distance.item())
                    if write_tb:
                        writer.add_scalar("Feature Space Distance",
                                        distance.item(), n_steps)
                        if use_vae: writer.add_scalar("Latent Space Norm",
                                        torch.norm(new_latent_x).item(), n_steps)

                    if plausibility_reg == "sphere": loss = distance + beta*(torch.norm(new_latent_x)-latent_radius)**2
                    elif plausibility_reg == "center": loss = distance + beta*(torch.norm(new_latent_x))**2
                    elif plausibility_reg == "none": loss = distance

                if explanation_type == "relative":
                    opt_crit = relative_optimality_criterion(new_cp.squeeze(), w_init, w_alt, task)
                elif explanation_type == "absolute":
                    opt_crit = absolute_optimality_criterion(new_cp.squeeze(), w_alt, knapsack_model, task)
                elif explanation_type == "epsilon":
                    opt_crit = epsilon_optimality_criterion(new_cp.squeeze(), w_init, epsilon, knapsack_model, task)

                optimalities.append(opt_crit.item())
                if write_tb:
                    writer.add_scalar(
                    explanation_type + " optimality", opt_crit.item(), n_steps)

                if first_feasible:
                    if opt_crit.item() < best_feasibility:
                        best_feasibility = opt_crit.item()
                        best_feasibility_loss = loss.item()
                        best_feasibility_sol = torch.clone(new_x)

                if opt_crit.item() <= 1e-5:
                    if first_feasible:
                        first_feasible_index = n_steps
                    first_feasible = False

                    counting_down = True

                    optimal_indexes.append(n_steps)

                    if loss.item() < best_loss:
                        best_loss = loss.item()
                        best_sol = torch.clone(new_x)
                        best_index = n_steps

                    if loss.item() < update_tol * checkpoint_loss:
                        countdown = max_countdown
                        checkpoint_loss = loss.item()

                mdmm_return = mdmm_module(loss)

                if not mdmm_return.value.isfinite():
                    break

                opt.zero_grad()
                mdmm_return.value.backward(retain_graph=True)
                opt.step()

            if plot_result:

                if explanation_type == "epsilon":
                    if best_sol is not None:
                        c_alt = pred_model(best_sol).squeeze()
                        w_alt, _ = solve(c_alt, knapsack_model, task)
                    else:
                        c_alt = c_init
                        w_alt = w_init

                plot_explanation_knapsack(m, c_init.cpu().detach().numpy(), w_init.cpu().detach().numpy(), 
                                          c_alt.cpu().detach().numpy(), w_alt.cpu().detach().numpy(), 
                                          explanation_type, objective, distance_from_x, best_sol, 
                                          best_index, first_feasible, first_feasible_index, 
                                          optimalities, lambdas, weights, caps)

            if best_loss == float('inf'):
                best_loss = best_feasibility_loss
            
            if return_sol and not return_value_feasibility_and_nsteps:
                if first_feasible:
                    return best_feasibility_sol
                return best_sol
            
            if return_value_feasibility_and_nsteps and not return_sol:
                if first_feasible:
                    return (best_feasibility_loss, best_feasibility, n_steps)
                return (best_loss, 0, n_steps)
            
            if return_sol and return_value_feasibility_and_nsteps:
                if first_feasible:
                    return (best_feasibility_sol, best_feasibility_loss, best_feasibility, n_steps)
                return (best_sol, best_loss, 0, n_steps)
            
            return (best_loss, n_steps, best_feasibility)