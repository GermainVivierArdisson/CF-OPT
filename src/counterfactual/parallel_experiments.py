import gurobipy as gp
import torch
import torch.nn as nn
from src.counterfactual.solve import solve
from src.optimization.ShortestPathModel import ShortestPathModel
from src.counterfactual.cf_opt import get_counterfactual_mdmm
import multiprocessing as mp
from multiprocessing.pool import ThreadPool
import numpy as np
import random as rd


def get_plausibility(vae, pred_model, counterfactual, shortest_path_model):

    with torch.no_grad():
        cp = pred_model(counterfactual)
        w, _ = solve(cp.squeeze(), shortest_path_model, task = "warcraft")

        latent_counterfactual = vae.encoder(counterfactual)[0]
        reconstructed_counterfactual = vae.decoder(latent_counterfactual)

        reconstructed_cp = pred_model(reconstructed_counterfactual)
        _, new_w_length = solve(reconstructed_cp.squeeze(), shortest_path_model, task = "warcraft")

        initial_w_length = torch.dot(w, reconstructed_cp.squeeze())

        reconstruction_error = nn.MSELoss()(reconstructed_counterfactual, counterfactual)

    return ((initial_w_length-new_w_length)/new_w_length, reconstruction_error)


def get_parallel_metrics(vae, pred_model, dataset, explanation_type, objective, epsilon, plausibility_reg, beta, step_size, validation_indices, vae_for_mdmm = None, metrics = ['stability', 'reconstruction', 'loss', 'feasibility', 'nsteps']):
    """
    Returns different metrics, with parallel computing.
    Args:
        vae (nn.module): VAE model used for computing metrics.
        pred_model (nn.module): prediction model of the structured learning pipeline to be explained.
        dataset (PyTorch dataset): dataset from which the samples used to compute metrics are taken.
        explanation_type (str): either 'relative', 'absolute', or 'epsilon'.
                                Indicates if the counterfactual
                                explanation must be relative, absolute, or an epsilon-explanation.
        objective (str): proximity metric used. 
                         Can be either 'latent', or 'feature'.
        epsilon (float): epsilon in epsilon-explanation (must be positive).
        plausibility_reg (float): plausibility regularization of the optimization objective. 
                                  Can be either 'sphere' for latent hypersphere regularization, 
                                  'center' for log-likelihood regularization, 
                                  'none' for no regularization.
        beta (float): weight of the plausibility regularization.
        step_size (float): step size used for MDMM algorithm.
        validation_indices (list of tuples of ints): indices of the samples used to compute metrics in dataset.
        vae_for_mdmm (nn.module): VAE model used to run MDMM. 
                                  If set to None, same as the one used for computing metrics.
        metrics (list of str): list of strings indicating the metrics to be computed.
                               Can contain 'stability', 'reconstruction', 'loss', 'feasibility', and 'nsteps'.
    """

    if vae_for_mdmm == 0:
        vae_for_mdmm = None

    elif vae_for_mdmm == None:
        vae_for_mdmm = vae
    
    def parallel_metrics(indices):

        xalt, loss, feasibility, nsteps = get_counterfactual_mdmm(vae=vae_for_mdmm, 
                                                                  pred_model=pred_model, 
                                                                  dataset=dataset, 
                                                                  initial_index=indices[0], 
                                                                  alternative_index=indices[1], 
                                                                  explanation_type=explanation_type, 
                                                                  objective=objective,
                                                                  epsilon = epsilon, 
                                                                  plausibility_reg=plausibility_reg,
                                                                  beta=beta,
                                                                  step_size=step_size, 
                                                                  plot_result=False, 
                                                                  return_sol = True,
                                                                  return_value_feasibility_and_nsteps = True,
                                                                  max_iter = 3000,
                                                                  max_countdown = 50)
        
        if 'stability' in metrics or 'reconstruction' in metrics:
        
            with gp.Env(empty=True) as env:
                env.setParam('OutputFlag', 0)
                env.start()
                with ShortestPathModel((12,12), env=env, task="warcraft") as short_path_model:

                    stability, reconstruction = get_plausibility(vae, pred_model, xalt, short_path_model)

        result = []
        if 'stability' in metrics:
            result.append(stability.item())
        else:
            result.append(0)
        if 'reconstruction' in metrics:
            result.append(reconstruction.item())
        else:
            result.append(0)
        if 'loss' in metrics:
            result.append(loss)
        else:
            result.append(0)
        if 'feasibility' in metrics:
            result.append(feasibility)
        else:
            result.append(0)
        if 'nsteps' in metrics:
            result.append(nsteps)
        else:
            result.append(0)
 
        return result[0], result[1], result[2], result[3],result[4]
                
    with ThreadPool(mp.cpu_count()) as pool:
        
        result = pool.map(parallel_metrics, validation_indices)

        pool.close()
        pool.join()

    return np.array(result)
