import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import colormaps


def plot_explanation(objective, distance_from_x, best_sol, best_index,
                     first_feasible, first_feasible_index, explanation_type,
                     optimalities, x_init, pred_model, w_init, w_alt, new_x,
                     lambdas=None, task = "warcraft", plot_style = "normal", use_vae=True):
    """
    Plots the result of counterfactual optimization with CF-OPT.
    """
    if task=="warcraft" and not use_vae:
        _, axs = plt.subplots(1, 3, figsize=(12, 4))

        # Initial map
        initial_map_array = x_init.permute(1, 2, 0).cpu().numpy()
        initial_path_array = (torch.nn.Unflatten(0, (12, 12))(w_init)).cpu().numpy()
        initial_masked_path = np.ma.masked_where(initial_path_array == 0, initial_path_array)
        initial_map_array = np.flipud(initial_map_array)
        path_extent = [0, 96, 0, 96]

        axs[0].axis('off')
        axs[0].imshow(initial_map_array)
        axs[0].imshow(initial_masked_path, cmap=mcolors.ListedColormap(['yellow']), interpolation='none', alpha=0.3, extent=path_extent)
        axs[0].set_title('Initial Map and Shortest Path')

        if best_sol is None:
            best_sol = new_x

        # Counterfactual map
        counterfactual_map_array = best_sol.squeeze().detach().permute(1, 2, 0).cpu().numpy()
        counterfactual_path_array = (torch.nn.Unflatten(0, (12, 12))(w_alt)).cpu().numpy()
        counterfactual_masked_path = np.ma.masked_where(counterfactual_path_array == 0, counterfactual_path_array)
        counterfactual_map_array = np.flipud(counterfactual_map_array)
        path_extent = [0, 96, 0, 96]
        axs[1].axis('off')
        axs[1].imshow(counterfactual_map_array)
        axs[1].imshow(counterfactual_masked_path, cmap=mcolors.ListedColormap(['yellow']), interpolation='none', alpha=0.3, extent=path_extent)
        axs[1].set_title('Counterfactual Map and Shortest Path')

        #Magnified difference
        diff = (10*(best_sol.squeeze()-x_init)).detach().permute(1, 2, 0).cpu().numpy()
        axs[2].axis('off')
        axs[2].imshow(diff)
        axs[2].set_title("Magnified Difference (x10)")
        plt.tight_layout()
        plt.show()
        return


    if task == "warcraft" and plot_style == "superposed":
        _, axs = plt.subplots(3, 2, figsize=(8, 12))

        # Distance from x_init
        if objective == "latent":
            adjective = "Latent"
        elif objective == "feature":
            adjective = "Feature"

        axs[0, 0].set_title(adjective + ' Space Distance From Initial Point')

        axs[0, 0].plot(distance_from_x, color='blue')

        if best_sol is not None:
            axs[0, 0].axvline(x=best_index,
                                linestyle='solid', color='pink')
            axs[0, 0].text(best_index + 5,  0.97*axs[0, 0].get_ylim()[1], 'Best Solution', rotation=90, color='red', verticalalignment='top', fontsize=8)

        if not first_feasible:
            axs[0, 0].axvline(x=first_feasible_index,
                                linestyle='dotted', color='pink')
            axs[0, 0].text(first_feasible_index - 5, 0.97*axs[0, 0].get_ylim()[1], 'First Feasible Solution', rotation=90, color='red', verticalalignment='top', horizontalalignment = "right", fontsize=8)

        # graphique 2 - optimality constraint
        if explanation_type == "relative":
            name = "Relative "
            adjective = "Better"
            axs[0, 1].plot(optimalities, color='red')

        elif explanation_type == "absolute":
            name = "Absolute "
            adjective = "Optimal"
            axs[0, 1].plot(optimalities, color='red')

        elif explanation_type == "epsilon":
            name = "Epsilon-"
            adjective = "Epsilon-Better"
            axs[0, 1].plot(optimalities, color='red')

        axs[0, 1].axhline(y=0, linestyle='--', color='green')

        if best_sol is not None:
            axs[0, 1].axvline(x=best_index, linestyle='solid', color='pink')
            axs[0, 1].text(best_index + 5,  0.97*axs[0, 1].get_ylim()[1], 'Best Solution', rotation=90, color='red', verticalalignment='top', fontsize=8)

        if not first_feasible:
            axs[0, 1].axvline(x=first_feasible_index,
                                linestyle='dotted', color='pink')
            axs[0, 1].text(first_feasible_index - 5,  0.97*axs[0, 1].get_ylim()[1], 'First Feasible Solution', rotation=90, color='red', verticalalignment='top', horizontalalignment = "right", fontsize=8)

        axs[0, 1].set_title(name + 'Optimality Constraint')

        # Initial map
        initial_map_array = x_init.permute(1, 2, 0).cpu().numpy()
        initial_path_array = (torch.nn.Unflatten(0, (12, 12))(w_init)).cpu().numpy()
        initial_masked_path = np.ma.masked_where(initial_path_array == 0, initial_path_array)
        initial_map_array = np.flipud(initial_map_array)
        path_extent = [0, 96, 0, 96]

        axs[1, 0].axis('off')
        axs[1, 0].imshow(initial_map_array)
        axs[1, 0].imshow(initial_masked_path, cmap=mcolors.ListedColormap(['yellow']), interpolation='none', alpha=0.3, extent=path_extent)
        axs[1, 0].set_title('(a) Initial Map and Shortest Path', size = 14)
        # Initial predicted costs
        axs[2, 0].axis('off')
        axs[2, 0].imshow((torch.nn.Unflatten(0, (12, 12))(
            pred_model(x_init.unsqueeze(0)).squeeze().detach())).cpu().numpy())
        axs[2, 0].set_title('Initial Predicted Costs')

        if best_sol is None:
            best_sol = new_x

        # Counterfactual map
        counterfactual_map_array = best_sol.squeeze().detach().permute(1, 2, 0).cpu().numpy()
        counterfactual_path_array = (torch.nn.Unflatten(0, (12, 12))(w_alt)).cpu().numpy()
        counterfactual_masked_path = np.ma.masked_where(counterfactual_path_array == 0, counterfactual_path_array)
        counterfactual_map_array = np.flipud(counterfactual_map_array)
        path_extent = [0, 96, 0, 96]
        axs[1, 1].axis('off')
        axs[1, 1].imshow(counterfactual_map_array)
        axs[1, 1].imshow(counterfactual_masked_path, cmap=mcolors.ListedColormap(['yellow']), interpolation='none', alpha=0.3, extent=path_extent)
        axs[1, 1].set_title('(b) Counterfactual Map and Shortest Path', size = 14)
        # New predicted costs
        axs[2, 1].axis('off')
        axs[2, 1].imshow((torch.nn.Unflatten(0, (12, 12))(
            pred_model(best_sol).squeeze().detach())).cpu().numpy())
        axs[2, 1].set_title('New Predicted Costs')
        plt.tight_layout()
        plt.show()
        return
    
    if task=="warcraft": _, axs = plt.subplots(3, 3, figsize=(12, 12))
    elif task== "grid": _, axs = plt.subplots(1, 3, figsize=(12, 4))

    # Distance from x_init
    if objective == "latent":
        adjective = "Latent"
    elif objective == "feature":
        adjective = "Feature"

    if task=="warcraft": axs[0, 0].set_title(adjective + ' Space Distance From Initial Point')
    elif task=="grid": axs[0].set_title(adjective + ' Space Distance From Initial Point')

    if task=="warcraft": axs[0, 0].plot(distance_from_x, color='blue')
    elif task=="grid": axs[0].plot(distance_from_x, color='blue')

    if best_sol is not None:
        if task=="warcraft": axs[0, 0].axvline(x=best_index,
                          linestyle='solid', color='pink')
        elif task=="grid": axs[0].axvline(x=best_index,
                          linestyle='solid', color='pink')
    if not first_feasible:
        if task=="warcraft": axs[0, 0].axvline(x=first_feasible_index,
                          linestyle='dotted', color='pink')
        elif task=="grid": axs[0].axvline(x=first_feasible_index,
                          linestyle='dotted', color='pink')

    # graphique 2 - optimality constraint
    if explanation_type == "relative":
        name = "Relative "
        adjective = "Better"
        if task=="warcraft": axs[0, 1].plot(optimalities, color='red')
        elif task=="grid": axs[1].plot(optimalities, color='red')

    elif explanation_type == "absolute":
        name = "Absolute "
        adjective = "Optimal"
        if task=="warcraft": axs[0, 1].plot(optimalities, color='red')
        elif task=="grid": axs[1].plot(optimalities, color='red')

    elif explanation_type == "epsilon":
        name = "Epsilon-"
        adjective = "Epsilon-Better"
        if task=="warcraft": axs[0, 1].plot(optimalities, color='red')
        elif task=="grid": axs[1].plot(optimalities, color='red')

    if task=="warcraft": axs[0, 1].axhline(y=0, linestyle='--', color='green')
    elif task=="grid": axs[1].axhline(y=0, linestyle='--', color='green')

    if best_sol is not None:
        if task=="warcraft": axs[0, 1].axvline(x=best_index, linestyle='solid', color='pink')
        elif task=="grid": axs[1].axvline(x=best_index, linestyle='solid', color='pink')

    if not first_feasible:
        if task=="warcraft": axs[0, 1].axvline(x=first_feasible_index,
                          linestyle='dotted', color='pink')
        elif task=="grid": axs[1].axvline(x=first_feasible_index,
                          linestyle='dotted', color='pink')
            
    if task=="warcraft": axs[0, 1].set_title(name + 'Optimality Constraint')
    elif task=="grid": axs[1].set_title(name + 'Optimality Constraint')

    # Lambdas

    if task=="warcraft": axs[0, 2].plot(lambdas, color='brown')
    elif task=="grid": axs[2].plot(lambdas, color='brown')

    if best_sol is not None:
        if task=="warcraft": axs[0, 2].axvline(x=best_index, linestyle='solid', color='pink')
        elif task=="grid": axs[2].axvline(x=best_index, linestyle='solid', color='pink')

    if not first_feasible:
        if task=="warcraft": axs[0, 2].axvline(x=first_feasible_index,
                            linestyle='dotted', color='pink')
        elif task=="grid": axs[2].axvline(x=first_feasible_index,
                            linestyle='dotted', color='pink')
            
    if task=="warcraft": axs[0, 2].set_title("Lambda")
    elif task=="grid": axs[2].set_title("Lambda")

    if task=="warcraft":

        # Initial map
        axs[1, 0].axis('off')
        axs[1, 0].imshow(x_init.permute(1, 2, 0).cpu().numpy())
        axs[1, 0].set_title('Initial Map')
        # Initial predicted costs
        axs[1, 1].axis('off')
        axs[1, 1].imshow((torch.nn.Unflatten(0, (12, 12))(
            pred_model(x_init.unsqueeze(0)).squeeze().detach())).cpu().numpy())
        axs[1, 1].set_title('Initial Predicted Costs')
        # Initial optimal path
        axs[1, 2].axis('off')
        axs[1, 2].imshow((torch.nn.Unflatten(0, (12, 12))(w_init)).cpu().numpy())
        axs[1, 2].set_title('Initial Optimal Path')
        if best_sol is not None:
            # Counterfactual map
            axs[2, 0].axis('off')
            axs[2, 0].imshow(best_sol.squeeze().detach().permute(
                1, 2, 0).cpu().numpy())
            axs[2, 0].set_title('Counterfactual Map')
            # New predicted costs
            axs[2, 1].axis('off')
            axs[2, 1].imshow((torch.nn.Unflatten(0, (12, 12))(
                pred_model(best_sol).squeeze().detach())).cpu().numpy())
            axs[2, 1].set_title('New Predicted Costs')
        else:
            # Counterfactual map
            axs[2, 0].axis('off')
            axs[2, 0].imshow(new_x.squeeze().detach().permute(
                1, 2, 0).cpu().numpy())
            axs[2, 0].set_title('Counterfactual Map')
            # New predicted costs
            axs[2, 1].axis('off')
            axs[2, 1].imshow((torch.nn.Unflatten(0, (12, 12))(
                pred_model(new_x).squeeze().detach())).cpu().numpy())
            axs[2, 1].set_title('New Predicted Costs')
        # New path
        axs[2, 2].axis('off')
        axs[2, 2].imshow((torch.nn.Unflatten(0, (12, 12))(w_alt)).cpu().numpy())
        axs[2, 2].set_title('New ' + adjective + ' Path')
    plt.tight_layout()
    plt.show()


def plot_sol_knapsack(m, c, w, weights, caps, caption=None, cp=None):
    # colors
    cmap = colormaps["plasma"](np.linspace(0, 1, m))
    # get list
    sol, val, cap1, cap2, cpred = [], [], [], [], []
    for i in range(m):
        sol.append(3)
        val.append(c[i] * w[i])
        cap1.append(weights[0,i] * w[i])
        cap2.append(weights[1,i] * w[i])
        if cp is not None:
          cpred.append(cp[i] * w[i])
    # init fig
    fig = plt.figure(figsize=(8,4))
    plt.gca().invert_yaxis()
    acc = [0, 0, 0, 0]
    # bar plot
    if cp is None:
      for i in range(m):
          bar = [sol[i], val[i], cap1[i], cap2[i]]
          plt.barh(range(4), bar, left=acc, color=cmap[i], height=0.75, edgecolor="w", linewidth=2)
          # not selected
          if not int(w[i]):
              # grey color
              bar = [sol[i], 0, 0, 0]
              plt.barh(range(4), bar, left=acc, color=cmap[i], height=0.75, edgecolor="w", linewidth=2)
              plt.barh(range(4), bar, left=acc, color="lightgrey", height=0.75, edgecolor="w", linewidth=2, alpha=0.9)
          acc = [acc[0]+sol[i], acc[1]+val[i], acc[2]+cap1[i], acc[3]+cap2[i]]
    else:
      acc = [0, 0, 0, 0, 0]
      for i in range(m):
          bar = [sol[i], val[i], cpred[i], cap1[i], cap2[i]]
          plt.barh(range(5), bar, left=acc, color=cmap[i], height=0.75, edgecolor="w", linewidth=2)
          # not selected
          if not int(w[i]):
              # grey color
              bar = [sol[i], 0, 0, 0, 0]
              plt.barh(range(5), bar, left=acc, color=cmap[i], height=0.75, edgecolor="w", linewidth=2)
              plt.barh(range(5), bar, left=acc, color="lightgrey", height=0.75, edgecolor="w", linewidth=2, alpha=0.9)
          acc = [acc[0]+sol[i], acc[1]+val[i], acc[2]+cpred[i], acc[3]+cap1[i], acc[4]+cap2[i]]
    # total value
    tval = sum(val)
    plt.text(tval+0.5, 1.1,  "%.2f"%tval, fontsize=12)
    # vertical line
    if cp is None:
      plt.axvline(x=caps[0], ymin=0.27, ymax=0.48, color="firebrick", linewidth=1.5)
      plt.text(caps[0]+0.5, 2.1, "Capacity 1", fontsize=12, color="firebrick")
      plt.axvline(x=caps[1], ymin=0.03, ymax=0.24, color="firebrick", linewidth=1.5)
      plt.text(caps[1]+0.5, 3.1, "Capacity 2", fontsize=12, color="firebrick")
    else:
      # total predicted value
      tpredval = sum(cpred)
      plt.text(tpredval+0.5, 2.1, "%.2f"%tpredval, fontsize=12)
      plt.axvline(x=caps[0], ymin=0.23, ymax=0.38, color="firebrick", linewidth=1.5)
      plt.text(caps[0]+0.5, 3.1, "Capacity 1", fontsize=12, color="firebrick")
      plt.axvline(x=caps[1], ymin=0.03, ymax=0.2, color="firebrick", linewidth=1.5)
      plt.text(caps[1]+0.5, 4.1, "Capacity 2", fontsize=12, color="firebrick")
    # labels and ticks
    plt.xticks([])
    bar_labels = ["Items Selection", "Items Predicted Value", "Resource 1", "Resource 2"] if cp is None else ["Items Selection", "Items True Value", "Items Predicted Value", "Resource 1", "Resource 2"]
    plt.yticks(range(len(bar_labels)), bar_labels, fontsize=16)
    plt.minorticks_off()
    plt.tick_params(axis='both', length=0)
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["bottom"].set_visible(False)
    plt.title(caption, fontsize=18)
    plt.show()


def plot_explanation_knapsack(m, c_init, w_init, c_alt, w_alt, explanation_type, objective, distance_from_x, best_sol, best_index, first_feasible, first_feasible_index, optimalities, lambdas, weights, caps):
    plot_sol_knapsack(m, c_init, w_init, weights, caps, caption="Initial context and solution")
    plot_sol_knapsack(m, c_alt, w_alt, weights, caps, caption="Alternative context and solution")

    _, axs = plt.subplots(1, 3, figsize=(12, 4))

    # Distance from x_init
    if objective == "latent":
        adjective = "Latent"
    elif objective == "feature":
        adjective = "Feature"

    axs[0].set_title(adjective + ' Space Distance From Initial Point')

    axs[0].plot(distance_from_x, color='blue')

    if best_sol is not None:
        axs[0].axvline(x=best_index,
                          linestyle='solid', color='pink')
    if not first_feasible:
        axs[0].axvline(x=first_feasible_index,
                          linestyle='dotted', color='pink')

    # graphique 2 - optimality constraint
    if explanation_type == "relative":
        name = "Relative "
        adjective = "Better"
        axs[1].plot(optimalities, color='red')

    elif explanation_type == "absolute":
        name = "Absolute "
        adjective = "Optimal"
        axs[1].plot(optimalities, color='red')

    elif explanation_type == "epsilon":
        name = "Epsilon-"
        adjective = "Epsilon-Better"
        axs[1].plot(optimalities, color='red')

    axs[1].axhline(y=0, linestyle='--', color='green')

    if best_sol is not None:
        axs[1].axvline(x=best_index, linestyle='solid', color='pink')

    if not first_feasible:
        axs[1].axvline(x=first_feasible_index,
                          linestyle='dotted', color='pink')
            
    axs[1].set_title(name + 'Optimality Constraint')

    # Lambdas

    axs[2].plot(lambdas, color='brown')

    if best_sol is not None:
        axs[2].axvline(x=best_index, linestyle='solid', color='pink')

    if not first_feasible:
        axs[2].axvline(x=first_feasible_index,
                            linestyle='dotted', color='pink')
            
    axs[2].set_title("Lambda")

    plt.tight_layout()
    plt.show()