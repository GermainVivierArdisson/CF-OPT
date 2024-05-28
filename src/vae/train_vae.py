import datetime
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


def vae_loss(x_recon, x, mu, logvar):
    """
    Traditional VAE loss: sum of reconstruction loss and
    Kullback-Leibler divergence.
    Args:
        x_recon (PyTorch tensor): reconstructed input.
        x (PyTorch tensor): input before reconstruction.
        mu (PyTorch tensor): mean of the encoded distribution.
        logvar (PyTorch tensor): log variance of the encoded distribution.
    """
    recon_loss = nn.MSELoss(reduction="mean")(x_recon, x)
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss, kl_div


def cost_aware_vae_loss(cnn, x_recon, x, mu, logvar, alpha=0.1):
    """
    Cost-aware loss: adds a penalty to loss function
    proportional to the error of the cost prediction before
    and after reconstructing the image. If alpha is 0, does not use cnn.
    Args:
        cnn (nn.module): CNN used to compute cost-aware regularization.
        x_recon (PyTorch tensor): reconstructed input.
        x (PyTorch tensor): input before reconstruction.
        mu (PyTorch tensor): mean of the encoded distribution.
        logvar (PyTorch tensor): log variance of the encoded distribution.
        alpha (float): weight of the cost-aware regularization.
    """
    recon_loss, kl_div = vae_loss(x_recon, x, mu, logvar)

    if alpha == 0:
        return recon_loss, kl_div, 0

    cost_loss = nn.MSELoss(reduction="mean")(cnn(x_recon), cnn(x))

    return recon_loss, kl_div, alpha*cost_loss

def train_cost_aware_vae(model, cnn, trainloader, testloader, start_epoch,
                            end_epoch, learning_rate,
                            device="cpu", write_tb=False, alpha=0.1,
                            save_frequency = 5,
                            log_directory=None, save_path=None):
    """
    Train a VAE model for CF-OPT.
    Args:
        model (nn.module): VAE to be trained.
        cnn (nn.module): CNN used to compute cost-aware loss.
        trainloader (DataLoader): pytorch DataLoader containing training set.
        testloader (DataLoader): pytorch DataLoader containing testing set.
        start_epoch (int): Index of the first epoch of the training process (useful if training an already trained model for other epochs, set to 0 else).
        end_epoch (int): Index of the last epoch of the training process (termination criterion of the process).
        learning_rate (float): Learning rate.
        device (str, optional): Device used by pytorch for training. Defaults to "cpu".
        write_tb (bool, optional): Write to tensorboard. Defaults to False.
        alpha (float): weight of the cost-aware regularization.
        save_frequency (int, optional): Number of epochs between each save of the model. Defaults to 5. Input 0 to never save weights.
        log_directory (str, optional): Log directory for tensorboard. Defaults to 'logs/'.
        save_path (str, optional): Saving path for the models. Defaults to 'temp_saves/'.
    """

    current_datetime = datetime.datetime.now().strftime("%m-%d_%H-%M")

    if write_tb:
        writer = SummaryWriter(log_dir=log_directory+current_datetime)

    # Set up the optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Move the models to the device
    model.to(device)
    cnn.to(device)

    epoch_recon_loss = 0
    epoch_kl = 0
    epoch_cost_loss = 0
    epoch_total_loss = 0

    # Train the model
    for epoch in range(start_epoch, end_epoch):

        print('Epoch {} out of {}.'.format(epoch+1, end_epoch))

        for _, data in enumerate(trainloader):

            # Get the inputs and move them to the device
            inputs, _ = data
            inputs = inputs.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs, mu, logvar = model(inputs)

            # Compute the loss
            r_loss, k_loss, c_loss = cost_aware_vae_loss(
                cnn, outputs, inputs, mu, logvar, alpha=alpha)

            epoch_recon_loss += r_loss.cpu().item()
            epoch_kl += k_loss.cpu().item()
            epoch_cost_loss += c_loss.cpu().item()

            loss = r_loss + k_loss + c_loss

            epoch_total_loss += loss.cpu().item()

            # Backward pass
            loss.backward()

            # Update the parameters
            optimizer.step()

        if write_tb :

            writer.add_scalar("Reconstruction Train", epoch_recon_loss, epoch)
            writer.add_scalar("KL Divergence Train", epoch_kl, epoch)
            writer.add_scalar("Cost Stability Train", epoch_cost_loss, epoch)
            writer.add_scalar("Total Loss Train", epoch_total_loss, epoch)
            
        # - Save model parameters -
        if (epoch+1) % save_frequency == 0:
            torch.save(model.state_dict(),
                    save_path + "_VAE_{}_cost_{}_epoch_{}.pth".format(
                        model.latent_dim, alpha, epoch+1))
            
            # - Evaluate test loss -
            if write_tb:

                with torch.no_grad():
                    test_r_loss = 0
                    test_k_loss = 0
                    test_c_loss = 0
                    test_total_loss = 0
                    for _, test_data in enumerate(testloader):


                        # Get the test inputs and move them to the device
                        test_inputs, _ = test_data
                        test_inputs = test_inputs.to(device)

                        # Forward pass
                        test_outputs, test_mu, test_logvar = model(test_inputs)

                        # Compute the loss
                        (test_r_loss_batch, test_k_loss_batch,
                         test_c_loss_batch) = cost_aware_vae_loss(
                            test_outputs, test_inputs, test_mu, test_logvar)

                        test_total_loss_batch = test_r_loss+test_k_loss+test_c_loss

                        test_r_loss += test_r_loss_batch.cpu().item()
                        test_k_loss += test_k_loss_batch.cpu().item()
                        test_c_loss += test_c_loss_batch.cpu().item()
                        test_total_loss += test_total_loss_batch.cpu().item()

                    # Write loss in SummaryWriter
                    writer.add_scalar("Reconstruction Test", test_r_loss, epoch)
                    writer.add_scalar("KL Divergence Test", test_k_loss, epoch)
                    writer.add_scalar("Cost Stability Test", test_c_loss, epoch)
                    writer.add_scalar( "Total Loss Test", test_total_loss, epoch)

    print('Finished training')
