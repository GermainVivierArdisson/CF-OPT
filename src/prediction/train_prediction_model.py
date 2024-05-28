import datetime
import torch
import torch.nn as nn
import pyepo
from tqdm import tqdm
from src.prediction.prediction_models import PartialResNet, LinearRegression, LinearRegression_KnapSack
from torch.utils.tensorboard import SummaryWriter


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def train_prediction_model(predModel, optModel, trainloader, testloader,
                           start_epoch, end_epoch, learning_rate,
                           device="cpu", use_spo_loss=False, write_tb=False,
                           save_frequency = 5, test_frequency = 5,
                           log_directory='logs/', save_path='temp_saves/', use_early_stopping = False, patience = 3, min_delta = 0):
    """Train the prediction model of a structured learning pipeline. 
    The training loss can be the SPO+ loss (end-to-end training)
    or the standard MSE loss.

    Args:
        predModel (nn.Module): prediction model used.
        optModel (ShortestPathModel): optimization model used.
        trainloader (DataLoader): pytorch DataLoader containing training set.
        testloader (DataLoader): pytorch DataLoader containing testing set.
        start_epoch (int): Index of the first epoch of the training process (useful if training an already trained model for other epochs, set to 0 else).
        end_epoch (int): Index of the last epoch of the training process (termination criterion of the process).
        learning_rate (float): Learning rate.
        device (str, optional): Device used by pytorch for training. Defaults to "cpu".
        use_spo_loss (bool, optional): Use an SPO loss for training. Defaults to False.
        write_tb (bool, optional): Write to tensorboard. Defaults to False.
        save_frequency (int, optional): Number of epochs between each save of the model. Defaults to 5. Input 0 to never save weights.
        log_directory (str, optional): Log directory for tensorboard. Defaults to 'logs/'.
        save_path (str, optional): Saving path for the models. Defaults to 'temp_saves/'.
        use_early_stopping (bool, optional): Use early stopping. Defaults to False.
        patience (int, optional): Number of non-improving epochs if using early stopping. Defaults to 3.
        min_delta (float, optional): Maximal non-improvement if using early stopping. Defaults to 0.
    """

    current_datetime = datetime.datetime.now().strftime("%m-%d_%H-%M")

    if write_tb:
        writer = SummaryWriter(log_dir=log_directory+current_datetime)

    # Set up the optimizer
    optimizer = torch.optim.Adam(predModel.parameters(), lr=learning_rate)
    if use_spo_loss:
        # Set up loss
        spoploss = pyepo.func.SPOPlus(optModel, processes=1)

    # Set up early stopper
    if use_early_stopping:
        early_stopper = EarlyStopper(patience, min_delta)

    # Move the model to the device
    predModel.to(device)

    # Train the model
    for epoch in range(start_epoch, end_epoch):
        print('Epoch {} out of {}.'.format(epoch+1, end_epoch))
        train_loss = 0
        for _, data in tqdm(enumerate(trainloader)):
            # Get the inputs and move them to the device
            x, c, w, z = data
            x, c, w, z = x.to(device), c.to(device), w.to(device), z.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass: predicted cost
            cp = predModel(x)
            if use_spo_loss:
                # Compute SPO+ loss
                loss = spoploss(cp, c, w, z).mean()
            else:
                # Compute MSE loss
                loss = nn.MSELoss(reduction="mean")(cp, c)
            train_loss += loss.cpu().item()

            # Backward pass and gradient step
            loss.backward()
            optimizer.step()

        # Store training loss in SummaryWRiter
        if write_tb:
            writer.add_scalar("Loss Train", train_loss, epoch)
            

        # - Save model parameters -
        if save_frequency != 0:
            if (epoch+1) % save_frequency == 0:
                
                if use_spo_loss:
                    if isinstance(predModel, PartialResNet):
                        torch.save(predModel.state_dict(),
                                save_path+"CNN_SPO_epoch_{}.pth".format(epoch+1))
                    elif isinstance(predModel, LinearRegression) :
                        torch.save(predModel.state_dict(),
                                save_path+"LinReg_SPO_epoch_{}.pth".format(epoch+1))
                    elif isinstance(predModel, LinearRegression_KnapSack) :
                        torch.save(predModel.state_dict(),
                                save_path+"LinReg_KnapSack_SPO_epoch_{}.pth".format(epoch+1))
                else:
                    if isinstance(predModel, PartialResNet):
                        torch.save(predModel.state_dict(),
                                save_path+"CNN_MSE_epoch_{}.pth".format(epoch+1))
                    elif isinstance(predModel, LinearRegression):
                        torch.save(predModel.state_dict(),
                                save_path+"LinReg_MSE_epoch_{}.pth".format(epoch+1))
                    elif isinstance(predModel, LinearRegression_KnapSack):
                        torch.save(predModel.state_dict(),
                                save_path+"LinReg_KnapSack_MSE_epoch_{}.pth".format(epoch+1))
                    
        
        # - Evaluate test loss -
        if test_frequency != 0:
            if (epoch+1) % test_frequency == 0:
                
                with torch.no_grad():
                    print('Evaluating test loss...')
                    test_loss = 0
                    for _, test_data in enumerate(testloader):
                        # Get the test inputs and move them to the device
                        x_test, c_test, w_test, z_test = test_data
                        x_test, c_test, w_test, z_test = x_test.to(device), c_test.to(device), w_test.to(device), z_test.to(device)

                        cp_test = predModel(x_test)
                        if use_spo_loss:
                            # Compute SPO+ loss
                            test_loss += spoploss(cp_test, c_test, w_test, z_test).mean()
                        else:
                            # Compute MSE loss
                            test_loss += nn.MSELoss(reduction="mean")(cp_test, c_test)

                if write_tb: 

                    # Store epoch loss in SummaryWriter
                    writer.add_scalar("Loss Test", test_loss.cpu().item(), epoch)

                if use_early_stopping:
                    if early_stopper.early_stop(test_loss.cpu().item()):             
                        break

    print('Finished training')
