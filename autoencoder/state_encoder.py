import torch
import matplotlib.pyplot as plt
from pprint import pp
# create an auto encoder 
# shape N - state_vars -> N/2 -> [N/4 (encoded state)] -> N/2 -> N - state_vars 
# step find examples of SLAs 
# make an SLA table and generate more data that is like the other SLAs 


#state table
general_state_var = [
    "n_mmtc",
    "n_embb",
    "ave_users_mmtc",
    "ave_cbr",
    "ave_vbr",
    "n_remaining_slices"
]

embb_state_var = ['cbr_traffic','cbr_th', 'cbr_prb',
                  'cbr_queue', 'cbr_snr', 'vbr_traffic',
                  'vbr_th', 'vbr_prb', 'vbr_queue', 'vbr_snr']

mmtc_state_var = ['devices', 'avg_rep', 'delay']

# Creating a PyTorch class
# 28*28 ==> 9 ==> 28*28
class AE(torch.nn.Module):
    """
        Autoencoder

    Args:
        torch : module base class
    """
    def __init__(self, dim):
        super().__init__()
         
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(dim, 10),
            torch.nn.Linear(10, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 6),
            torch.nn.ReLU(),
            torch.nn.Linear(6, 3)
        )
         
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(3, 6),
            torch.nn.ReLU(),
            torch.nn.Linear(6, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 10),
            torch.nn.Linear(10, dim),
            torch.nn.Sigmoid()
        )
 
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class Train():
    """
        Model training class
    """
    def __init__(self, dim):
        # Model Initialization
        self.model = AE(dim)
        
        # Validation using MSE Loss function
        self.loss_function = torch.nn.MSELoss()
        
        # Using an Adam Optimizer with lr = 0.1
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                    lr = 1e-1,
                                    weight_decay = 1e-8)
    
    def train(self,epochs, datas):
        """
            Training function

        Args:
            epochs (int): _description_
            datas (list[float]): _description_
        """
        outputs = []
        losses = []
        for epochs in range(epochs):
            for data in datas:
                # Output of Autoencoder
                reconstructed = self.model(data)
                
                # Calculating the loss function
                loss = self.loss_function(reconstructed, data)
                
                # The gradients are set to zero,
                # the gradient is computed and stored.
                # .step() performs parameter update
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # Storing the losses in a list for plotting
                losses.append(loss)
                outputs.append((epochs, data, reconstructed))
        
        return outputs, losses


if __name__ == "__main__":
    data = [[10e6, 20, 10e4, 15e6, 30, 15e4],
            [9e6, 20, 10e4, 15e6, 28, 14e4],
            [11e6, 13, 10e4, 15e6, 37, 14e4],
            [14e6, 15, 10e4, 15e6, 15, 13e4],
            [8e6, 20, 10e4, 15e6, 20, 17e4],
            [10e6, 22, 10e4, 15e6, 35, 19e4]            
            ]
    tensor1 = torch.FloatTensor(data)
    autoencoder_trainer = Train(len(data[0]))
    output, losses = autoencoder_trainer.train(10, tensor1)
    pp(losses)
