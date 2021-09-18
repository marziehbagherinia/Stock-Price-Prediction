import torch
from torch.nn import Module, LSTM, Linear
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

class Net(Module):
    def __init__(self, config):
        super(Net, self).__init__()
        self.lstm = LSTM(input_size = config.input_size, hidden_size = config.hidden_size,
                         num_layers = config.lstm_layers, batch_first = True, dropout = config.dropout_rate)
        self.linear = Linear(in_features = config.hidden_size, out_features = config.output_size)

    def forward(self, x, hidden = None):
        lstm_out, hidden = self.lstm(x, hidden)
        linear_out = self.linear(lstm_out)
        return linear_out, hidden

def train(config, train_and_valid_data):
    train_X, train_Y, valid_X, valid_Y = train_and_valid_data
    train_X, train_Y = torch.from_numpy(train_X).float(), torch.from_numpy(train_Y).float()
    train_loader = DataLoader(TensorDataset(train_X, train_Y), batch_size=config.batch_size)    # DataLoader can automatically generate trainable batch data

    valid_X, valid_Y = torch.from_numpy(valid_X).float(), torch.from_numpy(valid_Y).float()
    valid_loader = DataLoader(TensorDataset(valid_X, valid_Y), batch_size=config.batch_size)

    device = torch.device("cuda: 0" if config.use_cuda and torch.cuda.is_available() else "cpu") # CPU training or GPU
    
    model = Net(config).to(device)      # If it is GPU training, .to(device) will copy the model/data to the GPU memory

    if config.add_train:                # If it is incremental training, the original model parameters will be loaded first
        model.load_state_dict(torch.load(config.model_save_path + config.model_name))
    
    optimizer = torch.optim.Adam(model.parameters(), lr = config.learning_rate)
    criterion = torch.nn.MSELoss()      # These two lines define the optimizer and loss

    valid_loss_min = float("inf")
    bad_epoch = 0
    global_step = 0
    
    for epoch in range(config.epoch):
        print("Epoch {}/{}".format(epoch, config.epoch))

        model.train()                   
        train_loss_array = []
        hidden_train = None
        for i, _data in enumerate(train_loader):
            _train_X, _train_Y = _data[0].to(device),_data[1].to(device)
            optimizer.zero_grad()               # Set the gradient information to 0 before training
            pred_Y, hidden_train = model(_train_X, hidden_train)    # Here is the forward calculation forward function

            if not config.do_continue_train:
                hidden_train = None             # Reset hidden
            else:
                h_0, c_0 = hidden_train
                h_0.detach_(), c_0.detach_()    # Remove gradient information
                hidden_train = (h_0, c_0)

            loss = criterion(pred_Y, _train_Y)  # Calculate loss
            loss.backward()                     # Backpropagate loss
            optimizer.step()                    # Update parameters with optimizer
            train_loss_array.append(loss.item())
            global_step += 1

        # The following is the early stopping mechanism. 
        # When the model training for consecutive config.patience epochs does not improve the prediction effect of the validation set, 
        # it stops to prevent overfitting        
        model.eval()                    
        valid_loss_array = []
        hidden_valid = None
        for _valid_X, _valid_Y in valid_loader:
            _valid_X, _valid_Y = _valid_X.to(device), _valid_Y.to(device)
            pred_Y, hidden_valid = model(_valid_X, hidden_valid)
            if not config.do_continue_train: hidden_valid = None
            loss = criterion(pred_Y, _valid_Y)  # The verification process only has forward calculation, no back propagation process
            valid_loss_array.append(loss.item())

        train_loss_cur = np.mean(train_loss_array)
        valid_loss_cur = np.mean(valid_loss_array)
        print("The train loss is {:.6f}. ".format(train_loss_cur) + "The valid loss is {:.6f}.".format(valid_loss_cur))

        if valid_loss_cur < valid_loss_min:
            valid_loss_min = valid_loss_cur
            bad_epoch = 0
            torch.save(model.state_dict(), config.model_save_path + config.model_name)  # 模型保存
        else:
            bad_epoch += 1
            if bad_epoch >= config.patience:    
                print(" The training stops early in epoch {}".format(epoch))                
                break


def predict(config, test_X):
    # Get test data
    test_X = torch.from_numpy(test_X).float()
    test_set = TensorDataset(test_X)
    test_loader = DataLoader(test_set, batch_size=1)

    # Load the model
    device = torch.device("cuda:0" if config.use_cuda and torch.cuda.is_available() else "cpu")
    model = Net(config).to(device)
    model.load_state_dict(torch.load(config.model_save_path + config.model_name))   # Load model parameters

    # First define a tensor to save the prediction results
    result = torch.Tensor().to(device)

    # Forecasting process
    model.eval()
    hidden_predict = None
    for _data in test_loader:
        data_X = _data[0].to(device)
        pred_X, hidden_predict = model(data_X, hidden_predict)
        cur_pred = torch.squeeze(pred_X, dim=0)
        result = torch.cat((result, cur_pred), dim=0)

    return result.detach().cpu().numpy()
