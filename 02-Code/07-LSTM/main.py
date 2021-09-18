# -*- coding: UTF-8 -*-
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

frame = "keras"
if frame == "pytorch":
    from model.model_pytorch import train, predict
elif frame == "keras":
    from model.model_keras import train, predict
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
elif frame == "tensorflow":
    from model.model_tensorflow import train, predict
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
else:
    raise Exception("Wrong frame seletion")

class Config:
    # Data parameters
    feature_columns = list(range(2, 23))
    label_columns = [21]                 
    label_in_feature_index = (lambda x,y: [x.index(i) for i in y])(feature_columns, label_columns)

    predict_day = 1

    # Network parameters
    input_size = len(feature_columns)
    output_size = len(label_columns)

    hidden_size = 128           # The hidden layer size of LSTM is also the output size
    lstm_layers = 2             # LSTM stacking layers
    dropout_rate = 0.2          # dropout probability
    time_step = 20     

    # Training parameters
    do_train = True
    do_predict = True
    add_train = False         
    shuffle_train_data = True   
    use_cuda = False            # Whether to use GPU training

    train_data_rate = 0.95   
    valid_data_rate = 0.15     

    batch_size = 64
    learning_rate = 0.001
    epoch = 20                 
    patience = 5                
    random_seed = 42           

    do_continue_train = False   # Each training uses the last final_state as the next init_state, which is only used for RNN type models!
                                # currently only supports pytorch

    continue_flag = ""          
    if do_continue_train:
        shuffle_train_data = False
        batch_size = 1
        continue_flag = "continue_"

    # Frame parameters
    used_frame = frame
    model_postfix = {"pytorch": ".pth", "keras": ".h5", "tensorflow": ".ckpt"}
    model_name = "model_" + continue_flag + used_frame + model_postfix[used_frame]

    # Path parameter
    train_data_path = "./data/stock_data.csv"
    model_save_path = "./checkpoint/" + used_frame + "/"
    figure_save_path = "./figure/"
    do_figure_save = True

    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)   

    if not os.path.exists(figure_save_path):
        os.mkdir(figure_save_path)

class Data:
    def __init__(self, config):
        self.config = config
        self.data, self.data_column_name = self.read_data()

        self.data_num = self.data.shape[0]
        self.train_num = int(self.data_num * self.config.train_data_rate)

        self.mean = np.mean(self.data, axis=0)             
        self.std = np.std(self.data, axis=0)
        self.norm_data = (self.data - self.mean)/self.std   # Normalization, to dimension

        self.start_num_in_test = 0    

        label_data = self.data[self.train_num + self.start_num_in_test :, config.label_in_feature_index]



    def read_data(self):               
        init_data = pd.read_csv(self.config.train_data_path, usecols=self.config.feature_columns)
        return init_data.values, init_data.columns.tolist()     

    def get_train_and_valid_data(self):
        feature_data = self.norm_data[:self.train_num]
        label_data = self.norm_data[self.config.predict_day : self.config.predict_day + self.train_num, self.config.label_in_feature_index]   

        if not self.config.do_continue_train:
            train_x = [feature_data[i:i+self.config.time_step] for i in range(self.train_num-self.config.time_step)]
            train_y = [label_data[i:i+self.config.time_step] for i in range(self.train_num-self.config.time_step)]
        else:
            train_x = [feature_data[start_index + i*self.config.time_step : start_index + (i+1)*self.config.time_step]
                       for start_index in range(self.config.time_step)
                       for i in range((self.train_num - start_index) // self.config.time_step)]
            train_y = [label_data[start_index + i*self.config.time_step : start_index + (i+1)*self.config.time_step]
                       for start_index in range(self.config.time_step)
                       for i in range((self.train_num - start_index) // self.config.time_step)]

        train_x, train_y = np.array(train_x), np.array(train_y)
        train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=self.config.valid_data_rate,
                                                              random_state=self.config.random_seed,
                                                              shuffle=self.config.shuffle_train_data)  
        return train_x, valid_x, train_y, valid_y

    def get_test_data(self, return_label_data = False):
        feature_data = self.norm_data[self.train_num:]
        sample_interval = min(feature_data.shape[0], self.config.time_step)    
        self.start_num_in_test = feature_data.shape[0] % sample_interval 
        time_step_size = feature_data.shape[0] // sample_interval

        test_x = [feature_data[self.start_num_in_test+i*sample_interval : self.start_num_in_test+(i+1)*sample_interval]
                   for i in range(time_step_size)]
        
        if return_label_data:
            label_data = self.norm_data[self.train_num + self.start_num_in_test:, self.config.label_in_feature_index]
            return np.array(test_x), label_data
        
        return np.array(test_x)

def draw(config: Config, origin_data: Data, predict_norm_data: np.ndarray):
    label_data = origin_data.data[origin_data.train_num + origin_data.start_num_in_test :, config.label_in_feature_index]
    predict_data = predict_norm_data * origin_data.std[config.label_in_feature_index] + origin_data.mean[config.label_in_feature_index]  
   
    assert label_data.shape[0] == predict_data.shape[0], "The element number in origin and predicted data is different"

    label_name = [origin_data.data_column_name[i] for i in config.label_in_feature_index]
    label_column_num = len(config.label_columns)

    loss = np.mean((label_data[config.predict_day:] - predict_data[:-config.predict_day] ) ** 2, axis=0)
    loss_norm = loss/(origin_data.std[config.label_in_feature_index] ** 2)
    print("The mean squared error of stock {} is ".format(label_name) + str(loss_norm))

    label_X = range(origin_data.data_num - origin_data.train_num - origin_data.start_num_in_test)
    predict_X = [ x + config.predict_day for x in label_X]
    

    for i in range(label_column_num):
        plt.figure(i + 1)
        plt.plot(label_X, label_data[:, i], label='label')
        plt.plot(predict_X, predict_data[:, i], label='predict')
        plt.title("Predict stock {} price with {}".format(label_name[i], config.used_frame))
        plt.legend(loc='upper left')

        print("The predicted stock {} for the next {} day(s) is: ".format(label_name[i], config.predict_day) + 
                str(np.squeeze(predict_data[-config.predict_day:, i])))
        
        if config.do_figure_save:
            plt.savefig(config.figure_save_path+"{}predict_{}_with_{}.png".format(config.continue_flag, label_name[i], config.used_frame))

    plt.show()

def main(config):
    np.random.seed(config.random_seed)
    data_gainer = Data(config)

    if config.do_train:
        train_X, valid_X, train_Y, valid_Y = data_gainer.get_train_and_valid_data()
        train(config, [train_X, train_Y, valid_X, valid_Y])

    if config.do_predict:
        test_X, test_Y = data_gainer.get_test_data(return_label_data=True)
        pred_result = predict(config, test_X)
        draw(config, data_gainer, pred_result)

if __name__=="__main__":
    con = Config()
    main(con)
