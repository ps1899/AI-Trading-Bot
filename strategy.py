# Importing various modules
import threading
import os
import abc
import time
import yfinance as yf
from alpaca_trade_api import REST
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.layers import Dense
from keras.models import Sequential, model_from_json

# AlpacaPaperSocket class for the connection to Alpaca API using paper trading key_id, secret_id & base_url
class AlpacaPaperSocket(REST):
    def __init__(self):
        super().__init__(
            key_id = 'PKARJ9A9ZP1H8K5A58TF',
            secret_key = '68Bl3zDxWDGGT1l30y*******NdwtEEe6nRAxAzW',
            base_url = 'https://paper-api.alpaca.markets'
        )

# TradingSystem class with methods declared as abstract, so that we can change our implementations according to the need of the system computations 
class TradingSystem(abc.ABC):
    def __init__(self, api, symbol, time_frame, system_id, system_label):
        self.api = api
        self.symbol = symbol
        self.time_frame = time_frame
        self.system_id = system_id
        self.system_label = system_label
        thread = threading.Thread(target = self.system_loop)
        thread.start()
        
    @abc.abstractmethod
    def place_buy_order(self):
        pass
    
    @abc.abstractmethod
    def place_sell_order(self):
        pass
    
    @abc.abstractmethod
    def system_loop(self):
        pass

# AI Porfolio Management Model class which implements our AI model
class PMDevelopment:
    def __init__(self):
        data = pd.read_csv("stock_data.csv")
        # Seperating the Dependent & the Independent Model
        x = data['Delta Value']
        y = data.drop(['Delta Value'], axis = 1)       
        
        # Splitting the Train & Test DataSet
        x_train, x_test, y_train, y_test = train_test_split(x, y)
        
        # Creating a Sequential Model
        network = Sequential()
        
        # Creating the Structure to our Neural network
        network.add(Dense(1, input_shape = (1,), activation = 'tanh'))
        network.add(Dense(3, activation = 'tanh'))
        network.add(Dense(3, activation = 'tanh'))
        #network.add(Dense(5, activation = 'tanh'))
        #network.add(Dense(3, activation = 'tanh'))
        network.add(Dense(3, activation = 'tanh'))
        network.add(Dense(1, activation = 'tanh'))
        
        #Compiling the network using rmsprop optimizer. We can also use Adam Optimizer
        network.compile(optimizer = 'rmsprop', loss = 'hinge', metrics = ['accuracy'])
        
        #Fitting(Training) the model to predict the Accuracy
        network.fit(x_train.values, y_train.values, epochs = 100)
        
        #Evaluaing our model predictions
        y_pred = network.predict(x_test.values)
        y_pred = np.around(y_pred, 0)
        print(classification_report(y_test, y_pred))
        
        #Saving the structure to our json
        strategy_model = network.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(strategy_model)
        
        #Saving our network weights to the HDF5
        network.save_weights("result.h5")    
#PMDevelopment()

# Portfolio Management Model class
class PortfolioMgmtModel:
    def __init__(self):
        data = pd.read_csv("stock_data.csv")
        x = data['Delta Value']
        y = data.drop(['Delta Value'], axis = 1)
        # Reading Structure from Json
        json_file = open("model.json", "r")
        json = json_file.read()
        json_file.close()
        self.network = model_from_json(json)
        
        # Reading weights from HDF5
        self.network.load_weights("result.h5")
        
        # Verifying weights & structure are loaded
        y_pred = self.network.predict(x.values)
        y_pred = np.around(y_pred, 0)
        print(classification_report(y, y_pred))
        
PortfolioMgmtModel()

# Portfolio ManagementSystem class where a vector is created for storing data
class PortfolioMgmtSystem(TradingSystem):
    def __init__(self):
        super().__init__(AlpacaPaperSocket(), 'IBM', 86400, 1, 'AI_PM')
        self.AI = PortfolioMgmtModel()
    
    # function for placing a buy order    
    def place_buy_order(self):
        self.api.submit_order(
            symbol = 'IBM',
            qty = 1,
            side = 'buy',
            type = 'market',
            time_in_force = 'day'
        )
    
    # function for placing a sell order
    def place_sell_order(self):
        self.api.submit_order(
            symbol = 'IBM',
            qty = 1,
            side = 'sell',
            type = 'market',
            time_in_force = 'day'
        )

    # An infinite loop which will systematically make the trades 
    def system_loop(self):
        this_week_close = 0
        last_week_close = 0
        delta = 0
        day_cnt = 0
        while(True):
            # Waiting a day fro requesting more data
            time.sleep(1440)
            # Requesting EOD from IBM
            data_req = self.api.get_barset('IBM', timeframe = '1D', limit = 1).df
            # Creating the dataframe to predict
            z = pd.DataFrame(
                data = [[data_req['IBM']['close'][0]]], columns = 'Close'.split()
            )
            if(day_cnt == 7):
                day_cnt = 0
                last_week_close = this_week_close
                this_week_close = z['Close']
                delta = this_week_close - last_week_close
                
                # AI will choose whether to Buy, Sell or Hold Stock
                if(np.around(self.AI.network.predict([delta])) <= -0.5):
                    self.place_sell_order()
                
                elif(np.around(self.AI.network.predict([delta])) >= 0.5):
                    self.place_buy_order()
                    
PortfolioMgmtSystem()
