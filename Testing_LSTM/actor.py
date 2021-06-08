import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.keras.initializers import RandomUniform
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Reshape, LSTM, Lambda, BatchNormalization, GaussianNoise, Flatten

class ActorNetwork:
    """ Actor Network for the DDPG Algorithm
    """

    def __init__(self, inp_dim, out_dim, lr,num_steps=5):
        self.inp_dim = inp_dim
        self.act_dim = out_dim
        self.lr = lr
        self.num_steps=num_steps
        self.model = self.network()
        self.target_model = self.network()



    def network(self):
        """ Actor Network for Policy function Approximation, using a tanh
        activation for conti/nuous control. We add parameter noise to encourage
        exploration, and balance it with Layer Normalization.
         """
        inp = Input((self.num_steps , self.inp_dim,))
        """
        # DNN
        output = Dense(256, activation='sigmoid', 
            kernel_initializer='random_normal')(inp)
        output = BatchNormalization()(output)
        output = Dropout(0.1)(output)
        output = Dense(128, activation='sigmoid', 
            kernel_initializer='random_normal')(output)
        output = BatchNormalization()(output)
        output = Dropout(0.1)(output)
        output = Dense(64, activation='sigmoid', 
            kernel_initializer='random_normal')(output)
        output = BatchNormalization()(output)
        output = Dropout(0.1)(output)
        output = Dense(32, activation='sigmoid', 
            kernel_initializer='random_normal')(output)
        output = BatchNormalization()(output)
        output = Dropout(0.1)(output)

        """
        # LSTM
        output = LSTM(256, dropout=0.1,
                      return_sequences=True, stateful=False,
                      kernel_initializer='random_normal')(inp)
        output = BatchNormalization()(output)
        output = LSTM(128, dropout=0.1,
                      return_sequences=True, stateful=False,
                      kernel_initializer='random_normal')(output)
        output = BatchNormalization()(output)
        output = LSTM(64, dropout=0.1,
                      return_sequences=True, stateful=False,
                      kernel_initializer='random_normal')(output)
        output = BatchNormalization()(output)
        output = LSTM(32, dropout=0.1,
                      stateful=False,
                      kernel_initializer='random_normal')(output)
        output = BatchNormalization()(output)

        """
        # ORIGINAL
        # x = Dense(256, activation='relu')(inp)
        # x = GaussianNoise(1.0)(x)
        # #
        # #x = Flatten()(x)
        # x = Dense(128, activation='relu')(x)
        # x = GaussianNoise(1.0)(x)

        """
        output = Dense(self.act_dim, activation='sigmoid', kernel_initializer='random_normal')(output)

        # out = Lambda(lambda i: i)(out)

        return Model(inp, output)

    def predict(self, sample):
        """ Action prediction
        """
        sample = np.array(sample).reshape(-1, self.num_steps, self.inp_dim)
        return self.model.predict(sample)


    def load_model(self, model_path):

        self.model.load_weights(model_path)
        # self.target_model.load_weights(path)

