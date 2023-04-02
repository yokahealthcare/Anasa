import pandas as pd
import numpy as np
import tensorflow as tf

df = pd.read_csv("breathing_waveform_data.csv").iloc[:, :-1] # get rid of last column ("notes")

X = df.iloc[:, :-1]
Y = df.iloc[:, -1]



# fix random seed for reproducibility
seed = 21
tf.random.set_seed(seed)


from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

# encode class values as integers [0,0,0,0,0,0,0,1,1,1,1,1,2,2,2,2]
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)

# convert integers to dummy variables (i.e. one hot encoded)
hot_y = np_utils.to_categorical(encoded_Y)




from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)


feature = 5
X = np.reshape(X, (X.shape[0], int(85/feature), feature))
# (26400, 17, 5)
# 5 indicator will be used per sequence/timestep per sample/row


from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout



# Configuration for Model Structure
from keras.optimizers import Adam
_optimizer = Adam()
_loss = "categorical_crossentropy"
_metric = ["accuracy"]



def create_model(dropout_rate=0.2, init_mode='glorot_uniform', init_recurrent='orthogonal', init_units=60):
    classifier = Sequential()

    # first layer
    classifier.add(LSTM(units=init_units, kernel_initializer=init_mode, recurrent_initializer=init_recurrent, return_sequences=True, input_shape=(17, 5)))
    classifier.add(Dropout(dropout_rate))    # Ignore xx% of the neuron (ex. 50 * 20% = 10 neuoron will be ignored) 

    # second layer
    classifier.add(LSTM(units=init_units, return_sequences=True))
    classifier.add(Dropout(dropout_rate))

    # third layer
    # classifier.add(LSTM(units=20, return_sequences=True))
    # classifier.add(Dropout(dropout_rate))

    # fourth layer
    classifier.add(LSTM(units=init_units))
    classifier.add(Dropout(dropout_rate))

    # last layer
    classifier.add(Dense(units=5, activation='softmax'))

    # Compile
    classifier.compile(optimizer=_optimizer, loss=_loss, metrics=_metric)
    
    return classifier



import multiprocessing
cpu_count = multiprocessing.cpu_count()
print(f"Number of CPU cores: {cpu_count}")



from sklearn.model_selection import GridSearchCV
from scikeras.wrappers import KerasClassifier

model = KerasClassifier(model=create_model)

param_grid = {
    'epochs': [15, 20],
    'batch_size': [32, 64],
    'model__dropout_rate': [0.2, 0.3],
    'model__init_mode': ['glorot_uniform', 'he_uniform'],
    'model__init_recurrent': ['glorot_uniform', 'orthogonal'],
    'model__init_units': [17, 30, 60]
}

grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=10, verbose=5, refit=True, n_jobs=12)


with tf.device('/device:CPU:0'):
    grid_result = grid.fit(X, hot_y)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))



print("Best Estimator")
print(grid_result.best_estimator_)
print("Best Param")
print(grid_result.best_params_)
print("Best Score")
print(grid_result.best_score_)


res = pd.DataFrame(GS.cv_results_)
res.to_csv("GridSearch_results.csv")