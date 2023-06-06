#!/usr/bin/env python
# coding: utf-8

# # BREATHING WAVE
# ## DEEP LEARNING - LSTM
# ### 04 March 2023

# In[1]:


import pandas as pd
import numpy as np
import tensorflow as tf

df = pd.read_csv("breathing_waveform_data.csv").iloc[:, :-1] # get rid of last column ("notes")

X = df.iloc[:, :-1]
Y = df.iloc[:, -1]


# In[2]:


# Check if the data do not have any NULL 
print("X have a null? \t{}".format(X.isnull().values.any()))
print("Y have a null? \t{}".format(Y.isnull().values.any()))


# In[3]:


X


# In[4]:


Y.value_counts()


# ## Fix Random Seed for Reproducibility

# In[6]:


# fix random seed for reproducibility
seed = 21
tf.random.set_seed(seed)


# ### Program Starting
# # PART 1 : Data Preprocessing

# ## Importing Imbalanced Libraries

# In[7]:


import imblearn
print(imblearn.__version__)


# ## Removing Class Overlapped (removing Tomek Links)
# > **Tomek links** identify pairs of samples from different classes that are close to each other and potentially contribute to class overlap or ambiguity.
# >
# > **Conclusion** : Nothing is removed. Indicate that the data is good and there is no ambiguity

# In[10]:


def tomek_links(X, Y):
    # define the undersampling method
    undersample = imblearn.under_sampling.TomekLinks()
    # transform the dataset
    return undersample.fit_resample(X, Y)


# In[11]:


X, Y = tomek_links(X, Y)


# In[14]:


Y.value_counts()


# ## Undersampling

# In[15]:


# NearMiss
def near_miss(X, Y, version, neighbors=3):
    # define the undersampling method
    undersample = imblearn.under_sampling.NearMiss(version=version, n_neighbors=3)
    # transform the dataset
    return undersample.fit_resample(X, Y)

# RandomUnderSample
labels = {
    "normal" : 800,
    "quick" : 800,
    "hold" : 800,
    "deep" : 800,
    "deep_quick" : 800
}

def rus(X, Y, strategy=labels):
    # define the undersampling method
    undersample = imblearn.under_sampling.RandomUnderSampler(sampling_strategy=strategy)
    # transform the dataset
    return undersample.fit_resample(X, Y)

## CNN (CondensedNearestNeighbour) error


# In[16]:


X, Y = near_miss(X, Y, version=1)
Y.value_counts()


# ## Hot Encoded The Label Data 

# In[ ]:


from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

# encode class values as integers [0,0,0,0,0,0,0,1,1,1,1,1,2,2,2,2]
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)

# convert integers to dummy variables (i.e. one hot encoded)
hot_y = np_utils.to_categorical(encoded_Y)


# In[ ]:


hot_y


# ## Scale The Training Data (STD)

# In[ ]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)


# ## Reshaping The Training Data to 3-Dimensional Numpy Array
# ### STRUCTURE : (batch_size, timestep, feature)

# In[ ]:


feature = 5
X = np.reshape(X, (X.shape[0], int(85/feature), feature))
# (26400, 17, 5)
# 5 indicator will be used per sequence/timestep per sample/row


# # PART 2 : Building The RNN

# In[ ]:


from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout


# ## Creating Layer of RNN

# In[ ]:


# Configuration for Model Structure
from keras.optimizers import Adam
_optimizer = Adam()
_loss = "categorical_crossentropy"
_metric = ["accuracy"]


# In[ ]:


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


# # PART 3 : Training Time

# ## Setting up the GridSearchCV

# In[ ]:


import multiprocessing

cpu_count = multiprocessing.cpu_count()

print(f"Number of CPU cores: {cpu_count}")


# In[ ]:


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


# ## Training

# In[ ]:


with tf.device('/device:CPU:0'):
    grid_result = grid.fit(X, hot_y)


# ## Summarize the Result

# In[ ]:


# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


# ## Plot The Best Estimator, Param, and Score

# In[ ]:


print("Best Estimator")
print(grid_result.best_estimator_)
print("Best Param")
print(grid_result.best_params_)
print("Best Score")
print(grid_result.best_score_)


# ## Save the Result to CSV

# In[ ]:


res = pd.DataFrame(GS.cv_results_)
res.to_csv("GridSearch_results.csv")


# ## Save the BEST Model

# In[ ]:


import os
import pickle
filename = "{}\\{}\\{}.pickle".format(os.getcwd(), "MODELS\\[3-layer] - 3L1\\CV\\GridSearchCV", "best_model")
best_model = grid_result.best_estimator_

# save the model to pickle file
with open(filename, 'wb') as o:
    pickle.dump(best_model, o)
    # close the file
    o.close()


# ## Evaluate Model

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, hot_y, test_size=.2, random_state=21)


# In[ ]:


# load the model to pickle file
filename = "{}\\{}\\{}.pickle".format(os.getcwd(), "MODELS\\[3-layer] - 3L1\\CV\\GridSearchCV", "best_model")
with open(filename, 'rb') as o:
    # dump information to that file
    data = pickle.load(o)

    # close the file
    file.close()


# In[ ]:


# evaluate the model
score = classifier.evaluate(X_test, Y_test)
print("Accuracy \t: {:.2f}".format(score[1]*100))
print("Loss \t\t: {:.2f}".format(score[0]*100))


# In[ ]:


pred = classifier.predict(X_test)


# In[ ]:


y_true = np.argmax(Y_test, axis=1)
y_pred = np.argmax(pred, axis=1)


# ## Plot Confusion Matrix

# In[ ]:


from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# Define the confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred)

# Plot the confusion matrix
plt.imshow(conf_matrix, cmap=plt.cm.Greens)

# Add labels to the plot
tick_marks = np.arange(len(conf_matrix))
plt.xlabel('Predicted label')
plt.ylabel('True label')

# Add values to the plot
for i in range(len(conf_matrix)):
    for j in range(len(conf_matrix)):
        plt.text(j, i, conf_matrix[i, j], ha='center', va='center')

# Show the plot
plt.show()


# ## Saving the model into a file

# In[ ]:


import os

# saving the model
filename = "{}\\{}\\{}.h5".format(os.getcwd(), "MODELS\\[3-layer] - 3L1", _optimizer)
classifier.save(filename)


# # PART 4 : Testing the Loaded Model

# In[ ]:


from tensorflow.keras.models import load_model
filename = "{}\\{}\\{}.h5".format(os.getcwd(), "MODELS\\[3-layer] - 3L1", _optimizer)

# load model
loaded_model = load_model(filename)


# ## evaluate the model

# In[ ]:


score = loaded_model.evaluate(X_test, Y_test)
print("Accuracy \t: {:.2f}".format(score[1]*100))
print("Loss \t\t: {:.2f}".format(score[0]*100))

