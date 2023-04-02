
# Anasa

![Logo](brand.png)

![GitHub repo size](https://img.shields.io/github/repo-size/yokahealthcare/Anasa?style=for-the-badge)

Project Anasa is a model for predicting a person's respiratory type with wave data using the RNN (Recurrent Neural Network) and LSTM (Long Short-Term Memory) deep learning methods.


## Authors

- [Ariana Tulus Purnomo](https://github.com/)
- [Erwin Yonata](https://github.com/yokahealthcare)

## Dataset
![Logo](https://websitev3-p-eu.figstatic.com/assets-v3/bee2b12a367b114cc0f33f2f24c15a70b76227db/static/media/defaultLogo.30adffde.png)

The dataset used in this project comes from figshare and is clean data https://figshare.com/articles/dataset/BWF_Breathing_Waveform_Dataset/20001326


## Requirements

- ![Pandas](https://img.shields.io/badge/pandas-v1.3.5-lightgrey)
- ![Numpy](https://img.shields.io/badge/numpy-v1.22.4-blue)
- ![Tensorflow](https://img.shields.io/badge/tensorflow-v2.11.0-orange)


## Installation
There are several files that you can operate on
### [Augmented]
the data used is the result of augmentation or is made to add data intentionally
### [Hold-Out] [StratifiedKFold] [GridSearchCV]
the model in using this type of cross-validation method.
### [TestingArea]
This is a testing area for you to customize yourself.
### [CPU] [GPU]
Using what processor is the model compiled. (NB. If there is no sign of anything, then it is run by the CPU.)

#### if you want to do the training yourself, you can be advised to use the command prompt, don't use a python notebook, because if it takes too long the training notebook process can stop for some reason. You can copy the code from the python notebook to a regular python file, then run it in the command prompt.


## Models
In the "Models" menu, there are several types of models that you can use directly to make predictions.

## Usage
This program is used to classify the types of people in breathing, which include,
- normal
- quick
- hold
- deep
- deep_quick


## More Information

You can check more details on this website

http://erwinyonata.com/Anasa/
