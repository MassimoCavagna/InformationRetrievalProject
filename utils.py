from numba import cuda 

from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, ReLU, PReLU, Concatenate, Layer, Rescaling, GaussianNoise, Resizing
from tensorflow.keras.layers import Conv2D, MaxPool2D, AveragePooling2D, GlobalAveragePooling2D, GlobalMaxPool2D, Flatten, Reshape, InputLayer

from tensorflow.keras import Sequential, layers, initializers, activations
from tensorflow.keras.models import Model

from tensorflow.keras.optimizers import Optimizer
from tensorflow.keras.optimizers import SGD, Nadam, Adam

from tensorflow.keras.losses import Loss
from tensorflow.keras.losses import MeanSquaredError, CategoricalCrossentropy, CategoricalHinge

from tensorflow.keras.metrics import Metric
from tensorflow.keras.metrics import Accuracy, CategoricalAccuracy, TruePositives, TrueNegatives, FalsePositives, FalseNegatives, BinaryAccuracy, Precision, Recall, AUC 

from tensorflow.keras.activations import tanh
from sklearn.model_selection import KFold

from tensorflow.data import Dataset
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.preprocessing.image import load_img

from sklearn.metrics.pairwise import cosine_similarity

from tensorflow.keras.callbacks import History

from sklearn.preprocessing import LabelBinarizer

import tensorflow as tf

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import svm
from sklearn.model_selection import train_test_split
import pickle
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from joblib import dump, load

from InformationRetrievalProject.preprocessing import tfidf_vectorize

def setup(data, t_size, sentiments = ["fear", "anger", "joy", "sadness"]):
  """
  The function preprocess the data and return the train and test split in order to
  train the ensemble of 4 svm based on the sentiments passed
  Params:
    data: the corpus of documents
    t_size: the test size
    sentiments: the list of sentiments assigned to the corpus' documents
  Return:
    X_train, X_test : lists of vectorized documents
    y_train, y_test, y_t, lists of : lists of labels of the relative sample
  """

  X, y, _ = tfidf_vectorize(data)

  map = [ {s : 1 if i == pos else 0 for i, s in enumerate(sentiments)} for pos in range(len(sentiments))]
  test_map = {s : i for i, s in enumerate(sentiments)}

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = t_size, random_state = 42)

  y_train = [[m[y] for y in y_train] for m in map]
  y_t = [[m[y] for y in y_test] for m in map]
  y_test = [test_map[y] for y in y_test]

  return X_train, X_test, y_train, y_test, y_t

def svm_train(X_train, y_train, sentiments = ["fear", "anger", "joy", "sadness"]):
  """
  The function train multiple SVM with the training data passed for each sentiment
  Params:
    X_train: the vectorized documents lists of samples
    y_train: list of the labels realted to the X_train samples
    sentiments: the sentiments for which the SVMs are trained
  Return:
    A list of (binary) SVMs trained for the different sentiments
  """
    
  clf = [svm.SVC(C = 1.5, kernel = 'rbf', probability = True) for _ in range(len(sentiments))]
  
  for i, clf_s in enumerate(clf):
    clf_s.fit(X_train, y_train[i])
  
  return clf

def compute_svm_prediction(clf, X_test):
  """
  The function compute the predictions of the X_test according to the different SVM in clf
  Params:
    clf: a list containing the different SVMs
    X_test: the test sample of vectorized documents
  Return:
    the list of prediction according to the more probabile result among the different SVM predictions
  """
  y_pred = [clf_s.predict(X_test) for clf_s in clf]
  y_prob = [[max(a, b) for [a,b] in clf_s.predict_proba(X_test)] for clf_s in clf]

  predictions = np.array(y_pred).T
  probabilities = np.array(y_prob).T
  res = []

  for i, prediction in enumerate(predictions):
    positions = np.argwhere(prediction == np.amax(prediction)).flatten()
    comparison = np.argmax if max(prediction) == 1 else np.argmin
    prob = probabilities[i]
    res.append(comparison(prob[positions]))
  
  return res

def five_fold_cross_svm(parameters : dict, hyper_name : str, hyper_value, X, y):
  """
  The function compute the Accuracy on the train and test for the specific configuration
  of a SVM
  Params:
    parameters: a dictionary containing the configuration for the SVM
    hyper_name: the hyperparameter to which the cross validation is computed
    hyper_value: a list containing the different values of the hypervalue 
    X: list of samples
    y: the labels realtive to the samples
  Return:
    It prints the train and test accuracy
  """
  kf = KFold(n_splits=5, shuffle=True)
  cclf = svm.SVC(**parameters)
  train_accuracy = []
  test_accuracy = []

  for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    cclf.fit(X_train, y_train)
    
    res = cclf.predict(X_train)
    train_accuracy.append(metrics.accuracy_score(y_train, res))
    

    res = cclf.predict(X_test)
    test_accuracy.append(metrics.accuracy_score(y_test, res))
  print(f"{hyper_name}: {hyper_value}")
  print(f"Train accuracy: {np.round(np.mean(train_accuracy), 3)}")
  print(f"Test accuracy: {np.round(np.mean(test_accuracy), 3)} \n")
  
def compute_recall(cm):
  """
  The function compute the recall fiven a confusion matrics
  Params:
    cm: the confusion matrix (np.array) 
  Return:
    The recall
  """
  true_pos = np.diag(cm)
  false_neg = np.sum(cm, axis=1) - true_pos

  recall = np.sum(true_pos) / np.sum(true_pos + false_neg)

  return recall

def five_fold_cross_RF(parameters : dict, hyper_name : str, hyper_value, X, y):
  
  """
  The function compute the Accuracy on the train and test for the specific configuration
  of a Random Forest
  Params:
    parameters: a dictionary containing the configuration for the Random Forest
    hyper_name: the hyperparameter to which the cross validation is computed
    hyper_value: a list containing the different values of the hypervalue 
    X: list of samples
    y: the labels realtive to the samples
  Return:
    It prints the train and test accuracy
  """
  kf = KFold(n_splits=5, shuffle=True)
  cclf = RandomForestClassifier(**parameters)
  train_accuracy = []
  test_accuracy = []

  for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    cclf.fit(X_train, y_train)
    
    res = cclf.predict(X_train)
    train_accuracy.append(metrics.accuracy_score(y_train, res))
    

    res = cclf.predict(X_test)
    test_accuracy.append(metrics.accuracy_score(y_test, res))
  print(f"{hyper_name}: {hyper_value}")
  print(f"Train accuracy: {np.round(np.mean(train_accuracy), 3)}")
  print(f"Test accuracy: {np.round(np.mean(test_accuracy), 3)} \n")

def NN_model( input_shape,
              hidden_layers: list,
              hid_layers_act: str = 'ReLU',
              outp_layer_act: str = 'softmax',
              optimizer : Optimizer = SGD(learning_rate = 0.1),
              loss: Loss = CategoricalCrossentropy(),
              metrs: list = [
                                Accuracy(),
                                Precision(),
                                Recall(),
                                AUC()
                              ],
              dropout = .0
              
              ) -> Model:
  """ 
  Build the structure of the ffnn model
  Parameters:
    - input_shape: the number of input that the model must handle
    - hidden_layers: an iterator containing the amount of neurons in each hidden layer
    - hid_layers_act: the activation function of the neurons in the hidden layers,
    - outp_layer_act: the activation function of the neurons in the output layer,
    - optimizer : The optimizer that will be used,
    - metrs : the list of metrics used to evaluate the model's performance
  Return:
    The compiled model.
    The input must be a vectorized document according to the tfidf vocabulary
    The ouput is a vector of probabilities of the input belongs to a specific sentiment
  """

  # Definition of the input and output (dense) layer
  input_layer = Input(shape = input_shape)

  hidden = Dense(hidden_layers[0], activation = hid_layers_act)(input_layer)

  for h_layer in hidden_layers[1:]:
    hidden = Dense(h_layer, activation = hid_layers_act)(hidden)
    
  output_layer = Dense(4, activation = "softmax")(hidden)

  mlnn = Model(inputs = input_layer, outputs = output_layer)


  mlnn.compile(
        optimizer = optimizer,
        loss = loss,
        metrics = metrs
    )
  return mlnn

def model_evaluation(X, y, save_path, QE):
  kf = KFold(n_splits = 5, shuffle=True)

  input_shape = max(X[0].shape)

  models = [
            svm.SVC(C = 1.5, kernel = 'rbf'), 
            RandomForestClassifier(n_estimators = 200, criterion = "entropy", max_depth = 300), 
            NN_model(input_shape = input_shape,
                              hidden_layers = [1024, 256, 128, 64],
                              metrs = [Precision()]
                    )
            ]

  m_names = ["SVM", "Random Forest", "Neural Network"]

  ev_df = pd.DataFrame(columns = ["Model", "Phase", "QE", "Accuracy", "Precision", "Recall"])

  lb = LabelBinarizer()

  # Evaluations
  i = 0
  lb.fit(y)
  for train_index, test_index in kf.split(X):
    print(f"Round {i}")
    i += 1
    X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]

    # Train phase
    for mn, m in list(zip(m_names, models)):
      print(f"\t {mn}", end = "")

      # Training
      if mn == "Neural Network":
        X_train, X_test, y_train, y_test = X_train.toarray(), X_test.toarray(), lb.fit_transform(y_train), lb.fit_transform(y_test)
        m.fit(X_train, y_train, epochs= 50, verbose= 0, batch_size = 100)
      else:
        m.fit(X_train, y_train)

      # Evaluation
      res = m.predict(X_train)

      if mn == "Neural Network":
        res = [[1 if np.argmax(pred) == pos else 0 for pos in range(len(pred))] for pred in res]

      recall = np.mean(metrics.recall_score(y_train, res, average = None))
      accuracy = metrics.accuracy_score(y_train, res)
      precision = np.mean(metrics.precision_score(y_train, res, average = None))

      ev_df.loc[len(ev_df)] = [mn, "train", QE, accuracy, precision, recall]

      print("\t Train completed")

      # Test phase
      print(f"\t {mn}", end = "")

      res = m.predict(X_test)

      if mn == "Neural Network":
        res = [[1 if np.argmax(pred) == pos else 0 for pos in range(len(pred))] for pred in res]
        
      accuracy = metrics.accuracy_score(y_test, res)
      precision = np.mean(metrics.precision_score(y_test, res, average = None))
      recall = np.mean(metrics.recall_score(y_test, res, average = None))

      ev_df.loc[len(ev_df)] = [mn, "test", QE, accuracy, precision, recall]

      print("\t Test completed")
  ev_df.to_csv(save_path)
  
def query_expansion(query: list, query_expansion_lenght, tk_tk_count: pd.DataFrame):
  """
  The function add new tokens according to the query_expansion_lenght passed and add them to the
  original query text
  Params:
    query: a list of tokens
    query_expansion_lenght: the number of tokens that should be added to pad the query
    tk_tk_count: the padaframe containing the co-frequencies of the tokens
  Return:
    The list of the expanded query
  """
  closest_grams = pd.Series(dtype = int)
  distances = pd.Series(dtype = float)
  for token in query:
    
    # Fix exception if gram not in dataframe
    try:
      token_vector = np.array([tk_tk_count[token]])
      distances = tk_tk_count.apply(lambda row: cosine_similarity(token_vector, 
                                                                        np.array([row]))[0][0]
                                      )
      distances.drop(query, axis = 0, inplace = True)
      
      new_dists = distances.nlargest(query_expansion_lenght)
      closest_grams = pd.concat([closest_grams, new_dists], axis = 1)
      # print(closest_grams)
    except:
      pass

  expanded_query = list(closest_grams.max(axis = 1).sort_values(ascending=False)[:query_expansion_lenght].index)
  # print(query, expanded_query)
  return query + expanded_query

def plot_ch_ch_sentiment(cc_time_sentiments, ch1, ch2):
  """
  This function plot the timeline of the sentiment from ch1 to ch2
  according to the cc_time_sentiment data
  Params:
    cc_time_sentiments: a 2 level dict that contains for each pair of 
                        characters the sequence sentiment from the 
                        first in recpect to the second
    ch1: the first character name
    ch2: the second character name
  Return:
    None, it displays the time plot of the sentiment relation
  """
  plt.figure(figsize = (30, 8))
  y_map = {
      "anger" : 1, 
      "fear" : 2, 
      "joy" : 3, 
      "sadness" : 4
  }

  palette = ["#5e6480"]
  sns.set_palette(palette)

  character_y = [y_map[sent] for sent in cc_time_sentiments[ch1][ch2]]
  plt.title(ch1 + " -> " + ch2, fontsize = 25)
  character_x = [i for i in range(len(character_y))]
  sns.lineplot(x = character_x, y = character_y)

  palette = {
      1 : "#b50f0d", 
      2 : "#a30bb8", 
      3 : "#fafa00", 
      4 : "#0226db"
  }
  ax = sns.scatterplot(x = character_x, y = character_y, hue = character_y, s = 150, legend = False, palette = palette)
  ax.set_yticks(range(1, 5))
  ax.set_yticklabels(["anger", "fear" , "joy", "sadness"], fontsize = 12)
  plt.grid(alpha = 0.5)
