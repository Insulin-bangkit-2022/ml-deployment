# -*- coding: utf-8 -*-
"""Copy of Capstone Deep Neural Network

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/github/Insulin-bangkit-2022/ml-deployment/blob/main/model.ipynb

**IMPORT LIBRARIES**
"""

# Commented out IPython magic to ensure Python compatibility.
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import numpy as np
import os
import pickle
from google.colab import drive
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score
from keras.models import load_model
# %matplotlib inline

"""**Import Diabetes Dataset from Kaggle**"""

! pip install kaggle

drive.mount('/content/gdrive')

os.environ['KAGGLE_CONFIG_DIR'] = "/content/gdrive/Shared drives/Capstone Project/Product Based"

# Commented out IPython magic to ensure Python compatibility.
#changing the working directory
# %cd /content/gdrive/Shared drives/Capstone Project/Product Based

!kaggle datasets download -d andrewmvd/early-diabetes-classification

"""**Read Diabetes Dataset**"""

data = pd.read_csv('diabetes_data.csv', delimiter = ';')
data

data["gender"] = data["gender"].apply({"Male":1, "Female":0}.get)
data.head()

"""Plotting Heat Map"""

tc = data.corr()
sns.heatmap(tc,annot = False,cmap="coolwarm")

data.info()

"""**Split the Dataset Into Training and Test Set**"""

x = data[data.columns[:-1]]
y = data[data.columns[-1]]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.15,random_state = 10)
print(x_train)
print(y_train)

"""**Training Dataset**"""

model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(units = 128, activation = "relu"),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(units = 32, activation = "relu"),
  tf.keras.layers.Dropout(0.1),
  tf.keras.layers.Dense(units = 1,activation = "sigmoid")
  ])

model.compile(optimizer = "adam", 
              loss = "binary_crossentropy" , 
              metrics=["accuracy"])

#Here we train our model.
history = model.fit(x_train,y_train,epochs = 100,validation_data = (x_test,y_test))
#This the inference phase.We try our model on test data.
y_pred = model.predict(x_test)
y_pred = (y_pred > 0.5)

"""**Plot Accuracy and Loss**"""

#-----------------------------------------------------------
# Retrieve a list of list results on training and test data
# sets for each training epoch
#-----------------------------------------------------------
acc      = history.history[     'accuracy' ]
val_acc  = history.history[ 'val_accuracy' ]
loss     = history.history[    'loss' ]
val_loss = history.history['val_loss' ]

epochs   = range(len(acc)) # Get number of epochs

#------------------------------------------------
# Plot training and validation accuracy per epoch
#------------------------------------------------
plt.plot  ( epochs,     acc )
plt.plot  ( epochs, val_acc )
plt.title ('Training and validation accuracy')
plt.figure()

#------------------------------------------------
# Plot training and validation loss per epoch
#------------------------------------------------
plt.plot  ( epochs,     loss )
plt.plot  ( epochs, val_loss )
plt.title ('Training and validation loss'   )

"""**Plotting Confusion Matrix**"""

cm = confusion_matrix(y_test,y_pred)
print(cm)
accuracy_score(y_test,y_pred)

def normalized_confusion_matrix(y_test, conf_mat, model):
    _ , counts = np.unique(y_test,return_counts=True)
    conf_mat = conf_mat/counts
    plt.figure(figsize=(6,5))
    ax=sns.heatmap(conf_mat,fmt='.2f',annot=True,annot_kws={'size':20},lw=2, cbar=True, cbar_kws={'label':'% Class accuracy'})
    plt.title(f'Confusion Matrix ({model})',size=22)
    plt.xticks(size=20)
    plt.yticks(size=20)
    ax.figure.axes[-1].yaxis.label.set_size(20) ##colorbar label
    cax = plt.gcf().axes[-1]  ##colorbar ticks
    cax.tick_params(labelsize=20) ## colorbar ticks
    plt.savefig(f'confusion-matrix-{model}.png',dpi=300)

conf_mat = confusion_matrix(y_test,y_pred)
normalized_confusion_matrix(y_test,conf_mat, 'Diabetes Pred Model')

model.summary()

"""**Save the Model**

save as model.pb
"""

tf.saved_model.save(
    model,
    export_dir = "/tmp/myModel",
)

"""Save as H5"""

model.save('model.h5')

"""save as model.pkl"""

pickle.dump(model,open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))

