import pandas as pd
import numpy as np
import tensorflow
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
from keras.metrics import Precision,Recall,F1Score


df = pd.read_csv(r"Assessment - Form Responses.csv")
df.drop(columns=["Timestamp","Email Address"],axis=1,inplace=True)
for col in df.columns:
    le = LabelEncoder()
    le = le.fit(df[col])
    df[col] = le.fit_transform(df[col])

from sklearn.cluster import KMeans
obj = KMeans(n_clusters=2,n_init=5).fit(df)
clusters = obj.labels_

df["labels"] = clusters
requirements = keras.callbacks.ModelCheckpoint(
    r"C:\Users\prade\OneDrive\Desktop\git quantization\normal.h5",
    monitor="val_loss",
    verbose=0,
    save_best_only=True,
    save_weights_only=False,
    mode="min",
    save_freq="epoch")

x = df.drop(columns="labels",axis=1)
y = df["labels"]

model = keras.Sequential()
model.add(keras.layers.Input(shape=(10,)))
model.add(keras.layers.Dense(8,activation="relu",
                             kernel_initializer=keras.initializers.GlorotNormal(seed=42),
                             ))
model.add(keras.layers.Dense(3,activation="relu",
                             kernel_initializer=keras.initializers.GlorotNormal(seed=42),
                            kernel_regularizer = keras.regularizers.L2()))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(1,activation="sigmoid",
                            kernel_initializer=keras.initializers.HeNormal(seed=42)))

model.compile(loss="binary_crossentropy",optimizer="adam",metrics=["accuracy"])
history = model.fit(x,y,validation_split=0.2,batch_size=8,epochs=30,callbacks=[requirements],verbose=0)