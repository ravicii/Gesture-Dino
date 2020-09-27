import tensorflow as tf
import numpy as np
X=np.load('X.npy',allow_pickle=True)
Y=np.load('Y.npy',allow_pickle=True)
model=tf.keras.Sequential([
    tf.keras.layers.Conv2D(16,(3,3),input_shape=(64,64,1),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(16,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(20,activation='relu'),
    tf.keras.layers.Dense(1,activation='sigmoid')
])
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
history=model.fit(X,Y,32,3,validation_split=0.1)
model.save('gestureDino.h5')