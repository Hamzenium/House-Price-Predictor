import tensorflow as tf
import keras
import numpy as np

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):

    # Check accuracy
    if(logs.get('loss') < 0.4):

      # Stop if threshold is met
      print("\nLoss is lower than 0.4 so cancelling training!")
      self.model.stop_training = True

# Instantiate class
callbacks = myCallback()

def house_model():
    xs = np.array([1.0,2.0,3.0,4.0,5.0,6.0],dtype=float)
    ys = np.array([1.0,1.5,2.0,2.5,3.0,3.5],dtype=float)
    model = keras.Sequential([keras.layers.Dense(units =1 , input_shape=[1])])
    model.compile(optimizer='sgd', loss='mean_squared_error')
    model.fit(xs, ys, epochs=1000, callbacks=[callbacks])
    return model

model = house_model()
new_y = 7.0
prediction = model.predict([new_y])[0]
print(prediction)