from keras.models import load_model
from keras.utils import CustomObjectScope
import tensorflow as tf
with CustomObjectScope({'tf': tf}):
    model = load_model('keras-facenet/model/facenet_keras.h5')
    print(model.inputs)
    print(model.outputs)