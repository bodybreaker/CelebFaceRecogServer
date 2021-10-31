import tensorflow as tf
from tensorflow import keras

new_model = tf.keras.models.load_model('face_model.h5')

# 모델 구조를 출력합니다
new_model.summary()