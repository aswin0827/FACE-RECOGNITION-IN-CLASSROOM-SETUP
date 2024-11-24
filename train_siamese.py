from keras.models import Model
from keras.layers import Input, Lambda, Dense, Flatten
from keras.optimizers import Adam
from keras.models import load_model
from keras.utils import register_keras_serializable
from keras import layers
import pickle
import numpy as np
import tensorflow as tf

from generate_embeddings import collect_embeddings
from create_pairs import create_pairs
# Build the Siamese Network
class DistanceLayer(layers.Layer):
    def call(self, inputs):
        return tf.abs(inputs[0] - inputs[1])
def create_siamese_model(input_shape):

    
    input1 = Input(input_shape)
    input2 = Input(input_shape)

    

    distance = DistanceLayer()([input1,input2])
    output = Dense(1, activation='sigmoid')(distance)

    model = Model(inputs=[input1, input2], outputs=output)
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])
    return model

# Train the Siamese network
def train_siamese_network(embeddings, labels, epochs=10, batch_size=32):
    input_shape = (2048,)  # Image shape for ResNet50
    model = create_siamese_model(input_shape)
    
    pairs = []
    targets = []
    
    for i in range(len(embeddings)):
        for j in range(len(embeddings)):
            pairs.append([embeddings[i], embeddings[j]])
            targets.append(1 if labels[i] == labels[j] else 0)
    
    pairs = np.array(pairs)
    targets = np.array(targets)
    x1 = np.array([pair[0] for pair in pairs])
    x2 = np.array([pair[1] for pair in pairs])

    model.fit([x1,x2], targets, epochs=epochs, batch_size=batch_size)
    
    model.save("siamese_model.h5")
    print("Model trained and saved as siamese_model.h5")

# Example usage
if __name__ == "__main__":
    embeddings = np.load("embeddings.npy", allow_pickle=True)
    labels = np.load("labels.npy", allow_pickle=True)
    
    train_siamese_network(embeddings, labels)
