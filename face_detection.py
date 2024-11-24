import cv2
import numpy as np
import tensorflow as tf


# Load Haar Cascade Classifier
haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
resnet_model = tf.keras.applications.ResNet50(
    include_top=False,  # Exclude classification head
    input_shape=(224, 224, 3),
    pooling="avg"  # Use global average pooling to get a feature vector
)

# Function to detect faces using Haar Cascade and extract embeddings
def extract_face_embeddings(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    face_embeddings =[]
    face_locations = []

    for (x, y, w, h) in faces:
        # Crop the detected face
        face_img = img[y:y + h, x:x + w]
        face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face_resized = cv2.resize(face_img_rgb, (224, 224))
        face_normalized = np.expand_dims(face_resized / 255.0, axis=0)

        # Extract the face encoding
        face_embedding = resnet_model.predict(face_normalized)
        face_embeddings.append(face_embedding[0])
        face_locations.append((y, x + w, y + h, x))

    return face_embeddings, face_locations
