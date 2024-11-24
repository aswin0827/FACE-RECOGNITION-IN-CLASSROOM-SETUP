import cv2
from keras.models import load_model
from keras import layers

import numpy as np
from face_detection import extract_face_embeddings
import tensorflow as tf
from keras.utils import register_keras_serializable
from keras.utils import custom_object_scope

class DistanceLayer(layers.Layer):
    def call(self, inputs):
        return tf.abs(inputs[0] - inputs[1])

known_embeddings = np.load("embeddings.npy", allow_pickle=True)
known_labels = np.load("labels.npy", allow_pickle=True)
# Load the trained Siamese model
with custom_object_scope({'DistanceLayer': DistanceLayer}):
    model = load_model("siamese_model.h5")
   

# Recognize faces in a new image
def recognize_face(img, threshold=0.5):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Detect faces in the image using RetinaFace
    face_encodings, face_locations = extract_face_embeddings(img)
    print(f"Detected {len(face_encodings)} faces.")

    assigned_names = set()
    for face_encoding, face_location in zip(face_encodings, face_locations):
        
        # Compare face encoding using the Siamese network
        distances =[]
        for known_encoding in known_embeddings:
            
            
            
            distance = model.predict([face_encoding.reshape(1, 2048), known_encoding.reshape(1, 2048)])
            distances.append(distance[0][0])
          # Compare face to itself (or another face)
        min_distance = min(distances)
        
        if min_distance <= threshold:
            index = distances.index(min_distance)
            label = known_labels[index]
        else:
            label = "Unknown"
        
        # Draw rectangle and label the recognized face
        (top, right, bottom, left) = face_location
        cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(img, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    return img

# Example usage to recognize face in a new image
if __name__ == "__main__":
    # Load the test image
    test_img = cv2.imread("test_image3.jpg")
    
    # Perform face recognition
    recognized_img = recognize_face(test_img, threshold=0.6)
    
    # Show the result
    cv2.imshow("Recognized Faces", recognized_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Save the recognized image with faces marked
    cv2.imwrite("output_image.jpg", recognized_img)  # Save result as 'output_image.jpg'
