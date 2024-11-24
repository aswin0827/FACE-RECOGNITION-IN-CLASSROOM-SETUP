import os
import cv2
import numpy as np
from face_detection import extract_face_embeddings
import tensorflow as tf



# Collect face embeddings and labels from the dataset
def collect_embeddings(dataset_path):
    embeddings = []
    labels = []
    
    # Loop through the dataset folder
    for person_name in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, person_name)
        
        # Loop over each image in the person's folder
        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)
            print(f"Processing: {img_path}")
            img = cv2.imread(img_path)
            if img is None:
                print(f"Failed to load image: {img_path}")
                continue  # Skip this image if it fails to load
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Extract face embeddings using RetinaFace
            encodings, _ = extract_face_embeddings(img)
            if encodings:  # If faces are found
                embeddings.append(encodings[0])  # Taking the first face (if multiple)
                labels.append(person_name)
    
    return embeddings, labels

# Example usage to collect embeddings and labels from the dataset
if __name__ == "__main__":
    dataset_path = "dataset"  # Update with your dataset path
    embeddings, labels = collect_embeddings(dataset_path)
    print(f"Collected {len(embeddings)} embeddings and {len(labels)} labels.")
    np.save("embeddings.npy", embeddings)
    np.save("labels.npy",labels)

