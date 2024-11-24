# create_pairs.py
import os
import random
import numpy as np

def create_pairs(dataset_path):
    """Create pairs of images for training Siamese network."""
    # Create positive and negative pairs
    pairs = []
    labels = []
    
    # Get the list of all people (folders)
    people = os.listdir(dataset_path)
    
    # Create positive pairs (same person)
    for person in people:
        person_path = os.path.join(dataset_path, person)
        images = os.listdir(person_path)
        for i in range(len(images)):
            for j in range(i + 1, len(images)):
                img1_path = os.path.join(person_path, images[i])
                img2_path = os.path.join(person_path, images[j])
                pairs.append([img1_path, img2_path])
                labels.append(1)  # Label 1 for positive pairs (same person)
    
    # Create negative pairs (different persons)
    for i in range(len(people)):
        for j in range(i + 1, len(people)):
            person1 = people[i]
            person2 = people[j]
            person1_path = os.path.join(dataset_path, person1)
            person2_path = os.path.join(dataset_path, person2)
            img1 = random.choice(os.listdir(person1_path))
            img2 = random.choice(os.listdir(person2_path))
            img1_path = os.path.join(person1_path, img1)
            img2_path = os.path.join(person2_path, img2)
            pairs.append([img1_path, img2_path])
            labels.append(0)  # Label 0 for negative pairs (different persons)
    
    # Convert to numpy arrays
    pairs = np.array(pairs)
    labels = np.array(labels)
    
    return pairs, labels
