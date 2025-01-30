import os
import cv2
import pickle
import face_recognition
import numpy as np
from sklearn.metrics import accuracy_score
from tqdm import tqdm

def augment_image(image, num_augmentations=10):
    augmented_images = []

    for _ in range(num_augmentations):
        scale = np.random.uniform(0.7, 1.3)
        resized = cv2.resize(image, None, fx=scale, fy=scale)

        angle = np.random.uniform(-30, 30)
        (h, w) = resized.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(resized, rotation_matrix, (w, h))

        brightness = np.random.uniform(0.5, 1.5)
        brightened = cv2.convertScaleAbs(rotated, alpha=brightness, beta=0)

        noise = np.random.normal(0, 15, brightened.shape).astype(np.uint8)
        noisy = cv2.add(brightened, noise)

        augmented_images.append(noisy)

    return augmented_images

def evaluate_threshold(knownFaceEncodings, knownFaceNames, threshold, test_images_dir, num_tests=100):
    true_labels = []
    predicted_labels = []

    for filename in tqdm(os.listdir(test_images_dir)[:num_tests], desc="Evaluating"):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(test_images_dir, filename)
            image = cv2.imread(image_path)

            if image is None:
                print(f"Image could not be loaded: {filename}")
                continue

            augmented_images = augment_image(image, num_augmentations=10)

            for aug_image in augmented_images:
                rgb_image = cv2.cvtColor(aug_image, cv2.COLOR_BGR2RGB)
                face_encodings = face_recognition.face_encodings(rgb_image)

                if len(face_encodings) > 0:
                    face_encoding = face_encodings[0]
                    face_distances = face_recognition.face_distance(knownFaceEncodings, face_encoding)
                    best_match_index = face_distances.argmin()
                    best_match_distance = face_distances[best_match_index]

                    if best_match_distance < threshold:
                        predicted_label = knownFaceNames[best_match_index]
                    else:
                        predicted_label = "Unknown"

                    true_labels.append(os.path.splitext(filename)[0])
                    predicted_labels.append(predicted_label)

    accuracy = accuracy_score(true_labels, predicted_labels)
    return accuracy

def find_best_threshold(test_images_dir, num_tests=100):
    with open('FaceEncoding.pickle', 'rb') as f:
        knownFaceEncodings, knownFaceNames = pickle.load(f)

    thresholds = np.arange(0.3, 0.8, 0.05)
    best_threshold = 0.5
    best_accuracy = 0

    for threshold in thresholds:
        accuracy = evaluate_threshold(knownFaceEncodings, knownFaceNames, threshold, test_images_dir, num_tests)
        print(f"Threshold: {threshold:.2f}, Accuracy: {accuracy:.2f}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold

    print(f"Best Threshold: {best_threshold:.2f} with Accuracy: {best_accuracy:.2f}")

test_images_dir = 'test_images'
find_best_threshold(test_images_dir, num_tests=100)