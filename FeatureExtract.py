import os
import pickle
import cv2
import face_recognition

def ExtractAndSaveFaces():
    knownFaceEncodings = []
    knownFaceNames = []
    validExtensions = ['.jpg', '.jpeg', '.png']

    for filename in os.listdir('students'):
        if any(filename.endswith(ext) for ext in validExtensions):
            try:
                imagePath = os.path.join('students', filename)
                image = cv2.imread(imagePath)

                if image is None:
                    print(f"Image could not be loaded: {filename}")
                    continue

                rgbImage = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                faceLocations = face_recognition.face_locations(rgbImage)
                faceEncodings = face_recognition.face_encodings(rgbImage, faceLocations)

                if len(faceEncodings) > 0:
                    for faceEncoding in faceEncodings:
                        knownFaceEncodings.append(faceEncoding)
                        student_name = os.path.splitext(filename)[0]
                        knownFaceNames.append(student_name)
                else:
                    print(f"No face found in image: {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    with open('FaceEncoding.pickle', 'wb') as f:
        pickle.dump((knownFaceEncodings, knownFaceNames), f)

    print("Face encodings saved successfully!")

ExtractAndSaveFaces()