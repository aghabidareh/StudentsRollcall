import face_recognition
import os
import pickle
import cv2


# Load student images and encode faces
def ExtractAndSaveFaces():
    knownFaceEncodings = []
    knownFaceNames = []
    validExtentions = ['.jpg', '.jpeg', '.png']  # Valid image formats

    for filename in os.listdir('students'):
        if any(filename.endswith(ext) for ext in validExtentions):
            try:
                imagePath = os.path.join('students', filename)
                image = cv2.imread(imagePath)  # Load image with OpenCV

                if image is None:
                    print(f"Image could not be loaded: {filename}")
                    continue

                # Convert to RGB
                rgbImage = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # Encode the face
                faceEncodings = face_recognition.face_encodings(rgbImage)
                if faceEncodings:
                    knownFaceEncodings.append(faceEncodings[0])
                    student_name = os.path.splitext(filename)[0]  # Extract name from filename
                    knownFaceNames.append(student_name)
                else:
                    print(f"No face found in image: {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    # Save the encodings to a file
    with open('FaceEncoding.pickle', 'wb') as f:
        pickle.dump((knownFaceEncodings, knownFaceNames), f)

    print("Face encodings saved successfully!")


# Call this function once to extract and save the faces
ExtractAndSaveFaces()