import dlib
import os
import pickle
import cv2

def ExtractAndSaveFaces():
    knownFaceEncodings = []
    knownFaceNames = []
    validExtensions = ['.jpg', '.jpeg', '.png']
    faceDetector = dlib.get_frontal_face_detector()
    shapePredictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    faceRecognizer = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')

    for filename in os.listdir('students'):
        if any(filename.endswith(ext) for ext in validExtensions):
            try:
                imagePath = os.path.join('students', filename)
                image = cv2.imread(imagePath)

                if image is None:
                    print(f"Image could not be loaded: {filename}")
                    continue

                rgbImage = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                faces = faceDetector(rgbImage)

                if len(faces) > 0:
                    for face in faces:
                        shape = shapePredictor(rgbImage, face)
                        faceEncoding = faceRecognizer.compute_face_descriptor(rgbImage, shape)
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
