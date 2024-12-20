import dlib
import cv2
import pickle
import pandas as pd
import numpy as np

def markAttendance():
    with open('FaceEncoding.pickle', 'rb') as f:
        knownFaceEncodings, knownFaceNames = pickle.load(f)

    attendanceFile = 'attendance.xlsx'
    allStudents = set(knownFaceNames)
    studentsMarked = set()

    faceDetector = dlib.get_frontal_face_detector()
    shapePredictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    faceRecognizer = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')

    videoCapture = cv2.VideoCapture(0)

    while True:
        ret, frame = videoCapture.read()
        if not ret:
            print("Failed to grab frame.")
            break

        rgbFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = faceDetector(rgbFrame)

        for face in faces:
            shape = shapePredictor(rgbFrame, face)
            faceEncoding = faceRecognizer.compute_face_descriptor(rgbFrame, shape)

            distances = [np.linalg.norm(np.array(faceEncoding) - np.array(knownFace)) for knownFace in knownFaceEncodings]
            minDistance = min(distances) if distances else float('inf')
            threshold = 0.6
            name = "Unknown"

            if minDistance < threshold:
                bestMatchIndex = distances.index(minDistance)
                name = knownFaceNames[bestMatchIndex]
                if name not in studentsMarked:
                    studentsMarked.add(name)

            left, top, right, bottom = face.left(), face.top(), face.right(), face.bottom()
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        cv2.imshow('Attendance System', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    videoCapture.release()
    cv2.destroyAllWindows()

    attendanceData = []
    for student in allStudents:
        status = "Present" if student in studentsMarked else "Absent"
        attendanceData.append({'Name': student, 'Status': status})

    attendanceDF = pd.DataFrame(attendanceData)
    attendanceDF.to_excel(attendanceFile, index=False)
    print(f"Attendance saved to {attendanceFile}")

markAttendance()
