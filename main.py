import cv2
import pickle
import pandas as pd
import face_recognition
import os
from datetime import datetime

def markAttendance():
    with open('FaceEncoding.pickle', 'rb') as f:
        knownFaceEncodings, knownFaceNames = pickle.load(f)

    attendanceFile = 'attendance.xlsx'
    allStudents = set(knownFaceNames)
    studentsMarked = {}

    studentRecords = {}

    videoCapture = cv2.VideoCapture(0)

    while True:
        ret, frame = videoCapture.read()
        if not ret:
            print("Failed to grab frame.")
            break

        rgbFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faceLocations = face_recognition.face_locations(rgbFrame)
        faceEncodings = face_recognition.face_encodings(rgbFrame, faceLocations)

        for faceEncoding, faceLocation in zip(faceEncodings, faceLocations):
            faceDistances = face_recognition.face_distance(knownFaceEncodings, faceEncoding)
            bestMatchIndex = faceDistances.argmin()
            bestMatchDistance = faceDistances[bestMatchIndex]

            threshold = 0.55
            name = "Unknown"

            if bestMatchDistance < threshold:
                name = knownFaceNames[bestMatchIndex]

                if name not in studentRecords:
                    studentRecords[name] = {'Name': name, 'First_Registration': datetime.now(), 'Last_Registration': datetime.now()}
                else:
                    studentRecords[name]['Last_Registration'] = datetime.now()

                studentsMarked[name] = studentRecords[name]

            s = f'{name} , {bestMatchDistance:.2f}'

            top, right, bottom, left = faceLocation
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(frame, s, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        cv2.imshow('Attendance System', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    videoCapture.release()
    cv2.destroyAllWindows()

    attendanceData = []
    for student in allStudents:
        status = "Present" if student in studentsMarked else "Absent"
        first_registration = studentRecords[student]['First_Registration'] if student in studentRecords else None
        last_registration = studentRecords[student]['Last_Registration'] if student in studentRecords else None

        attendanceData.append({
            'Name': student,
            'Status': status,
            'First_Registration': first_registration,
            'Last_Registration': last_registration
        })

    attendanceDF = pd.DataFrame(attendanceData)
    attendanceDF.to_excel(attendanceFile, index=False)
    print(f"Attendance saved to {attendanceFile}")

markAttendance()