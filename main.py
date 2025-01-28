import cv2
import pickle
import pandas as pd
import face_recognition

def markAttendance():
    with open('FaceEncoding.pickle', 'rb') as f:
        knownFaceEncodings, knownFaceNames = pickle.load(f)

    attendanceFile = 'attendance.xlsx'
    allStudents = set(knownFaceNames)
    studentsMarked = set()

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
            matches = face_recognition.compare_faces(knownFaceEncodings, faceEncoding)
            name = "Unknown"

            if True in matches:
                firstMatchIndex = matches.index(True)
                name = knownFaceNames[firstMatchIndex]
                if name not in studentsMarked:
                    studentsMarked.add(name)

            top, right, bottom, left = faceLocation
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