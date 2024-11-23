import face_recognition
import cv2
import pickle
import pandas as pd

def markAttendance():
    with open('FaceEncoding.pickle', 'rb') as f:
        knownFaceEncodings, knownFaceNames = pickle.load(f)

    # Initialize attendance data
    attendanceFile = 'attendance.xlsx'
    allStudents = set(knownFaceNames)
    studentsMarked = set()

    videoCapture = cv2.VideoCapture(0)

    while True:
        ret, frame = videoCapture.read()
        if not ret:
            print("Failed to grab frame.")
            break

        smallFrame = cv2.resize(frame, (0, 0), fx=1, fy=1)
        enhancedFrame = cv2.cvtColor(smallFrame, cv2.COLOR_BGR2GRAY)
        rgbSmallFrame = cv2.cvtColor(cv2.equalizeHist(enhancedFrame), cv2.COLOR_GRAY2RGB)

        faceLocations = face_recognition.face_locations(rgbSmallFrame)
        faceEncodings = face_recognition.face_encodings(rgbSmallFrame, faceLocations)

        for faceEncoding, face_location in zip(faceEncodings, faceLocations):
            matches = face_recognition.compare_faces(knownFaceEncodings, faceEncoding)
            name = "Unknown"

            faceDistances = face_recognition.face_distance(knownFaceEncodings, faceEncoding)
            bestMatchIndex = faceDistances.argmin() if matches else None

            if bestMatchIndex is not None and matches[bestMatchIndex]:
                name = knownFaceNames[bestMatchIndex]
                if name not in studentsMarked:
                    studentsMarked.add(name)

            top, right, bottom, left = [v for v in face_location]
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