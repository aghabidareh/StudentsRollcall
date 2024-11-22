import face_recognition
import cv2
import pickle
import pandas as pd
import os

def markAttendance():
    # Load face encodings
    with open('FaceEncoding.pickle', 'rb') as f:
        knownFaceEncodings, knownFaceNames = pickle.load(f)

    # Initialize attendance data
    attendanceFile = 'attendance.xlsx'
    allStudents = set(knownFaceNames)  # Set of all registered students
    studentsMarked = set()  # Set to keep track of marked students

    # Open webcam
    videoCapture = cv2.VideoCapture(0)

    while True:
        ret, frame = videoCapture.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Resize frame for faster processing
        smallFrame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        enhancedFrame = cv2.cvtColor(smallFrame, cv2.COLOR_BGR2GRAY)
        rgbSmallFrame = cv2.cvtColor(cv2.equalizeHist(enhancedFrame), cv2.COLOR_GRAY2RGB)

        # Find all face encodings in the current frame
        faceLocations = face_recognition.face_locations(rgbSmallFrame)
        faceEncodings = face_recognition.face_encodings(rgbSmallFrame, faceLocations)

        for faceEncoding, face_location in zip(faceEncodings, faceLocations):
            matches = face_recognition.compare_faces(knownFaceEncodings, faceEncoding)
            name = "Unknown"  # Default name if no match found

            # Use the known face with the smallest distance
            faceDistances = face_recognition.face_distance(knownFaceEncodings, faceEncoding)
            bestMatchIndex = faceDistances.argmin() if matches else None

            if bestMatchIndex is not None and matches[bestMatchIndex]:
                name = knownFaceNames[bestMatchIndex]

                # Mark the student as present if not already marked
                if name not in studentsMarked:
                    studentsMarked.add(name)

            # Draw rectangle and name on frame
            top, right, bottom, left = [v * 4 for v in face_location]
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        # Display the frame
        cv2.imshow('Attendance System', frame)

        # Break on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Close the webcam
    videoCapture.release()
    cv2.destroyAllWindows()

    # Prepare attendance data
    attendance_data = []
    for student in allStudents:
        status = "Present" if student in studentsMarked else "Absent"
        attendance_data.append({'Name': student, 'Status': status})

    # Save attendance to Excel
    attendanceDF = pd.DataFrame(attendance_data)
    attendanceDF.to_excel(attendanceFile, index=False)
    print(f"Attendance saved to {attendanceFile}")

# Run the attendance system
markAttendance()