import os
import cv2
import numpy as np
import face_recognition
import pandas as pd
from datetime import datetime
from threading import Thread

class VideoStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        self.grabbed, self.frame = self.stream.read()
        self.stopped = False

    def start(self):
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while not self.stopped:
            self.grabbed, self.frame = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True

class AttendanceSystem:
    def __init__(self, students_folder="students", threshold=0.5):
        self.students_folder = students_folder
        self.threshold = threshold
        self.known_encodings = {}
        self.attendance = {}
        self.load_student_encodings()
        
    def load_student_encodings(self):
        for filename in os.listdir(self.students_folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                name = os.path.splitext(filename)[0]
                image_path = os.path.join(self.students_folder, filename)
                image = face_recognition.load_image_file(image_path)
                encodings = face_recognition.face_encodings(image)
                if encodings:
                    self.known_encodings[name] = encodings[0]
                else:
                    print(f"⚠️ No face detected in {filename}")

    def process_frame(self, frame):
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5) 
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        face_locations = face_recognition.face_locations(rgb_small_frame, model="hog") 
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            name = self.recognize_face(face_encoding)
            self.update_attendance(name)
            self.draw_face_box(frame, top * 2, right * 2, bottom * 2, left * 2, name) 
        
        return frame

    def recognize_face(self, face_encoding):
        distances = face_recognition.face_distance(
            list(self.known_encodings.values()), 
            face_encoding
        )
        min_index = np.argmin(distances)
        min_distance = distances[min_index]
        
        if min_distance < self.threshold:
            return list(self.known_encodings.keys())[min_index]
        return "Unknown"

    def update_attendance(self, name):
        if name != "Unknown":
            now = datetime.now()
            if name not in self.attendance:
                self.attendance[name] = {
                    'status': 'Present',
                    'first_seen': now,
                    'last_seen': now
                }
            else:
                self.attendance[name]['last_seen'] = now

    def draw_face_box(self, frame, top, right, bottom, left, name):
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left + 6, top - 6), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    def generate_report(self):
        all_students = set(self.known_encodings.keys())
        present_students = set(self.attendance.keys())
        absent_students = all_students - present_students
        
        data = []
        for student in all_students:
            status = 'Present' if student in present_students else 'Absent'
            record = {
                'Name': student,
                'Status': status,
                'First_Seen': self.attendance.get(student, {}).get('first_seen', 'N/A'),
                'Last_Seen': self.attendance.get(student, {}).get('last_seen', 'N/A')
            }
            data.append(record)
        
        df = pd.DataFrame(data)
        df.to_excel("attendance.xlsx", index=False)
        print("✅ Report generated successfully!")

def main():
    system = AttendanceSystem()
    video_stream = VideoStream(src=0).start() 

    try:
        while True:
            frame = video_stream.read()
            processed_frame = system.process_frame(frame)
            cv2.imshow('Attendance System', processed_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        video_stream.stop()
        cv2.destroyAllWindows()
        system.generate_report()

if __name__ == "__main__":
    main()