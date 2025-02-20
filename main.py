import os
import cv2
import face_recognition
import pandas as pd
from datetime import datetime

# تنظیمات اولیه
STUDENTS_DIR = "students"
OUTPUT_EXCEL = "attendance.xlsx"
UNKNOWN_NAME = "Unknown"
THRESHOLD = 0.5  # حد آستانه شباهت

# آماده سازی دیتابیس
def load_student_data():
    known_faces = []
    known_names = []
    
    for filename in os.listdir(STUDENTS_DIR):
        if filename.startswith('.'):
            continue  # Skip hidden files
        name = os.path.splitext(filename)[0]
        image_path = os.path.join(STUDENTS_DIR, filename)
        image = face_recognition.load_image_file(image_path)
        encoding = face_recognition.face_encodings(image)[0]
        known_faces.append(encoding)
        known_names.append(name)
    
    return known_faces, known_names

# ذخیره اطلاعات در اکسل
def save_to_excel(attendance_dict):
    df = pd.DataFrame.from_dict(attendance_dict, orient='index',
                               columns=['Status', 'First Seen', 'Last Seen'])
    df.to_excel(OUTPUT_EXCEL)
    print(f"Attendance saved to {OUTPUT_EXCEL}")

# تشخیص و ثبت حضور
def main():
    known_faces, known_names = load_student_data()
    video_capture = cv2.VideoCapture(0)
    attendance = {}
    
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        # پردازش فریم
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # مقایسه چهره‌ها
            matches = face_recognition.compare_faces(known_faces, face_encoding, THRESHOLD)
            name = UNKNOWN_NAME
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            if True in matches:
                first_match_index = matches.index(True)
                name = known_names[first_match_index]

                # به روزرسانی اطلاعات حضور
                if name not in attendance:
                    attendance[name] = ['Present', current_time, current_time]
                else:
                    attendance[name][2] = current_time

            # رسم مستطیل و نام
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(frame, name, (left + 6, bottom - 6), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # نمایش خروجی
        cv2.imshow('Face Recognition Attendance', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # ذخیره نهایی و تمیزکاری
    video_capture.release()
    cv2.destroyAllWindows()
    
    # ثبت غیبت برای دانش‌آموزان شناسایی نشده
    for name in known_names:
        if name not in attendance:
            attendance[name] = ['Absent', '-', '-']
    
    save_to_excel(attendance)

if __name__ == "__main__":
    main()