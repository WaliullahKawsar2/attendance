from flask import Flask, render_template, request, jsonify, send_file
import cv2
import pandas as pd
from datetime import datetime
import os
import numpy as np
import base64
import io
from PIL import Image

app = Flask(__name__)

# Directory containing known face images
known_faces_dir = 'images'

# Load known faces using OpenCV
known_faces = []
known_names = []
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load known faces
def load_known_faces():
    global known_faces, known_names
    known_faces = []
    known_names = []
    
    # Load actual face images from the images directory
    for filename in os.listdir(known_faces_dir):
        if filename.endswith(('.jpg', '.png', '.jpeg')):
            image_path = os.path.join(known_faces_dir, filename)
            image = cv2.imread(image_path)
            if image is not None:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                if len(faces) > 0:
                    x, y, w, h = faces[0]
                    face_roi = gray[y:y+h, x:x+w]
                    face_roi = cv2.resize(face_roi, (100, 100))
                    known_faces.append(face_roi)
                    known_names.append(filename.split('.')[0])
                    print(f"Loaded face: {filename.split('.')[0]}")
    
    print(f"Loaded {len(known_faces)} known faces: {known_names}")

# Function to recognize face using template matching
def recognize_face(image_data):
    try:
        # Decode base64 image
        image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert PIL image to OpenCV format
        opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
        
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) == 0:
            return None, "No faces detected in the image."
        
        # Get the first detected face
        x, y, w, h = faces[0]
        face_roi = gray[y:y+h, x:x+w]
        face_roi = cv2.resize(face_roi, (100, 100))
        
        # Compare with known faces using template matching
        best_match = None
        best_score = 0
        
        for i, known_face in enumerate(known_faces):
            result = cv2.matchTemplate(face_roi, known_face, cv2.TM_CCOEFF_NORMED)
            score = np.max(result)
            
            if score > best_score and score > 0.5:  # Threshold for matching
                best_score = score
                best_match = known_names[i]
        
        if best_match:
            return best_match, f"Recognized as {best_match} (confidence: {best_score:.2f})"
        else:
            return None, "Face not recognized. Please try again."
            
    except Exception as e:
        return None, f"Error processing image: {str(e)}"

# Function to mark attendance
def mark_attendance(student_name, file='attendance.xlsx'):
    now = datetime.now()
    current_date = now.strftime("%Y-%m-%d")
    current_time = now.strftime("%H:%M:%S")
    
    try:
        # For Vercel deployment, we'll use a simple in-memory storage
        # In production, you'd use a database
        attendance_data = getattr(app, 'attendance_data', [])
        
        # Check if attendance already marked for today
        today_attendance = [record for record in attendance_data 
                          if record['Name'] == student_name and record['Date'] == current_date]
        
        if len(today_attendance) > 0:
            return False, f"Attendance already marked for {student_name} today."
        
        new_record = {"Name": student_name, "Date": current_date, "Time": current_time}
        attendance_data.append(new_record)
        app.attendance_data = attendance_data
        
        return True, f"Attendance marked for {student_name} at {current_time}"
        
    except Exception as e:
        return False, f"Error marking attendance: {str(e)}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/recognize', methods=['POST'])
def recognize():
    try:
        data = request.get_json()
        image_data = data.get('image')
        
        if not image_data:
            return jsonify({'success': False, 'message': 'No image data received'})
        
        student_name, message = recognize_face(image_data)
        
        if student_name:
            success, attendance_message = mark_attendance(student_name)
            return jsonify({
                'success': True,
                'student_name': student_name,
                'message': message,
                'attendance_marked': success,
                'attendance_message': attendance_message
            })
        else:
            return jsonify({
                'success': False,
                'message': message
            })
            
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})

@app.route('/api/attendance')
def get_attendance():
    try:
        # Return in-memory attendance data
        attendance_data = getattr(app, 'attendance_data', [])
        return jsonify({'success': True, 'data': attendance_data})
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})

@app.route('/api/known-faces')
def get_known_faces():
    return jsonify({'success': True, 'faces': known_names})

# Initialize the app
if __name__ == '__main__':
    load_known_faces()
    print(f"Loaded {len(known_names)} known faces: {known_names}")
    app.run(debug=True, host='0.0.0.0', port=5000)
else:
    # For Vercel deployment
    load_known_faces()
