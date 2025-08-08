from flask import Flask, render_template, request, jsonify
import pandas as pd
from datetime import datetime
import os
import base64
import io
from PIL import Image
import hashlib

app = Flask(__name__)

# Directory containing known face images
known_faces_dir = 'images'

# Store known face hashes for lightweight comparison
known_face_hashes = {}
known_names = []

# Load known faces using PIL (lighter than OpenCV)
def load_known_faces():
    global known_face_hashes, known_names
    known_face_hashes = {}
    known_names = []
    
    # Load actual face images from the images directory
    for filename in os.listdir(known_faces_dir):
        if filename.endswith(('.jpg', '.png', '.jpeg')):
            image_path = os.path.join(known_faces_dir, filename)
            try:
                image = Image.open(image_path)
                # Convert to grayscale and resize for consistent hashing
                image = image.convert('L').resize((64, 64))
                
                # Create a simple hash of the image
                image_bytes = image.tobytes()
                image_hash = hashlib.md5(image_bytes).hexdigest()
                
                known_face_hashes[image_hash] = filename.split('.')[0]
                known_names.append(filename.split('.')[0])
                print(f"Loaded face: {filename.split('.')[0]}")
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    
    print(f"Loaded {len(known_names)} known faces: {known_names}")

# Function to recognize face using image hashing
def recognize_face(image_data):
    try:
        # Decode base64 image
        image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to grayscale and resize for consistent hashing
        image = image.convert('L').resize((64, 64))
        
        # Create hash of the uploaded image
        image_bytes = image.tobytes()
        image_hash = hashlib.md5(image_bytes).hexdigest()
        
        # Compare with known faces
        best_match = None
        best_similarity = 0
        
        for known_hash, name in known_face_hashes.items():
            # Simple similarity based on hash difference
            similarity = sum(a == b for a, b in zip(image_hash, known_hash)) / len(image_hash)
            
            if similarity > best_similarity and similarity > 0.8:  # 80% similarity threshold
                best_similarity = similarity
                best_match = name
        
        if best_match:
            return best_match, f"Recognized as {best_match} (similarity: {best_similarity:.2f})"
        else:
            return None, "Face not recognized. Please try again."
            
    except Exception as e:
        return None, f"Error processing image: {str(e)}"

# Function to mark attendance
def mark_attendance(student_name):
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
