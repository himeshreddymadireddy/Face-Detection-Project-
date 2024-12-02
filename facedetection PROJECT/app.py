from flask import Flask, request, jsonify, render_template, send_file, redirect, url_for
import os
import cv2
import face_recognition
import numpy as np

app = Flask(__name__)

# Configure upload and output folders
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
STUDENT_FACES_FOLDER = 'student_faces'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Preload student face encodings
student_encodings = {}
for student_id in os.listdir(STUDENT_FACES_FOLDER):
    folder_path = os.path.join(STUDENT_FACES_FOLDER, student_id)
    if os.path.isdir(folder_path):
        encodings = []
        for file in os.listdir(folder_path):
            if file.endswith(('.jpg', '.png', '.pgm')):
                image_path = os.path.join(folder_path, file)
                image = face_recognition.load_image_file(image_path)
                face_locations = face_recognition.face_locations(image)
                if face_locations:  # If a face is detected
                    encoding = face_recognition.face_encodings(image, face_locations)[0]
                    encodings.append(encoding)
        if encodings:
            student_encodings[student_id] = encodings

def detect_and_match_faces(image):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_image)
    face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

    detected_students = set()
    all_students = set(student_encodings.keys())
    not_detected_students = all_students.copy()

    for face_encoding, face_location in zip(face_encodings, face_locations):
        matched = False
        for student_id, encodings in student_encodings.items():
            results = face_recognition.compare_faces(encodings, face_encoding, tolerance=0.6)
            if any(results):  # If at least one encoding matches
                detected_students.add(student_id)
                not_detected_students.discard(student_id)
                matched = True
                break

        # Draw bounding boxes
        top, right, bottom, left = face_location
        color = (0, 255, 0) if matched else (0, 0, 255)
        label = student_id if matched else "Unknown"
        cv2.rectangle(image, (left, top), (right, bottom), color, 2)
        cv2.putText(image, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    return image, list(detected_students), list(not_detected_students)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    # Save uploaded file
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    # Process image
    image = cv2.imread(file_path)
    processed_image, detected_students, not_detected_students = detect_and_match_faces(image)

    # Save output image
    output_file_path = os.path.join(app.config['OUTPUT_FOLDER'], f"output_{file.filename}")
    cv2.imwrite(output_file_path, processed_image)

    # Render results page
    return render_template('results.html',
                           detected=detected_students,
                           not_detected=not_detected_students,
                           output_image=f"/outputs/output_{file.filename}")


@app.route('/outputs/<filename>')
def output_image(filename):
    return send_file(os.path.join(OUTPUT_FOLDER, filename), mimetype='image/jpeg')


if __name__ == '__main__':
    app.run(debug=True)
