import cv2
import mediapipe as mp
from flask import Flask, render_template, Response, request, jsonify
from flask_cors import CORS
import threading
from collections import deque
import time
import speech_recognition as sr
import pyttsx3
import json
import os
from deepface import DeepFace
from twilio.rest import Client
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize Mediapipe Hand Detector
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Initialize face detector for emotion detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Global variables
selected_button = None
detected_button = None
previous_button = None  # to prevent duplicate notifications
cap = None  # We'll initialize the camera when needed
engine = pyttsx3.init()

# Action mapping for notifications
action_mapping = {
    "1": "Call Nurse",
    "2": "Need Water",
    "3": "Need Food",
    "4": "Need Bathroom Assistance",
    "5": "Emergency!"
}

# Button names
button_names = ["CALL NURSE", "WATER", "FOOD", "BATHROOM", "EMERGENCY"]

# Patient information
patient_info = {"name": "", "bed_number": ""}

# Active features
active_features = set()

# Gesture buffer
gesture_buffer = deque(maxlen=20)  # 2 seconds buffer (20 frames at ~10 fps)

# Eye gaze tracking variables
gaze_buffer = deque(maxlen=30)  # Buffer for eye gaze stabilization
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Emotion to command mapping
emotion_to_command = {
    'angry': {'name': 'Anger', 'command': 'Patient is frustrated, needs assistance or calm down'},
    'disgust': {'name': 'Disgust', 'command': 'Patient is uncomfortable, needs hygiene or cleaning supplies'},
    'fear': {'name': 'Fear', 'command': 'Patient feels anxious, needs comfort or reassurance'},
    'happy': {'name': 'Happiness', 'command': 'Patient is comfortable, no immediate action needed'},
    'sad': {'name': 'Sadness', 'command': 'Patient is feeling down, may need emotional support or medication'},
    'surprise': {'name': 'Surprise', 'command': 'Patient is startled, check if there is an emergency or sudden change'},
    'neutral': {'name': 'Neutral', 'command': 'Patient is calm, routine check-up may be needed'}
}


# === EMAIL Notification Function ===
def send_email_notification(subject, body):
    sender_email = os.getenv("EMAIL_SENDER")
    receiver_email = os.getenv("CAREGIVER_EMAIL")
    password = os.getenv("EMAIL_PASSWORD")

    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = subject
    message.attach(MIMEText(body, "plain"))

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, message.as_string())
        server.quit()
        print("✅ Email sent!")
    except Exception as e:
        print(f"❌ Email error: {e}")


# === SMS Notification Function ===
def send_sms_notification(message_body):
    try:
        client = Client(
            os.getenv("TWILIO_ACCOUNT_SID"),
            os.getenv("TWILIO_AUTH_TOKEN")
        )
        message = client.messages.create(
            body=message_body,
            from_=os.getenv("TWILIO_PHONE"),
            to=os.getenv("CAREGIVER_PHONE")
        )
        print(f"✅ SMS sent! SID: {message.sid}")
    except Exception as e:
        print(f"❌ SMS error: {e}")


# === Trigger Notifications if Needed ===
def notify_caregiver(action_text, patient_name="", bed_number=""):
    subject = f"Patient Request: {action_text}"

    # Include patient details if available
    if patient_name and bed_number:
        body = f"Patient {patient_name} (Bed #{bed_number}) has requested: {action_text}. Please respond immediately."
    else:
        body = f"A patient has requested: {action_text}. Please respond immediately."

    send_email_notification(subject, body)
    send_sms_notification(body)


# Function to detect eye gaze direction
def detect_eye_gaze(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)

    if len(eyes) == 0:
        return None

    # Get largest eye area
    largest_eye = max(eyes, key=lambda eye: eye[2] * eye[3])
    x, y, w, h = largest_eye

    eye_roi = gray[y:y + h, x:x + w]
    _, threshold = cv2.threshold(eye_roi, 70, 255, cv2.THRESH_BINARY_INV)

    # Find the iris by identifying the largest contour
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest_contour = max(contours, key=cv2.contourArea)
    M = cv2.moments(largest_contour)

    if M["m00"] == 0:
        return None

    # Calculate center of the iris
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    # Determine gaze direction based on iris position
    eye_center_x = w // 2

    # Draw the eye ROI and iris center for visualization
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.circle(frame, (x + cx, y + cy), 2, (0, 0, 255), 3)

    # Determine gaze direction
    if cx < eye_center_x - 5:
        return "left"
    elif cx > eye_center_x + 5:
        return "right"
    else:
        return "center"


# Function to get stable eye gaze direction
def get_stable_gaze():
    if not gaze_buffer:
        return None

    # Filter out None values
    valid_gazes = [g for g in gaze_buffer if g is not None]
    if not valid_gazes:
        return None

    return max(set(valid_gazes), key=valid_gazes.count)


# Function to count raised fingers
def count_raised_fingers(hand_landmarks):
    if not hand_landmarks:
        return 0
    fingers_up = 0
    tips = [8, 12, 16, 20]  # Tip landmarks for index, middle, ring, and pinky fingers
    base = [6, 10, 14, 18]  # Base landmarks for fingers

    # Thumb check (special case)
    if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x:
        fingers_up += 1

    # Other fingers
    for tip, base_point in zip(tips, base):
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[base_point].y:
            fingers_up += 1

    return fingers_up


# Function to stabilize gesture recognition
def get_stable_gesture():
    if not gesture_buffer:
        return None
    return max(set(gesture_buffer), key=gesture_buffer.count)


# Function to process camera feed and recognize gestures, emotions, and eye gaze
def process_frame():
    global selected_button, cap, detected_button, previous_button

    if cap is None or not cap.isOpened():
        return None

    ret, frame = cap.read()
    if not ret:
        return None

    # Create a copy of the frame for display
    display_frame = frame.copy()

    # Flip the frame for better user experience
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process hand detection
    if "Hand Sign Detection" in active_features:
        results = hands.process(rgb_frame)

        # Draw hand landmarks and process gestures
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                fingers_count = count_raised_fingers(hand_landmarks)
                gesture_buffer.append(fingers_count)

            # Determine stable gesture
            stable_gesture = get_stable_gesture()
            if stable_gesture:
                selected_button = stable_gesture if 1 <= stable_gesture <= 5 else None

                # Set detected button for notification system
                if selected_button != detected_button:
                    detected_button = selected_button

                    # Send notification if button changed
                    if detected_button is not None and detected_button != previous_button:
                        button_idx = detected_button - 1
                        action_text = button_names[button_idx] if 0 <= button_idx < len(
                            button_names) else "Unknown Request"
                        notify_caregiver(action_text, patient_info["name"], patient_info["bed_number"])
                        previous_button = detected_button

    # Process emotion detection
    if "Emotion Detection" in active_features:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face_roi = frame[y:y + h, x:x + w]

            try:
                result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)

                dominant_emotion = result[0]['dominant_emotion']
                emotion = emotion_to_command.get(dominant_emotion, {'name': 'Unknown', 'command': 'Unknown emotion'})
                emotion_name = emotion['name']
                command = emotion['command']

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f'Patient: {patient_info["name"]}', (x, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                            (0, 255, 0), 2)
                cv2.putText(frame, f'Emotion: {emotion_name}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0),
                            2)
                cv2.putText(frame, f'Command: {command}', (x, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0),
                            2)

                # If emotion is fear, anger, or surprise - treat as emergency
                if dominant_emotion in ['fear', 'angry', 'surprise']:
                    # Only notify if not already notified for this emotion
                    if detected_button != 5:  # 5 is EMERGENCY
                        detected_button = 5
                        if detected_button != previous_button:
                            notify_caregiver(f"EMERGENCY: Patient showing {emotion_name}",
                                             patient_info["name"], patient_info["bed_number"])
                            previous_button = detected_button

            except Exception as e:
                print(f"Error in emotion detection: {e}")

    # Process eye gaze detection
    if "Eye Gaze Detection" in active_features:
        gaze_direction = detect_eye_gaze(frame)
        if gaze_direction:
            gaze_buffer.append(gaze_direction)
            stable_gaze = get_stable_gaze()

            if stable_gaze:
                cv2.putText(frame, f'Gaze: {stable_gaze}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

                # Map gaze direction to buttons
                new_button = None
                if stable_gaze == "left":
                    new_button = 1  # CALL NURSE
                elif stable_gaze == "right":
                    new_button = 5  # EMERGENCY
                elif stable_gaze == "center":
                    new_button = 3  # FOOD

                # Update selected button
                if new_button is not None:
                    selected_button = new_button

                    # Set detected button for notification system
                    if selected_button != detected_button:
                        detected_button = selected_button

                        # Send notification if button changed
                        if detected_button != previous_button:
                            button_idx = detected_button - 1
                            action_text = button_names[button_idx] if 0 <= button_idx < len(
                                button_names) else "Unknown Request"
                            notify_caregiver(action_text, patient_info["name"], patient_info["bed_number"])
                            previous_button = detected_button

    # Convert back to BGR for video streaming
    return frame


# Generate frames for video streaming
def generate_frames():
    while True:
        frame = process_frame()
        if frame is None:
            continue

        # Encode the frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue

        # Yield the frame in byte format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

        # Control frame rate
        time.sleep(0.1)


# Voice recognition function
def voice_recognition():
    recognizer = sr.Recognizer()

    try:
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source, timeout=5)

        command = recognizer.recognize_google(audio).lower()

        if "water" in command:
            return {"status": "success", "command": "water", "button": 2}
        elif "food" in command:
            return {"status": "success", "command": "food", "button": 3}
        elif "emergency" in command:
            return {"status": "success", "command": "emergency", "button": 5}
        elif "help" in command or "nurse" in command:
            return {"status": "success", "command": "help", "button": 1}
        elif "bathroom" in command or "toilet" in command:
            return {"status": "success", "command": "bathroom", "button": 4}
        else:
            return {"status": "unknown", "command": command}

    except sr.WaitTimeoutError:
        return {"status": "timeout"}
    except sr.UnknownValueError:
        return {"status": "error", "message": "Could not understand audio"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


# Routes
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/start_monitoring', methods=['POST'])
def start_monitoring():
    global active_features, patient_info, cap

    data = request.json
    patient_info = {
        "name": data.get("patientName", ""),
        "bed_number": data.get("bedNumber", "")
    }
    active_features = set(data.get("features", []))

    # Initialize camera if needed
    if cap is None:
        cap = cv2.VideoCapture(0)

    # Return success message
    return jsonify({"status": "success", "message": "Monitoring started"})


@app.route('/get_selected_button')
def get_selected_button():
    global detected_button, previous_button, selected_button

    # Use the selected button from the current frame processing
    current_button = selected_button

    if current_button is not None:
        button_idx = current_button - 1
        button_text = button_names[button_idx] if 0 <= button_idx < len(button_names) else "Unknown Request"

        # Set for notification system
        if current_button != detected_button:
            detected_button = current_button

            # Send notification if button changed
            if detected_button != previous_button:
                notify_caregiver(button_text, patient_info["name"], patient_info["bed_number"])
                previous_button = detected_button

        return jsonify({"button": current_button, "text": button_text})

    return jsonify({"button": None, "text": "None"})


@app.route('/set_button', methods=['POST'])
def set_button():
    global detected_button, previous_button

    data = request.get_json()
    button = data.get("button")

    if button != detected_button:
        detected_button = button

        # Send notification if button changed
        if detected_button is not None and detected_button != previous_button:
            # Convert to int for proper indexing if it's a string
            if isinstance(detected_button, str) and detected_button.isdigit():
                detected_button = int(detected_button)

            if isinstance(detected_button, int) and 1 <= detected_button <= 5:
                button_idx = detected_button - 1
                action_text = button_names[button_idx]
                notify_caregiver(action_text, patient_info["name"], patient_info["bed_number"])
            else:
                # Handle string-based action mapping
                action_text = action_mapping.get(detected_button, "Unknown Request")
                notify_caregiver(action_text, patient_info["name"], patient_info["bed_number"])

            previous_button = detected_button

    return jsonify({"status": "success"})


@app.route('/listen_voice', methods=['POST'])
def listen_voice():
    global detected_button, previous_button

    if "Voice Recognition" not in active_features:
        return jsonify({"status": "error", "message": "Voice recognition not active"})

    result = voice_recognition()

    if result.get("status") == "success" and result.get("button"):
        button = result.get("button")

        if button != detected_button:
            detected_button = button

            # Send notification if button changed
            if detected_button != previous_button:
                button_idx = detected_button - 1
                action_text = button_names[button_idx] if 0 <= button_idx < len(button_names) else "Unknown Request"
                notify_caregiver(action_text, patient_info["name"], patient_info["bed_number"])
                previous_button = detected_button

    return jsonify(result)


@app.route('/stop_monitoring', methods=['POST'])
def stop_monitoring():
    global cap, active_features, detected_button, previous_button

    if cap is not None:
        cap.release()
        cap = None

    active_features.clear()
    detected_button = None
    previous_button = None

    return jsonify({"status": "success", "message": "Monitoring stopped"})


# Clean up resources when the app is shutting down
def cleanup():
    global cap
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()


# Main
if __name__ == "__main__":
    try:
        # Create a templates directory if it doesn't exist
        if not os.path.exists('templates'):
            os.makedirs('templates')

        # Save the HTML template
        with open('templates/index.html', 'w') as f:
            f.write("""<!DOCTYPE html>
<html>
<head>
    <title>Hospital Assistant System</title>
    <style>
        /* CSS styles from your frontend */
        body {
            margin: 0;
            padding: 0;
            height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            background-color: #1a1f2c;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        }

        .monitor {
            position: relative;
            width: 90%;
            max-width: 1000px;
            aspect-ratio: 16/9;
            background-color: #2a3347;
            border-radius: 15px;
            border: 3px solid #3d4557;
            box-shadow: 0 0 30px rgba(0, 0, 0, 0.5);
            overflow: hidden;
        }

        .screen {
            height: 100%;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            color: #fff;
            padding: 20px;
        }

        .patient-screen {
            background: linear-gradient(135deg, #2a3347 0%, #1a1f2c 100%);
        }

        .feature-screen {
            background: linear-gradient(135deg, #2a3347 0%, #1a1f2c 100%);
            display: none;
        }

        .monitoring-screen {
            background: linear-gradient(135deg, #2a3347 0%, #1a1f2c 100%);
            display: none;
        }

        .user-container {
            text-align: center;
            padding: 30px;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 15px;
            width: 80%;
            max-width: 400px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }

        .icon {
            width: 80px;
            height: 80px;
            background: linear-gradient(135deg, #3498db, #2980b9);
            border-radius: 50%;
            margin: 0 auto 20px;
            display: flex;
            justify-content: center;
            align-items: center;
            box-shadow: 0 4px 15px rgba(52, 152, 219, 0.3);
        }

        .icon svg {
            width: 40px;
            height: 40px;
            color: white;
        }

        .input-field {
            width: 100%;
            padding: 12px;
            margin: 10px 0;
            background: rgba(255, 255, 255, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 8px;
            color: white;
            font-size: 16px;
            transition: all 0.3s ease;
        }

        .input-field:focus {
            outline: none;
            border-color: #3498db;
            background: rgba(255, 255, 255, 0.15);
        }

        .input-field::placeholder {
            color: rgba(255, 255, 255, 0.5);
        }

        .error-message {
            color: #e74c3c;
            font-size: 14px;
            margin-top: 5px;
            display: none;
        }

        .button {
            margin-top: 20px;
            padding: 12px 30px;
            background: linear-gradient(135deg, #3498db, #2980b9);
            color: white;
            border: none;
            border-radius: 25px;
            font-size: 16px;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(52, 152, 219, 0.3);
        }

        .button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(52, 152, 219, 0.4);
        }

        .feature-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 30px;
            padding: 20px;
            max-width: 800px;
            width: 90%;
            margin-top: 20px;
        }

        .feature-card {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 15px;
            padding: 20px;
            text-align: center;
            border: 1px solid rgba(255, 255, 255, 0.1);
            transition: all 0.3s ease;
            cursor: pointer;
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 15px;
        }

        .feature-card.selected {
            background: rgba(52, 152, 219, 0.2);
            border: 1px solid #3498db;
            box-shadow: 0 0 15px rgba(52, 152, 219, 0.3);
        }

        .feature-card:hover {
            transform: translateY(-5px);
            background: rgba(255, 255, 255, 0.08);
        }

        .feature-icon {
            width: 48px;
            height: 48px;
            color: #3498db;
        }

        .feature-title {
            font-size: 16px;
            font-weight: 500;
        }

        .process-container {
            width: 100%;
            text-align: center;
            margin-top: 20px;
            display: none;
        }

        .selected-features {
            margin: 15px 0;
            color: #3498db;
        }

        .process-button {
            padding: 12px 40px;
            background: linear-gradient(135deg, #2ecc71, #27ae60);
            color: white;
            border: none;
            border-radius: 25px;
            font-size: 16px;
            cursor: pointer;
            transition: all 0.3s ease;
        }

        .loading {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(26, 31, 44, 0.9);
            justify-content: center;
            align-items: center;
            z-index: 1000;
        }

        .loading-spinner {
            width: 50px;
            height: 50px;
            border: 5px solid #f3f3f3;
            border-top: 5px solid #3498db;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }

        .video-container {
            width: 100%;
            max-width: 640px;
            border-radius: 15px;
            overflow: hidden;
            margin-bottom: 20px;
        }

        .video-feed {
            width: 100%;
            height: auto;
            border-radius: 15px;
        }

        .action-buttons {
            display: flex;
            gap: 15px;
            flex-wrap: wrap;
            justify-content: center;
            margin-top: 20px;
        }

        .action-button {
            padding: 10px 20px;
            background: rgba(255, 255, 255, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 20px;
            color: white;
            transition: all 0.3s ease;
        }

        .action-button.active {
            background: rgba(52, 152, 219, 0.6);
            border-color: #3498db;
        }

        .status-panel {
            background: rgba(0, 0, 0, 0.2);
            padding: 15px;
            border-radius: 10px;
            margin-top: 20px;
            width: 100%;
            max-width: 640px;
        }

        .notification-badge {
            background-color: #e74c3c;
            color: white;
            border-radius: 50%;
            width: 20px;
            height: 20px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-size: 12px;
            margin-left: 8px;
            opacity: 0;
            transition: opacity 0.3s ease;
        }

        .notification-badge.show {
            opacity: 1;
        }

        .voice-button {
            padding: 12px 30px;
            background: linear-gradient(135deg, #e74c3c, #c0392b);
            color: white;
            border: none;
            border-radius: 25px;
            font-size: 16px;
            cursor: pointer;
            transition: all 0.3s ease;
            margin-top: 20px;
        }

        .voice-button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(231, 76, 60, 0.4);
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="monitor">
        <!-- First Page -->
        <div class="screen patient-screen" id="patientScreen">
            <div class="user-container">
                <div class="icon">
                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M18 20a6 6 0 0 0-12 0"/>
                        <circle cx="12" cy="10" r="4"/>
                        <circle cx="12" cy="12" r="10"/>
                    </svg>
                </div>
                <h2 style="margin: 0; color: #fff;">Patient Details</h2>
                <div class="status">Enter Information</div>
                <input type="text" id="patientName" class="input-field" placeholder="Enter Patient Name"required>
                <input type="text" id="bedNumber" class="input-field" placeholder="Enter Bed Number"required>
                <div id="errorMessage" class="error-message">Please fill in all fields</div>
                <button class="button" onclick="validateAndProceed()">Proceed</button>
            </div>
        </div>

        <!-- Second Page -->
        <div class="screen feature-screen" id="featureScreen">
            <h1 id="patientHeader" style="margin: 0; color: #fff;"></h1>
            <div id="bedInfo" style="color: #3498db; margin-bottom: 20px;"></div>
            <div class="feature-grid">
                <div class="feature-card" data-feature="Hand Sign Detection">
                    <svg xmlns="http://www.w3.org/2000/svg" class="feature-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M12 1v9M9 2v8M6 3v7M15 2v8M18 5v5M7 10h10v5c0 2.5-1.5 4-5 4s-5-1.5-5-4v-5z"/>
                    </svg>
                    <div class="feature-title">Hand Sign Detection</div>
                </div>
                <div class="feature-card" data-feature="Voice Recognition">
                    <svg xmlns="http://www.w3.org/2000/svg" class="feature-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/>
                        <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
                        <line x1="12" y1="19" x2="12" y2="22"/>
                    </svg>
                    <div class="feature-title">Voice Recognition</div>
                </div>
                <div class="feature-card" data-feature="Emotion Detection">
                    <svg xmlns="http://www.w3.org/2000/svg" class="feature-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <circle cx="12" cy="12" r="10"/>
                        <path d="M8 14s1.5 2 4 2 4-2 4-2"/>
                        <line x1="9" y1="9" x2="9.01" y2="9"/>
                        <line x1="15" y1="9" x2="15.01" y2="9"/>
                    </svg>
                    <div class="feature-title">Emotion Detection</div>
                </div>
                <div class="feature-card" data-feature="Eye Gaze Detection">
                    <svg xmlns="http://www.w3.org/2000/svg" class="feature-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M2 12s3-7 10-7 10 7 10 7-3 7-10 7-10-7-10-7Z"/>
                        <circle cx="12" cy="12" r="3"/>
                    </svg>
                    <div class="feature-title">Eye Gaze Detection</div>
                </div>
            </div>
            <div class="process-container" id="processContainer">
                <div class="selected-features" id="selectedFeatures">No features selected</div>
                <button class="process-button" onclick="startProcess()">Start Monitoring</button>
            </div>
        </div>

        <!-- Monitoring Screen -->
        <div class="screen monitoring-screen" id="monitoringScreen">
            <h1 id="monitoringHeader" style="margin: 0; color: #fff;"></h1>
            <div id="monitoringBedInfo" style="color: #3498db; margin-bottom: 10px;"></div>

            <div class="video-container" id="videoContainer">
                <img src="/video_feed" class="video-feed" id="videoFeed">
            </div>

            <div class="action-buttons" id="actionButtons">
                <div class="action-button" data-action="1">CALL NURSE</div>
                <div class="action-button" data-action="2">WATER</div>
                <div class="action-button" data-action="3">FOOD</div>
                <div class="action-button" data-action="4">BATHROOM</div>
                <div class="action-button" data-action="5">EMERGENCY</div>
            </div>

            <button class="voice-button" id="voiceButton" style="display: none;">Listen for Voice Command</button>

            <div class="status-panel">
                <div id="statusText">Ready for monitoring...</div>
            </div>

            <button class="button" style="margin-top: 20px; background: linear-gradient(135deg, #e74c3c, #c0392b);" onclick="stopMonitoring()">Stop Monitoring</button>
        </div>
    </div>

    <div class="loading" id="loadingScreen">
        <div class="loading-spinner"></div>
    </div>

    <script>
        let selectedFeatures = new Set();
        let buttonCheckInterval;

        function validateAndProceed() {
            const patientName = document.getElementById('patientName').value;
            const bedNumber = document.getElementById('bedNumber').value;
            const errorMessage = document.getElementById('errorMessage');

            if (!patientName || !bedNumber) {
                errorMessage.style.display = 'block';
                return;
            }

            errorMessage.style.display = 'none';
            document.getElementById('patientHeader').textContent = `Patient: ${patientName}`;
            document.getElementById('bedInfo').textContent = `Bed Number: ${bedNumber}`;
            document.getElementById('patientScreen').style.display = 'none';
            document.getElementById('featureScreen').style.display = 'flex';
        }

        document.querySelectorAll('.feature-card').forEach(card => {
            card.addEventListener('click', () => {
                const feature = card.getAttribute('data-feature');

                if (card.classList.contains('selected')) {
                    card.classList.remove('selected');
                    selectedFeatures.delete(feature);
                } else {
                    card.classList.add('selected');
                    selectedFeatures.add(feature);
                }

                updateSelectedFeatures();
            });
        });

        function updateSelectedFeatures() {
            const processContainer = document.getElementById('processContainer');
            const selectedFeaturesElement = document.getElementById('selectedFeatures');

            if (selectedFeatures.size > 0) {
                processContainer.style.display = 'block';
              selectedFeaturesElement.textContent = `Selected: ${Array.from(selectedFeatures).join(', ')}`;
            } else {
                processContainer.style.display = 'none';
                selectedFeaturesElement.textContent = 'No features selected';
            }
        }

        function startProcess() {
            // Show loading screen
            document.getElementById('loadingScreen').style.display = 'flex';

            const patientName = document.getElementById('patientName').value;
            const bedNumber = document.getElementById('bedNumber').value;

            // Set up monitoring screen
            document.getElementById('monitoringHeader').textContent = `Patient: ${patientName}`;
            document.getElementById('monitoringBedInfo').textContent = `Bed Number: ${bedNumber}`;

            // Check if voice recognition is enabled
            if (selectedFeatures.has('Voice Recognition')) {
                document.getElementById('voiceButton').style.display = 'block';
            }

            // Send data to backend
            fetch('/start_monitoring', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    patientName: patientName,
                    bedNumber: bedNumber,
                    features: Array.from(selectedFeatures)
                }),
            })
            .then(response => response.json())
            .then(data => {
                console.log('Success:', data);

                // Hide loading and feature screens, show monitoring screen
                document.getElementById('loadingScreen').style.display = 'none';
                document.getElementById('featureScreen').style.display = 'none';
                document.getElementById('monitoringScreen').style.display = 'flex';

                // Start polling for selected button
                startButtonPolling();
            })
            .catch((error) => {
                console.error('Error:', error);
                document.getElementById('loadingScreen').style.display = 'none';
                alert('An error occurred when starting the monitoring system.');
            });
        }

        function startButtonPolling() {
            // Clear any existing interval
            if (buttonCheckInterval) {
                clearInterval(buttonCheckInterval);
            }

            // Poll for selected button every second
            buttonCheckInterval = setInterval(() => {
                fetch('/get_selected_button')
                    .then(response => response.json())
                    .then(data => {
                        // Reset all buttons
                        document.querySelectorAll('.action-button').forEach(btn => {
                            btn.classList.remove('active');
                        });

                        // Activate selected button if any
                        if (data.button !== null) {
                            const activeBtn = document.querySelector(`.action-button[data-action="${data.button}"]`);
                            if (activeBtn) {
                                activeBtn.classList.add('active');
                                document.getElementById('statusText').textContent = `Selected action: ${data.text}`;
                            }
                        }
                    })
                    .catch(error => {
                        console.error('Error polling for button:', error);
                    });
            }, 1000);
        }

        // Voice button functionality
        document.getElementById('voiceButton').addEventListener('click', function() {
            document.getElementById('statusText').textContent = 'Listening...';

            fetch('/listen_voice', {
                method: 'POST',
            })
            .then(response => response.json())
            .then(data => {
                if (data.status === 'success') {
                    document.getElementById('statusText').textContent = `Voice command detected: ${data.command}`;

                    // Activate the corresponding button
                    if (data.button) {
                        document.querySelectorAll('.action-button').forEach(btn => {
                            btn.classList.remove('active');
                        });

                        const activeBtn = document.querySelector(`.action-button[data-action="${data.button}"]`);
                        if (activeBtn) {
                            activeBtn.classList.add('active');
                        }
                    }
                } else if (data.status === 'timeout') {
                    document.getElementById('statusText').textContent = 'Voice command timed out. Please try again.';
                } else if (data.status === 'unknown') {
                    document.getElementById('statusText').textContent = `Unrecognized command: "${data.command}"`;
                } else {
                    document.getElementById('statusText').textContent = `Error: ${data.message}`;
                }
            })
            .catch(error => {
                console.error('Error with voice recognition:', error);
                document.getElementById('statusText').textContent = 'Error with voice recognition service.';
            });
        });

        function stopMonitoring() {
            // Show loading screen
            document.getElementById('loadingScreen').style.display = 'flex';

            // Clear polling interval
            if (buttonCheckInterval) {
                clearInterval(buttonCheckInterval);
                buttonCheckInterval = null;
            }

            // Call backend to stop monitoring
            fetch('/stop_monitoring', {
                method: 'POST',
            })
            .then(response => response.json())
            .then(data => {
                console.log('Monitoring stopped:', data);

                // Reset UI
                document.getElementById('loadingScreen').style.display = 'none';
                document.getElementById('monitoringScreen').style.display = 'none';
                document.getElementById('patientScreen').style.display = 'flex';

                // Clear form fields
                document.getElementById('patientName').value = '';
                document.getElementById('bedNumber').value = '';

                // Reset selected features
                selectedFeatures.clear();
                document.querySelectorAll('.feature-card').forEach(card => {
                    card.classList.remove('selected');
                });
                updateSelectedFeatures();

                // Reset status text
                document.getElementById('statusText').textContent = 'Ready for monitoring...';
            })
            .catch((error) => {
                console.error('Error stopping monitoring:', error);
                document.getElementById('loadingScreen').style.display = 'none';
                alert('An error occurred when stopping the monitoring system.');
            });
        }
    </script>
</body>
</html>""")

        app.run(debug=True)
    finally:
        cleanup()