import tkinter as tk
from tkinter import messagebox
import cv2
from __init__ import BarcodeDetector  # Assuming barcode_scanner/__init__.py is renamed to barcode_scanner.py
# Load face, eye, and smile cascades

root = tk.Tk()
root.title("Barcode Scanner and Face Detection")

face_cap = cv2.CascadeClassifier("D:/kishan/Git_and_Github/face-detect/haarcascade_frontalface_default.xml")
eye_cap = cv2.CascadeClassifier("D:/kishan/Git_and_Github/face-detect/haarcascade_eye.xml")
smile_cap = cv2.CascadeClassifier("D:/kishan/Git_and_Github/face-detect/haarcascade_smile.xml")

scan_barcode_button = tk.Button(root, text="Scan Barcode", command=lambda: start_capture("b"))
scan_barcode_button.pack()

face_detection_button = tk.Button(root, text="Face/Feature Detection", command=lambda: start_capture("f"))
face_detection_button.pack()

stop_button = tk.Button(root, text="Stop", command=lambda: stop_capture("c"))
stop_button.pack()
# Initialize video capture
video_cap = cv2.VideoCapture(0)
# Function for face/eye/smile detection
video_data = video_cap.read()
def detect_features(col, faces):
    eyes = []
    smiles = []
    for (x, y, w, h) in faces:
        cv2.rectangle(video_data, (x, y), (x + w, y + h), (0, 254, 0), 2)

        # Detect eyes within the face region
        roi_gray = col[y:y+h, x:x+w]
        eyes.extend(eye_cap.detectMultiScale(roi_gray, scaleFactor=1.1, minSize=(10, 10)))

        # Detect smiles within the face region (reuse grayscale ROI)
        smiles.extend(smile_cap.detectMultiScale(roi_gray, scaleFactor=1.1, minSize=(20, 20)))

    # Draw rectangles around detected eyes and smiles
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(video_data, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (0, 0, 255), 2)
    for (sx, sy, sw, sh) in smiles:
        cv2.rectangle(video_data, (x + sx, y + sy), (x + sx + sw, y + sy + sh), (253, 0, 0), 2)
# Global variables (accessed by button functions)
current_mode = tk.StringVar()
is_running = False

def start_capture():
    global is_running
    if not is_running:
        is_running = True
        current_mode.set(selected_mode.get())  # type: ignore # Set detection mode when starting capture
        capture_thread = threading.Thread(target=capture_loop) # type: ignore
        capture_thread.start()

def stop_capture():
    global is_running
    is_running = False

def capture_loop():
    global is_running

    while is_running:
        ret, video_data = video_cap.read()

        col = cv2.cvtColor(video_data, cv2.COLOR_BGR2GRAY)

        detection_mode = current_mode.get()
        if detection_mode == 'f':
            # Detect faces
            faces = face_cap.detectMultiScale(
                col,
                scaleFactor=1.1,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )

            # Call function for face/eye/smile detection
            detect_features(col, faces)

        elif detection_mode == 'b':
            # Use barcode scanner (assuming a working `BarcodeDetector` class)
            barcode_detector = BarcodeDetector()  # Initialize detector if needed
            results = barcode_detector.detectAndDecodeWithType(video_data)
            # Process barcode results (display data, take actions)
            if results[0]:  # Check success flag
                print("Barcode detected:", results[1][0])  # Assuming first barcode information

        else:
            print("Invalid mode:", detection_mode)

        if not ret:
            print("Error: Can't retrieve frame from video stream.")
            break

        cv2.imshow("Video_on", video_data)

        if cv2.waitKey(1) == ord("q"):
            break

    # Release resources after stopping capture
    video_cap.release()
    cv2.destroyAllWindows()
    is_running = False

# Create the tkinter window
