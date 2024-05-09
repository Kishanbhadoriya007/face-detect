import cv2
from __init__ import BarcodeDetector
#face, eyes, smile detection ke liye
face_cap = cv2.CascadeClassifier("D:/kishan/Git_and_github/face-detect/haarcascade_frontalface_default.xml")
eye_cap = cv2.CascadeClassifier("D:/kishan/Git_and_github/face-detect/haarcascade_eye.xml")
smile_cap = cv2.CascadeClassifier("D:/kishan/Git_and_github/face-detect/haarcascade_smile.xml")

#video capture ke liye
video_cap = cv2.VideoCapture(0)

while True:
    ret, video_data = video_cap.read()

    # grayscale convert for better detection
    col = cv2.cvtColor(video_data, cv2.COLOR_BGR2GRAY)

    # Using face data to detect face data
    faces = face_cap.detectMultiScale(
        col,
        scaleFactor=1.1,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    #rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(video_data, (x, y), (x + w, y + h), (0, 254, 0), 2)

        # Detect eyes within the face region
        roi_gray = col[y:y+h, x:x+w]
        eyes = eye_cap.detectMultiScale(roi_gray, scaleFactor=1.1, minSize=(10, 10))

        # Draw rectangles around detected eyes (optional)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(video_data, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (0, 0, 255), 2)

            # Detect smiles within the face region
        roi_gray = col[y:y+h, x:x+w]  # Reuse the grayscale face ROI
        smiles = smile_cap.detectMultiScale(roi_gray, scaleFactor=1.1, minSize=(20, 20))

        # Draw rectangles around detected smiles (optional)
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(video_data, (x + sx, y + sy), (x + sx + sw, y + sy + sh), (253, 0, 0), 2)  # Red for smiles

    # Error handling and window control
    if not ret:
        print("Error: Can't retrieve frame from video stream.")
        break

    cv2.imshow("Video_on", video_data)

    if cv2.waitKey(10) == ord("c"):
        break

# Release resources
video_cap.release()
cv2.destroyAllWindows()



#code for video capture only 
# video_cap = cv2.VideoCapture(0)
# while True:
#     ret, video_data = video_cap.read()

#     if not ret:
#         print("Error: Can't retrieve frame from video stream.")
#         break

#     cv2.imshow("Video_on", video_data)
#     if cv2.waitKey(10) == ord("c"):
#         break

# video_cap.release()
# cv2.destroyAllWindows()

#code for face detection, we will add new variable named face_cap
