import cv2
import numpy as np

# Specify the index of your external webcam (try different indices if needed)
external_webcam_index = 1

cap = cv2.VideoCapture(external_webcam_index)

while True:
    ret, frame = cap.read()

    # Convert frame to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define range of blue color in HSV
    lower_blue = np.array([10, 50, 50])
    upper_blue = np.array([130, 255, 255])

    # Define range of white color in HSV
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 30, 255])

    # Threshold the HSV images to get masks for blue and white colors
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    white_mask = cv2.inRange(hsv, lower_white, upper_white)

    # Combine the masks to get a final mask
    final_mask = cv2.bitwise_or(blue_mask, white_mask)

    # Apply GaussianBlur to the blue mask to reduce noise
    blurred_mask = cv2.GaussianBlur(final_mask, (3, 3), 0)
    # Bitwise-AND mask and original image
    result = cv2.bitwise_and(frame, frame, mask=blurred_mask)

    cv2.imshow('External Webcam', result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()