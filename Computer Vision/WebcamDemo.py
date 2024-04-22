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
    lower_blue = np.array([80, 50, 50])
    upper_blue = np.array([130, 255, 255])

    # Define range of green color in HSV
    lower_green = np.array([50, 50, 50])
    upper_green = np.array([80, 255, 255])

    # Define range of pink color in HSV
    lower_pink = np.array([130, 50, 50])
    upper_pink = np.array([200, 255, 255])

    # Define range of white color in HSV
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 30, 255])

    # Threshold the HSV images to get masks for blue and white colors
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    pink_mask = cv2.inRange(hsv, lower_pink, upper_pink)
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    white_mask = cv2.inRange(hsv, lower_white, upper_white)

    blue_white_mask = cv2.bitwise_or(blue_mask, white_mask)
    green_white_mask = cv2.bitwise_or(green_mask, white_mask)
    pink_white_mask = cv2.bitwise_or(pink_mask, white_mask)
    # Combine the masks to get a final mask
    final_mask = cv2.bitwise_or(blue_white_mask, pink_white_mask)
    final_mask = cv2.bitwise_or(final_mask, green_white_mask)

    # Apply GaussianBlur to the final mask to reduce noise
    blurred_mask = cv2.GaussianBlur(final_mask, (3, 3), 0)

    # Use Hough Circle Transform to detect circles in the final mask
    circles = cv2.HoughCircles(
        blurred_mask,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=20,
        param1=50,
        param2=20,
        minRadius=0,
        maxRadius=50,
    )

    if circles is not None:
        # Convert circle coordinates to integers
        circles = np.uint16(np.around(circles))

        for i in circles[0, :]:
            center = (i[0], i[1])  # Center coordinates
            radius = i[2]  # Radius

            # Extract the circular region from the final mask
            roi_final_mask = final_mask[i[1] - radius : i[1] + radius, i[0] - radius : i[0] + radius]

            
            

            roi_blue_mask = blue_mask[i[1] - radius : i[1] + radius, i[0] - radius : i[0] + radius]
            # Calculate the total pixel count within the circular region
            total_pixel_count = roi_blue_mask.size
            # Calculate the blue pixel count within the circular region
            blue_pixel_count = np.sum(roi_blue_mask == 255)

             # Adjust the threshold based on your requirements
            if (blue_pixel_count / total_pixel_count) > 0.25:  
                # Draw the circle outline on the original frame
                cv2.circle(frame, center, radius, (255, 0, 0), 2)

                # Draw the center of the circle on the original frame
                cv2.circle(frame, center, 2, (0, 0, 255), 3)

                # Print the center coordinates
                print("Center coordinates:", center, "blue with: ", (blue_pixel_count / total_pixel_count *100))

            roi_pink_mask = pink_mask[i[1] - radius : i[1] + radius, i[0] - radius : i[0] + radius]
            
            # Calculate the total pixel count within the circular region
            total_pixel_count = roi_pink_mask.size

             # Calculate the blue pixel count within the circular region
            pink_pixel_count = np.sum(roi_pink_mask == 255)

             # Adjust the threshold based on your requirements
            if (pink_pixel_count / total_pixel_count) > 0.25:  
                # Draw the circle outline on the original frame
                cv2.circle(frame, center, radius, (255, 110, 200), 2)

                # Draw the center of the circle on the original frame
                cv2.circle(frame, center, 2, (0, 0, 255), 3)

                # Print the center coordinates
                print("Center coordinates:", center, "pink with: ", (pink_pixel_count / total_pixel_count *100))
                
            roi_green_mask = green_mask[i[1] - radius : i[1] + radius, i[0] - radius : i[0] + radius]

            # Calculate the total pixel count within the circular region
            total_pixel_count = roi_green_mask.size
            # Calculate the blue pixel count within the circular region
            green_pixel_count = np.sum(roi_green_mask == 255)

             # Adjust the threshold based on your requirements
            if (green_pixel_count / total_pixel_count) > 0.25:  
                # Draw the circle outline on the original frame
                cv2.circle(frame, center, radius, (0, 255, 0), 2)

                # Draw the center of the circle on the original frame
                cv2.circle(frame, center, 2, (0, 0, 255), 3)

                # Print the center coordinates
                print("Center coordinates:", center, "green with: ", (green_pixel_count / total_pixel_count *100))
        cv2.imshow('Blue', blue_mask)
        cv2.imshow('Pink', pink_mask)
        cv2.imshow('Green', green_mask) 
        cv2.imshow('Final', final_mask)


    
    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()