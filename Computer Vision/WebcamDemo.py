import cv2
import numpy as np

class Sphere:
    def __init__(self, color):
        self.color = color
        self.center = None
        self.direction = None
        self.radius = None

    def update_center(self, all_circles):
        print(f"Attempting to update {self.color} with all circles {all_circles}")
        if len(all_circles) == 0:
            pass
        elif len(all_circles) == 1:
            self.center = all_circles[0][0]
            self.radius = all_circles[0][1]
        else:  # Multiple circles are detected, we must find the most accurate circle
            new_circle = Sphere(self.color)  # Create a new Sphere instance for comparison
            new_circle.center = self.center  # Copy the current center to the new instance for comparison
            new_circle.radius = self.radius  # Copy the current radius to the new instance for comparison

            # Check if the radius is available for the new circle
            if new_circle.radius is not None:
                # Iterate over all circles and find the most similar one
                min_similarity_score = float('inf')
                most_similar_circle = None

                for circle_info in all_circles:
                    existing_circle = Sphere(self.color)
                    existing_circle.center, existing_circle.radius = circle_info

                    # Check if the radius is available for the existing circle
                    if existing_circle.radius is not None:
                        similarity_score = similarity_measure(new_circle, existing_circle)

                        if similarity_score < min_similarity_score:
                            min_similarity_score = similarity_score
                            most_similar_circle = existing_circle

                # Update the current Sphere with the most similar circle's properties
                if most_similar_circle is not None and min_similarity_score <100:
                    self.center = most_similar_circle.center
                    self.radius = most_similar_circle.radius
                    print(f"{self.color} ball has been updated to have a center of {self.center} and a radius of {self.radius} which had a similarity score of {min_similarity_score}")

    def update_direction(self, frame, prev_frame, sphere_mask):
        if prev_frame is not None:
            
            prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv2.imshow("Frame",frame_gray)
            cv2.imshow("Mask",sphere_mask)
            
            prev_frame_gray = cv2.bitwise_and(prev_frame_gray, sphere_mask)
            frame_gray = cv2.bitwise_and(frame_gray, sphere_mask)
            
            cv2.imshow("Frame2", frame_gray)
    # Calculate optical flow using Lucas-Kanade method
            optical_flow = cv2.calcOpticalFlowPyrLK(
            prev_frame_gray, frame_gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.1, flags=0
            )

        # Calculate the average optical flow vector
            avg_optical_flow = np.mean(optical_flow, axis=(0, 1))

        # Normalize the optical flow vector
            norm_optical_flow = avg_optical_flow / np.linalg.norm(avg_optical_flow)

        # Update the direction based on the average optical flow
            self.direction = norm_optical_flow
            print(f"Direction{self.direction}")


def euclidean_distance(center1, center2):
    return np.linalg.norm(np.array(center1) - np.array(center2))

def similarity_measure(new_circle, existing_circle):
    # Adjust the weight as needed based on the importance of each property
    weight_radius = 0.4
    weight_distance = 0.6

    radius_difference = abs(new_circle.radius - existing_circle.radius)
    distance = euclidean_distance(new_circle.center, existing_circle.center)

    # Combine the measures with weights
    similarity_score = (weight_radius * radius_difference) + (weight_distance * distance)

    return similarity_score
def capture_video(webcam_index):
    cap = cv2.VideoCapture(webcam_index)
    return cap

def process_frame(frame):
    # Convert frame to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define color ranges
    color_ranges = {
        'blue': ((90, 50, 50), (120, 255, 255)),  # Adjusted green channel values
        'green': ((50, 50, 50), (80, 255, 255)),
        'pink': ((130, 50, 50), (200, 255, 255)),
        'white': ((0, 0, 200), (180, 30, 255)),
    }

    # Threshold the HSV images to get masks
    masks = {color: cv2.inRange(hsv, lower, upper) for color, (lower, upper) in color_ranges.items()}
    cv2.imshow('Pink',masks['pink'])
    cv2.imshow('Green',masks['green'])
    cv2.imshow('Blue',masks['blue'])
    cv2.imshow('White',masks['white'])
    # Combine the masks
    final_mask = cv2.bitwise_or(masks['blue'] | masks['white'], masks['pink'] | masks['white'] | masks['green'])

    # Apply GaussianBlur to the final mask to reduce noise
    blurred_mask = cv2.GaussianBlur(final_mask, (3, 3), 0)
    cv2.imshow("Blur", blurred_mask)
    
    
    return blurred_mask, masks  # Returning masks for further use

def detect_circles(frame, mask, masks):
    # Use Hough Circle Transform to detect circles in the mask
    circles = cv2.HoughCircles(
        mask,
        cv2.HOUGH_GRADIENT,
        dp=5,
        minDist=20,
        param1=40,
        param2=10,
        minRadius=0,
        maxRadius=50,
    )
    detected_circles = [[], [], []]
    if circles is not None:
        circles = np.uint16(np.around(circles))

        for i in circles[0, :]:
            center = (i[0], i[1])
            radius = i[2]

            # Extract the circular region from the final mask
            roi_mask = mask[i[1] - radius : i[1] + radius, i[0] - radius : i[0] + radius]

            roi_blue_mask = masks['blue'][i[1] - radius : i[1] + radius, i[0] - radius : i[0] + radius]
            # Calculate the total pixel count within the circular region
            total_pixel_count = roi_blue_mask.size
            # Calculate the blue pixel count within the circular region
            blue_pixel_count = np.sum(roi_blue_mask == 255)

             # Adjust the threshold based on your requirements
            if (blue_pixel_count / total_pixel_count) > 0.25:  
                detected_circles[0].append((center, radius))
                # Draw the circle outline on the original frame
                #cv2.circle(frame, center, radius, (255, 0, 0), 2)

                # Draw the center of the circle on the original frame
                #cv2.circle(frame, center, 2, (0, 0, 255), 3)

                # Print the center coordinates
                #print("Center coordinates:", center, "blue with: ", (blue_pixel_count / total_pixel_count *100))

            roi_pink_mask = masks['pink'][i[1] - radius : i[1] + radius, i[0] - radius : i[0] + radius]
            
            # Calculate the total pixel count within the circular region
            total_pixel_count = roi_pink_mask.size

             # Calculate the blue pixel count within the circular region
            pink_pixel_count = np.sum(roi_pink_mask == 255)

             # Adjust the threshold based on your requirements
            if (pink_pixel_count / total_pixel_count) > 0.25:  
                detected_circles[1].append((center, radius))
                

                # Print the center coordinates
                #print("Center coordinates:", center, "pink with: ", (pink_pixel_count / total_pixel_count *100))
                
            roi_green_mask = masks['green'][i[1] - radius : i[1] + radius, i[0] - radius : i[0] + radius]

            # Calculate the total pixel count within the circular region
            total_pixel_count = roi_green_mask.size
            # Calculate the blue pixel count within the circular region
            green_pixel_count = np.sum(roi_green_mask == 255)

             # Adjust the threshold based on your requirements
            if (green_pixel_count / total_pixel_count) > 0.25:  
                detected_circles[2].append((center, radius))
                # Draw the circle outline on the original frame
                #cv2.circle(frame, center, radius, (0, 255, 0), 2)

                # Draw the center of the circle on the original frame
                #cv2.circle(frame, center, 2, (0, 0, 255), 3)

                # Print the center coordinates
                #print("Center coordinates:", center, "green with: ", (green_pixel_count / total_pixel_count *100))
        
    return frame, detected_circles

def updateSpheres(spheres, allCircles, frame, prev_frame, masks):
    for sphere in spheres:
        match sphere.color:
            case "blue":
                sphere.update_center(allCircles[0])
                sphere.update_direction(frame, prev_frame, masks['blue'])

                draw_arrow(frame, sphere, (255, 0, 0))  
                # Draw the circle outline on the original frame
                cv2.circle(frame, sphere.center, sphere.radius, (255, 0, 0), 2)
                # Draw the center of the circle on the original frame
                cv2.circle(frame, sphere.center, 2, (255, 0, 0), 3)
            case "pink":
                sphere.update_center(allCircles[1])
                sphere.update_direction(frame, prev_frame, masks['pink'])
                
                draw_arrow(frame, sphere, (255, 110, 200))  
                # Draw the circle outline on the original frame
                cv2.circle(frame, sphere.center, sphere.radius, (255, 110, 200), 2)

                # Draw the center of the circle on the original frame
                cv2.circle(frame, sphere.center, 2, (0, 0, 255), 3)
            case "green":
                sphere.update_center(allCircles[2])
                sphere.update_direction(frame, prev_frame, masks['green'])
                draw_arrow(frame, sphere, (0, 0, 255))  
                # Draw the circle outline on the original frame
                cv2.circle(frame, sphere.center, sphere.radius, (0, 255, 0), 2)
                # Draw the center of the circle on the original frame
                cv2.circle(frame, sphere.center, 2, (0, 0, 255), 3)
        print(f"{sphere.color} ball updated")
def draw_arrow(frame, sphere,color):
    arrow_length = 30  # You can adjust the arrow length as needed
    print("Gets here")
    if sphere.direction is not None and sphere.center is not None:
        print("Gets there")
        print(sphere.center)
        print(sphere.direction[0])
    # Calculate the endpoint of the arrow
        endpoint = (int(sphere.center[0] + sphere.direction[0] * arrow_length), int(sphere.center[1] + sphere.direction[1] * arrow_length))

    # Draw the arrow on the frame
        cv2.arrowedLine(frame, sphere.center, endpoint, color, 2)        
def main():
    external_webcam_index = 1
    cap = capture_video(external_webcam_index)

    blueSphero = Sphere("blue")
    pinkSphero = Sphere("pink")
    greenSphero = Sphere("green")
    spheres = [blueSphero, pinkSphero,greenSphero]
    prev_frame = None
    frame_skip = 5  # Adjust the skip factor as needed
    frame_count = 0
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Failed to capture frame")
            break
        
        # Append the current frame to the buffer
        #frame_buffer.append(frame.copy())

# Limit the buffer size to 3 frames (adjust as needed)
        #frame_buffer = frame_buffer[-2:]

# Use the last frame in the buffer as the current frame
        #current_frame = frame_buffer[-1]

# Use the second-to-last frame in the buffer as the previous frame
        

        frame_count += 1
        if frame_count % frame_skip != 0:
            # Skip frames
            continue
        frame = cv2.resize(frame, None, fx=0.2, fy=0.2)
        #prev_frame = frame_buffer[-2] if len(frame_buffer) >= 2 else None
        processed_mask, masks = process_frame(frame)
        result_frame,allCircles = detect_circles(frame, processed_mask, masks)
        updateSpheres(spheres, allCircles, frame, prev_frame, masks)
        prev_frame = frame
        cv2.imshow('Result', result_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()