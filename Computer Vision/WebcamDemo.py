import pygame
import sys
import cv2
import numpy as np
import threading
import neat
import pickle
import math

from spherov2 import scanner
from spherov2.sphero_edu import SpheroEduAPI, Color

class Sphero:
    
    def __init__(self):
        self.center = None
        self.direction = None 
        self.radius = None
        self.speed = 0
        self.radars = []
        self.direction_to_goal = 0
        self.remaining_distance = 0
        self.prev_frame = None

    def get_data(self):
        # Get Distances To Border
        radars = self.radars
        #return_values = []
        return_values = [0, 0, 0, 0, 0]
        for i, radar in enumerate(radars):
            return_values[i] = int(radar[1] / 30)

        #direction_to_goal
        return_values.append(self.direction_to_goal)
        
        #distance to goal
        return_values.append(self.remaining_distance)

        # FACING
        return_values.append(self.direction)
        # Current speed
        return_values.append(self.speed)
        
        return return_values
    
    def update_center(self, all_circles):
        print(f"Attempting to update sphero with all circles {all_circles}")
        if all_circles is None:
            print("No circles")
            pass
        elif len(all_circles) == 1:
            self.center = all_circles[0][0]
            self.radius = all_circles[0][1]
        else:  # Multiple circles are detected, we must find the most accurate circle
            #current_circle_colour = self.colour  # Create a new Sphere instance for comparison
            current_circle_center = self.center  # Copy the current center to the new instance for comparison
            current_circle_radius = self.radius  # Copy the current radius to the new instance for comparison

                # Check if the radius is available for the new circle
           
                # Iterate over all circles and find the most similar one
            min_similarity_score = float('inf')
            most_similar_circle_center = None
            for circle_info in all_circles:
                new_circle_center, new_circle_radius = circle_info

                    # Check if the radius is available for the existing circle
                if new_circle_radius is not None:
                    similarity_score = similarity_measure(current_circle_center, current_circle_radius, new_circle_center, new_circle_radius)
                    
                    if similarity_score < min_similarity_score:
                        min_similarity_score = similarity_score
                        most_similar_circle_center = new_circle_center
                        most_similar_circle_radius = new_circle_radius

                # Update the current Sphere with the most similar circle's properties
            if most_similar_circle_center is not None and min_similarity_score <50000:
                self.center = most_similar_circle_center
                self.radius = most_similar_circle_radius
                #print(f"Replacing because the most similar had a score of {min_similarity_score}")

    def update_direction(self, frame):
        if self.prev_frame is None:
            self.prev_frame = frame
            return

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        colour_ranges = {
            'blue': ((90, 50, 50), (120, 255, 255)),  # Adjusted blue channel values
            #'green': ((50, 50, 50), (80, 255, 255)),
            #'pink': ((130, 50, 50), (200, 255, 255)),
            'white': ((0, 0, 200), (180, 30, 255)),
            }
        masks = {colour: cv2.inRange(hsv, lower, upper) for colour, (lower, upper) in colour_ranges.items()}
        mask = cv2.inRange(hsv,(85, 50, 20), (120, 255, 255) )
        final_mask = cv2.bitwise_or(mask , masks['white'] )
        blurred_mask = cv2.GaussianBlur(final_mask, (3, 3), 0)

        # Convert frames to grayscale
        prev_gray = cv2.cvtColor(self.prev_frame, cv2.COLOR_RGB2GRAY)
        current_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)


        # Calculate optical flow
        optical_flow = cv2.calcOpticalFlowFarneback(prev_gray, current_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        # Calculate the mean optical flow
        mean_flow = optical_flow.mean(axis=0)

        # Convert mean optical flow to direction
        angle = np.arctan2(mean_flow[1], mean_flow[0])
        #self.direction = np.degrees(angle)

        # Update the previous frame
        self.prev_frame = frame
    
    def draw_arrow(self, frame, colour):
        arrow_length = 30  # You can adjust the arrow length as needed
        #print("Gets here")
        if self.direction is not None and self.center is not None:
            #print("Gets there")
            #print(self.center)
            #print(self.direction[0])
            # Calculate the endpoint of the arrow
            #endpoint = (int(self.center[0] + self.direction[0] * arrow_length), int(self.center[1] + self.direction[1] * arrow_length))

            # Draw the arrow on the frame
            angle_rad = np.radians(self.direction)
            print(f"Angle_rad {angle_rad}")
            # Calculate direction components using sine and cosine
            direction_x = np.cos(angle_rad)
            direction_y = np.sin(angle_rad)
            endpoint = (int(self.center[0] + direction_x * arrow_length), int(self.center[1] + direction_y * arrow_length))
            print(f"endpoint {endpoint}")
            print(self.direction)
            print("XX")
            cv2.arrowedLine(frame, self.center, endpoint, colour, 2) 
    
    def draw_radar(self, frame):
    # Optionally Draw All Sensors / Radars
        #cv2.line(frame, self.center, self.starting_position, (255, 0, 0), 1)
        #cv2.line(frame, self.center, self.end_goal, (0, 255, 0), 1)
        for radar in self.radars:
            position = radar[0]
            cv2.line(frame, self.center, position, (0, 255, 0), 1)
            cv2.circle(frame, position, 5, (0, 255, 0), -1)

    def calibrate(self, api):
        print("YIPPEE2", api.get_heading())
        start_location = sphero.center
        api.roll(0, 10, 2)
        end_location = sphero.center
        print(f"Start {start_location}, end {end_location}")

        # Calculate vector between start and end locations
        vector = np.array(end_location) - np.array(start_location)

        # Calculate angle of the vector in radians
        angle_rad = np.arctan2(vector[1], vector[0])

        # Convert angle to degrees and ensure it's in the range of 0 to 360
        direction_degrees = np.degrees(angle_rad) % 360
        if direction_degrees >180:
            direction_degrees = direction_degrees -180
        elif direction_degrees < 180: 
            direction_degrees = direction_degrees +180
        
        # Update the direction attribute
        self.direction = direction_degrees

        print(f"Calibrated direction: {direction_degrees}")

    def check_radar(self, degree, frame):
        length = 0
        while length < 300:
            # Calculate the coordinates of the point at the given degree and distance
            x = int(self.center[0] + np.cos(np.radians(360 - (self.direction + degree))) * length)
            y = int(self.center[1] + np.sin(np.radians(360 - (self.direction + degree))) * length)
        
        # Ensure the coordinates are within the frame boundaries
            if x >= 0 and x < frame.shape[1] and y >= 0 and y < frame.shape[0]:
                # Get the BGR color values of the pixel at (x, y)
                b, g, r = frame[y, x]

                # Check if the color matches the BORDER_COLOR
                if (b, g, r) == BORDER_COLOR:
                    break

            length += 1

        # Calculate Distance To Border And Append To Radars List
        dist = int(np.sqrt(np.power(x - self.center[0], 2) + np.power(y - self.center[1], 2)))
        self.radars.append([(x, y), dist])


def euclidean_distance(center1, center2):
    return np.linalg.norm(np.array(center1) - np.array(center2))
def similarity_measure(current_circle_center, current_circle_radius, new_circle_center, new_circle_radius):
    # Adjust the weight as needed based on the importance of each property
    weight_radius = 0.6
    weight_distance = 0.4
    if current_circle_center is None:
        current_circle_center = new_circle_center
        current_circle_radius = new_circle_radius
    radius_difference = abs(new_circle_radius - current_circle_radius)
    distance = euclidean_distance(new_circle_center, current_circle_center)

    # Combine the measures with weights
    similarity_score = (weight_radius * radius_difference) + (weight_distance * distance)

    return similarity_score
def load_genome(filename):
  with open(filename, 'rb') as f:
        genome = pickle.load(f)
  return genome

def get_position():
    return 4
def get_angle():
    return 0
# Function to capture video from the camera

def detect_circles(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    colour_ranges = {
        'blue': ((90, 50, 50), (120, 255, 255)),  # Adjusted blue channel values
        #'green': ((50, 50, 50), (80, 255, 255)),
        #'pink': ((130, 50, 50), (200, 255, 255)),
        'white': ((0, 0, 200), (180, 30, 255)),
        }
    masks = {colour: cv2.inRange(hsv, lower, upper) for colour, (lower, upper) in colour_ranges.items()}
    mask = cv2.inRange(hsv,(85, 50, 20), (120, 255, 255) )
    final_mask = cv2.bitwise_or(mask , masks['white'] )

    # Apply GaussianBlur to the final mask to reduce noise
    blurred_mask = cv2.GaussianBlur(final_mask, (3, 3), 0)
    circles = cv2.HoughCircles(
        blurred_mask,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=20,
        param1=50,
        param2=15,
        minRadius=0,
        maxRadius=50,
    )
    filtered_circles = []

    if circles is not None:
        circles = np.uint16(np.around(circles))

        for i in circles[0, :]:
            center = (i[0], i[1])
            radius = i[2]

            
            
            roi_blue_mask = masks['blue'][i[1] - radius : i[1] + radius, i[0] - radius : i[0] + radius]
            cv2.imshow("Blue", masks["blue"])
            # Calculate the total pixel count within the circular region
            total_pixel_count = roi_blue_mask.size
            # Calculate the blue pixel count within the circular region
            blue_pixel_count = np.sum(roi_blue_mask == 255)

             # Adjust the threshold based on your requirements
            if (blue_pixel_count / total_pixel_count) > 0.25:  
                filtered_circles.append((center, radius))

        return filtered_circles



def update_data(sphero):
    video = cv2.VideoCapture(0)
    image = cv2.imread('mapempty.png')
    while True:
        ret, frame = video.read()
        frame = cv2.resize(frame, (1920, 1080))
        image = cv2.resize(image, (1920, 1080))
        frame_HSV = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

        # HSV MASK
        mask = cv2.inRange(frame_HSV, (35, 50, 20), (80, 255, 255))

        res = cv2.bitwise_and(frame, frame, mask=mask)

        f = frame - res
        f = np.where(f == 0, image, f)

        

        all_circles = detect_circles(frame)
        
        sphero.update_center(all_circles)
        cv2.circle(frame, sphero.center, sphero.radius, (255, 0, 0), 2)
        # Draw the center of the circle on the original frame
        cv2.circle(frame, sphero.center, 2, (255, 0, 0), 3)

        sphero.update_direction(frame)
        sphero.draw_arrow(f, (255, 0, 0))  

        sphero.radars.clear()
        if sphero.direction is not None:
        # From -90 To 120 With Step-Size 45 Check Radar
            for d in range(-90, 120, 45):  
                sphero.check_radar(d, f)
        sphero.draw_radar(f)


        cv2.imshow("video", frame)
        cv2.imshow("mask", f)

        if cv2.waitKey(25) == 27:
            break
    video.release()
    cv2.destroyAllWindows()

# Function to handle Pygame logic
def decide_action(sphero):
    # Initialize Pygame
    pygame.init()

    # Set up display
    width, height = 800, 600
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Pygame Input Example")

    # Set up player
    player_size = 50.0
    player_x = width // 2 - player_size // 2
    player_y = height // 2 - player_size // 2
    player_speed = 0

    print("Finding toy")
    toy = scanner.find_toy()
    print("Found toy")
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    with SpheroEduAPI(toy) as api:
        sphero.calibrate(api)
        while True:
            #api.set_speed(player_speed)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            # Get key states
            keys = pygame.key.get_pressed()

            # Update player position based on keys
            if keys[pygame.K_w]:
                player_y -= player_speed
                #api.set_main_led(Color(r=0, g=255, b=0))
                api.set_speed(10)
            if keys[pygame.K_s]:
                player_y += player_speed
                #api.set_main_led(Color(r=255, g=0, b=0))
                api.set_speed(0)
            if keys[pygame.K_a]:
                player_x -= player_speed
                if sphero.direction == 360:
                    sphero.direction=15
                else:
                    sphero.direction -= 15 # Left
                api.spin(-15, 0)
            if keys[pygame.K_d]:
                player_x += player_speed
                if sphero.direction==0:
                    sphero.direction=345
                else:
                    sphero.direction += 15 # Right
                api.spin(15, 0)
                




            output = net.activate(sphero.get_data())
            choice = output.index(max(output))
            print("Neural network recommends:", choice)
            if choice == 0:
                player_x += player_speed
                if sphero.direction==0:
                    sphero.direction=345
                else:
                    sphero.direction += 15 # Right
                api.spin(15, 0)
            elif choice == 1:
                player_x -= player_speed
                if sphero.direction == 360:
                    sphero.direction=15
                else:
                    sphero.direction -= 15 # Left
                api.spin(-15, 0)
            elif choice == 2:
            #    if car.speed > 4:
                sphero.speed = 5 #Go
                api.set_speed(10)
            else:
                sphero.speed = 0 # Stop
                api.set_speed(0)



                

            # Clear the screen
            screen.fill((255, 255, 255))

            # Draw the player
            pygame.draw.circle(screen, (0, 0, 255,), (int(player_x), int(player_y)), int(player_size))

            # Update the display
            pygame.display.flip()

            # Set the frames per second
            pygame.time.Clock().tick(30)

config_path = "./config.txt"
config = neat.config.Config(neat.DefaultGenome,
    neat.DefaultReproduction,
    neat.DefaultSpeciesSet,
    neat.DefaultStagnation,
    config_path)

filename = "./best_genome_1.pickle"
genome= load_genome(filename)
sphero = Sphero()

BORDER_COLOR = (234, 206, 183, 255)

# Create and start threads
data_thread = threading.Thread(target=update_data, args=(sphero,))
logic_thread = threading.Thread(target=decide_action, args=(sphero,))

data_thread.start()
logic_thread.start()
