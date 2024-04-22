# This Code is Heavily Inspired By The YouTuber: Cheesy AI
# Code Changed, Optimized And Commented By: NeuralNine (Florian Dedov)

import math
import random
import sys
import os

import neat
import pygame
import pickle
import graphviz
import visualize


import cv2
import numpy as np

WIDTH = 1920
HEIGHT = 1080

CAR_SIZE_X = 40

BORDER_COLOR = (234, 206, 183, 255) 

current_generation = 0 # Generation counter
top_performing_genomes=[]

class Car:

    

    def __init__(self, game_map):
        # Load Car Sprite and Rotate
        self.sprite = sprite # Convert Speeds Up A Lot
        self.sprite = pygame.transform.scale(self.sprite, (CAR_SIZE_X, CAR_SIZE_X))

        self.dead_sprite = dead_sprite # Convert Speeds Up A Lot
        self.dead_sprite = pygame.transform.scale(self.dead_sprite, (CAR_SIZE_X, CAR_SIZE_X))

        self.success_sprite = success_sprite # Convert Speeds Up A Lot
        self.success_sprite = pygame.transform.scale(self.success_sprite, (CAR_SIZE_X, CAR_SIZE_X))

        self.goal_sprite = goal_sprite # Convert Speeds Up A Lot
        self.goal_sprite = pygame.transform.scale(self.goal_sprite, (CAR_SIZE_X, CAR_SIZE_X))

        self.rotated_sprite = self.sprite 


        if spawn_style == "group":
            STARTING_POSITIONS = [
            [630, 670],
            [1450, 360],
            [300, 450],
            [1229, 690]]
            self.starting_position = random.choice(STARTING_POSITIONS)

            self.end_goal = random.choice(STARTING_POSITIONS)
            while ( math.isclose(self.end_goal[0], self.starting_position[0], abs_tol=100)) or ( math.isclose(self.end_goal[1], self.starting_position[1], abs_tol=100)) or (game_map.get_at((self.end_goal[0], self.end_goal[1])) == BORDER_COLOR):
                self.end_goal = random.choice(STARTING_POSITIONS)
        
        elif spawn_style =="random":
            x_range = [CAR_SIZE_X, 1920-CAR_SIZE_X]
            y_range = [CAR_SIZE_X, 1080-CAR_SIZE_X]

            self.starting_position = [random.randint(x_range[0], x_range[1]), random.randint(y_range[0], y_range[1])]
            self.starting_center = [int(self.starting_position[0]) + CAR_SIZE_X / 2, int(self.starting_position[1]) + CAR_SIZE_X / 2]
            self.angle=0
            # Calculate Four Corners
            length = 0.5 * CAR_SIZE_X
            left_top = [self.starting_center[0] + math.cos(math.radians(360 - (self.angle + 30))) * length, self.starting_center[1] + math.sin(math.radians(360 - (self.angle + 30))) * length]
            right_top = [self.starting_center[0] + math.cos(math.radians(360 - (self.angle + 150))) * length, self.starting_center[1] + math.sin(math.radians(360 - (self.angle + 150))) * length]
            left_bottom = [self.starting_center[0] + math.cos(math.radians(360 - (self.angle + 210))) * length, self.starting_center[1] + math.sin(math.radians(360 - (self.angle + 210))) * length]
            right_bottom = [self.starting_center[0] + math.cos(math.radians(360 - (self.angle + 330))) * length, self.starting_center[1] + math.sin(math.radians(360 - (self.angle + 330))) * length]
            self.corners = [left_top, right_top, left_bottom, right_bottom]
            print("1")
            corners_invalid = True
            while corners_invalid:
              self.starting_position = [random.randint(x_range[0], x_range[1]), random.randint(y_range[0], y_range[1])]
              self.starting_center = [int(self.starting_position[0]) + CAR_SIZE_X / 2, int(self.starting_position[1]) + CAR_SIZE_X / 2]
            
              # Calculate Four Corners
              left_top = [self.starting_center[0] + math.cos(math.radians(360 - (self.angle + 30))) * length, self.starting_center[1] + math.sin(math.radians(360 - (self.angle + 30))) * length]
              right_top = [self.starting_center[0] + math.cos(math.radians(360 - (self.angle + 150))) * length, self.starting_center[1] + math.sin(math.radians(360 - (self.angle + 150))) * length]
              left_bottom = [self.starting_center[0] + math.cos(math.radians(360 - (self.angle + 210))) * length, self.starting_center[1] + math.sin(math.radians(360 - (self.angle + 210))) * length]
              right_bottom = [self.starting_center[0] + math.cos(math.radians(360 - (self.angle + 330))) * length, self.starting_center[1] + math.sin(math.radians(360 - (self.angle + 330))) * length]
              self.corners = [left_top, right_top, left_bottom, right_bottom]
              
              print("2")
              corners_invalid = False
              for point in self.corners:
                print(point)
                if game_map.get_at((int(point[0]), int(point[1]))) == BORDER_COLOR:
                  corners_invalid = True
                  print("one corner invalid")
                else:
                    print("One corner valid")
            print("2")

            




            print("3")
            self.end_goal = [random.randint(x_range[0], x_range[1]), random.randint(y_range[0], y_range[1])]
            while ( math.isclose(self.end_goal[0], self.starting_position[0], abs_tol=200)) or ( math.isclose(self.end_goal[1], self.starting_position[1], abs_tol=200)) or (game_map.get_at((self.end_goal[0], self.end_goal[1])) == BORDER_COLOR):
                self.end_goal = [random.randint(x_range[0], x_range[1]), random.randint(y_range[0], y_range[1])]
        
        self.position = self.starting_position.copy()
        
        #self.end_goal = [random.randint(x_range[0], x_range[1]), random.randint(y_range[0], y_range[1])]
        #while (not math.isclose(self.end_goal[0], self.position[0], abs_tol=100)) and (not math.isclose(self.end_goal[1], self.position[1], abs_tol=100)) or (game_map.get_at((self.end_goal[0], self.end_goal[1])) == BORDER_COLOR):

        #    self.end_goal = [random.randint(x_range[0], x_range[1]), random.randint(y_range[0], y_range[1])]


        print(f"Start:{self.starting_position} End: {self.end_goal}")
        self.prev_remaining_distance = math.sqrt((self.position[0] - self.end_goal[0])**2 + (self.position[1] - self.end_goal[1])**2)
        self.direction_to_goal = 0
        ###################################################################################################################################################################
        starting_angles = [0,90,180,270]
        self.angle = random.choice(starting_angles)
        self.speed = 0

        self.completed = False
        self.crashed = False

        self.speed_set = False # Flag For Default Speed Later on

        self.center = [self.position[0] + CAR_SIZE_X / 2, self.position[1] + CAR_SIZE_X / 2] # Calculate Center

        self.radars = [] # List For Sensors / Radars
        self.drawing_radars = [] # Radars To Be Drawn

        self.alive = True # Boolean To Check If Car is Crashed

        self.distance = 0 # Distance Driven
        self.time = 0 # Time Passed

    def draw(self, screen, alive):
        if(alive ==True):
            screen.blit(self.rotated_sprite, self.position) # Draw Sprite
            #self.draw_radar(screen) #OPTIONAL FOR SENSORS
            screen.blit(self.goal_sprite, self.end_goal)
            #pygame.draw.rect()
        else:
            if(self.completed):
                screen.blit(self.success_sprite, self.position)
            else:
                screen.blit(self.dead_sprite, self.position)

        
    def draw_radar(self, screen):
        # Optionally Draw All Sensors / Radars
        pygame.draw.line(screen, (255,0,0), self.center, self.starting_position, 1)
        pygame.draw.line(screen, (0,255,0), self.center, self.end_goal, 1)
        for radar in self.radars:
            position = radar[0]
            pygame.draw.line(screen, (0, 0, 255), self.center, position, 1)
            pygame.draw.circle(screen, (0, 0, 255), position, 5)

    def check_collision(self, game_map):
        self.alive = True
        for point in self.corners:
            # If Any Corner Touches Border Color -> Crash
            if point[0] >= 1920 or point[0] <= 0 or point[1] >= 1080 or point[1] <= 0:
                self.alive = False
                print("Car made it out of bounds somehow")
                return True
            else:
                if game_map.get_at((int(point[0]), int(point[1]))) == BORDER_COLOR:
                    self.alive = False
                    return True
                    
        return False
    def check_complete(self):
        # Check if the car has reached its end_goal
        for point in self.corners:
            if point[0] > 1920 or point[0] <0:
                self.alive = False
                print("Car made it out of bounds somehow")
            elif point[1] >1080 or point[1] <0:
                self.alive = False
                print("Car made it out of bounds somehow")
            else:
                if math.isclose(int(point[0]), self.end_goal[0],abs_tol=100) and math.isclose( int(point[1]), self.end_goal[1],abs_tol=100):
                    print(point[0],point[1])
                    print(self.end_goal)
                    # Car has reached the end_goal
                    self.alive = False  # Stop the car (you may want to modify this behavior based on your requirements)
                    return True  

        return False  # Car hasn't completed the goal
    
    def check_radar(self, degree, game_map):
        length = 0
        x = int(self.center[0] + math.cos(math.radians(360 - (self.angle + degree))) * length)
        y = int(self.center[1] + math.sin(math.radians(360 - (self.angle + degree))) * length)

        # While We Don't Hit BORDER_COLOR AND length < 300 (just a max) 
        
        #if wit these  and x<1920 and y<1080
        #if x<1920 and x>0 and y<1080 and y>0:
        while x<1920 and x>0 and y<1080 and y>0 and not game_map.get_at((x, y)) == BORDER_COLOR and length < 300:
            length = length + 1
            x = int(self.center[0] + math.cos(math.radians(360 - (self.angle + degree))) * length)
            y = int(self.center[1] + math.sin(math.radians(360 - (self.angle + degree))) * length)

        # Calculate Distance To Border And Append To Radars List
        dist = int(math.sqrt(math.pow(x - self.center[0], 2) + math.pow(y - self.center[1], 2)))
        self.radars.append([(x, y), dist])
    
    def update(self, game_map):
         # Calculate direction to goal
        self.direction_to_goal = math.atan2(self.end_goal[1] - self.position[1], self.end_goal[0] - self.position[0])
        self.direction_to_goal = math.degrees(self.direction_to_goal)
    
    # Convert angle to range [0, 360)
        self.direction_to_goal %= 360
    
    # Convert negative angles to positive
        if self.direction_to_goal < 0:
            self.direction_to_goal += 360
    
        self.position[0] = max(min(self.position[0], game_map.get_width() - 1), 0)
        self.position[1] = max(min(self.position[1], game_map.get_height() - 1), 0)
        # Set The Speed To 20 For The First Time
        # Only When Having 4 Output Nodes With Speed Up and Down
        if not self.speed_set:
            self.speed = 5
            self.speed_set = True

        # Get Rotated Sprite And Move Into The Right X-Direction
        self.rotated_sprite = self.rotate_center(self.sprite, self.angle)
        self.position[0] += math.cos(math.radians(360 - self.angle)) * self.speed
        
        if math.cos(math.radians(360 - self.angle))==0:
            self.position[0] +=10

        # Increase Distance and Time
        self.distance += self.speed
        self.time += 1
        
        # Same For Y-Position
        self.position[1] += math.sin(math.radians(360 - self.angle)) * self.speed

        # Calculate New Center
        self.center = [int(self.position[0]) + CAR_SIZE_X / 2, int(self.position[1]) + CAR_SIZE_X / 2]

        # Calculate Four Corners
        # Length Is Half The Side
        length = 0.5 * CAR_SIZE_X
        left_top = [self.center[0] + math.cos(math.radians(360 - (self.angle + 30))) * length, self.center[1] + math.sin(math.radians(360 - (self.angle + 30))) * length]
        right_top = [self.center[0] + math.cos(math.radians(360 - (self.angle + 150))) * length, self.center[1] + math.sin(math.radians(360 - (self.angle + 150))) * length]
        left_bottom = [self.center[0] + math.cos(math.radians(360 - (self.angle + 210))) * length, self.center[1] + math.sin(math.radians(360 - (self.angle + 210))) * length]
        right_bottom = [self.center[0] + math.cos(math.radians(360 - (self.angle + 330))) * length, self.center[1] + math.sin(math.radians(360 - (self.angle + 330))) * length]
        self.corners = [left_top, right_top, left_bottom, right_bottom]

        # Check Collisions And Clear Radars
        
        self.radars.clear()

        # From -90 To 120 With Step-Size 45 Check Radar
        for d in range(-90, 120, 45):
            self.check_radar(d, game_map)

        # Check for completion
        
        self.crashed = self.check_collision(game_map)
        self.completed = self.check_complete()

        reward = self.get_reward()
        # Update fitness based on completion reward
        if self.completed:
            self.radars.clear()  # Clear radars when the car completes its goal
            return (reward + 1000)
        if self.crashed:
            return (reward - 1000)
        return reward

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
        return_values.append(self.prev_remaining_distance)

        # FACING
        return_values.append(self.angle)
        # Current speed
        return_values.append(self.speed)

            
        
        return return_values

    def is_alive(self):
        # Basic Alive Function
        return self.alive
    def is_success(self):
        return self.completed

    def get_reward(self):
        
        displacement = math.sqrt((self.position[0] - self.starting_position[0])**2 + (self.position[1] - self.starting_position[1])**2)
        remaining_distance = math.sqrt((self.position[0] - self.end_goal[0])**2 + (self.position[1] - self.end_goal[1])**2)
        
        self.prev_remaining_distance = remaining_distance
        
        return displacement - 10/self.prev_remaining_distance

    def rotate_center(self, image, angle):
        # Rotate The Rectangle
        rectangle = image.get_rect()
        rotated_image = pygame.transform.rotate(image, angle)
        rotated_rectangle = rectangle.copy()
        rotated_rectangle.center = rotated_image.get_rect().center
        rotated_image = rotated_image.subsurface(rotated_rectangle).copy()
        return rotated_image

def run_simulation(genomes, config):
    #map="detect"
    # Empty Collections For Nets and Cars
    nets = []
    cars = []

    global top_performing_genomes
    global sprite, dead_sprite, success_sprite, goal_sprite
    
    # Initialize PyGame And The Display
    pygame.init()
    WINDOW_SIZE = (1920, 1080)
    screen = pygame.display.set_mode(WINDOW_SIZE,pygame.RESIZABLE)

    game_map = process_map(map)
    sprite = pygame.image.load('car.png').convert()
    dead_sprite = pygame.image.load('dead.png').convert()
    success_sprite =pygame.image.load('success.png').convert()
    goal_sprite =pygame.image.load('goal.png').convert()
    
    # For All Genomes Passed Create A New Neural Network
    for i, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        g.fitness = 0

        cars.append(Car(game_map))

    # Clock Settings
    # Font Settings & Loading Map
    clock = pygame.time.Clock()
    generation_font = pygame.font.SysFont("Arial", 30)
    alive_font = pygame.font.SysFont("Arial", 20)

    #game_map = map.convert()
    print("Dimensions of the game map:")
    print("Width:", game_map.get_width())
    print("Height:", game_map.get_height())
    
    global current_generation
    current_generation += 1

    # Simple Counter To Roughly Limit Time (Not Good Practice)
    counter = 0

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit(0)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    pygame.quit()
                    sys.exit(0)

        # For Each Car Get The Acton It Takes
        for i, car in enumerate(cars):
            if random.random() < 0.2:
                choice = random.randint(0, 3)  # Choose a random action
            else:
                output = nets[i].activate(car.get_data())
                choice = output.index(max(output))

            if choice == 0:
                if car.angle == 360:
                    car.angle=15
                else:
                    car.angle += 15 # Left
            elif choice == 1:
                if car.angle==0:
                    car.angle=345
                else:
                    car.angle -= 15 # Right
            elif choice == 2:
            #    if car.speed > 4:
                car.speed = 5 #Go
            else:
                car.speed = 0 # Stop
        
        # Check If Car Is Still Alive
        # Increase Fitness If Yes And Break Loop If Not
        still_alive = 0
        made_it = 0
        for i, car in enumerate(cars):
            if car.is_alive():
                still_alive += 1
                reward = car.update(game_map)
                
                genomes[i][1].fitness =  reward


        if still_alive == 0:
            break

        counter += 1
        if counter == 1000: 
            break

        # Draw Map And All Cars That Are Alive
        screen.blit(game_map, (0, 0))
        for car in cars:
            car.draw(screen, car.is_alive())

        # Display Info
        info_surface = pygame.Surface((400, 100))  # Surface for displaying info
        info_surface.fill((255, 255, 255))  # Fill the surface with white color

        # Render and blit text onto the info surface
        text = generation_font.render("Generation: " + str(current_generation), True, (0, 0, 0))
        info_surface.blit(text, (10, 10))  # Position the text within the info surface

        text = alive_font.render("Still Alive: " + str(still_alive), True, (0, 0, 0))
        info_surface.blit(text, (10, 40))  # Position the text within the info surface

        # Blit the info surface onto the main screen
        screen.blit(info_surface, (10, WINDOW_SIZE[1] - 110))  # Position the info surface at the bottom left corner

        pygame.display.flip()
    
    generations_best_fitness=0    
    for genome_id, genome in genomes:
        # Evaluate the genome's fitness
        if genome.fitness>generations_best_fitness:
            generations_best_fitness = genome.fitness

        # Add the genome to the list of top performing genomes
    top_performing_genomes.append((genome_id, genome, generations_best_fitness))
    print(top_performing_genomes)
    print(f"Cars: {cars}")
    for car in cars:
        if car.is_success():
          made_it += 1
    print(f"Successful cars: {made_it}")
    return made_it
        
def overlay_image_on_screen(background, overlay, x, y, alpha, flip_horizontal=False):
  overlay_width = overlay.get_width()
  overlay_height = overlay.get_height()
    
  if flip_horizontal:
    overlay = pygame.transform.flip(overlay, True, False)  # Flip the overlay horizontally
    
  overlay_copy = overlay.copy()
  overlay_copy.set_alpha(alpha)
  background.blit(overlay_copy, (x - overlay_width // 2, y - overlay_height // 2))

def process_map(map):
  if map == "overlay":
    cap = cv2.VideoCapture(0)  # Capture webcam feed
    overlay_img = pygame.image.load('mapcity11.png').convert_alpha()  # Load overlay image with alpha channel
    ret, frame = cap.read()
    if ret:
      frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      frame = cv2.flip(frame, 1)  # Flip the frame horizontally
      frame = cv2.resize(frame, (1920, 1080))  # Resize the frame to match window size
      frame = np.rot90(frame)
      map_surface = pygame.surfarray.make_surface(frame)
      map_surface.blit(overlay_img, (0, 0))  # Apply the overlay image to the entire surface
      return map_surface
    else:
      print("No webcam detected")
      sys.exit(0)
  elif map == "live":
    print("Live feed selected. Not implemented yet.")
  elif map =="detect":
    cap = cv2.VideoCapture(0)  # Capture webcam feed
    for _ in range(10):
        ret, frame = cap.read()
    if ret:
      # Convert BGR to RGB
      frame_rgb = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
      frame_HSV = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

      #HSV MASK

      mask = cv2.inRange(frame_HSV,(35, 50, 20), (80, 255, 255) )
      
      
      
      
      # Apply mask to replace green color
      #lower_green = np.array([30, 30, 50],np.uint8)  # Lower bound for green color in BGR
      #upper_green = np.array([104, 153, 70],np.uint8)  # Upper bound for green color in BGR
            
      #WORKS WITH LIGHT GREEN / ALMOST WHITE
      #lower_green = np.array([0, 238, 0])  # Lower bound for green color in BGR
      #upper_green = np.array([255, 255, 255])  # Upper bound for green color in BGR
      
            
      #mask = cv2.inRange(frame_rgb, lower_green, upper_green)




      frame_rgb[mask != 0] = [234, 206, 183]  # Replace green pixels with specified color

      # Flip the frame horizontally
      frame_rgb = cv2.flip(frame_rgb, 1)

      # Resize the frame to match window size
      frame_rgb = cv2.resize(frame_rgb, (1920, 1080))

      # Rotate the frame
      frame_rgb = np.rot90(frame_rgb)

      # Convert the frame to a surface for pygame
      map_surface = pygame.surfarray.make_surface(frame_rgb)

      return map_surface
    else:
      print("No webcam detected")
      sys.exit(0)
  elif map == "simulation":
    return pygame.image.load('mapcity9.png').convert()
  else:
    print("Invalid map name")
  return map.convert()

def run_training(generations):
  # Load Config
    config_path = "./config.txt"
    config = neat.config.Config(neat.DefaultGenome,
                                neat.DefaultReproduction,
                                neat.DefaultSpeciesSet,
                                neat.DefaultStagnation,
                                config_path)

    # Create Population And Add Reporters
    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
    
    population.run(run_simulation, generations)
     # Sort the top performing genomes based on fitness scores
    top_performing_genomes.sort(key=lambda x: x[2], reverse=True)

    # Select the top 3 genomes
    top_3_genomes = top_performing_genomes[:3]

    # Save the neural networks corresponding to the top 3 genomes
    for i, (genome_id, genome, fitness) in enumerate(top_3_genomes):
        filename = f"best_genome_{i+1}.pickle"
        with open(filename, 'wb') as f:
            print(f"Attempting to save genome: {genome}")
            pickle.dump(genome, f)
    pygame.quit()  

def load_genome(filename):
  with open(filename, 'rb') as f:
        genome = pickle.load(f)
  return genome

def visualize_genome(genome, filename="genome"):
    # Create a directed graph
    graph = graphviz.Graph()

    # Add nodes
    for node_key, node in genome.nodes.items():
        if node.type != 'bias':  # Exclude bias nodes
            graph.add_node(node_key, label=str(node_key))

    # Add connections
    for conn_key, conn in genome.connections.items():
        if conn.enabled:
            graph.add_edge(conn.in_node, conn.out_node, label=str(round(conn.weight, 2)))

    # Save the graph to a file
    graph.layout(prog="dot")  # Use "dot" layout engine
    graph.draw(f"{filename}.png")

def test_genome(filename):
  
    config_path = "./config.txt"
    config = neat.config.Config(neat.DefaultGenome,
                                neat.DefaultReproduction,
                                neat.DefaultSpeciesSet,
                                neat.DefaultStagnation,
                                config_path)
    genome1 = load_genome(filename)
    
    #visualize_genome(genome1, filename="genome_visualization")
    genomes = [("1", genome1)]
    run_simulation(genomes, config)
    
    pygame.quit()  

def demo_test_genome(filename, generations):
  
    config_path = "./config.txt"
    config = neat.config.Config(neat.DefaultGenome,
                                neat.DefaultReproduction,
                                neat.DefaultSpeciesSet,
                                neat.DefaultStagnation,
                                config_path)
    genome1 = load_genome(filename)
    genomes = [("1", genome1)]
    counter=0
    successful_generations = 0
    while counter < generations:
      
      successful_cars = run_simulation(genomes, config)
      counter = counter+1
      if successful_cars >0:
          successful_generations += 1
    print(f"Accuracy of chosen model: {successful_generations/generations *100}")
    pygame.quit()  

if __name__ == "__main__":
    global map
    global spawn_style
    #random grouped
    spawn_style = "group"
    #detect overlay simulation
    map = "simulation"
    generations = 10
    #run_training(generations)
    #demo_test_genome("./genome_1.pickle",generations)
    demo_test_genome("./decent_genome.pickle",generations)
    #test_genome("./genome_1.pickle")
    #demo_test_genome("./best_genome_3.pickle",generations)
    #test_genome("./best_genome_1.pickle")