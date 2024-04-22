import pygame
import sys

from spherov2 import scanner
from spherov2.sphero_edu import SpheroEduAPI, Color

import cv2 
import numpy as np 

video = cv2.VideoCapture(0) 
image = cv2.imread('mapempty.png') 
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

# Main game loop
print("Finding toy")
toy = scanner.find_toy()
print("Found toy")
with SpheroEduAPI(toy) as api:
  
  
  while True:
      ret, frame = video.read() 
      frame = cv2.resize(frame, (640, 480)) 
      image = cv2.resize(image, (640, 480)) 
      frame_rgb = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
      frame_HSV = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

  #HSV MASK
      mask = cv2.inRange(frame_HSV,(35, 50, 20), (80, 255, 255) )
      
      res = cv2.bitwise_and(frame, frame, mask = mask) 

	    
      f = frame - res 
      f = np.where(f == 0, image, f) 

      cv2.imshow("video", frame) 
      cv2.imshow("mask", f) 


      if cv2.waitKey(25) == 27: 
         break

      player_speed = 1
      api.set_speed(player_speed)
      for event in pygame.event.get():
          if event.type == pygame.QUIT:
              pygame.quit()
              sys.exit()

    # Get key states
      keys = pygame.key.get_pressed()

    # Update player position based on keys
      if keys[pygame.K_w]:
          player_y -= player_speed
          api.set_main_led(Color(r=255, g=0, b=0))
          #api.raw_motor(255, 255, 4) #DANGER THIS GOES FULL POWER 
          #player_speed = player_speed + 1
          api.set_speed(player_speed)
      if keys[pygame.K_s]:
          player_y += player_speed
          api.set_main_led(Color(r=0, g=255, b=0))
          if player_speed ==0:
              pass
          else:
            #player_speed = player_speed - 1
            api.set_speed(player_speed)    
          
      if keys[pygame.K_a]:
          player_x -= player_speed
          api.set_main_led(Color(r=0, g=0, b=255))
          api.spin(-30,0.1)
      if keys[pygame.K_d]:
          player_x += player_speed
          api.set_main_led(Color(r=0, g=255, b=255))
          api.spin(30,0.1)

    # Clear the screen
      screen.fill((255, 255, 255))

    # Draw the player
      #pygame.draw.rect(screen, (0, 0, 255), (player_x, player_y, player_size, player_size))
      pygame.draw.circle(screen, (0, 0, 255,), (player_x, player_y), player_size)

    # Update the display
      pygame.display.flip()

    # Set the frames per second
      pygame.time.Clock().tick(30)