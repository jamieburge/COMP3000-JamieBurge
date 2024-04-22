import random
import keyboard
from spherov2 import scanner
from spherov2.sphero_edu import SpheroEduAPI, Color

def connect():
    print("Connect Mode")
    try:
      print("Finding all toys")
      toys = scanner.find_toys()
      print("Found:")
      print(toys)
      print(toys[0])
      print(toys[1])
      print("Connecting to one")
      toy = scanner.find_toy()
      print(f"Connected to {toy}")
      with SpheroEduAPI(toy) as api:
         api.set_main_led(Color(r=255, g=0, b=0))
         api.set_main_led(Color(r=0, g=255, b=0))
         api.set_main_led(Color(r=0, g=0, b=0))
         print(f"Colour: {api.get_color()}")
      print(f"Returning: {toy}")
      return toy
    except Exception as e:
      print(f"An error occurred: {e}")

def move(mini):
      toy = scanner.find_toy(toy_name="")
          #toy= scanner.find_toy(toy_name: Optional[str] = "SM-D692")
      print(f"Connected to :{toy}")
      while True:
        try:
          
          # Check if any key is being pressed
          if keyboard.is_pressed('w'):
                  print(f'Move in the direction: w')
                  try:
                    print(toy)
                    with SpheroEduAPI(toy) as api:
                     # print("Test")
                      api.set_main_led(Color(r=255, g=0, b=255))
                  except Exception as e:
                    print(f"An error occurred: {e}")
                  
          elif keyboard.is_pressed('s'):
                  print(f'Move in the direction: s')

              # You can add more conditions for other keys if needed
          elif keyboard.is_pressed('q'):
            print('Quitting the program...')
            break  # Exit the while loop

        except Exception as e:
          print(f"An error occurred: {e}")
          break

def main():
    while True:
        print("\nSelect a mode:")
        print("1. Connect")
        print("2. Move")
        print("3. Exit")

        choice = input("Enter the mode number: ")

        if choice == '1':
            toy=connect()
        elif choice == '2':
            move(toy)
        elif choice == '3':
            print("Exiting the program. Goodbye!")
            break
        else:
            print("Invalid choice. Please enter a valid mode number.")

if __name__ == "__main__":
    main()