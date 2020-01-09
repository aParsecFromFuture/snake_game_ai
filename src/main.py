import threading
import time
from src.Snake import *

GENERATION_COUNT = 105
POPULATION_COUNT = 100


def run():
    brain = get_brain_from_file("brains/golden_generation.pkl")
    snake = Snake(3, 3, brain)
    generate_apple()
    while not snake.is_dead:
        snake.move()
        time.sleep(0.1)


t = threading.Thread(target=run)
t.daemon = True
t.start()

WINDOW.mainloop()
