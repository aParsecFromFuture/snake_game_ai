import numpy as np
from tkinter import *

CANVAS_WIDTH, CANVAS_HEIGHT = 600, 600
MAP_WIDTH, MAP_HEIGHT = 20, 20
SQUARE_WIDTH, SQUARE_HEIGHT = CANVAS_WIDTH / MAP_WIDTH, CANVAS_HEIGHT / MAP_HEIGHT

MAP = np.zeros(shape=(MAP_WIDTH, MAP_HEIGHT))
for x in range(MAP_WIDTH):
    for y in range(MAP_HEIGHT):
        if x == 0 or y == 0 or x == MAP_WIDTH - 1 or y == MAP_HEIGHT - 1:
            MAP[x][y] = 1

WINDOW = Tk()
WINDOW.title("Snake Game")

CANVAS = Canvas(WINDOW, width=CANVAS_WIDTH, height=CANVAS_HEIGHT, bg="black")
for x in range(MAP_WIDTH):
    for y in range(MAP_HEIGHT):
        if MAP[x][y] == 1:
            pos_x = SQUARE_WIDTH * x
            pos_y = SQUARE_HEIGHT * y
            CANVAS.create_rectangle(pos_x, pos_y, pos_x + SQUARE_WIDTH, pos_y + SQUARE_HEIGHT, fill="gray", outline="")

APPLE = CANVAS.create_rectangle(-1, -1, 0, 0, fill="orange", outline="")
SCORE_TEXT = CANVAS.create_text(2 * SQUARE_WIDTH, SQUARE_HEIGHT / 2, text="Score : 0", fill="white")
SCORE = 0
CANVAS.pack()

UP = np.array([0, 1])
RIGHT = np.array([1, 0])
DOWN = np.array([0, -1])
LEFT = np.array([-1, 0])


def rotate_left(v):
    swap = v[0]
    v[0] = -v[1]
    v[1] = swap


def rotate_right(v):
    swap = v[0]
    v[0] = v[1]
    v[1] = -swap


def get_rotated_left(v):
    v2 = np.array(v)
    swap = v2[0]
    v2[0] = -v2[1]
    v2[1] = swap
    return v2


def get_rotated_right(v):
    v2 = np.array(v)
    swap = v2[0]
    v2[0] = v2[1]
    v2[1] = -swap
    return v2


def relative_to(v1, v2):
    v3 = np.array(v1)
    if np.array_equal(v2, UP):
        pass
    elif np.array_equal(v2, LEFT):
        rotate_right(v3)
    elif np.array_equal(v2, DOWN):
        v3 = -v3
    else:
        rotate_left(v3)
    return v3


def generate_apple():
    rx, ry = 1 + np.random.randint(MAP_WIDTH - 2), 1 + np.random.randint(MAP_HEIGHT - 2)
    while MAP[rx][ry] != 0:
        rx = 1 + np.random.randint(MAP_WIDTH - 2)
        ry = 1 + np.random.randint(MAP_HEIGHT - 2)
    MAP[rx][ry] = 3
    CANVAS.coords(APPLE, rx * SQUARE_WIDTH, ry * SQUARE_HEIGHT, rx * SQUARE_WIDTH + SQUARE_WIDTH,
                  ry * SQUARE_HEIGHT + SQUARE_HEIGHT)


def generate_apple_for(habitat):
    rx, ry = 1 + np.random.randint(MAP_WIDTH - 2), 1 + np.random.randint(MAP_HEIGHT - 2)
    while habitat[rx][ry] != 0:
        rx = 1 + np.random.randint(MAP_WIDTH - 2)
        ry = 1 + np.random.randint(MAP_HEIGHT - 2)
    return [rx, ry]


def upgrade_the_score():
    global SCORE
    SCORE = SCORE + 10
    CANVAS.dchars(SCORE_TEXT, 8, 20)
    CANVAS.insert(SCORE_TEXT, 8, SCORE)


SEARCH_UP = np.array([get_rotated_left(UP), get_rotated_left(UP) + UP, UP,
                      get_rotated_right(UP) + UP, get_rotated_right(UP)])

SEARCH_RIGHT = np.array([get_rotated_left(RIGHT), get_rotated_left(RIGHT) + RIGHT,
                         RIGHT, get_rotated_right(RIGHT) + RIGHT, get_rotated_right(RIGHT)])

SEARCH_DOWN = np.array([get_rotated_left(DOWN), get_rotated_left(DOWN) + DOWN, DOWN,
                        get_rotated_right(DOWN) + DOWN, get_rotated_right(DOWN)])

SEARCH_LEFT = np.array([get_rotated_left(LEFT), get_rotated_left(LEFT) + LEFT,
                        LEFT, get_rotated_right(LEFT) + LEFT, get_rotated_right(LEFT)])
