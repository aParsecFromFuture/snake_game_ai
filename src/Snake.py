from src.Game import *
from src.NeuralNetwork import *
import pickle

SNAKE_INPUT_COUNT = 19
SNAKE_LAYER_STRUCTURE = [13, 6, 2]
SNAKE_MUTATION_EFFECT = 5
SNAKE_HEALTH = 80
SNAKE_VISION_RANGE = 20


def generate_input(habitat, position, direction, memory):
    output_wall = [MAP_WIDTH, MAP_WIDTH, MAP_WIDTH, MAP_WIDTH, MAP_WIDTH]
    output_body = [MAP_WIDTH, MAP_WIDTH, MAP_WIDTH, MAP_WIDTH, MAP_WIDTH]
    output_apple = [MAP_WIDTH, MAP_WIDTH, MAP_WIDTH, MAP_WIDTH, MAP_WIDTH]

    if np.array_equal(direction, UP):
        search_vector = SEARCH_UP
    elif np.array_equal(direction, RIGHT):
        search_vector = SEARCH_RIGHT
    elif np.array_equal(direction, DOWN):
        search_vector = SEARCH_DOWN
    else:
        search_vector = SEARCH_LEFT

    for i in range(5):
        v = position
        for j in range(SNAKE_VISION_RANGE):
            v = v + search_vector[i]
            if 0 <= v[0] < MAP_WIDTH and 0 <= v[1] < MAP_HEIGHT:
                if habitat[v[0]][v[1]] == 1:
                    output_wall[i] = j
                    break
                if habitat[v[0]][v[1]] == 2:
                    output_body[i] = j
                    memory[2] = 3
                    tmp = relative_to(search_vector[i], direction)
                    memory[3] = tmp[0]
                    memory[4] = tmp[1]
                    break
                if habitat[v[0]][v[1]] == 3:
                    output_apple[i] = j
                    break
    if memory[2] != 0:
        memory[2] = memory[2] - 1
    else:
        memory[3] = memory[4] = 0

    return np.array([output_wall + output_body + output_apple + [memory[0], memory[1], memory[3], memory[4]]])


def evolution(parent, generation_count, population_count):
    for generation in range(generation_count):
        population = get_generation(parent, population_count)
        parent = natural_selection(population)

    return parent[0]


def natural_selection(population):
    best_snake = population[0]
    for member in population:
        if best_snake[1] <= member[1]:
            best_snake = member
    return best_snake


def get_generation(parent, snake_count):
    habitat = np.array(MAP)
    brain_list = []

    for i in range(snake_count - 1):
        brain = mutation(parent[0])
        try_1 = let_her_try_best(brain, habitat)
        habitat = habitat * MAP
        try_2 = let_her_try_best(brain, habitat)
        habitat = habitat * MAP
        try_3 = let_her_try_best(brain, habitat)
        habitat = habitat * MAP

        brain_list.append([brain, (try_1 + try_2 + try_3) / 3])

    return brain_list


def let_her_try_best(brain, habitat):
    point = 0
    health = SNAKE_HEALTH
    position = [3, 3]
    body = [position, [4, 3], [5, 3]]
    direction = np.array([-1, 0])
    habitat[3][3] = habitat[4][3] = habitat[5][3] = 2
    memory = [0, 0, 0, 0, 0]
    rx, ry = generate_apple_for(habitat)
    habitat[rx][ry] = 3
    while 1:
        output = brain.get_output(generate_input(habitat, position, direction, memory))

        if output[0][0] > 0.5 and output[0][1] > 0.5:
            memory[0] = memory[1] = 0
        elif output[0][0] > 0.5:
            rotate_left(direction)
            memory[0], memory[1] = 1, 0
        elif output[0][1] > 0.5:
            rotate_right(direction)
            memory[0], memory[1] = 0, 1

        tail = body[-1]
        habitat[tail[0]][tail[1]] = 0
        position = position + direction
        body[0] = position

        for i in range(len(body) - 1, 0, -1):
            body[i] = body[i - 1]

        if health < 0:
            break
        elif habitat[position[0]][position[1]] == 1:
            break
        elif habitat[position[0]][position[1]] == 2:
            break
        elif point > 10000:
            break
        if habitat[position[0]][position[1]] == 3:
            health = SNAKE_HEALTH
            point = point + 10
            rx, ry = generate_apple_for(habitat)
            habitat[rx][ry] = 3
            body.append(tail)
        else:
            habitat[tail[0]][tail[1]] = 0

        habitat[position[0]][position[1]] = 2
        health = health - 1

    return point


def mutation(parent):
    brain = NeuralNetwork(SNAKE_INPUT_COUNT, SNAKE_LAYER_STRUCTURE)
    for i in range(len(SNAKE_LAYER_STRUCTURE)):
        mutation_m = (0.5 - np.random.uniform(size=parent.weights[i].shape))
        brain.weights[i] = parent.weights[i] + SNAKE_MUTATION_EFFECT * mutation_m
        mutation_m = (0.5 - np.random.uniform(size=parent.bias[i].shape))
        brain.bias[i] = parent.bias[i] + SNAKE_MUTATION_EFFECT * mutation_m
    return brain


def save_brain_to_file(brain, filename):
    with open(filename, "wb",) as output:
        pickle.dump(brain, output, pickle.HIGHEST_PROTOCOL)


def get_brain_from_file(filename):
    with open(filename, "rb", ) as inp:
        return pickle.load(inp)


class Snake:
    def __init__(self, x, y, brain):
        self.brain = brain
        self.position = np.array([x, y])
        self.direction = np.array([-1, 0])
        self.canvas_body = []
        self.body = []
        self.memory = [0, 0, 0, 0, 0]
        self.eat(x, y)
        self.eat(x + 1, y)
        self.eat(x + 2, y)
        self.is_dead = False
        self.health = SNAKE_HEALTH

    def eat(self, x, y):
        self.body.append([x, y])
        self.health = SNAKE_HEALTH
        MAP[x][y] = 2
        x = x * SQUARE_WIDTH
        y = y * SQUARE_HEIGHT
        self.canvas_body.append(CANVAS.create_rectangle(x, y, x + SQUARE_WIDTH, y + SQUARE_HEIGHT, fill="white", outline=""))

    def make_a_decision(self, habitat):
        decision = self.brain.get_output(generate_input(habitat, self.position, self.direction, self.memory))[0]
        if decision[0] > 0.5 and decision[1] > 0.5:
            self.memory[0] = self.memory[1] = 0
        elif decision[0] > 0.5:
            rotate_left(self.direction)
            self.memory[0], self.memory[1] = 1, 0
        elif decision[1] > 0.5:
            rotate_right(self.direction)
            self.memory[0], self.memory[1] = 0, 1

    def move(self):
        self.make_a_decision(MAP)

        tail = self.body[-1]
        MAP[tail[0]][tail[1]] = 0

        for i in range(len(self.body) - 1, 0, -1):
            self.body[i] = self.body[i - 1]
            CANVAS.coords(self.canvas_body[i], CANVAS.coords(self.canvas_body[i - 1]))

        self.position = self.position + self.direction
        self.body[0] = self.position

        if self.health < 0:
            self.is_dead = True
            print("The snake has died because of hunger")
            return
        elif MAP[self.position[0]][self.position[1]] == 1:
            print("The snake has crossed the wall")
            self.is_dead = True
            return
        elif MAP[self.position[0]][self.position[1]] == 2:
            print("The snake has crossed her body")
            self.is_dead = True
            return
        elif MAP[self.position[0]][self.position[1]] == 3:
            self.eat(tail[0], tail[1])
            upgrade_the_score()
            generate_apple()
        else:
            self.health = self.health - 1

        MAP[self.position[0]][self.position[1]] = 2
        c_pos_x = SQUARE_WIDTH * self.position[0]
        c_pos_y = SQUARE_HEIGHT * self.position[1]
        CANVAS.coords(self.canvas_body[0], c_pos_x, c_pos_y, c_pos_x + SQUARE_WIDTH, c_pos_y + SQUARE_HEIGHT)