import numpy as np
import random

class SnakeGame:
    def __init__(self, width=10, height=10):
        self.width = width
        self.height = height
        self.snake = [(0, 0)]  # Snake starts at the top-left corner
        self.apple = self.generate_apple()
        self.direction = (0, 1)  # Initial direction: down

    def generate_apple(self):
        while True:
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)
            if (x, y) not in self.snake:
                return (x, y)

    def move(self, action):
        # Actions: 0 = up, 1 = down, 2 = left, 3 = right
        if action == 0:
            self.direction = (-1, 0)
        elif action == 1:
            self.direction = (1, 0)
        elif action == 2:
            self.direction = (0, -1)
        elif action == 3:
            self.direction = (0, 1)

        # Move the snake
        new_head = (self.snake[0][0] + self.direction[0], self.snake[0][1] + self.direction[1])
        self.snake.insert(0, new_head)

        # Check for collision
        if (new_head[0] < 0 or new_head[0] >= self.height or
            new_head[1] < 0 or new_head[1] >= self.width or
            new_head in self.snake[1:]):
            return -1  # Game over

        # Check if the snake has eaten the apple
        if new_head == self.apple:
            self.apple = self.generate_apple()
            return 1  # Positive reward
        else:
            self.snake.pop()
            return 0  # No reward

    def get_state(self):
        # Encode the state as a 1D array
        state = [0] * (self.width * self.height)
        for i, (x, y) in enumerate(self.snake):
            state[x * self.width + y] = -1  # Snake body
        x, y = self.apple
        state[x * self.width + y] = 1  # Apple
        return state

class QLearningAgent:
    def __init__(self, num_actions, learning_rate=0.1, discount_factor=0.99, exploration_rate=0.1):
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.q_table = {}

    def choose_action(self, state):
        if random.random() < self.exploration_rate:
            return random.randint(0, self.num_actions - 1)
        else:
            if state not in self.q_table:
                return random.randint(0, self.num_actions - 1)
            else:
                return np.argmax(self.q_table[state])

    def update_q_table(self, state, action, reward, next_state):
        if state not in self.q_table:
            self.q_table[state] = [0] * self.num_actions
        if next_state not in self.q_table:
            self.q_table[next_state] = [0] * self.num_actions

        # Q-learning update rule
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.discount_factor * self.q_table[next_state][best_next_action]
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.learning_rate * td_error

# Main loop
if __name__ == "__main__":
    env = SnakeGame(width=5, height=5)
    agent = QLearningAgent(num_actions=4)

    num_episodes = 1000
    for episode in range(num_episodes):
        state = tuple(env.get_state())

        total_reward = 0
        while True:
            action = agent.choose_action(state)
            reward = env.move(action)
            total_reward += reward
            next_state = tuple(env.get_state())
            agent.update_q_table(state, action, reward, next_state)
            state = next_state

            if reward == -1:
                break  # Game over

        print(f"Episode {episode + 1}: Total Reward = {total_reward}")

    # You can test the learned policy here by running the agent in the environment and observing its performance.
