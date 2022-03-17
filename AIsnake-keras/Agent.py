import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from helper import plot
from model import QTrainer_NN

MAX_MEMORY = 100000
BATCH_SIZE = 1000
LR= 0.001

class Agent:
    
    def __init__(self,):
        self.num_games = 0
        self.epsilon = 0  # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.DQN = QTrainer_NN(input_size=11, hidden_size=128, output_size=3,
                 learn_rate=0.0001, gamma=0.9, load_path='./model_weight/model.h5')
        
        
    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
            ]

        return np.array(state, dtype=int)
        
        
    def remeber(self, state, action ,reward, next_state, done):
        self.memory.append((state, action ,reward, next_state, done)) # popleft if MAX_MEMORY is reached
        
    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory
        
        #print(type(mini_sample))
        #print(mini_sample)
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        states = np.array(states)
        next_states = np.array(next_states)
        self.DQN.train_step(states, actions ,rewards, next_states, dones)
        
        
    # only one step   
    def train_short_memory(self, state, action ,reward, next_state, done):
        #done = np.array([done])
        self.DQN.train_step(state, action ,reward, next_state, done)
        
    def get_action(self, state):
        # random moves: tradoff exploration / exploitation
        self.epsilon = 80 - self.num_games
        final_move = [0,0,0]
        if random.randint(0,300) < self.epsilon:
            move = random.randint(0,2)
            final_move[move] = 1
        else:
            state = state.reshape((1,state.shape[0]))
            prediction = self.DQN.model.predict(state)  
            move = np.argmax(prediction)
            final_move[move] = 1
         
        return final_move
        
        
def train():
    plot_scores = []
    plot_mean_scores = [] 
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    while agent.num_games <= 2000:
        # get old state
        state_old = agent.get_state(game)
        
        # get move
        final_move = agent.get_action(state_old)
        #print(final_move)
        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)
        
        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        
        # remeber
        agent.remeber(state_old, final_move, reward, state_new, done)
        
        if done:  # end game
            # train long memory. plot result
            game.reset()
            agent.num_games += 1
            agent.train_long_memory()
            
            if score > record:
                record = score
                agent.DQN.model.save("model_weight/model.h5")
                
            print('Game', agent.num_games, 'Score', score, 'Record', record)
            
            # plot
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.num_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)
        
    
    
if __name__ == '__main__':
    train()