import numpy as np
from dqn_agent import DQNAgent
from utils import make_env

if __name__ == '__main__':
    env = make_env('PongNoFrameskip-v4')
    best_score = -np.inf
    load_checkpoint = False
    n_games = 500
    agent = DQNAgent(gamma=0.99, epsilon=1.0, lr=0.0001,
                     n_actions=env.action_space.n,
                     input_dims=(env.observation_space.shape),
                     batch_size=32, memory_size=40000, epsilon_min=0.1,
                     replace_target_count=1000, epsilon_decay=1e-5,
                     checkpoint_dir='models/', algo='DQNAgent',
                     env_name='PongNoFrameskip-v4')

    if load_checkpoint:
        agent.load_models()

    filename = agent.algo + '_' + agent.env_name \
                + '_lr' + str(agent.lr) + '_' + str(n_games) \
                + '_' + 'games'

    n_steps = 0
    scores, eps_history, steps_array = [], [], []

    for i in range(n_games):
        done = False
        score = 0
        observation = env.reset()

        while not done:
            action = agent.choose_action(observation)
            observation, reward, done, info = env.step(action)
            score += reward

            scores.append(score)
            steps_array.append(n_steps)

            avg_score = np.mean(scores[-100:])

            print(f'episode {i}, score {score} \
                    average score {avg_score} best score {best_score} \
                    epsilon {agent.epsilon} steps {n_steps}')

            if avg_score > best_score:
                if not load_checkpoint:
                    agent.save_models()
                best_score = avg_score

            eps_history.append(agent.epsilon)