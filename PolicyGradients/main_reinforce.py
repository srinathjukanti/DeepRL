import numpy as np
import argparse
import sys
from utils import make_env, wrap_env, show_video
from torch.utils.tensorboard import SummaryWriter
from pyvirtualdisplay import Display
from reinforce_agent import ReinforceAgent


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_dir", help="Checkpoint directory to load and save models")
    parser.add_argument("env_name", help="Environment name to run the agent on")
    parser.add_argument("--test", help="Test the trained dqn agent")
    parser.add_argument("--render", help="Render the environment")
    parser.add_argument("--notebook_render", help="Render environment in notebook")
    args = parser.parse_args()

    env = make_env(args.env_name)
    if args.render and args.notebook_render:
        display = Display(visible=0, size=(1400, 900))
        display.start()
        env = wrap_env(env)
    writer = SummaryWriter('runs/policy_gradient_reinforce')
    best_score = -np.inf
    checkpoint_directory = args.checkpoint_dir
    n_games = 500
    agent = ReinforceAgent(gamma=0.99, lr=0.0001, n_actions=env.action_space.n,
                          input_dims=env.observation_space.shape, algo='ReinforceAgent', 
                          env_name=args.env_name, checkpoint_dir=checkpoint_directory)
    if args.test:
        try:
            agent.load_models()
        except:
            print(f'Agent model not found in {checkpoint_directory}')
            sys.exit()
        finally:
            env.close()

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
            if args.render:
                env.render()

            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            agent.rewards.append(reward)
            score += reward

            if not args.test:
                agent.learn()

            observation = observation_
            n_steps += 1

        scores.append(score)
        steps_array.append(n_steps)

        avg_score = np.mean(scores[-100:])

        print(f'episode {i}, score {score}, '
              f'average score {avg_score}, best score {best_score} '
              f'epsilon {agent.epsilon} steps {n_steps}')

        writer.add_scalar('Scores/n_steps', score, n_steps)
        writer.add_scalar('Average Scores/n_games', score, i)
        writer.add_scalar('epsilon/n_games', agent.epsilon, i)

        if avg_score > best_score:
            if not args.test:
                agent.save_models()
            best_score = avg_score

        eps_history.append(agent.epsilon)
        if args.notebook_render:
            show_video()

    env.close()
    writer.close()