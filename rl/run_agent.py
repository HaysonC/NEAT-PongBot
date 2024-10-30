import argparse
import gymnasium as gym
import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or Inference VPG Agent")
    parser.add_argument("--mode", type=str, choices=["train", "inference"], required=True, help="Mode to run the agent in: 'train' or 'inference'")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes for training or inference")
    parser.add_argument("--render", action="store_true", help="Render the environment during inference")
    parser.add_argument("--agent", type=int, choices=[1, 2, 3], default=1, help="1: VPG (2D), 2: VPG(3D), 3: DDQN")
    args = parser.parse_args()

    device = torch.device("mps" if torch.has_mps else "cuda" if torch.cuda.is_available() else "cpu")

    # Very crude, but I'm so depressed and hopeless with coding rn

    if args.agent == 3 and args.mode == "train":
        from rl.pong_dqn import DQNAgent as Agent
        env = gym.make("ALE/Pong-v5")
        agent = Agent(env, device)
        agent.train_agent(num_episodes=args.episodes)
    else:
        if args.agent == 1:
            from rl.pong_vpg_1 import VPGAgent as Agent
        elif args.agent == 2:
            from rl.pong_vpg_2 import VPGAgent as Agent

        if args.mode == "train":
            env = gym.make("ALE/Pong-v5")
            agent = Agent(action_dim=env.action_space.n)
            agent.train(env, num_episodes=args.episodes)
        elif args.mode == "inference":
            env = gym.make("ALE/Pong-v5", render_mode="human")
            agent = Agent(action_dim=env.action_space.n)
            agent.load_model()
            agent.inference(env, episode=args.episodes, render=args.render)