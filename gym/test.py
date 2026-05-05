import gymnasium as gym
import ale_py

gym.register_envs(ale_py)

env = gym.make("ALE/Pacman-v5")
obs, info = env.reset()

done = False
while not done:
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    print(obs)
    done = terminated or truncated

env.close()
