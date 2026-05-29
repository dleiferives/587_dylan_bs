import gymnasium as gym
import ale_py
import neat
import numpy as np
from tqdm import tqdm

gym.register_envs(ale_py)

def eval_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    env = gym.make("ALE/Pacman-v5", obs_type="ram")
    obs, info = env.reset()

    fitness = 0.0
    for _ in range(2000):
        x = obs.astype(np.float32) / 255.0
        y = net.activate(x)
        action = int(np.argmax(y))
        obs, reward, terminated, truncated, info = env.step(action)
        fitness += float(reward)
        if terminated or truncated:
            break

    env.close()
    return fitness

def eval_genomes(genomes, config):
    for genome_id, genome in tqdm(genomes):
        genome.fitness = eval_genome(genome, config)

if __name__ == "__main__":
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        "neat-config.txt",
    )
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    p.add_reporter(neat.StatisticsReporter())
    winner = p.run(eval_genomes, 50)
    print("Done:", winner)
