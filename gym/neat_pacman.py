"""
NEAT training for ALE Pac-Man using structured state inputs.

Input vector (26 values):
  [0-1]   Pacman absolute position (norm_r, norm_c)
  [2-5]   Distance to nearest wall in each direction (up, down, left, right), normalized
  [6-13]  Ghost 0-3 relative to Pacman (delta_r, delta_c), 0.0 if absent
  [14]    Ghost scared flag
  [15-22] Power pellet 0-3 relative to Pacman (delta_r, delta_c), 0.0 if eaten
  [23-24] Closest regular pellet relative to Pacman (delta_r, delta_c)
  [25]    Fraction of pellets remaining

Output vector (5 values): NOOP, UP, RIGHT, LEFT, DOWN — argmax selects action.
"""

import os
import pickle
import numpy as np
import gymnasium as gym
import ale_py
import neat
from neat.reporting import BaseReporter
from tqdm import tqdm

from state_extractor import PacmanStateExtractor, GameState, MAZE_H, MAZE_W

gym.register_envs(ale_py)

NUM_INPUTS  = 32
NUM_OUTPUTS = 5  # NOOP, UP, RIGHT, LEFT, DOWN
MAX_STEPS   = 2000
NUM_GENERATIONS = 100
CHECKPOINT_DIR = "checkpoints"


# ---------------------------------------------------------------------------
# Input vector construction
# ---------------------------------------------------------------------------

def _wall_distances(wall_mask: np.ndarray, r: int, c: int) -> list[float]:
    """
    Scan from (r, c) in each cardinal direction until hitting a wall.
    Returns [up, down, left, right] distances normalized to [0, 1].
    """
    h, w = wall_mask.shape
    r = int(np.clip(r, 0, h - 1))
    c = int(np.clip(c, 0, w - 1))

    def scan(dr, dc, max_d):
        for d in range(1, max_d + 1):
            nr, nc = r + dr * d, c + dc * d
            if nr < 0 or nr >= h or nc < 0 or nc >= w or wall_mask[nr, nc]:
                return (d - 1) / max_d
        return 1.0

    return [
        scan(-1,  0, h),   # up
        scan( 1,  0, h),   # down
        scan( 0, -1, w),   # left
        scan( 0,  1, w),   # right
    ]


def _closest_pellet_delta(pellet_mask: np.ndarray, pr: float, pc: float) -> list[float]:
    """
    Find the closest pellet pixel to Pacman and return the normalized delta.
    Returns [0.0, 0.0] if no pellets remain.
    """
    locs = np.argwhere(pellet_mask)
    if len(locs) == 0:
        return [0.0, 0.0]
    dists = np.hypot(locs[:, 0] - pr, locs[:, 1] - pc)
    nearest = locs[dists.argmin()]
    return [
        float((nearest[0] - pr) / MAZE_H),
        float((nearest[1] - pc) / MAZE_W),
    ]


def build_inputs(
    state: GameState,
    maze_frame: np.ndarray,   # (MAZE_H, MAZE_W, 3) cropped frame
    wall_mask: np.ndarray,    # (MAZE_H, MAZE_W) bool
    pellet_mask: np.ndarray,  # (MAZE_H, MAZE_W) bool
) -> np.ndarray:
    """Construct the 32-element NEAT input vector from extracted game state."""
    if state.pacman:
        pr, pc = state.pacman.center_r, state.pacman.center_c
    else:
        pr, pc = MAZE_H / 2, MAZE_W / 2

    inputs: list[float] = []

    # [0-3] Wall distances
    inputs.extend(_wall_distances(wall_mask, pr, pc))

    # [4-11] Ghost relative positions (nearest-first, zero-padded)
    for i in range(4):
        if i < len(state.ghosts):
            g = state.ghosts[i]
            inputs.append(float((g.center_r - pr) / MAZE_H))
            inputs.append(float((g.center_c - pc) / MAZE_W))
        else:
            inputs.extend([0.0, 0.0])

    # [12-19] Ghost velocities (same order, zero-padded)
    for i in range(4):
        if i < len(state.ghost_velocities):
            vr, vc = state.ghost_velocities[i]
            inputs.append(float(vr / MAZE_H))
            inputs.append(float(vc / MAZE_W))
        else:
            inputs.extend([0.0, 0.0])

    # [20] Scared timer (0-1 continuous)
    inputs.append(float(state.scared_timer_norm))

    # [21-28] Power pellet relative positions (zero if eaten)
    for i in range(4):
        if i < len(state.power_pellets):
            pp = state.power_pellets[i]
            inputs.append(float((pp.center_r - pr) / MAZE_H))
            inputs.append(float((pp.center_c - pc) / MAZE_W))
        else:
            inputs.extend([0.0, 0.0])

    # [29-30] Closest regular pellet delta
    inputs.extend(_closest_pellet_delta(pellet_mask, pr, pc))

    # [31] Pellet fraction
    inputs.append(state.pellet_fraction)

    assert len(inputs) == NUM_INPUTS, f"Expected {NUM_INPUTS} inputs, got {len(inputs)}"
    return np.array(inputs, dtype=np.float32)


# ---------------------------------------------------------------------------
# Genome evaluation
# ---------------------------------------------------------------------------

def eval_genome(genome, config) -> float:
    from state_extractor import _mask, COLOR_PELLET

    net = neat.nn.FeedForwardNetwork.create(genome, config)
    env = gym.make("ALE/Pacman-v5", obs_type="rgb")
    extractor = PacmanStateExtractor()

    obs, _ = env.reset()
    total_score = 0.0
    frames_survived = 0

    for step in range(MAX_STEPS):
        state = extractor.extract(obs)
        maze = obs[18:200]

        wall_mask   = extractor._wall_mask
        pellet_mask = _mask(maze, COLOR_PELLET)

        net_inputs = build_inputs(state, maze, wall_mask, pellet_mask)
        output = net.activate(net_inputs)
        action = int(np.argmax(output))

        obs, reward, terminated, truncated, _ = env.step(action)
        total_score += float(reward)
        frames_survived += 1

        if terminated or truncated:
            break

    env.close()

    # Fitness: score + small survival bonus
    fitness = total_score + frames_survived * 0.01
    return fitness


def eval_genomes(genomes, config):
    for genome_id, genome in tqdm(genomes):
        genome.fitness = eval_genome(genome, config)


# ---------------------------------------------------------------------------
# Reporter: save best genome each generation
# ---------------------------------------------------------------------------

class CheckpointReporter(BaseReporter):
    def __init__(self, directory: str):
        os.makedirs(directory, exist_ok=True)
        self.directory = directory
        self.generation = 0

    def post_evaluate(self, config, population, species_set, best_genome):
        path = os.path.join(self.directory, f"best_gen_{self.generation:04d}.pkl")
        with open(path, "wb") as f:
            pickle.dump(best_genome, f)
        self.generation += 1


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    config_path = os.path.join(os.path.dirname(__file__), "neat_pacman_config.ini")
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )

    pop = neat.Population(config)
    pop.add_reporter(neat.StdOutReporter(True))
    pop.add_reporter(neat.StatisticsReporter())
    pop.add_reporter(CheckpointReporter(CHECKPOINT_DIR))

    winner = pop.run(eval_genomes, NUM_GENERATIONS)

    with open("winner.pkl", "wb") as f:
        pickle.dump(winner, f)
    print("\nWinner saved to winner.pkl")
    print(winner)


if __name__ == "__main__":
    main()
