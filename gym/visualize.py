import gymnasium as gym
import ale_py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from state_extractor import (
    PacmanStateExtractor, GameState, Blob,
    MAZE_ROW_START, MAZE_ROW_END, MAZE_H, MAZE_W,
)

gym.register_envs(ale_py)


def _to_frame_coords(blob: Blob):
    """Convert maze-relative blob center to full-frame pixel coords."""
    r = blob.center_r + MAZE_ROW_START
    c = blob.center_c
    return r, c


def draw_box(ax, r, c, h, w, label, color):
    rect = patches.Rectangle(
        (c - w/2, r - h/2), w, h,
        linewidth=2, edgecolor=color, facecolor='none',
    )
    ax.add_patch(rect)
    ax.text(c, r - h/2 - 2, label, color=color, fontsize=7, ha='center', va='bottom',
            bbox=dict(boxstyle='round,pad=0.1', fc='black', alpha=0.6))


def render_with_detections(frame: np.ndarray, state: GameState, save_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 8))

    axes[0].imshow(frame)
    axes[0].set_title('Raw Frame', color='white')
    axes[0].axis('off')

    axes[1].imshow(frame)
    axes[1].set_title('Detected Entities', color='white')
    axes[1].axis('off')
    ax = axes[1]

    # Maze boundary
    ax.add_patch(patches.Rectangle(
        (0, MAZE_ROW_START), MAZE_W, MAZE_H,
        linewidth=1, edgecolor='gray', facecolor='none', linestyle='--'
    ))

    # Pacman
    if state.pacman:
        r, c = _to_frame_coords(state.pacman)
        draw_box(ax, r, c, state.pacman.height + 2, state.pacman.width + 2,
                 'Pac-Man', 'yellow')

    # Ghosts
    ghost_color = 'cyan' if state.ghost_scared else 'magenta'
    for i, g in enumerate(state.ghosts):
        r, c = _to_frame_coords(g)
        draw_box(ax, r, c, g.height + 2, g.width + 2, f'Ghost {i+1}', ghost_color)

    # Power pellets
    for i, pp in enumerate(state.power_pellets):
        r, c = _to_frame_coords(pp)
        ax.add_patch(plt.Circle((c, r), 6, color='white', fill=False, linewidth=1.5))
        ax.text(c, r, 'PP', color='white', fontsize=6, ha='center', va='center')

    # Stats
    neat = state.to_neat_inputs()
    stats = (
        f"Pellets left: {state.pellet_fraction:.1%}\n"
        f"Ghosts detected: {len(state.ghosts)}\n"
        f"Power pellets: {len(state.power_pellets)}\n"
        f"Ghost scared: {state.ghost_scared}\n"
        f"NEAT vec ({len(neat)}): {np.round(neat, 2)}"
    )
    fig.text(0.01, 0.01, stats, fontsize=7, color='white', va='bottom', ha='left',
             bbox=dict(boxstyle='round', fc='#111111', alpha=0.85))

    fig.patch.set_facecolor('#111111')
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight', facecolor='#111111')
    plt.close()
    print(f'Saved: {save_path}')


def main():
    env = gym.make('ALE/Pacman-v5', obs_type='rgb')
    extractor = PacmanStateExtractor()

    obs, _ = env.reset()
    # Warm up: let ghosts leave ghost house
    for _ in range(80):
        obs, _, term, trunc, _ = env.step(0)
        if term or trunc:
            obs, _ = env.reset()

    save_steps = {0, 60, 120, 180, 240}
    saved = 0
    for step in range(300):
        obs, reward, term, trunc, info = env.step(np.random.randint(5))
        if term or trunc:
            obs, _ = env.reset()
            extractor = PacmanStateExtractor()
            continue

        state = extractor.extract(obs)

        if step in save_steps:
            render_with_detections(obs, state, f'/tmp/pacman_det_{step:03d}.png')
            saved += 1

    env.close()
    print(f'\nDone — {saved} frames saved to /tmp/pacman_det_*.png')


if __name__ == '__main__':
    main()
