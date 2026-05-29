import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Optional

# Frame layout: full frame is 250x160, maze is rows 18-200
MAZE_ROW_START = 18
MAZE_ROW_END   = 200
FRAME_H        = 250
FRAME_W        = 160
MAZE_H         = MAZE_ROW_END - MAZE_ROW_START  # 182
MAZE_W         = FRAME_W                         # 160

# Entity colors (RGB uint8)
COLOR_WALL   = np.array([50,  50,  176], dtype=np.uint8)
COLOR_BG     = np.array([0,   0,   0],   dtype=np.uint8)
COLOR_PELLET = np.array([223, 192, 111], dtype=np.uint8)
COLOR_PINK   = np.array([252, 144, 200], dtype=np.uint8)  # ghosts + power pellets
COLOR_PACMAN = np.array([252, 224, 144], dtype=np.uint8)
COLOR_SCARED = np.array([66,  72,  200], dtype=np.uint8)  # ghosts when scared

# Blob size thresholds (measured from actual frames)
# Power pellet: ~40px, w=4 (narrow)
# Ghost:        ~96px, w=8 (wider)
POWER_PELLET_MAX_WIDTH = 5   # px; blobs narrower than this are power pellets


@dataclass
class Blob:
    pixels: int
    height: int
    width: int
    center_r: float  # in maze coords (0..MAZE_H)
    center_c: float  # in maze coords (0..MAZE_W)

    @property
    def norm_r(self) -> float:
        return self.center_r / MAZE_H

    @property
    def norm_c(self) -> float:
        return self.center_c / MAZE_W


MAX_SCARED_FRAMES = 180  # ~3s at 60fps

@dataclass
class GameState:
    pacman:            Optional[Blob]
    ghosts:            list   # list of Blob, up to 4, nearest-first; each has vel_r, vel_c
    ghost_velocities:  list   # list of (vel_r, vel_c) tuples, same order as ghosts
    power_pellets:     list   # list of Blob (fixed corner positions)
    ghost_scared:      bool
    scared_timer_norm: float  # 0-1, counts down from 1 when power pellet eaten
    pellet_fraction:   float  # fraction of pellets remaining [0,1]

    def to_neat_inputs(self) -> np.ndarray:
        """
        Fixed-length float32 vector for NEAT.

        Index  Description
        -----  -----------
        0-1    Pacman (norm_r, norm_c), -1 if not found
        2-9    Ghost 0-3 (norm_r, norm_c), -1 if absent
        10     ghost_scared flag
        11     pellet fraction remaining
        12-15  power pellet present flags (slots 0-3)
        16-19  Euclidean distance from pacman to each ghost (norm), -1 if absent
        """
        v = np.full(20, -1.0, dtype=np.float32)

        if self.pacman:
            v[0] = self.pacman.norm_r
            v[1] = self.pacman.norm_c

        for i, g in enumerate(self.ghosts[:4]):
            v[2 + i*2]     = g.norm_r
            v[2 + i*2 + 1] = g.norm_c

        v[10] = 1.0 if self.ghost_scared else 0.0
        v[11] = self.pellet_fraction

        for i, pp in enumerate(self.power_pellets[:4]):
            v[12 + i] = 1.0

        if self.pacman:
            pr, pc = self.pacman.norm_r, self.pacman.norm_c
            for i, g in enumerate(self.ghosts[:4]):
                dist = np.hypot(g.norm_r - pr, g.norm_c - pc)
                v[16 + i] = float(np.clip(dist, 0.0, 1.0))

        return v


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _mask(frame: np.ndarray, color: np.ndarray, tol: int = 12) -> np.ndarray:
    """Boolean mask where frame pixels match color within per-channel tolerance."""
    diff = np.abs(frame.astype(np.int16) - color.astype(np.int16))
    return np.all(diff <= tol, axis=2)


def _find_blobs(mask: np.ndarray, min_pixels: int = 8) -> list[Blob]:
    """Connected-component blob detection via BFS."""
    visited = np.zeros_like(mask, dtype=bool)
    blobs: list[Blob] = []

    for sr, sc in zip(*np.where(mask)):
        if visited[sr, sc]:
            continue
        q = deque([(sr, sc)])
        pixels: list[tuple[int, int]] = []
        while q:
            r, c = q.popleft()
            if r < 0 or r >= mask.shape[0] or c < 0 or c >= mask.shape[1]:
                continue
            if visited[r, c] or not mask[r, c]:
                continue
            visited[r, c] = True
            pixels.append((r, c))
            q.extend([(r+1, c), (r-1, c), (r, c+1), (r, c-1)])

        if len(pixels) < min_pixels:
            continue

        arr = np.array(pixels, dtype=float)
        r_min, r_max = arr[:, 0].min(), arr[:, 0].max()
        c_min, c_max = arr[:, 1].min(), arr[:, 1].max()
        blobs.append(Blob(
            pixels=len(pixels),
            height=int(r_max - r_min + 1),
            width=int(c_max - c_min + 1),
            center_r=float(arr[:, 0].mean()),
            center_c=float(arr[:, 1].mean()),
        ))

    return blobs


# ---------------------------------------------------------------------------
# Main extractor
# ---------------------------------------------------------------------------

class PacmanStateExtractor:
    """
    Extracts structured game state from raw RGB frames (250x160x3).

    Pink blobs are classified by width:
      width <= POWER_PELLET_MAX_WIDTH  →  power pellet
      width >  POWER_PELLET_MAX_WIDTH  →  ghost

    Pacman is detected via its distinct yellow color.
    Regular pellets are tan corridors; eaten ones turn black.
    """

    def __init__(self):
        self._initial_pellet_px: Optional[int] = None
        self._wall_mask: Optional[np.ndarray] = None
        self._prev_ghost_positions: Optional[list] = None  # [(r,c), ...] sorted nearest-first
        self._scared_remaining: int = 0

    def _init_from_frame(self, maze: np.ndarray):
        self._wall_mask = _mask(maze, COLOR_WALL)
        self._initial_pellet_px = int(_mask(maze, COLOR_PELLET).sum())

    def extract(self, frame: np.ndarray) -> GameState:
        """frame: (250, 160, 3) uint8 RGB array from the ALE environment."""
        maze = frame[MAZE_ROW_START:MAZE_ROW_END]   # crop to maze area

        if self._wall_mask is None:
            self._init_from_frame(maze)

        # --- Pacman (yellow) ---
        pac_blobs = _find_blobs(_mask(maze, COLOR_PACMAN), min_pixels=8)
        pacman = pac_blobs[0] if pac_blobs else None

        # --- Pink blobs → ghosts or power pellets ---
        scared_mask = _mask(maze, COLOR_SCARED)
        ghost_scared = bool(scared_mask.any())

        pink_mask = _mask(maze, COLOR_PINK) | scared_mask
        pink_blobs = _find_blobs(pink_mask, min_pixels=8)

        ghosts: list[Blob] = []
        power_pellets: list[Blob] = []
        for b in pink_blobs:
            if b.width <= POWER_PELLET_MAX_WIDTH:
                power_pellets.append(b)
            else:
                ghosts.append(b)

        # Sort ghosts by distance to pacman (nearest first)
        if pacman:
            ghosts.sort(key=lambda g: np.hypot(g.center_r - pacman.center_r,
                                                g.center_c - pacman.center_c))

        # Ghost velocities: match current sorted slots to previous sorted slots
        ghost_velocities = []
        prev = self._prev_ghost_positions or []
        for i, g in enumerate(ghosts[:4]):
            if i < len(prev):
                ghost_velocities.append((g.center_r - prev[i][0],
                                         g.center_c - prev[i][1]))
            else:
                ghost_velocities.append((0.0, 0.0))
        self._prev_ghost_positions = [(g.center_r, g.center_c) for g in ghosts[:4]]

        # Scared timer
        if ghost_scared:
            if self._scared_remaining == 0:
                self._scared_remaining = MAX_SCARED_FRAMES
            else:
                self._scared_remaining = max(0, self._scared_remaining - 1)
        else:
            self._scared_remaining = 0
        scared_timer_norm = self._scared_remaining / MAX_SCARED_FRAMES

        # --- Regular pellet fraction ---
        current_px = int(_mask(maze, COLOR_PELLET).sum())
        if self._initial_pellet_px and self._initial_pellet_px > 0:
            pellet_fraction = current_px / self._initial_pellet_px
        else:
            pellet_fraction = 1.0

        return GameState(
            pacman=pacman,
            ghosts=ghosts,
            ghost_velocities=ghost_velocities,
            power_pellets=power_pellets,
            ghost_scared=ghost_scared,
            scared_timer_norm=scared_timer_norm,
            pellet_fraction=pellet_fraction,
        )
