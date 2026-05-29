#pragma once
#include <ale/ale_interface.hpp>
#include <ale/environment/ale_ram.hpp>
#include <ale/environment/ale_screen.hpp>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

// ─── Frame constants ──────────────────────────────────────────────────────────
static constexpr int FRAME_H = 250;
static constexpr int FRAME_W = 160;
static constexpr int MAZE_R0 = 18;
static constexpr int MAZE_R1 = 200;
static constexpr int MAZE_H  = MAZE_R1 - MAZE_R0;  // 182
static constexpr int MAZE_W  = FRAME_W;             // 160
static constexpr int MAZE_SZ = MAZE_H * MAZE_W;

// PAL palette indices (verified from ALE's PALPalette[] table)
static constexpr uint8_t PAL_WALL   = 212;
static constexpr uint8_t PAL_PELLET =  42;
static constexpr uint8_t PAL_PINK   = 110;
static constexpr uint8_t PAL_PACMAN =  46;
static constexpr uint8_t PAL_SCARED = 214;

static constexpr int PP_MAX_WIDTH = 5;

// RAM address layout from datacrystal.tcrf.net Pac-Man Atari 2600 RAM map
//   0x08-0x17 (16 bytes) = dot bitmap     ← used for accurate dot count
//   0x18 = lives
//   0x31 = Pac-Man X       0x36 = Pac-Man Y
//   0x32-0x35 = Ghost X    0x37-0x3A = Ghost Y
//   0x4C-0x51 = score
//   0x68 = sound slot (0x01=ghost, 0x02/0x04=power, 0x08=dot)
static constexpr uint8_t RAM_PAC_X      = 0x31;
static constexpr uint8_t RAM_PAC_Y      = 0x36;
static constexpr uint8_t RAM_GHOST_X[4] = {0x32, 0x33, 0x34, 0x35};   // BUGFIX: was {0x30,0x32,0x33,0x34}
static constexpr uint8_t RAM_GHOST_Y[4] = {0x37, 0x38, 0x39, 0x3A};
static constexpr uint8_t RAM_DOTS_BASE  = 0x08;
static constexpr uint8_t RAM_DOTS_END   = 0x17;   // inclusive (16 bytes total)
static constexpr uint8_t RAM_SOUND_SLOT = 0x68;

// ─── Screen helper ────────────────────────────────────────────────────────────
inline uint8_t screen_px(const ale::ALEScreen& s, int r, int c) {
    return s.get(r, c);
}

static constexpr int   MAX_SCARED_FRAMES = 180;  // ~3s at 60fps

// ─── Blob ─────────────────────────────────────────────────────────────────────
struct Blob {
    float center_r, center_c;
    int   width, height, npix;
    float vel_r = 0.f, vel_c = 0.f;  // pixels/frame, set by Extractor
    float norm_r() const { return center_r / MAZE_H; }
    float norm_c() const { return center_c / MAZE_W; }
};

// ─── State ───────────────────────────────────────────────────────────────────
struct GameState {
    bool  pacman_found = false;
    float pacman_r = 0.f, pacman_c = 0.f;
    bool  ghost_scared = false;
    float scared_timer_norm = 0.f;   // 0-1, decounts from 1 when power pellet eaten
    std::array<Blob,4> ghosts;
    int   n_ghosts = 0;
    std::vector<Blob> power_pellets;
    float pellet_fraction = 1.f;
};

// ─── Extractor ────────────────────────────────────────────────────────────────
class Extractor {
public:
    Extractor() {
        pel_mask_ .resize(MAZE_SZ, false);
        wall_mask_.resize(MAZE_SZ, false);
        // wall_dist_table_[r*MAZE_W+c][dir]: precomputed at init
        wall_dist_table_.resize(MAZE_SZ);
    }

    // Call once with the first frame
    void init(const ale::ALEScreen& screen) {
        pellet_coords_.clear();
        power_pellet_coords_.clear();

        for (int r = 0; r < MAZE_H; r++) {
            for (int c = 0; c < MAZE_W; c++) {
                uint8_t px = screen_px(screen, r+MAZE_R0, c);
                int i = r*MAZE_W+c;
                wall_mask_[i] = (px == PAL_WALL);
                if (px == PAL_PELLET) {
                    pel_mask_[i] = true;
                    pellet_coords_.push_back(i);
                }
                if (px == PAL_PINK) {
                    power_pellet_coords_.push_back(i);
                }
            }
        }
        initial_pellet_count_ = std::max((int)pellet_coords_.size(), 1);
        precompute_wall_distances();
        initialized_ = true;
    }

    // Extract state from RAM (positions) + screen palette (pellets + scared only)
    GameState extract(const ale::ALEScreen& screen, const ale::ALERAM& ram) {
        if (!initialized_) init(screen);

        GameState state;

        // ── Pacman from RAM ──────────────────────────────────────────────────
        state.pacman_found = true;
        state.pacman_c = (float)ram.get(RAM_PAC_X);
        state.pacman_r = 2.f * (float)ram.get(RAM_PAC_Y) + 15.f;

        // ── Ghosts from RAM; scared flag via single palette byte per ghost ───
        state.ghost_scared = false;
        state.n_ghosts = 4;
        for (int i = 0; i < 4; i++) {
            float gx = (float)ram.get(RAM_GHOST_X[i]);
            float gy = 2.f * (float)ram.get(RAM_GHOST_Y[i]) + 15.f;
            // Velocity from previous frame (RAM index is stable, no sorting confusion)
            float vr = prev_valid_ ? (gy - prev_ghost_r_[i]) : 0.f;
            float vc = prev_valid_ ? (gx - prev_ghost_c_[i]) : 0.f;
            state.ghosts[i] = {gy, gx, 8, 8, 8, vr, vc};
            prev_ghost_r_[i] = gy;
            prev_ghost_c_[i] = gx;

            int pr = std::clamp((int)gy + MAZE_R0, 0, FRAME_H-1);
            int pc = std::clamp((int)gx,            0, FRAME_W-1);
            if (screen_px(screen, pr, pc) == PAL_SCARED)
                state.ghost_scared = true;
        }
        prev_valid_ = true;

        // Scared timer: reset to MAX when scared starts, count down each frame
        if (state.ghost_scared) {
            if (scared_remaining_ == 0) scared_remaining_ = MAX_SCARED_FRAMES;
            else scared_remaining_ = std::max(0, scared_remaining_ - 1);
        } else {
            scared_remaining_ = 0;
        }
        state.scared_timer_norm = (float)scared_remaining_ / MAX_SCARED_FRAMES;

        // Sort ghosts nearest-first to pacman (velocity travels with the blob)
        float pr = state.pacman_r, pc = state.pacman_c;
        std::sort(state.ghosts.begin(), state.ghosts.begin() + state.n_ghosts,
            [pr, pc](const Blob& a, const Blob& b) {
                float da = (a.center_r-pr)*(a.center_r-pr) + (a.center_c-pc)*(a.center_c-pc);
                float db = (b.center_r-pr)*(b.center_r-pr) + (b.center_c-pc)*(b.center_c-pc);
                return da < db;
            });

        // ── Pellets: positions still from screen (geometric input) but COUNT from RAM ──
        // Screen scan tells us which specific positions still have a dot (used for
        // closest-pellet vector). Brief Pacman occlusion causes flicker but that's OK
        // for direction-finding. The TRUE remaining-count comes from the RAM bitmap.
        for (int idx : pellet_coords_) pel_mask_[idx] = false;   // clear only known positions
        for (int idx : pellet_coords_) {
            int r = idx / MAZE_W, c = idx % MAZE_W;
            if (screen_px(screen, r+MAZE_R0, c) == PAL_PELLET)
                pel_mask_[idx] = true;
        }
        if (ram_dot_initial_ < 0) ram_dot_initial_ = std::max(1, dot_count_from_ram(ram));
        int current_dots_ram = dot_count_from_ram(ram);
        state.pellet_fraction = (float)current_dots_ram / (float)ram_dot_initial_;

        // ── Power pellets: check only known positions ────────────────────────
        state.power_pellets.clear();
        for (int idx : power_pellet_coords_) {
            int r = idx / MAZE_W, c = idx % MAZE_W;
            if (screen_px(screen, r+MAZE_R0, c) == PAL_PINK)
                state.power_pellets.push_back({(float)r, (float)c, 3, 3, 9});
        }

        return state;
    }

    const std::vector<bool>& wall_mask()     const { return wall_mask_; }
    const std::vector<bool>& pel_mask()      const { return pel_mask_; }
    const std::vector<int>&  pellet_coords() const { return pellet_coords_; }

    // RAM-based dot count: popcount of the 16-byte dot bitmap. Clean, no screen artifacts.
    // Initial value (full level) is the bit count of all dots present.
    static int dot_count_from_ram(const ale::ALERAM& ram) {
        int n = 0;
        for (int i = RAM_DOTS_BASE; i <= RAM_DOTS_END; i++)
            n += __builtin_popcount(ram.get(i));
        return n;
    }

    // Sound slot tells us what was just eaten this frame.
    static uint8_t sound_event(const ale::ALERAM& ram) {
        return ram.get(RAM_SOUND_SLOT);
    }

    // Precomputed wall distances lookup: [cell_idx][dir] where dir=0up,1down,2left,3right
    const std::array<float,4>& wall_dist(int r, int c) const {
        return wall_dist_table_[r*MAZE_W+c];
    }

private:
    std::vector<bool>              wall_mask_, pel_mask_;
    std::vector<int>               pellet_coords_, power_pellet_coords_;
    std::vector<std::array<float,4>> wall_dist_table_;
    int  initial_pellet_count_ = 1;
    bool initialized_          = false;
    // Velocity tracking — stored by RAM ghost index (stable across frames)
    std::array<float,4> prev_ghost_r_{}, prev_ghost_c_{};
    bool prev_valid_ = false;
    int  scared_remaining_ = 0;
    int  ram_dot_initial_  = -1;   // popcount of dot bitmap on first extract()

    void precompute_wall_distances() {
        for (int r = 0; r < MAZE_H; r++) {
            for (int c = 0; c < MAZE_W; c++) {
                auto& d = wall_dist_table_[r*MAZE_W+c];
                d[0] = scan(r, c, -1,  0, MAZE_H); // up
                d[1] = scan(r, c,  1,  0, MAZE_H); // down
                d[2] = scan(r, c,  0, -1, MAZE_W); // left
                d[3] = scan(r, c,  0,  1, MAZE_W); // right
            }
        }
    }

    float scan(int r, int c, int dr, int dc, int max_d) const {
        for (int d = 1; d <= max_d; d++) {
            int nr = r + dr*d, nc = c + dc*d;
            if (nr < 0 || nr >= MAZE_H || nc < 0 || nc >= MAZE_W
                || wall_mask_[nr*MAZE_W+nc])
                return (float)(d-1) / max_d;
        }
        return 1.f;
    }
};

// ─── Build 32-element NEAT input vector ──────────────────────────────────────
// [0-3]   Wall distances up/down/left/right
// [4-11]  Ghost 0-3 relative position (delta_r, delta_c), nearest-first
// [12-19] Ghost 0-3 velocity (vel_r, vel_c), same order
// [20]    Scared timer normalized 0-1
// [21-28] Power pellet 0-3 relative position (delta_r, delta_c)
// [29-30] Closest pellet (delta_r, delta_c)
// [31]    Pellet fraction
inline std::vector<float> build_inputs(
    const GameState& state,
    const Extractor& extractor)
{
    std::vector<float> v;
    v.reserve(32);

    float pr = state.pacman_r, pc = state.pacman_c;

    // [0-3] Wall distances
    int ir = std::clamp((int)pr, 0, MAZE_H-1);
    int ic = std::clamp((int)pc, 0, MAZE_W-1);
    const auto& wd = extractor.wall_dist(ir, ic);
    v.insert(v.end(), wd.begin(), wd.end());

    // [4-11] Ghost relative positions (nearest-first)
    for (int i = 0; i < 4; i++) {
        if (i < state.n_ghosts) {
            v.push_back((state.ghosts[i].center_r - pr) / MAZE_H);
            v.push_back((state.ghosts[i].center_c - pc) / MAZE_W);
        } else {
            v.push_back(0.f); v.push_back(0.f);
        }
    }

    // [12-19] Ghost velocities (same order as positions)
    for (int i = 0; i < 4; i++) {
        if (i < state.n_ghosts) {
            v.push_back(state.ghosts[i].vel_r / MAZE_H);
            v.push_back(state.ghosts[i].vel_c / MAZE_W);
        } else {
            v.push_back(0.f); v.push_back(0.f);
        }
    }

    // [20] Scared timer
    v.push_back(state.scared_timer_norm);

    // [21-28] Power pellet relative positions
    for (int i = 0; i < 4; i++) {
        if (i < (int)state.power_pellets.size()) {
            v.push_back((state.power_pellets[i].center_r - pr) / MAZE_H);
            v.push_back((state.power_pellets[i].center_c - pc) / MAZE_W);
        } else {
            v.push_back(0.f); v.push_back(0.f);
        }
    }

    // [29-30] Closest pellet
    const auto& pel_mask   = extractor.pel_mask();
    const auto& pel_coords = extractor.pellet_coords();
    float best_d2 = 1e18f, cpr = 0.f, cpc = 0.f;
    for (int idx : pel_coords) {
        if (!pel_mask[idx]) continue;
        float r = idx / MAZE_W, c = idx % MAZE_W;
        float dr = r - pr, dc = c - pc;
        float d2 = dr*dr + dc*dc;
        if (d2 < best_d2) { best_d2 = d2; cpr = r; cpc = c; }
    }
    v.push_back((cpr - pr) / MAZE_H);
    v.push_back((cpc - pc) / MAZE_W);

    // [31] Pellet fraction
    v.push_back(state.pellet_fraction);

    return v;
}
