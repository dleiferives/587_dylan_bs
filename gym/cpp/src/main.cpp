#include <ale/ale_interface.hpp>
#include <ale/common/Log.hpp>
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#ifdef USE_OPENMP
#include <omp.h>
#endif

#include <random>
#include <unordered_set>

#include "neat.hpp"
#include "extractor.hpp"

// Thread-local RNG for action exploration noise. Without this, a near-tied output
// causes the agent to pick the same action every step → walks into a wall, never explores.
static thread_local std::mt19937 g_action_rng{std::random_device{}()};

// ─── Config ──────────────────────────────────────────────────────────────────
// MAX_STEPS is adaptive: 5 × previous-gen val_mean steps (top-5 capability), floor 500.
// train_mean is dominated by 300 dud genomes that die in 30 steps — useless as a signal.
// val_mean reflects what the BEST agents can do, so the cap grows with real capability.
static constexpr int   NUM_GENERATIONS = 100000;
static constexpr int   EVAL_TRIALS     = 3;    // 3 trials; selection uses MEDIAN (robust to one bad seed)
static constexpr int   VAL_TRIALS      = 5;    // fixed held-out eval set for true progress measurement
static constexpr int   VAL_TOP_K       = 5;    // re-evaluate top-K genomes on val set (robust progress signal)
static const char*     CHECKPOINT_DIR  = "checkpoints_cpp";

// ─── State cache: skip the ROM boot sequence by restoring a snapshot ────────
struct AleCache {
    int last_seed = -2;
    ale::ALEState saved{};
    bool has_saved = false;
};

// ─── Breakdown for debug print ────────────────────────────────────────────────
struct EvalBreakdown {
    float ale_score        = 0.f;
    int   pellets_eaten    = 0;
    int   power_eaten      = 0;
    int   ghosts_eaten     = 0;
    float novelty          = 0.f;
    float step_cost        = 0.f;
    float death_penalty    = 0.f;
    float shaping          = 0.f;
    int   steps            = 0;
    bool  died             = false;
};

// ─── Evaluate one genome ──────────────────────────────────────────────────────
float evaluate(const neat::Genome& genome,
               ale::ALEInterface& ale,
               uint32_t seed,
               AleCache& cache,
               int max_steps,
               int* out_steps = nullptr,
               EvalBreakdown* out_breakdown = nullptr)
{
    // Cached reset: if we already booted this seed on this ALE instance,
    // restore the post-boot state instead of re-running reset_game().
    if (cache.has_saved && cache.last_seed == (int)seed) {
        ale.restoreState(cache.saved);
    } else {
        ale.setInt("random_seed", (int)seed);
        ale.reset_game();
        // Warm up past the attract/title sequence: NOOP frames until lives() is set
        // and the score has stopped jumping. Without this we capture state mid-boot
        // and the first act() returns a huge junk score from the demo screen.
        const auto& acts = ale.getMinimalActionSet();
        for (int w = 0; w < 100; w++) {
            ale.act(acts[0]);              // NOOP (first action is always no-op)
            if (ale.lives() > 0 && w >= 30) break;
        }
        cache.saved = ale.cloneState();
        cache.last_seed = (int)seed;
        cache.has_saved = true;
    }

    const auto& actions = ale.getMinimalActionSet();  // 5 actions for Pac-Man

    neat::Network net = neat::Network::build(genome);
    Extractor extractor;
    extractor.init(ale.getScreen());

    const int initial_lives = ale.lives();

    float total_score          = 0.f;
    int   frames_survived      = 0;
    int   pellets_eaten        = 0;   // from RAM dot-bitmap delta (exact)
    int   ghosts_eaten         = 0;   // from RAM sound slot 0x68 (rising-edge)
    int   power_pellets_eaten  = 0;   // from RAM sound slot 0x68 (rising-edge)
    int   prev_dot_count       = Extractor::dot_count_from_ram(ale.getRAM());
    uint8_t prev_sound         = Extractor::sound_event(ale.getRAM());
    float prev_pellet_dist     = -1.f;
    float shaping_total        = 0.f;
    bool  died                 = false;

    // Novelty: count unique maze tiles visited (16x16 grid → ~120 tiles).
    // Capped at +150 to prevent novelty-cycling between two tiles.
    std::unordered_set<int> visited_tiles;
    float novelty_bonus = 0.f;

    for (int step = 0; step < max_steps && !ale.game_over(); step++) {
        // Terminate on FIRST death — don't waste cycles on Pac-Man's 2 backup lives.
        if (ale.lives() < initial_lives) { died = true; break; }

        GameState state = extractor.extract(ale.getScreen(), ale.getRAM());
        auto inputs = build_inputs(state, extractor);
        auto output = net.activate(inputs);

        // Novelty: bonus for each unique 16x16 tile visited (cap at 300).
        // Bigger cap pushes agents to explore deeper into the maze where power pellets live.
        if (novelty_bonus < 300.f) {
            int tr = (int)state.pacman_r / 16;
            int tc = (int)state.pacman_c / 16;
            int key = tr * 256 + tc;
            if (visited_tiles.insert(key).second) novelty_bonus += 1.f;
        }

        // Softmax(τ) action selection. Per Sutton & Barto §2.3 and the noisy-EA
        // literature, softmax beats epsilon-greedy when the worst actions are very bad
        // (= running into a ghost). With τ=0.5 the best action is favored but lesser
        // actions still get tried with probability proportional to exp(o/τ).
        constexpr float TEMPERATURE = 0.5f;
        float max_out = *std::max_element(output.begin(), output.end());
        std::vector<float> probs(output.size());
        float psum = 0.f;
        for (size_t i = 0; i < output.size(); i++) {
            probs[i] = std::exp((output[i] - max_out) / TEMPERATURE);
            psum += probs[i];
        }
        std::uniform_real_distribution<float> u01(0.f, 1.f);
        float r = u01(g_action_rng) * psum;
        int best = 0;
        float cum = 0.f;
        for (size_t i = 0; i < probs.size(); i++) {
            cum += probs[i];
            if (r <= cum) { best = (int)i; break; }
        }
        ale::Action action = actions[std::min(best, (int)actions.size()-1)];

        float reward = ale.act(action);
        total_score += reward;
        frames_survived++;

        // *** EVENT DETECTION from RAM (post-act state) ***
        // Dots: diff the bitmap → exact count of dots cleared this frame_skip block.
        int curr_dot_count = Extractor::dot_count_from_ram(ale.getRAM());
        if (curr_dot_count < prev_dot_count) {
            pellets_eaten += (prev_dot_count - curr_dot_count);
            prev_dot_count = curr_dot_count;
        } else if (curr_dot_count > prev_dot_count) {
            // Level wrapped/reset — recalibrate baseline.
            prev_dot_count = curr_dot_count;
        }
        // Ghosts and power pellets: sound slot, RISING EDGE only.
        // The sound bit persists for ~30 game frames after the event. We poll every
        // frame_skip=8 frames, so without edge detection one ghost eat counts 3-4×.
        uint8_t sound = Extractor::sound_event(ale.getRAM());
        uint8_t newly_set = sound & ~prev_sound;
        if (newly_set & 0x01) ghosts_eaten++;
        if (newly_set & 0x06) power_pellets_eaten++;   // 0x02 OR 0x04
        prev_sound = sound;

        // Tiny dense shaping — just enough gradient for blind early agents,
        // not enough to farm by orbiting pellets without eating.
        float pdr = inputs[29] * MAZE_H;
        float pdc = inputs[30] * MAZE_W;
        float pellet_dist = std::hypot(pdr, pdc);
        if (prev_pellet_dist > 0.f && pellet_dist < prev_pellet_dist)
            shaping_total += (prev_pellet_dist - pellet_dist);
        prev_pellet_dist = pellet_dist;
    }

    // Fitness composition:
    //   ALE total_score:         10/pellet, 50/power, 200..1600/ghost, level bonus
    //   + ghosts_eaten * 100:    extra emphasis — ghost-eating is the high-skill behavior
    //   + pellet_rate bonus:     pellets per step, scaled — rewards efficient clearing
    //   + tiny shaping:          early-learning gradient only
    //   NO survival bonus:       was the lazy-corner-hugger attractor
    if (out_steps) *out_steps = frames_survived;

    // Per-step cost: punishes stalling more aggressively. -0.15 × 200 normal play = -30
    // (small vs pellet rewards). Stallers hitting cap=765 get -115, killing their score.
    float step_cost = (float)frames_survived * -0.15f;

    // Death penalty: REMOVED — was -300, which dominated all other signal and made
    // every agent's fitness ~-280 regardless of skill. Staying alive is already
    // implicitly rewarded by accumulating more ALE score, more novelty, more pellets.
    float death_penalty = 0.f;

    // Fitness composition (HRA-inspired ratios, scaled for our score range):
    //   ALE total_score:               10/pellet, 50/power, 200..1600/ghost, level bonus
    //   + ghosts_eaten * 500:          ONE ghost eat (~700 total) > entire pellet rush
    //   + power_pellets_eaten * 200:   makes triggering frighten mode worth ~25 pellets
    //   + pellets_eaten * 2:           tiny on top of ALE's 10
    //   + novelty_bonus (≤ 150):       breaks corner-hugger by rewarding maze exploration
    //   + step_cost (negative):        replaces survival bonus, punishes stalling
    //   + death_penalty (if died):     teaches ghost-avoidance
    //   + shaping (tiny):              early-learning gradient
    if (out_breakdown) {
        out_breakdown->ale_score     = total_score;
        out_breakdown->pellets_eaten = pellets_eaten;   // counted via reward magnitudes
        out_breakdown->power_eaten   = power_pellets_eaten;
        out_breakdown->ghosts_eaten  = ghosts_eaten;
        out_breakdown->novelty       = novelty_bonus;
        out_breakdown->step_cost     = step_cost;
        out_breakdown->death_penalty = death_penalty;
        out_breakdown->shaping       = shaping_total * 0.05f;
        out_breakdown->steps         = frames_survived;
        out_breakdown->died          = died;
    }

    // Pellet rewards come naturally from ALE total_score (10 pts each).
    // Bonuses emphasize the high-skill behaviors that ALE alone wouldn't make
    // visible to evolution at this score scale.
    // 2600 scoring is 10× smaller than arcade. Rescaling bonuses to match:
    //   ALE total_score: now ~10-40 per game (was thought to be 100-400)
    //   ghost: +50 bonus (1 ghost = ~16 ALE + 50 bonus = 66 — beats eating 20 pellets)
    //   power: +20 bonus (1 power = ~5 ALE + 20 bonus = 25 — beats 5 pellets, signals chain)
    //   novelty: keep ~300 cap, dominates pellets and pushes exploration
    //   step_cost: -0.15/step keeps stallers negative
    return total_score
         + ghosts_eaten * 50.f
         + power_pellets_eaten * 20.f
         + novelty_bonus
         + step_cost
         + death_penalty
         + shaping_total * 0.5f;
}

// ─── Save genome to text file ─────────────────────────────────────────────────
void save_genome(const neat::Genome& g, const std::string& path) {
    std::ofstream f(path);
    f << "fitness " << g.fitness << "\n";
    f << "nodes " << g.nodes.size() << "\n";
    for (auto& n : g.nodes)
        f << "n " << n.id << " " << (int)n.type << " " << n.bias << "\n";
    f << "conns " << g.conns.size() << "\n";
    for (auto& c : g.conns)
        f << "c " << c.from << " " << c.to << " " << c.weight
          << " " << c.enabled << " " << c.innov << " " << c.disabled_since << "\n";
}

// ─── RAM-diff diagnostic ──────────────────────────────────────────────────────
// Goal: find which RAM bytes hold the pellet bitmap. When a pellet is eaten, one
// bit in some byte gets cleared. We detect by playing random actions, watching
// for reward==10 events (pellet eat), and logging which RAM bytes lost exactly
// 1 bit on those events.
void run_ram_diff(const std::string& rom_path) {
    ale::Logger::setMode(ale::Logger::mode::Error);
    ale::ALEInterface ale;
    ale.setBool("display_screen", false);
    ale.setBool("sound",          false);
    ale.setInt("frame_skip",      4);   // need real movement to actually find pellets
    ale.setFloat("repeat_action_probability", 0.0f);
    ale.loadROM(rom_path);

    constexpr int RAM_SZ = 128;
    const auto& ram = ale.getRAM();
    const auto& acts = ale.getMinimalActionSet();
    std::mt19937 rng(123);
    std::uniform_int_distribution<int> pick(0, (int)acts.size()-1);

    // Per-byte stats — accumulated across multiple games
    std::array<int, RAM_SZ> pellet_clear_count{};
    std::array<int, RAM_SZ> any_change_count{};
    std::map<int, int> reward_histogram;   // reward value → count
    int pellet_events = 0, total_steps = 0, total_games = 0;

    std::cerr << "[ram-diff] running random play across multiple games...\n";

    for (int game = 0; game < 30 && pellet_events < 100; game++) {
        ale.setInt("random_seed", 42 + game);
        ale.reset_game();
        // Warm past boot
        for (int w = 0; w < 30; w++) ale.act(acts[0]);
        int initial_lives = ale.lives();
        total_games++;

        std::array<uint8_t, RAM_SZ> prev_ram{}, curr_ram{};
        for (int i = 0; i < RAM_SZ; i++) prev_ram[i] = ram.get(i);

        for (int step = 0; step < 500 && !ale.game_over(); step++) {
            if (ale.lives() < initial_lives) break;
            ale::Action a = acts[pick(rng)];
            float reward = ale.act(a);
            total_steps++;

            if (reward > 0.f) reward_histogram[(int)std::round(reward)]++;

            for (int i = 0; i < RAM_SZ; i++) curr_ram[i] = ram.get(i);
            for (int i = 0; i < RAM_SZ; i++) {
                if (curr_ram[i] != prev_ram[i]) any_change_count[i]++;
            }

            // Treat any small positive reward (likely pellet) as event
            if (reward > 0.f && reward < 30.f) {
                pellet_events++;
                for (int i = 0; i < RAM_SZ; i++) {
                    if (curr_ram[i] != prev_ram[i]) {
                        int dpop = __builtin_popcount(curr_ram[i])
                                 - __builtin_popcount(prev_ram[i]);
                        if (dpop == -1) pellet_clear_count[i]++;
                    }
                }
            }
            prev_ram = curr_ram;
        }
    }

    std::cerr << "\n[ram-diff] DONE.\n"
              << "  games=" << total_games
              << "  total_steps=" << total_steps
              << "  pellet_events=" << pellet_events << "\n\n";

    std::cerr << "Reward histogram (all positive rewards seen):\n";
    for (auto& [val, cnt] : reward_histogram)
        std::cerr << "  reward=" << val << "  count=" << cnt << "\n";

    if (pellet_events == 0) {
        std::cerr << "\nNo pellet events. The reward values above tell us what to look for.\n";
        return;
    }

    std::cerr << "\nRAM bytes that LOST EXACTLY 1 BIT on pellet events:\n";
    std::cerr << "  addr   pellet_clears / events    any_changes / steps\n";
    std::vector<std::pair<int,int>> candidates;
    for (int i = 0; i < RAM_SZ; i++) {
        if (pellet_clear_count[i] >= 2) {
            candidates.push_back({pellet_clear_count[i], i});
        }
    }
    std::sort(candidates.rbegin(), candidates.rend());
    for (auto& [score, addr] : candidates) {
        std::cerr << "  0x" << std::hex << std::setw(2) << std::setfill('0') << addr
                  << std::dec << std::setfill(' ')
                  << "      " << std::setw(3) << pellet_clear_count[addr]
                  << " / " << pellet_events
                  << "                "
                  << std::setw(5) << any_change_count[addr] << " / " << total_steps
                  << "\n";
    }
}

// ─── Main ─────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[]) {
    if (argc >= 3 && std::string(argv[2]) == "ram-diff") {
        run_ram_diff(argv[1]);
        return 0;
    }
    if (argc < 2) {
        std::cerr << "Usage: pacman_neat <path/to/pacman.bin> [seed]\n";
        return 1;
    }
    std::string rom_path = argv[1];
    uint32_t    seed     = (argc > 2) ? std::stoul(argv[2]) : 42;

    std::filesystem::create_directories(CHECKPOINT_DIR);

    neat::Config cfg;
    cfg.n_inputs          = 32;
    cfg.n_outputs         = 5;
    cfg.pop_size          = 300;
    cfg.compat_threshold  = 2.0f;   // larger species = more runway for structural mutations to survive
    cfg.add_conn_prob     = 0.25f;
    cfg.add_node_prob     = 0.10f;
    cfg.weight_mutate_rate = 0.8f;
    cfg.max_stagnation    = 35;     // structural innovations need more time to refine
    cfg.prune_stale_gens  = 40;
    cfg.elitism           = 5;      // protect more top genomes per species from mutation
    cfg.survival_threshold = 0.3f;  // broader breeding pool slows mean collapse
    cfg.init_conn         = neat::InitConn::PARTIAL_DIRECT;
    cfg.init_conn_frac    = 0.4f;   // 40% of input→output edges = ~64 initial connections, gives structural diversity from gen 0

    neat::Population pop(cfg, seed);

    // Create one ALE instance per thread (reused across all generations)
    ale::Logger::setMode(ale::Logger::mode::Error);
#ifdef USE_OPENMP
    int n_threads = omp_get_max_threads();
#else
    int n_threads = 1;
#endif
    std::vector<std::unique_ptr<ale::ALEInterface>> ale_pool(n_threads);
    std::vector<AleCache> cache_pool(n_threads);
    for (int t = 0; t < n_threads; t++) {
        ale_pool[t] = std::make_unique<ale::ALEInterface>();
        ale_pool[t]->setBool("display_screen", false);
        ale_pool[t]->setBool("sound",          false);
        ale_pool[t]->setInt("frame_skip",      8);
        ale_pool[t]->setFloat("repeat_action_probability", 0.0f);  // deterministic
        ale_pool[t]->loadROM(rom_path);
    }

    std::cout << "=== Pac-Man NEAT ===\n"
              << "Population: " << cfg.pop_size << "  Inputs: " << cfg.n_inputs
              << "  Outputs: " << cfg.n_outputs << "\n"
              << "ROM: " << rom_path << "\n\n";

    // Fixed held-out validation seed set — NEVER changes across generations.
    // This is the only reliable signal for "is the agent actually getting better?"
    std::vector<uint32_t> val_seeds(VAL_TRIALS);
    for (int t = 0; t < VAL_TRIALS; t++)
        val_seeds[t] = 0xDEADBEEF ^ (uint32_t)(t * 2654435761u);

    float best_val_ever = -1e9f;
    int   best_val_gen  = -1;
    int   adaptive_max_steps = 500;  // baseline for gen 0; updated from val_mean each gen

    // ─── Hall of Fame: preserve top-K genomes EVER, regardless of species stagnation ─
    // Counters the "lucky breakthrough lost" problem: a genome that scored well once
    // is kept and re-evaluated every gen with fresh seeds. Genes spread via crossover.
    constexpr int HOF_SIZE = 8;
    struct HofEntry {
        neat::Genome genome;
        float val_score = -1e9f;   // last re-evaluated val score
        int   first_gen = -1;       // when entered the hall
    };
    std::vector<HofEntry> hall_of_fame;

    for (int gen = 0; gen < NUM_GENERATIONS; gen++) {
        auto t0 = std::chrono::steady_clock::now();

        // Same trial seeds for every genome this generation — fair comparison.
        std::vector<uint32_t> trial_seeds(EVAL_TRIALS);
        for (int t = 0; t < EVAL_TRIALS; t++)
            trial_seeds[t] = seed ^ ((uint32_t)gen * 997u + (uint32_t)t * 31337u);

        int n = (int)pop.genomes.size();
        std::vector<int> train_steps(n, 0);

#ifdef USE_OPENMP
        #pragma omp parallel for schedule(dynamic,1)
#endif
        for (int i = 0; i < n; i++) {
#ifdef USE_OPENMP
            int tid = omp_get_thread_num();
#else
            int tid = 0;
#endif
            // Run each trial, collect per-trial fitness. Selection uses MEDIAN
            // not MEAN — robust to single lucky/unlucky seed dragging the average.
            std::array<float, EVAL_TRIALS> trial_fit{};
            int step_total = 0;
            for (int t = 0; t < EVAL_TRIALS; t++) {
                int s = 0;
                trial_fit[t] = evaluate(pop.genomes[i], *ale_pool[tid], trial_seeds[t],
                                        cache_pool[tid], adaptive_max_steps, &s);
                step_total += s;
            }
            // Median of EVAL_TRIALS=3 → sort and take middle element
            std::array<float, EVAL_TRIALS> sorted = trial_fit;
            std::sort(sorted.begin(), sorted.end());
            pop.genomes[i].fitness = sorted[EVAL_TRIALS / 2];
            train_steps[i] = step_total / EVAL_TRIALS;
        }

        // Pick top-K by training fitness as validation candidates.
        std::vector<int> idx(pop.genomes.size());
        for (int i = 0; i < (int)idx.size(); i++) idx[i] = i;
        int k = std::min(VAL_TOP_K, (int)idx.size());
        std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
            [&](int a, int b){ return pop.genomes[a].fitness > pop.genomes[b].fitness; });

        // Validate top-K on fixed held-out seeds (in parallel: K*VAL_TRIALS evals).
        std::vector<float> val_scores(k, 0.f);
        std::vector<int>   val_steps(k, 0);
#ifdef USE_OPENMP
        #pragma omp parallel for schedule(dynamic,1)
#endif
        for (int i = 0; i < k; i++) {
#ifdef USE_OPENMP
            int tid = omp_get_thread_num();
#else
            int tid = 0;
#endif
            float total = 0.f;
            int   step_total = 0;
            for (int t = 0; t < VAL_TRIALS; t++) {
                int s = 0;
                total += evaluate(pop.genomes[idx[i]], *ale_pool[tid], val_seeds[t], cache_pool[tid],
                                  adaptive_max_steps, &s);
                step_total += s;
            }
            val_scores[i] = total / VAL_TRIALS;
            val_steps[i]  = step_total / VAL_TRIALS;
        }

        float val_max = *std::max_element(val_scores.begin(), val_scores.end());
        float val_mean = 0.f;
        for (float v : val_scores) val_mean += v;
        val_mean /= k;

        int champ_idx = idx[(int)(std::max_element(val_scores.begin(), val_scores.end())
                              - val_scores.begin())];
        const auto& best = pop.genomes[champ_idx];   // true val champion, not lucky one

        bool new_best = false;
        if (val_max > best_val_ever) {
            best_val_ever = val_max;
            best_val_gen  = gen;
            new_best = true;
        }

        // ── Hall of Fame update ──
        // Re-evaluate existing HOF members on the SAME val seeds we just used,
        // so their scores reflect current selection pressure (not stale memories).
        for (auto& h : hall_of_fame) {
            float total = 0.f;
            for (int t = 0; t < VAL_TRIALS; t++)
                total += evaluate(h.genome, *ale_pool[0], val_seeds[t], cache_pool[0],
                                  adaptive_max_steps);
            h.val_score = total / VAL_TRIALS;
        }
        // Consider adding the top-K val genomes from this gen.
        for (int i = 0; i < k; i++) {
            float s = val_scores[i];
            if ((int)hall_of_fame.size() < HOF_SIZE) {
                HofEntry e;
                e.genome = pop.genomes[idx[i]];
                e.val_score = s;
                e.first_gen = gen;
                hall_of_fame.push_back(std::move(e));
            } else {
                // Find worst entry; replace if this candidate beats it
                auto worst = std::min_element(hall_of_fame.begin(), hall_of_fame.end(),
                    [](const HofEntry& a, const HofEntry& b){ return a.val_score < b.val_score; });
                if (s > worst->val_score) {
                    worst->genome = pop.genomes[idx[i]];
                    worst->val_score = s;
                    worst->first_gen = gen;
                }
            }
        }
        // Sort HOF best-first for display
        std::sort(hall_of_fame.begin(), hall_of_fame.end(),
            [](const HofEntry& a, const HofEntry& b){ return a.val_score > b.val_score; });

        auto t1 = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(t1-t0).count();

        // Stats
        float mean = 0.f;
        for (auto& g : pop.genomes) mean += g.fitness;
        mean /= pop.genomes.size();

        float train_steps_mean = 0.f;
        for (int s : train_steps) train_steps_mean += s;
        train_steps_mean /= std::max(1, (int)train_steps.size());

        // Cap update moved below — needs val_steps_mean which is computed after this.

        float val_steps_mean = 0.f;
        for (int s : val_steps) val_steps_mean += s;
        val_steps_mean /= std::max(1, (int)val_steps.size());

        // Update adaptive cap for NEXT generation: 5× val_mean (top-5 capability), floor 500.
        adaptive_max_steps = std::max(500, 5 * (int)val_steps_mean);

        int champ_val_pos = (int)(std::max_element(val_scores.begin(), val_scores.end())
                              - val_scores.begin());
        int best_steps = val_steps[champ_val_pos];

        std::cout << "Gen " << std::setw(3) << gen
                  << "  val_max=" << std::setw(7) << (int)val_max
                  << (new_best ? "*" : " ")
                  << "  val_top5_mean=" << std::setw(7) << (int)val_mean
                  << "  best_ever=" << std::setw(7) << (int)best_val_ever
                  << "@" << std::setw(4) << best_val_gen
                  << "  train_mean=" << std::setw(6) << (int)mean
                  << "  steps(train_mean/val_mean/best)="
                  << (int)train_steps_mean << "/"
                  << (int)val_steps_mean   << "/"
                  << best_steps
                  << "  cap=" << adaptive_max_steps
                  << "  hof=" << (hall_of_fame.empty() ? 0 : (int)hall_of_fame.front().val_score)
                  << "/" << (hall_of_fame.empty() ? 0 : (int)hall_of_fame.back().val_score)
                  << "(" << hall_of_fame.size() << ")"
                  << "  sp=" << pop.species.size()
                  << "  n=" << best.nodes.size()
                  << "  c=" << best.conns.size()
                  << "  " << std::setprecision(1) << std::fixed << elapsed << "s\n"
                  << std::flush;

        // Save only on validation improvement — stops disk spam, keeps real bests.
        if (new_best) {
            std::string cp = std::string(CHECKPOINT_DIR) + "/best_val_gen_"
                           + std::to_string(gen) + "_score_"
                           + std::to_string((int)val_max) + ".txt";
            save_genome(best, cp);

            // Diagnostic: re-run the champion on ALL val seeds, print per-trial
            // breakdown. val_max is the MEAN across these trials — individual trials
            // can vary wildly because Pac-Man starts are very seed-dependent.
            for (int t = 0; t < VAL_TRIALS; t++) {
                EvalBreakdown bd;
                float f = evaluate(best, *ale_pool[0], val_seeds[t], cache_pool[0],
                                   adaptive_max_steps, nullptr, &bd);
                std::cout << "    [trial " << t << "] fit=" << (int)f
                          << " ale=" << (int)bd.ale_score
                          << " p=" << bd.pellets_eaten
                          << " pwr=" << bd.power_eaten
                          << " g=" << bd.ghosts_eaten
                          << " nov=" << (int)bd.novelty
                          << " step=" << (int)bd.step_cost
                          << " shape=" << (int)bd.shaping
                          << " steps=" << bd.steps
                          << " died=" << bd.died
                          << "\n";
            }
            std::cout << std::flush;
        }

        pop.evolve();

        // ── Hall of Fame injection ──
        // After evolve() builds the new population, splice HOF genomes in to replace
        // the bottom-K by training fitness. They get re-evaluated next gen and their
        // genes propagate via species assignment + crossover.
        if (!hall_of_fame.empty()) {
            // Sort current new population by fitness ascending — worst at front.
            // (Note: fitness is 0 after evolve resets it, so we use insertion order as proxy.)
            // Simply overwrite the LAST K genomes with HOF copies. evolve() places elites
            // first, so the tail tends to be filler/mutated entries — safe to replace.
            int inject = std::min((int)hall_of_fame.size(), (int)pop.genomes.size());
            for (int i = 0; i < inject; i++) {
                auto& slot = pop.genomes[pop.genomes.size() - 1 - i];
                slot = hall_of_fame[i].genome;
                slot.id = pop.next_genome_id++;
                slot.fitness = 0.f;
            }
        }
    }

    std::cout << "\nDone. Best genomes saved to " << CHECKPOINT_DIR << "/\n";
    return 0;
}
