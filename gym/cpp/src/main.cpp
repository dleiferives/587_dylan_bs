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

#include "neat.hpp"
#include "extractor.hpp"

// ─── Config ──────────────────────────────────────────────────────────────────
static constexpr int   MAX_STEPS       = 10000; // safety cap only — real termination is lives() / game_over()
static constexpr int   NUM_GENERATIONS = 100000;
static constexpr int   EVAL_TRIALS     = 1;    // single noisy sample; rely on large pop for selection signal
static constexpr int   VAL_TRIALS      = 5;    // fixed held-out eval set for true progress measurement
static constexpr int   VAL_TOP_K       = 5;    // re-evaluate top-K genomes on val set (robust progress signal)
static const char*     CHECKPOINT_DIR  = "checkpoints_cpp";

// ─── State cache: skip the ROM boot sequence by restoring a snapshot ────────
struct AleCache {
    int last_seed = -2;
    ale::ALEState saved{};
    bool has_saved = false;
};

// ─── Evaluate one genome ──────────────────────────────────────────────────────
float evaluate(const neat::Genome& genome,
               ale::ALEInterface& ale,
               uint32_t seed,
               AleCache& cache)
{
    // Cached reset: if we already booted this seed on this ALE instance,
    // restore the post-boot state instead of re-running reset_game().
    if (cache.has_saved && cache.last_seed == (int)seed) {
        ale.restoreState(cache.saved);
    } else {
        ale.setInt("random_seed", (int)seed);
        ale.reset_game();
        cache.saved = ale.cloneState();
        cache.last_seed = (int)seed;
        cache.has_saved = true;
    }

    const auto& actions = ale.getMinimalActionSet();  // 5 actions for Pac-Man

    neat::Network net = neat::Network::build(genome);
    Extractor extractor;
    extractor.init(ale.getScreen());

    const int initial_lives = ale.lives();

    float total_score     = 0.f;
    int   frames_survived = 0;
    int   pellets_eaten   = 0;
    int   ghosts_eaten    = 0;
    float prev_pellet_frac = -1.f;
    float prev_pellet_dist = -1.f;
    float shaping_total    = 0.f;

    for (int step = 0; step < MAX_STEPS && !ale.game_over(); step++) {
        // Terminate on FIRST death — don't waste cycles on Pac-Man's 2 backup lives.
        if (ale.lives() < initial_lives) break;

        GameState state = extractor.extract(ale.getScreen(), ale.getRAM());
        auto inputs = build_inputs(state, extractor);
        auto output = net.activate(inputs);

        // Track pellets eaten via fraction-drop from extractor.
        if (prev_pellet_frac < 0.f) prev_pellet_frac = state.pellet_fraction;
        if (state.pellet_fraction < prev_pellet_frac) {
            float ate = (prev_pellet_frac - state.pellet_fraction)
                      * (float)extractor.pellet_coords().size();
            pellets_eaten += (int)std::round(ate);
            prev_pellet_frac = state.pellet_fraction;
        }

        // argmax → action
        int best = (int)(std::max_element(output.begin(), output.end()) - output.begin());
        ale::Action action = actions[std::min(best, (int)actions.size()-1)];

        float reward = ale.act(action);
        total_score += reward;
        frames_survived++;

        // Ghost-eating detection: scared-ghost rewards are 200/400/800/1600.
        if (reward >= 200.f) ghosts_eaten++;

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
    float pellet_rate = frames_survived > 0
        ? (float)pellets_eaten / (float)frames_survived : 0.f;

    return total_score
         + ghosts_eaten * 100.f
         + pellet_rate * 500.f
         + shaping_total * 0.05f;
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

// ─── Main ─────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[]) {
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

    for (int gen = 0; gen < NUM_GENERATIONS; gen++) {
        auto t0 = std::chrono::steady_clock::now();

        // Same trial seeds for every genome this generation — fair comparison.
        std::vector<uint32_t> trial_seeds(EVAL_TRIALS);
        for (int t = 0; t < EVAL_TRIALS; t++)
            trial_seeds[t] = seed ^ ((uint32_t)gen * 997u + (uint32_t)t * 31337u);

        int n = (int)pop.genomes.size();

#ifdef USE_OPENMP
        #pragma omp parallel for schedule(dynamic,1)
#endif
        for (int i = 0; i < n; i++) {
#ifdef USE_OPENMP
            int tid = omp_get_thread_num();
#else
            int tid = 0;
#endif
            float total = 0.f;
            for (int t = 0; t < EVAL_TRIALS; t++)
                total += evaluate(pop.genomes[i], *ale_pool[tid], trial_seeds[t], cache_pool[tid]);
            pop.genomes[i].fitness = total / EVAL_TRIALS;
        }

        // Pick top-K by training fitness as validation candidates.
        std::vector<int> idx(pop.genomes.size());
        for (int i = 0; i < (int)idx.size(); i++) idx[i] = i;
        int k = std::min(VAL_TOP_K, (int)idx.size());
        std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
            [&](int a, int b){ return pop.genomes[a].fitness > pop.genomes[b].fitness; });

        // Validate top-K on fixed held-out seeds (in parallel: K*VAL_TRIALS evals).
        std::vector<float> val_scores(k, 0.f);
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
            for (int t = 0; t < VAL_TRIALS; t++)
                total += evaluate(pop.genomes[idx[i]], *ale_pool[tid], val_seeds[t], cache_pool[tid]);
            val_scores[i] = total / VAL_TRIALS;
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

        auto t1 = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(t1-t0).count();

        // Stats
        float mean = 0.f;
        for (auto& g : pop.genomes) mean += g.fitness;
        mean /= pop.genomes.size();

        std::cout << "Gen " << std::setw(3) << gen
                  << "  val_max=" << std::setw(7) << (int)val_max
                  << (new_best ? "*" : " ")
                  << "  val_top5_mean=" << std::setw(7) << (int)val_mean
                  << "  best_ever=" << std::setw(7) << (int)best_val_ever
                  << "@" << std::setw(4) << best_val_gen
                  << "  train_mean=" << std::setw(6) << (int)mean
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
        }

        pop.evolve();
    }

    std::cout << "\nDone. Best genomes saved to " << CHECKPOINT_DIR << "/\n";
    return 0;
}
