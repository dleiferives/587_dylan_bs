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
static constexpr int   MAX_STEPS       = 500;   // with frame_skip=4, covers same game duration as 2000 steps
static constexpr int   NUM_GENERATIONS = 100000;
static constexpr int   EVAL_TRIALS     = 3;    // average over N runs to reduce noise
static const char*     CHECKPOINT_DIR  = "checkpoints_cpp";

// ─── Evaluate one genome ──────────────────────────────────────────────────────
float evaluate(const neat::Genome& genome,
               ale::ALEInterface& ale,
               uint32_t seed = 0)
{
    ale.setInt("random_seed", (int)seed);
    ale.reset_game();

    const auto& actions = ale.getMinimalActionSet();  // 5 actions for Pac-Man

    neat::Network net = neat::Network::build(genome);
    Extractor extractor;
    extractor.init(ale.getScreen());

    float total_score     = 0.f;
    int   frames_survived = 0;
    float prev_pellet_dist = -1.f;  // for dense shaping

    for (int step = 0; step < MAX_STEPS && !ale.game_over(); step++) {
        GameState state = extractor.extract(ale.getScreen(), ale.getRAM());
        auto inputs = build_inputs(state, extractor);
        auto output = net.activate(inputs);

        // argmax → action
        int best = (int)(std::max_element(output.begin(), output.end()) - output.begin());
        ale::Action action = actions[std::min(best, (int)actions.size()-1)];

        float reward = ale.act(action);
        total_score += reward;
        frames_survived++;

        // Dense shaping: small reward for moving closer to nearest pellet.
        // inputs[29-30] = (delta_r/MAZE_H, delta_c/MAZE_W) to closest pellet.
        float pdr = inputs[29] * MAZE_H;
        float pdc = inputs[30] * MAZE_W;
        float pellet_dist = std::hypot(pdr, pdc);
        if (prev_pellet_dist > 0.f && pellet_dist < prev_pellet_dist)
            total_score += 0.5f * (prev_pellet_dist - pellet_dist);
        prev_pellet_dist = pellet_dist;
    }

    return total_score + frames_survived * 0.04f;  // *4 to match pre-frameskip scale
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
    cfg.pop_size          = 150;
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
    for (int t = 0; t < n_threads; t++) {
        ale_pool[t] = std::make_unique<ale::ALEInterface>();
        ale_pool[t]->setBool("display_screen", false);
        ale_pool[t]->setBool("sound",          false);
        ale_pool[t]->setInt("frame_skip",      4);
        ale_pool[t]->setFloat("repeat_action_probability", 0.0f);  // deterministic
        ale_pool[t]->loadROM(rom_path);
    }

    std::cout << "=== Pac-Man NEAT ===\n"
              << "Population: " << cfg.pop_size << "  Inputs: " << cfg.n_inputs
              << "  Outputs: " << cfg.n_outputs << "\n"
              << "ROM: " << rom_path << "\n\n";

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
                total += evaluate(pop.genomes[i], *ale_pool[tid], trial_seeds[t]);
            pop.genomes[i].fitness = total / EVAL_TRIALS;
        }

        const auto& best = pop.best();
        auto t1 = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(t1-t0).count();

        // Stats
        float mean = 0.f;
        for (auto& g : pop.genomes) mean += g.fitness;
        mean /= pop.genomes.size();

        std::cout << "Gen " << std::setw(3) << gen
                  << "  best=" << std::setw(8) << (int)best.fitness
                  << "  mean=" << std::setw(8) << (int)mean
                  << "  species=" << pop.species.size()
                  << "  nodes=" << best.nodes.size()
                  << "  conns=" << best.conns.size()
                  << "  " << std::setprecision(1) << std::fixed << elapsed << "s\n"
                  << std::flush;

        // Save best genome
        std::string cp = std::string(CHECKPOINT_DIR) + "/best_gen_"
                       + std::to_string(gen) + ".txt";
        save_genome(best, cp);

        pop.evolve();
    }

    std::cout << "\nDone. Best genomes saved to " << CHECKPOINT_DIR << "/\n";
    return 0;
}
