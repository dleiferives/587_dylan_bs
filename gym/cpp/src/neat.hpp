#pragma once
#include <algorithm>
#include <cassert>
#include <cmath>
#include <functional>
#include <map>
#include <random>
#include <set>
#include <unordered_map>
#include <vector>

namespace neat {

// ─── Activation ──────────────────────────────────────────────────────────────
inline float activate(float x) { return std::tanh(x); }

// ─── Innovation tracker ──────────────────────────────────────────────────────
struct InnovTracker {
    int next = 1;
    std::map<std::pair<int,int>, int> history;

    int get(int from, int to) {
        auto k = std::make_pair(from, to);
        auto it = history.find(k);
        if (it != history.end()) return it->second;
        return history[k] = next++;
    }
};

// ─── Genes ───────────────────────────────────────────────────────────────────
enum class NodeType { INPUT, HIDDEN, OUTPUT };

struct NodeGene {
    int    id;
    NodeType type;
    float  bias = 0.f;
};

struct ConnGene {
    int   from, to, innov;
    float weight;
    bool  enabled;
    int   disabled_since = -1;  // generation when disabled; -1 means currently enabled
};

// ─── Initial connection topology ─────────────────────────────────────────────
enum class InitConn { FULL_DIRECT, UNCONNECTED, PARTIAL_DIRECT };

// ─── Genome ──────────────────────────────────────────────────────────────────
struct Genome {
    int id = -1;
    float fitness = 0.f;
    int next_node = 0;
    std::vector<NodeGene> nodes;
    std::vector<ConnGene> conns;

    // --- construction ---
    static Genome make_minimal(int id, int n_in, int n_out,
                               InnovTracker& innov, std::mt19937& rng,
                               InitConn init_conn = InitConn::FULL_DIRECT,
                               float partial_frac = 0.5f) {
        Genome g;
        g.id = id;
        std::uniform_real_distribution<float> wd(-1.f, 1.f);
        std::uniform_real_distribution<float> prob(0.f, 1.f);
        for (int i = 0; i < n_in;  i++) g.nodes.push_back({g.next_node++, NodeType::INPUT,  0.f});
        for (int i = 0; i < n_out; i++) g.nodes.push_back({g.next_node++, NodeType::OUTPUT, 0.f});
        if (init_conn != InitConn::UNCONNECTED) {
            for (int i = 0; i < n_in; i++)
                for (int j = n_in; j < n_in + n_out; j++) {
                    if (init_conn == InitConn::PARTIAL_DIRECT && prob(rng) >= partial_frac)
                        continue;
                    g.conns.push_back({i, j, innov.get(i, j), wd(rng), true, -1});
                }
        }
        return g;
    }

    // Remove connections disabled for >= stale_gens generations, then orphaned hidden nodes.
    void prune_stale(int current_gen, int stale_gens) {
        conns.erase(std::remove_if(conns.begin(), conns.end(), [&](const ConnGene& c) {
            return !c.enabled && c.disabled_since >= 0
                   && (current_gen - c.disabled_since) >= stale_gens;
        }), conns.end());

        std::set<int> referenced;
        for (auto& c : conns) { referenced.insert(c.from); referenced.insert(c.to); }
        nodes.erase(std::remove_if(nodes.begin(), nodes.end(), [&](const NodeGene& n) {
            return n.type == NodeType::HIDDEN && !referenced.count(n.id);
        }), nodes.end());
    }

    // --- mutations ---
    void mutate_weights(float rate, float replace_rate, float power, std::mt19937& rng) {
        std::uniform_real_distribution<float> u(0.f, 1.f);
        std::normal_distribution<float>       nd(0.f, power);
        std::uniform_real_distribution<float> wr(-1.f, 1.f);
        for (auto& c : conns) {
            if (u(rng) < rate)
                c.weight = (u(rng) < replace_rate) ? wr(rng)
                           : std::clamp(c.weight + nd(rng), -8.f, 8.f);
        }
        for (auto& n : nodes) {
            if (n.type == NodeType::INPUT) continue;
            if (u(rng) < rate)
                n.bias = (u(rng) < replace_rate) ? wr(rng)
                         : std::clamp(n.bias + nd(rng), -8.f, 8.f);
        }
    }

    void mutate_add_node(InnovTracker& innov, std::mt19937& rng, int current_gen) {
        std::vector<int> ei;
        for (int i = 0; i < (int)conns.size(); i++) if (conns[i].enabled) ei.push_back(i);
        if (ei.empty()) return;
        std::uniform_int_distribution<int> pick(0, ei.size()-1);
        auto& old = conns[ei[pick(rng)]];
        old.enabled = false;
        old.disabled_since = current_gen;
        int nid = next_node++;
        nodes.push_back({nid, NodeType::HIDDEN, 0.f});
        conns.push_back({old.from, nid,      innov.get(old.from, nid),      1.f,        true, -1});
        conns.push_back({nid,      old.to,   innov.get(nid,      old.to),   old.weight, true, -1});
    }

    bool mutate_add_conn(InnovTracker& innov, std::mt19937& rng) {
        std::set<std::pair<int,int>> existing;
        for (auto& c : conns) existing.insert({c.from, c.to});

        std::vector<int> src, dst;
        for (auto& n : nodes) {
            if (n.type != NodeType::OUTPUT) src.push_back(n.id);
            if (n.type != NodeType::INPUT)  dst.push_back(n.id);
        }
        std::shuffle(src.begin(), src.end(), rng);
        std::shuffle(dst.begin(), dst.end(), rng);

        std::uniform_real_distribution<float> wd(-1.f, 1.f);
        for (int f : src) for (int t : dst) {
            if (f == t) continue;
            if (existing.count({f, t})) continue;
            if (creates_cycle(f, t)) continue;
            conns.push_back({f, t, innov.get(f, t), wd(rng), true});
            return true;
        }
        return false;
    }

    void mutate_toggle(std::mt19937& rng, int current_gen) {
        if (conns.empty()) return;
        std::uniform_int_distribution<int> pick(0, conns.size()-1);
        auto& c = conns[pick(rng)];
        c.enabled ^= true;
        c.disabled_since = c.enabled ? -1 : current_gen;
    }

    // --- compatibility ---
    float distance(const Genome& o, float c1, float c2) const {
        std::map<int, const ConnGene*> mine, theirs;
        for (auto& c : conns)   mine[c.innov]   = &c;
        for (auto& c : o.conns) theirs[c.innov] = &c;

        int N = std::max({(int)conns.size(), (int)o.conns.size(), 1});
        if (N < 20) N = 1;

        int disjoint = 0; float wdiff = 0.f; int matching = 0;
        for (auto& [k, g] : mine) {
            auto it = theirs.find(k);
            if (it == theirs.end()) { disjoint++; }
            else { wdiff += std::abs(g->weight - it->second->weight); matching++; }
        }
        for (auto& [k, g] : theirs) if (!mine.count(k)) disjoint++;

        return c1 * disjoint / N + c2 * (matching ? wdiff / matching : 0.f);
    }

    // --- crossover ---
    static Genome crossover(const Genome& p1, const Genome& p2,
                            bool p1_fitter, std::mt19937& rng) {
        const Genome& better = p1_fitter ? p1 : p2;
        const Genome& worse  = p1_fitter ? p2 : p1;

        std::map<int, const ConnGene*> wmap;
        for (auto& c : worse.conns) wmap[c.innov] = &c;

        Genome child;
        child.nodes      = better.nodes;
        child.next_node  = better.next_node;

        std::uniform_real_distribution<float> u(0.f, 1.f);
        for (auto& c : better.conns) {
            auto it = wmap.find(c.innov);
            child.conns.push_back((it != wmap.end() && u(rng) < .5f) ? *it->second : c);
        }
        return child;
    }

private:
    bool creates_cycle(int from, int to) const {
        // DFS from 'to': if we reach 'from', adding from->to creates a cycle
        std::set<int> seen;
        std::vector<int> stack = {to};
        while (!stack.empty()) {
            int cur = stack.back(); stack.pop_back();
            if (cur == from) return true;
            if (!seen.insert(cur).second) continue;
            for (auto& c : conns)
                if (c.enabled && c.from == cur) stack.push_back(c.to);
        }
        return false;
    }
};

// ─── Feedforward Network ──────────────────────────────────────────────────────
struct Network {
    struct EvalNode {
        int id;
        float bias;
        std::vector<std::pair<int,float>> inputs;  // (from_id, weight)
    };

    std::vector<int>      input_ids, output_ids;
    std::vector<EvalNode> order;  // topological, inputs excluded

    static Network build(const Genome& g) {
        Network net;
        std::unordered_map<int, EvalNode> nmap;
        for (auto& ng : g.nodes) {
            EvalNode en; en.id = ng.id; en.bias = ng.bias;
            nmap[ng.id] = en;
            if (ng.type == NodeType::INPUT)  net.input_ids.push_back(ng.id);
            if (ng.type == NodeType::OUTPUT) net.output_ids.push_back(ng.id);
        }
        for (auto& c : g.conns)
            if (c.enabled && nmap.count(c.to))
                nmap[c.to].inputs.push_back({c.from, c.weight});

        // Kahn's topological sort
        std::unordered_map<int,int> indeg;
        std::unordered_map<int,std::vector<int>> adj;
        for (auto& ng : g.nodes) indeg[ng.id] = 0;
        for (auto& c  : g.conns) if (c.enabled) { adj[c.from].push_back(c.to); indeg[c.to]++; }

        std::vector<int> q;
        for (auto& [id, d] : indeg) if (d == 0) q.push_back(id);

        std::set<int> in_set(net.input_ids.begin(), net.input_ids.end());
        while (!q.empty()) {
            int cur = q.back(); q.pop_back();
            if (!in_set.count(cur) && nmap.count(cur))
                net.order.push_back(nmap[cur]);
            for (int nxt : adj[cur]) if (--indeg[nxt] == 0) q.push_back(nxt);
        }
        return net;
    }

    std::vector<float> activate(const std::vector<float>& inputs) const {
        std::unordered_map<int,float> vals;
        for (int i = 0; i < (int)input_ids.size(); i++)
            vals[input_ids[i]] = i < (int)inputs.size() ? inputs[i] : 0.f;

        for (auto& node : order) {
            float s = node.bias;
            for (auto& [fid, w] : node.inputs) {
                auto it = vals.find(fid);
                if (it != vals.end()) s += w * it->second;
            }
            vals[node.id] = neat::activate(s);
        }

        std::vector<float> out;
        for (int id : output_ids)
            out.push_back(vals.count(id) ? vals[id] : 0.f);
        return out;
    }
};

// ─── Config ──────────────────────────────────────────────────────────────────
struct Config {
    int      n_inputs             = 26;
    int      n_outputs            = 5;
    int      pop_size             = 150;
    float    compat_threshold     = 3.0f;
    float    c1                   = 1.0f;
    float    c2                   = 0.5f;
    int      max_stagnation       = 20;
    int      elitism              = 2;
    float    survival_threshold   = 0.2f;
    float    weight_mutate_rate   = 0.8f;
    float    weight_replace_rate  = 0.1f;
    float    weight_power         = 0.5f;
    float    add_node_prob        = 0.03f;
    float    add_conn_prob        = 0.05f;
    float    toggle_prob          = 0.01f;
    float    crossover_prob       = 0.75f;
    // Initial topology: FULL_DIRECT (default), UNCONNECTED, or PARTIAL_DIRECT
    InitConn init_conn            = InitConn::FULL_DIRECT;
    float    init_conn_frac       = 0.5f;   // fraction used for PARTIAL_DIRECT
    // Prune disabled connections after this many generations; 0 = disabled
    int      prune_stale_gens     = 20;
};

// ─── Species ─────────────────────────────────────────────────────────────────
struct Species {
    int            id;
    Genome         rep;
    std::vector<int> member_ids;
    float          best_fitness  = 0.f;
    float          shared_fit    = 0.f;
    int            stagnation    = 0;
};

// ─── Population ──────────────────────────────────────────────────────────────
struct Population {
    Config   cfg;
    InnovTracker innov;
    std::mt19937 rng;
    int generation       = 0;
    int next_genome_id   = 0;
    int next_species_id  = 0;

    std::vector<Genome>  genomes;
    std::vector<Species> species;

    explicit Population(Config cfg, uint32_t seed = 42) : cfg(cfg), rng(seed) {
        for (int i = 0; i < cfg.pop_size; i++)
            genomes.push_back(Genome::make_minimal(next_genome_id++,
                              cfg.n_inputs, cfg.n_outputs, innov, rng,
                              cfg.init_conn, cfg.init_conn_frac));
    }

    const Genome& best() const {
        return *std::max_element(genomes.begin(), genomes.end(),
            [](const Genome& a, const Genome& b){ return a.fitness < b.fitness; });
    }

    // Assign genomes to species
    void speciate() {
        for (auto& s : species) s.member_ids.clear();

        for (auto& g : genomes) {
            bool placed = false;
            for (auto& s : species) {
                if (g.distance(s.rep, cfg.c1, cfg.c2) < cfg.compat_threshold) {
                    s.member_ids.push_back(g.id);
                    placed = true;
                    break;
                }
            }
            if (!placed) {
                Species ns;
                ns.id  = next_species_id++;
                ns.rep = g;
                ns.member_ids.push_back(g.id);
                species.push_back(std::move(ns));
            }
        }

        // Drop empty species; update representatives
        std::erase_if(species, [](const Species& s){ return s.member_ids.empty(); });
        std::unordered_map<int, Genome*> gmap;
        for (auto& g : genomes) gmap[g.id] = &g;

        for (auto& s : species) {
            std::uniform_int_distribution<int> p(0, s.member_ids.size()-1);
            s.rep = *gmap.at(s.member_ids[p(rng)]);
        }
    }

    void compute_shared_fitness() {
        std::unordered_map<int, Genome*> gmap;
        for (auto& g : genomes) gmap[g.id] = &g;

        for (auto& s : species) {
            float max_fit = 0.f;
            s.shared_fit = 0.f;
            for (int gid : s.member_ids) {
                float f = gmap[gid]->fitness;
                s.shared_fit += f / s.member_ids.size();
                max_fit = std::max(max_fit, f);
            }
            if (max_fit > s.best_fitness) { s.best_fitness = max_fit; s.stagnation = 0; }
            else s.stagnation++;
        }
    }

    // Evolve one generation — call AFTER setting genome.fitness for all genomes
    void evolve() {
        speciate();
        compute_shared_fitness();

        std::unordered_map<int, Genome*> gmap;
        for (auto& g : genomes) gmap[g.id] = &g;

        // Total shared fitness (exclude stagnant species)
        float total = 0.f;
        for (auto& s : species)
            if (s.stagnation < cfg.max_stagnation) total += s.shared_fit;
        if (total <= 0.f) total = 1.f;

        std::vector<Genome> next;
        next.reserve(cfg.pop_size);

        std::uniform_real_distribution<float> u(0.f, 1.f);

        for (auto& s : species) {
            if (s.stagnation >= cfg.max_stagnation) continue;

            // Collect + sort members by fitness (desc)
            std::vector<Genome*> members;
            for (int gid : s.member_ids)
                if (gmap.count(gid)) members.push_back(gmap[gid]);
            std::sort(members.begin(), members.end(),
                [](Genome* a, Genome* b){ return a->fitness > b->fitness; });

            // Elite
            int elite = std::min(cfg.elitism, (int)members.size());
            for (int i = 0; i < elite; i++) {
                Genome e = *members[i]; e.id = next_genome_id++; e.fitness = 0.f;
                next.push_back(std::move(e));
            }

            // Offspring count (proportional to shared fitness)
            int n_off = std::max(0, (int)std::round(
                (s.shared_fit / total) * cfg.pop_size)) - elite;

            // Survival cut
            int survivors = std::max(1, (int)(members.size() * cfg.survival_threshold));
            members.resize(survivors);
            std::uniform_int_distribution<int> pick(0, members.size()-1);

            for (int i = 0; i < n_off; i++) {
                Genome child;
                if ((int)members.size() > 1 && u(rng) < cfg.crossover_prob) {
                    auto* p1 = members[pick(rng)];
                    auto* p2 = members[pick(rng)];
                    child = Genome::crossover(*p1, *p2, p1->fitness >= p2->fitness, rng);
                } else {
                    child = *members[pick(rng)];
                }
                child.id = next_genome_id++; child.fitness = 0.f;
                child.mutate_weights(cfg.weight_mutate_rate, cfg.weight_replace_rate,
                                     cfg.weight_power, rng);
                if (u(rng) < cfg.add_node_prob) child.mutate_add_node(innov, rng, generation);
                if (u(rng) < cfg.add_conn_prob) child.mutate_add_conn(innov, rng);
                if (u(rng) < cfg.toggle_prob)   child.mutate_toggle(rng, generation);
                next.push_back(std::move(child));
            }
        }

        // Fill remainder from entire population
        std::uniform_int_distribution<int> any(0, genomes.size()-1);
        while ((int)next.size() < cfg.pop_size) {
            Genome child = genomes[any(rng)]; child.id = next_genome_id++; child.fitness = 0.f;
            child.mutate_weights(cfg.weight_mutate_rate, cfg.weight_replace_rate,
                                 cfg.weight_power, rng);
            if (u(rng) < cfg.add_node_prob) child.mutate_add_node(innov, rng, generation);
            if (u(rng) < cfg.add_conn_prob) child.mutate_add_conn(innov, rng);
            next.push_back(std::move(child));
        }

        // Prune connections that have been disabled for too long, then orphaned hidden nodes
        if (cfg.prune_stale_gens > 0) {
            int next_gen = generation + 1;
            for (auto& g : next) g.prune_stale(next_gen, cfg.prune_stale_gens);
        }

        genomes = std::move(next);
        generation++;
    }
};

} // namespace neat
