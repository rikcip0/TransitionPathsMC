// ising_barrier.cpp
// g++ -O2 -std=c++17 ising_barrier.cpp -o ising_barrier
//
// Uso:
//   ./ising_barrier <graph_type> <c> <beta> [seed]
//
//           graph_type = 0  →  Random-Regular  (grado fisso = c)
//                        1  →  Erdős-Rényi     (grado ~ Poisson(c))
//
//                      c = grado medio (o fisso)  es. 3
//                  beta = 1/T                     es. 1
//                 seed  = (opzionale) intero per RNG

#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <cstdlib>

struct Params {
    int  graph_type;   // 0 = RRG  ,  1 = ER
    int  c;            // connettività media (o fissa)
    double beta;       // β = 1/T
};

// ------------------------------ clamp helpers ------------------------------
static inline double clamp(double x, double lo, double hi) {
    return (x < lo) ? lo : (x > hi) ? hi : x;
}

// ------------------------------ population dynamics ------------------------
void population_dynamics(std::vector<double>& h_pop,
                         const Params& p,
                         std::mt19937& gen)
{
    std::uniform_int_distribution<> idx_dist(0, (int)h_pop.size() - 1);
    std::poisson_distribution<>     deg_dist(p.c - 1);           // per ER
    const double J = 1.0;                                        // Ising J=1

    // bias iniziale piccolo per rompere la simmetria
    std::fill(h_pop.begin(), h_pop.end(), 0.1);

    const int iters = 1'500'000;                                 // PD steps
    for (int iter = 0; iter < iters; ++iter) {
        int i = idx_dist(gen);

        // numero di vicini per questo update
        int k = (p.graph_type == 0) ? (p.c - 1)                   // RRG
                                    : deg_dist(gen);             // ER

        if (k == 0) { h_pop[i] = 0.0; continue; }

        double sum_u = 0.0;
        for (int r = 0; r < k; ++r) {
            int j = idx_dist(gen);
            double tanh_term = std::tanh(p.beta * h_pop[j]);
            double arg = std::tanh(p.beta * J) * tanh_term;
            arg = clamp(arg, -1.0 + 1e-12, 1.0 - 1e-12);
            double u_k = std::atanh(arg) / p.beta;
            sum_u += u_k;
        }
        h_pop[i] = sum_u;
    }
}

// ------------------------------ free energy --------------------------------
double free_energy_ferro(const std::vector<double>& h_pop,
                         const Params& p,
                         std::mt19937& gen)
{
    const double beta = p.beta;
    const int    N    = (int)h_pop.size();
    const int    c    = p.c;

    std::uniform_int_distribution<> idx_dist(0, N - 1);
    std::poisson_distribution<>     deg_dist(p.c);               // per ER

    // ---- <ln Z_i> ----
    const int node_samples = 1'000'000;
    double sum_lnZi = 0.0;
    for (int s = 0; s < node_samples; ++s) {
        int k = (p.graph_type == 0) ? c : deg_dist(gen);         // k = grado
        double H_tot = 0.0;
        for (int m = 0; m < k; ++m) {
            int idx = idx_dist(gen);
            H_tot += h_pop[idx];
        }
        double arg = clamp(beta * H_tot, -50.0, 50.0);
        sum_lnZi += std::log( 2.0 * std::cosh(arg) );
    }
    double avg_lnZi = sum_lnZi / node_samples;

    // ---- <ln Z_ij> ----
    const int edge_samples = 1'000'000;
    double sum_lnZij = 0.0;
    for (int s = 0; s < edge_samples; ++s) {
        int a = idx_dist(gen);
        int b = idx_dist(gen);
        double u1 = h_pop[a];
        double u2 = h_pop[b];

        double arg_sum  = clamp(beta * (u1 + u2), -50.0, 50.0);
        double arg_diff = clamp(beta * (u1 - u2), -50.0, 50.0);

        double term_plus  = std::exp(beta)  * std::cosh(arg_sum);
        double term_minus = std::exp(-beta) * std::cosh(arg_diff);
        sum_lnZij += std::log( 2.0 * (term_plus + term_minus) );
    }
    double avg_lnZij = sum_lnZij / edge_samples;

    // F/N = (1/β) [ <ln Z_i> - (c/2) <ln Z_ij> ]
    return (1.0 / beta) * (avg_lnZi - 0.5 * c * avg_lnZij);
}

// ------------------------------ Fpara closed form --------------------------
double free_energy_para(const Params& p)
{
    const double beta = p.beta;
    const int    c    = p.c;

    /*  Vale sia per RRG sia per ER:
          F_para/N = -(1/β)·ln2  - (c/2β)·ln cosh(β)
       perché nei due casi:
          ln Z_i      = ln 2
          ln Z_ij     = ln 2 + ln cosh β
          F/N         = (1/β)( Σ_e ln Z_ij − Σ_i ln Z_i )
                       = (1/β)( Nc/2 ln Z_ij − N ln2 )
                       = −ln2/β − c/2β ln cosh β               */
    return -std::log(2.0) / beta
           -0.5 * c * std::log( std::cosh(beta) ) / beta;
}


// ------------------------------ magnetization ------------------------------
double magnetization(const std::vector<double>& h_pop,
                     const Params& p,
                     std::mt19937& gen)
{
    const int    c    = p.c;
    const double beta = p.beta;
    std::uniform_int_distribution<> idx_dist(0, (int)h_pop.size() - 1);
    std::poisson_distribution<>     deg_dist(p.c);

    const int samples = 300'000;
    double acc = 0.0;
    for (int s = 0; s < samples; ++s) {
        int k = (p.graph_type == 0) ? c : deg_dist(gen);
        if (k == 0) { acc += 0.0; continue; }
        double H_tot = 0.0;
        for (int m = 0; m < k; ++m) {
            int idx = idx_dist(gen);
            H_tot += h_pop[idx];
        }
        double arg = clamp(beta * H_tot, -50.0, 50.0);
        acc += std::tanh(arg);
    }
    return acc / samples;
}

// ==========================================================================
//                                    main
// ==========================================================================
int main(int argc, char* argv[])
{
    if (argc < 4 || argc > 5) {
        std::cerr << "Usage: " << argv[0]
                  << " <graph_type:0=RRG,1=ER> <c> <beta> [seed]\n";
        return 1;
    }

    Params p;
    p.graph_type = std::stoi(argv[1]);
    p.c          = std::stoi(argv[2]);
    p.beta       = std::stod(argv[3]);

    unsigned seed = (argc == 5) ? std::stoul(argv[4])
                                : std::random_device{}();
    std::mt19937 gen(seed);

    // popolazione di cavity-field
    const int POP_SIZE = 50'000;
    std::vector<double> h_pop(POP_SIZE);

    population_dynamics(h_pop, p, gen);

    double F_para  = free_energy_para(p);
    double F_ferro = free_energy_ferro(h_pop, p, gen);
    double dF      = F_para - F_ferro;
    double m       = magnetization(h_pop, p, gen);

    std::cout << (p.graph_type == 0 ? "RRG" : "ER") << "  (c = "
              << p.c << ", beta = " << p.beta << ")\n";
    std::cout << "Paramagnetic free energy (m=0): " << F_para  << '\n';
    std::cout << "Ferromagnetic free energy (m!=0): " << F_ferro << '\n';
    std::cout << "Free energy barrier per spin: " << dF << '\n';
    std::cout << "Average magnetization: " << m  << '\n';
}
