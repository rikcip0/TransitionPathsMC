#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <cassert>
#include <algorithm>

/*
  ===========================================================
  Ising Ferromagnet on an Erdős-Rényi graph (average degree c)
  ===========================================================
  
  We implement the Bethe (cavity) approximation using a population
  of cavity fields ("messages"). Then, we:
  
    1) Impose a target magnetization m_target via a Lagrange multiplier mu
       (found through a bisection procedure).
    2) Stabilize the population of fields for that mu.
    3) Measure the resulting magnetization and compute the Bethe free energy:
         f = - (1 / beta) * [ < ln(2 cosh( beta * H )) >  -  (c/2) * < ln Z_ij > ]

  where:
    - H = mu + sum_{neighbors} of their cavity fields,
    - Z_ij = sum_{s_i, s_j = ±1} exp( beta [ s_i u_{i->j} + s_j u_{j->i} + J s_i s_j ] ).

  Physics:
    - J > 0 => Ferromagnet
    - c=3, J=1 => T_c ~ 1.82 (approx)
      => T < 1.82 => ferromagnetic stable solution (|m| ~ 1),
         T > 1.82 => paramagnetic solution (m ~ 0).
  
  If everything is correct:
    * for T=1 < T_c => free energy is minimal around ±1,
    * for T=10 >> T_c => free energy is minimal around 0.
  
  Check that the sign is correct. If you see paramagnet stable at T=1,
  something is wrong in the sign or factor. If you see ferromagnet stable at T=10,
  again there's a bug in the formula or in how beta is used.

  NOTE on performance:
    - We have "Npop" cavity fields in a vector, updated randomly
      according to a Poisson(c) distribution for node degrees.
    - The run might be quite fast if Npop or N_ITER are not large.
      Increase them for more stable results.
*/

// ----------------- Global or adjustable parameters ---------------- //

// Interactions
static const double J = 1.0;    // Ferromagnetic coupling

// We'll set "c" and "T" in main, for clarity.

// Population size
static const int Npop = 30000;   // number of cavity fields
// Iterations to stabilize
static const int N_ITER = 100000;
// Number of sampling steps for measuring magnetization
static const int N_SAMP_M = 20000;
// Number of sampling steps for measuring free energy
static const int N_SAMP_F = 20000;

// Bisection on mu
static const double MU_MIN = -10.0;
static const double MU_MAX =  10.0;
static const int    MAX_BISEC_ITER = 30;
static const double TOL_M = 1e-4;  // tolerance on magnetization

// ------------------------ Helper inline functions ------------------------ //

// Clip for atanh to avoid numerical blowup near ±1
inline double atanh_clip(double x) {
    const double eps = 1e-14;
    if(x >  1.0 - eps) x =  1.0 - eps;
    if(x < -1.0 + eps) x = -1.0 + eps;
    return 0.5 * std::log( (1.0 + x) / (1.0 - x) );
}

// The standard cavity update for the Ising model on a random graph:
//   u_new = atanh[ tanh(beta * J) * tanh(H) ]
// where H = mu + sum_of_other_cavity_fields
inline double cavity_update(double H, double beta) {
    double x = std::tanh(beta * J) * std::tanh(H);
    return atanh_clip(x);
}

// We will sample node degrees from a Poisson(c) distribution
int sample_poisson(double c, std::mt19937 &rng) {
    std::poisson_distribution<int> pd(c);
    return pd(rng);
}

// ----------------- Population update / stabilization ----------------- //

// We do many random updates to "stabilize" the population of fields
// under a given Lagrange multiplier mu (which enforces magnetization).
void stabilize_population(
    std::vector<double> &pop, 
    double mu,
    double beta,
    double c,
    std::mt19937 &rng
){
    std::uniform_int_distribution<int> idx_dist(0, (int)pop.size()-1);

    for(int step = 0; step < N_ITER; step++) {
        // pick a random node degree k from Poisson(c)
        int k = sample_poisson(c, rng);
        // sum up k random fields + mu
        double H = mu;
        for(int i = 0; i < k; i++){
            int idx = idx_dist(rng);
            H += pop[idx];
        }
        double u_new = cavity_update(H, beta);

        // Replace one random element in the population
        int replace_idx = idx_dist(rng);
        pop[replace_idx] = u_new;
    }
}

// ---------------- Measure the magnetization with sampling --------------- //

// We'll do a certain number of "virtual node" samplings. For each sample:
//   - extract a random degree k
//   - sum k random fields + mu => H
//   - the spin is s = tanh( beta * H )
// We accumulate and average s.
double measure_magnetization(
    std::vector<double> &pop,
    double mu,
    double beta,
    double c,
    std::mt19937 &rng
){
    std::uniform_int_distribution<int> idx_dist(0, (int)pop.size()-1);

    double accum = 0.0;
    for(int step = 0; step < N_SAMP_M; step++){
        //int k = sample_poisson(c, rng);
        int k = c;
        double H = mu;
        for(int i = 0; i < k; i++){
            int idx = idx_dist(rng);
            H += pop[idx];
        }
        // The "node spin" is s = tanh(beta * H)
        double s = std::tanh(beta * H);
        accum += s;

        // Optional: keep updating the population in order to remain coherent
        double u_new = cavity_update(H, beta);
        int replace_idx = idx_dist(rng);
        pop[replace_idx] = u_new;
    }
    return accum / double(N_SAMP_M);
}

// ------------------- link partition function -------------------- //
//  Z_{ij} = sum_{s_i, s_j = ±1} exp[ beta ( s_i u_i_j + s_j u_j_i + J s_i s_j ) ]
inline double link_partition_function(double u_i_j, double u_j_i, double beta){
    double z = 0.0;
    for(int s_i = -1; s_i <= 1; s_i+=2){
        for(int s_j = -1; s_j <= 1; s_j+=2){
            // Because H = -J s_i s_j in standard Ising,
            //   => Boltzmann factor = exp(+ beta * J s_i s_j ) (since H is negative)
            // plus the cavity contributions s_i u_i_j + s_j u_j_i.
            double e = beta * ( s_i * u_i_j + s_j * u_j_i + J * s_i * s_j );
            z += std::exp(e);
        }
    }
    return z;
}

// ----------------- measure the Bethe free energy ------------------ //
// In the standard form:
//    f = - (1/beta) [ <ln(2 cosh( beta * H ))> - (c/2)* <ln Z_{ij}> ]
// where
//   H = mu + sum of incoming fields
//   Z_{ij} = sum_{s_i,s_j} exp( beta [ s_i u_{i->j} + s_j u_{j->i} + J s_i s_j ] )
double measure_free_energy(
    std::vector<double> &pop,
    double mu,
    double beta,
    double c,
    std::mt19937 &rng
){
    std::uniform_int_distribution<int> idx_dist(0, (int)pop.size()-1);

    // 1) < ln(2 cosh( beta * H )) >
    double accum_lnZi = 0.0;
    for(int step = 0; step < N_SAMP_F; step++){
        //int k = sample_poisson(c, rng);
        int k = c;
        double H = mu;
        for(int i = 0; i < k; i++){
            int idx = idx_dist(rng);
            H += pop[idx];
        }
        // contribution for a single node
        double val = std::log( 2.0 * std::cosh(beta * H) );
        accum_lnZi += val;
    }
    double avg_lnZi = accum_lnZi / double(N_SAMP_F);

    // 2) < ln Z_{ij} >
    double accum_lnZij = 0.0;
    for(int step = 0; step < N_SAMP_F; step++){
        // pick two random fields to mimic a link i->j, j->i
        int idx1 = idx_dist(rng);
        int idx2 = idx_dist(rng);
        double u1 = pop[idx1];
        double u2 = pop[idx2];

        double z = link_partition_function(u1, u2, beta);
        accum_lnZij += std::log(z);
    }
    double avg_lnZij = accum_lnZij / double(N_SAMP_F);

    // Final formula
    double f = - (1.0 / beta) * ( avg_lnZi - 0.5 * c * avg_lnZij );
    return f;
}

// --------------------- Bisection on mu to fix m --------------------- //
// We want to find mu such that the measured magnetization ~ m_target.
double find_mu_for_magnetization(
    double m_target,
    std::vector<double> &pop,
    double beta,
    double c,
    std::mt19937 &rng
){
    // We'll backup the initial population to restore it each time we test
    std::vector<double> pop_backup = pop;

    double left = MU_MIN;
    double right= MU_MAX;
    double mu_found = 0.5*(left + right);

    // Evaluate magnetization at left
    pop = pop_backup;
    stabilize_population(pop, left, beta, c, rng);
    double m_left = measure_magnetization(pop, left, beta, c, rng);

    // Evaluate magnetization at right
    pop = pop_backup;
    stabilize_population(pop, right, beta, c, rng);
    double m_right= measure_magnetization(pop, right, beta, c, rng);

    // We do a simple bisection
    for(int iter=0; iter<MAX_BISEC_ITER; iter++){
        double mid = 0.5*(left + right);

        pop = pop_backup;
        stabilize_population(pop, mid, beta, c, rng);
        double m_mid = measure_magnetization(pop, mid, beta, c, rng);

        if(std::fabs(m_mid - m_target) < TOL_M){
            mu_found = mid;
            break;
        }
        if(m_mid < m_target){
            left = mid;
            m_left = m_mid;
        } else {
            right= mid;
            m_right= m_mid;
        }
        mu_found = 0.5*(left + right);
    }

    // Finally, re-stabilize with the found mu
    pop = pop_backup;
    stabilize_population(pop, mu_found, beta, c, rng);

    return mu_found;
}

// --------------------- MAIN --------------------- //
int main(){

    // =========== You can modify these ============= //
    double c = 3.0;   // average connectivity
    double T = 1.0;   // temperature (try T=1 < Tc, or T=10 >> Tc)
    // ============================================== //

    double beta = 1.0 / T;

    // Random generator
    std::random_device rd;
    std::mt19937 rng(rd());

    // Prepare the population with small random fields
    std::uniform_real_distribution<double> init_dist(-0.01, 0.01);
    std::vector<double> population(Npop);
    for(int i=0; i<Npop; i++){
        population[i] = init_dist(rng);
    }

    // We'll do a simple loop over m_target from -1 to +1
    double m_min = -1.0;
    double m_max =  1.0;
    double dm = 0.2;

    // Print header
    std::cout << "# c=" << c << ", T=" << T 
              << ", (Tc~1.82 if c=3, J=1)\n";
    std::cout << "# m_target, f(m), mu_found, m_measured\n";

    for(double m_target = m_min; m_target <= m_max+1e-9; m_target += dm){

        // 1) Bisection to find mu s.t. magnetization ~ m_target
        double mu_found = find_mu_for_magnetization(
            m_target, population, beta, c, rng
        );

        // 2) measure the actual magnetization
        double m_meas = measure_magnetization(
            population, mu_found, beta, c, rng
        );

        // 3) measure the Bethe free energy
        double f_val = measure_free_energy(
            population, mu_found, beta, c, rng
        );

        // print one line per m_target
        std::cout << m_target << "  " 
                  << f_val << "  "
                  << mu_found << "  "
                  << m_meas << "\n";
    }

    return 0;
}
