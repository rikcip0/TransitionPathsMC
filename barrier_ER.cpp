/*  barrier_multiplets.cpp  –  ER (C = 3)  •  RS Bethe–Peierls con sampling su multiplette
    -------------------------------------------------------------------------------
    Warm‑up:   2000·NPOOL update
    Sampling:  SAMPLE valid update‐multiplets → accumulo dei termini f_site,f_edge,m
    Output per β:
      β   f_PM     f_FM     m_FM   β·Δf
*/

#include <bits/stdc++.h>
using Vec = std::vector<double>;
std::mt19937_64 rng(123456);

// —————————————————————————————————————————————————
//  Parametri
// —————————————————————————————————————————————————
constexpr double C          = 3.0;
constexpr size_t NPOOL      = 100'000;
constexpr size_t WARMUP     = 2000 * NPOOL;
constexpr size_t SAMPLE     = 1'00'000;
constexpr double BIAS       = 1e-6;    // bias micro FM
constexpr double EPS_ISO    = 1e-6;    // rumore nodi isolati

inline double sgn() { return std::bernoulli_distribution(0.5)(rng) ? 1.0 : -1.0; }

// —————————————————————————————————————————————————
//  Esegue prima WARMUP update, poi SAMPLE aggiornamenti
//  e accumula i termini di free‐energy su ciascuna multiplet.
//  Restituisce f, m.
// —————————————————————————————————————————————————
struct Stat { double f, m; };

Stat run_and_sample(double beta, bool ferro) {
    std::poisson_distribution<int> pois(C);
    std::uniform_int_distribution<size_t> ui(0, NPOOL-1);
    double tβ = std::tanh(beta);

    // Inizializzo il pool
    Vec pool(NPOOL);
    std::uniform_real_distribution<double> noise(-1e-3, 1e-3);
    for (double &u : pool)
        u = ferro ? (BIAS) : (noise(rng)*sgn());

    // Warm‑up
    for (size_t t = 0; t < WARMUP; ++t) {
        int k_full = pois(rng);
        size_t idx = ui(rng);
        if (k_full == 0) {
            pool[idx] = ferro ? BIAS : EPS_ISO*sgn();
        } else {
            int k = k_full - 1;
            double arg = 0;
            for (int r = 0; r < k; ++r) {
                double v = pool[ ui(rng) ];
                if (!ferro) v *= sgn();
                arg += std::atanh(std::clamp(v, -1.0+1e-12, 1.0-1e-12));
            }
            double u_new = std::tanh(arg);
            pool[idx] = ferro ? u_new : u_new*sgn();
        }
    }

    // Sampling sui multiplette
    double sum_edge = 0, sum_site = 0, sum_m = 0;
    size_t count = 0;
    while (count < SAMPLE) {
        int k_full = pois(rng);
        size_t idx = ui(rng);

        // prelevo la stessa multipletto per update e per free‐energy
        if (k_full == 0) {
            // update isolato
            pool[idx] = ferro ? BIAS : EPS_ISO*sgn();
            continue;
        }
        int k = k_full - 1;
        std::vector<double> us;
        us.reserve(k);
        double arg = 0;
        for (int r = 0; r < k; ++r) {
            double v = pool[ ui(rng) ];
            if (!ferro) v *= sgn();
            v = std::clamp(v, -1.0+1e-12, 1.0-1e-12);
            arg += std::atanh(v);
            us.push_back(v);
        }
        double u_new = std::tanh(arg);
        pool[idx] = ferro ? u_new : u_new*sgn();

        // accumulo solo se ho almeno 2 messaggi
        if (k >= 2) {
            // — edge term —
            double u = us[0], v = us[1];
            sum_edge += std::log(std::cosh(beta))
                      + std::log1p(tβ * u * v);

            // — site term (H=0) —
            double prod_p = 1, prod_m = 1;
            for (double uj : us) {
                prod_p *= (1 + tβ * uj);
                prod_m *= (1 - tβ * uj);
            }
            double Zp = prod_p, Zm = prod_m;
            sum_site += std::log(Zp + Zm);
            sum_m    += (Zp - Zm) / (Zp + Zm);

            ++count;
        }
    }

    double lnZ_edge = sum_edge / SAMPLE;
    double lnZ_site = sum_site / SAMPLE;
    double f  = -(lnZ_site - C*lnZ_edge/2.0)/beta;
    double m  =  sum_m  / SAMPLE;
    return {f, m};
}

int main(){
    std::cout << "# beta   f_PM      f_FM      m_FM     beta*Δf\n";
    for(double beta = 0.45; beta <= 1.00 + 1e-12; beta += 0.05) {
        auto statPM = run_and_sample(beta, /*ferro=*/false);
        auto statFM = run_and_sample(beta, /*ferro=*/true );
        double delta = beta*(statPM.f - statFM.f);
        std::cout << std::fixed<<std::setprecision(3)
                  << beta << "   "
                  << statPM.f << "   "
                  << statFM.f << "   "
                  << std::abs(statFM.m) << "   "
                  << delta  << "\n";
    }
    return 0;
}
