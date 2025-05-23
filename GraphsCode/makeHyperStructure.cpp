#include <iostream>
#include <fstream>
#include "../Generic/fileSystemUtil.h"
#include "../Generic/random.h"
#include <cmath>
#include <vector>
#include <algorithm>

using namespace std;

unsigned long long binomial_coefficient(int n, int k) {
    if (k > n) return 0;  // Se k > n, il coefficiente è 0
    if (k == 0 || k == n) return 1;  // Se k == 0 o k == n, il coefficiente è 1
    
    // Calcolare il coefficiente binomiale in modo efficiente
    if (k > n - k)  // Usa la simmetria: nCk = nC(n-k)
        k = n - k;
    
    unsigned long long result = 1;
    for (int i = 1; i <= k; ++i) {
        result *= n--;
        result /= i;
    }
    
    return result;
}

int main(int argc, char **argv)
{

int type = atoi(argv[1]);
if (argc != 3 && argc != 5 && type != -4)
{
    cout << "Usage: type(-3: doublePlantedRRG -2:ER -1:RRG 1:1dChain k(>1):k-dim SqLatt) N (C) (p)" << endl;
    exit(1);
}
if (argc <4 && type == -4)
{
    cout << "Usage: type(-4: ErdosRenyFixedPMax) N C2 C3 ... CPmax" << endl;
    exit(1);
}

int seed = init_random(0, 0);
int N = atoi(argv[2]);
int C = 0;
int p = 2; // Imposta un valore di default per p
int pmax = 2;
vector<int> CC;
vector<int> deg(N, 0);
vector<int> NumInterac(pmax-1,0);

if (argc == 5 && type != -4) {
  C = atoi(argv[3]);
  if (C >= N)
    error("C must be smaller than N");
  p = atoi(argv[4]); 
  if (p <= 1)
    error("p must be greater than 1");
  if (p > N)
    error("C must be smaller or equal than N");
}
if (type == -4) {
    pmax = argc - 2;
    for (int i = 3; i < argc; ++i) {
        CC.push_back(atoi(argv[i]));
    }

}

char simulationType[40];
string folder;
string nomeFile;

if (type == -1) {
    sprintf(simulationType, "p%iC%i", p, C);
    folder = "../Data/Graphs/RRG/" + string(simulationType);
} else if (type == -2) {
    sprintf(simulationType, "p%iC%i", p, C);
    folder = "../Data/Graphs/ER/" + string(simulationType);
} else if (type == -3) {
    sprintf(simulationType, "p%iC%i", p, C);
    folder = "../Data/Graphs/DPRRG/" + string(simulationType);
} else if (type == -4) {
    string simulationType = "";
    for (int i = 0; i < pmax-1; ++i) {
        char buffer[70];
        sprintf(buffer, "C%i_%i", i+2, CC[i]);
        simulationType += string(buffer);
    }
    folder = "../Data/Graphs/ERPMAX/" + simulationType;
} else if (type == 1) {
    folder = "../Data/Graphs/1dChain";
} else {
    folder = "../Data/Graphs/SqLatt/" + to_string(type) + "d";
}

folder += "/N" + to_string(N);
if (type < 0)
    folder += "/structure" + to_string(seed % 1000000);

vector<int> list;

  if (type == -3)
  {
    int ind[C], v[C], i, site, q, graphLen, t, listLen;
    int newSite, flag, sum, prod, max, indMax;

    int nPatt = 3;
    double factor = 0.05;
    vector<int> s[nPatt];

    for (int i = 0; i < nPatt; i++)
    {
      s[i].assign(N, 0);
    }

    printf(".");
    // generate 2 orthogonal configurations
    s[0].assign(N,1 );
    for (int i = 1; i < nPatt; i++)
    {
      for (int j = 0; j < N; j++)
      {
        s[i][j] = Xrandom() > 0.5 ? 1 : -1;
      }
    }

    double prob = (double)C / (double)(N - 1);
    double thisProb;

    for (int i = 0; i < N; i++)
    {
      for (int j = 0; j < i; j++)
      {
        thisProb = prob;
        for (int k = 0; k < nPatt; k++)
        {
          for(int l = 0; l < k; l++)
            {
                if (s[k][i]*s[k][j] != s[l][i]*s[l][j])
                {
                  thisProb *= factor;
                }
            }
        }

        cout << "prob è" << thisProb << endl;
        if (prob == thisProb)
        {
          cout << "E UGUALEEEEEEEEE" << endl;
        }
        if (Xrandom() < thisProb)
        {
          list.push_back(i);
          list.push_back(j);
        }
      }
    }
  }
  else if (type == -2)//da rifare ancora
  {

    double prob = C / (double)(N - 1);
    int site = 0;

    for (int i = 0; i < N; i++)
      for (int j = 0; j < i; j++)

        if (Xrandom() < prob)
        {
          list.push_back(i);
          list.push_back(j);
          deg[i]++;
          deg[j]++;
        }

    // check self-links
    for (int i = 0; i < list.size() / 2; i++)
      if (list[2 * i] == list[2 * i + 1])
      {
        cout << i << " " << list[2 * i] << " " << list[2 * i + 1] << endl;
        error("self-loop");
      }
  }
  else if (type == -4)
  { 
    for (int pp = 2; pp < pmax+1; pp++){

      double prob = CC[pp-2] / (double)(binomial_coefficient(N-1, pp-1));
      vector<int> comb(pp);
      for (int i = 0; i < pp; ++i) comb[i] = i; // inizializza con 0, 1, ..., p-1

      while (true) {
        if (Xrandom() < prob)
            {
                for (int i = 0; i < pp; ++i) {
                    list.push_back(comb[i]);
                }       
                NumInterac[pp-2]++;
            }

        // trova la prossima combinazione
        int i = pp - 1;
        while (i >= 0 && comb[i] == N - pp + i) {
            --i;
        }

        if (i < 0) break; // tutte le combinazioni esaurite

        ++comb[i]; // se non esce va a quella dopo
        for (int j = i + 1; j < pp; ++j) {
            comb[j] = comb[j - 1] + 1;
        }
      }
    }

  }
  else if (type == -1)
  {
    if (N * C % p)
      error("N*C/p is not integer");
    int n_int, i, j, k, kk, counts, a, b, flag, tmp;
    //site = 0;
    
    n_int = (C * N) / p; // number of interactions

    // fill list
    for (i = 0; i < N; i++)
      for (j = 0; j < C; j++)
        list.push_back(i); // so I'll have list = [000 111 222 333 ... NNN] if C=3

    // randomize list
    for (i = 0; i < C * N; i++)
    {
      j = (int)(Xrandom() * C * N);
      tmp = list[i];
      list[i] = list[j];
      list[j] = tmp;
    }
    // remove self-links
    do
    {
      flag = 0;
      for (i = 0; i < n_int; i++) // per ogni p-pletta
      {
        for (a = 0; a < p - 1; a++)
        {
          for (b = a + 1; b < p; b++)
          {
            if (list[p * i + a] == list[p * i + b]) // duplicato!
            {
              // scegli posizione globale casuale j ≠ p*i + b
              do
              {
                j = (int)(Xrandom() * (n_int * p));
                counts = 0;
                for (kk = 0; kk < p; kk++)
                  if (list[p*(j/p)+kk]==list[p*i+b] || list[p*i+kk]==list[j]){
                    counts++;
                  }
              } while (counts > 0);  // evita swap inutili

              // swap tra elemento duplicato e uno casuale
              tmp = list[p * i + b];
              list[p * i + b] = list[j];
              list[j] = tmp;

              flag = 1;
              a = -1; // restart check for this p-pletta
              break; // esci dal ciclo interno (b)
            }
          }
          if (a == -1) break; // se abbiamo riavviato il controllo
        }
      }
    } while (flag);

  }
  else
  {
    int d = type;
    int L = (int)round(pow(N, 1.0 / d));

    if ((int)pow(L, d) != N)
    {
      cout << "N=" << N << " value is not allowed for d= " << d << " square lattice!" << endl;
      exit(1);
    }

    for (int i = 0; i < N; i++)
    {
      int x = i;
      vector<int> coords(d);
      for (int j = 0; j < d; j++)
      {
        coords[j] = x % L;
        x /= L;
      }

      for (int sign = 1; sign >= -1; sign -= 2)
      {

        for (int j = 0; j < d; j++)
        {
          vector<int> neighborCords = coords;
          neighborCords[j] += sign;
          if (neighborCords[j] == L)
            neighborCords[j] = 0;
          if (neighborCords[j] == -1)
            neighborCords[j] = L - 1;

          int neighbor = 0;
          for (int k = 0; k < d; k++)
            neighbor += pow(L, k) * neighborCords[k];

          if (neighbor > i)
          {
            list.push_back(i);
            list.push_back(neighbor);
          }
        }
      }
    }
  }
  createFolder(folder);
  // Open a file for writing
  nomeFile = folder + "/structure.txt";
  std::ofstream outputFile(nomeFile);

  // Check if the file is open
  if (!outputFile.is_open())
  {
    std::cerr << "Error opening the file " << nomeFile << " for writing!" << std::endl;
    return 1;
  }

  if (type < 0 && type != -4)
  {
    outputFile << N << " " << p << " " << C << " " << seed << "\n";
  }
  if (type == -4) {
    outputFile << N;
    for (int i = 0; i < pmax - 1; ++i) {
        outputFile << " " << CC[i];
    }
    outputFile << " " << seed << "\n";
}
  // Write the array to the file
  for (int i = 0; i < list.size(); i++)
  {
    outputFile << list[i] << " ";
  }

  // Close the file
  outputFile.close();

  std::cout << "Array has been written to file " << nomeFile << std::endl;

  if (type == -2)
  {

    nomeFile = folder + "/structure_degreeDistribution.txt";
    outputFile.open(nomeFile);

    // Check if the file is open
    if (!outputFile.is_open())
    {
      std::cerr << "Error opening the file " << nomeFile << " for writing!" << std::endl;
      return 1;
    }

    for (int i = 0; i < N; i++)
    {
      outputFile << deg[i] << endl;
    }

    // Close the file
    outputFile.close();
  }
  if (type == -4)
  {

    nomeFile = folder + "/structure_Num_Interactions.txt";
    outputFile.open(nomeFile);

    // Check if the file is open
    if (!outputFile.is_open())
    {
      std::cerr << "Error opening the file " << nomeFile << " for writing!" << std::endl;
      return 1;
    }

    for (int i = 0; i < pmax-1; i++)
    {
      outputFile << NumInterac[i] << endl;
    }

    // Close the file
    outputFile.close();
  }
  return 0;
}
