#include <iostream>
#include <fstream>
#include "../MCtrajs/MCdyn_classi/Generic/fileSystemUtil.h"
#include "../Generic/random.h"

#define p 2 // as of now, the dynamics generation (and hence the code) only works for p=2 (RC)

using namespace std;

int main(int argc, char **argv)
{
  if (argc != 3 && argc != 4)
  {
    cout << "Usage: type(-2:ER -1:RRG 1:1dChain k(>1):k-dim SqLatt) N (C)" << endl;
    exit(1);
  }

  int seed = init_random(0, 0);
  int type = atoi(argv[1]);
  int N = atoi(argv[2]);
  int C = 0;
  if (argc == 4)
  {
    C = atoi(argv[3]);
    if (C >= N)
      error("C must be smaller than N");
  }
  char simulationType[40];
  string folder;
  string nomeFile;

  if (type == -1)
  {
    sprintf(simulationType, "p%iC%i", p, C);
    folder = "../Data/Graphs/RRG/" + string(simulationType);
  }
  else if (type == -2)
  {
    sprintf(simulationType, "p%iC%i", p, C);
    folder = "../Data/Graphs/ER/" + string(simulationType);
  }
  else if (type == 1)
  {
    folder = "../Data/Graphs/1dChain";
  }
  else
  {
    folder = "../Data/Graphs/SqLatt/" + to_string(type) + "d";
  }
  
  folder += "/N" + to_string(N);
  if (type < 0)
    folder += "/structure" + to_string(seed % 1000000);

  vector<int> list;
  vector<int> deg(N, 0);
  if (type == -2)
  {
    double prob = C / (double)(N - 1);
    int **neigh, site = 0;

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
    for (int i = 0; i < list.size() / 2; i += 2)
      if (list[2 * i] == list[2 * i + 1])
      {
        cout << i << " " << list[2 * i] << " " << list[2 * i + 1] << endl;
        error("self-loop");
      }
  }
  else if (type == -1)
  {
    if (N * C % 2)
      error("RRG with both C and N odd do not exist!");
    int n_int, i, j, k, l, flag, tmp, i1, i2, e1, e2;
    int *J, **neigh, site = 0;
    int **edge;

    neigh = (int **)calloc(N, sizeof(int *));         // one array for each spin
    neigh[0] = (int *)calloc(N * p * C, sizeof(int)); // for each spin there is a
    for (int i = 1; i < N; i++)
      neigh[i] = neigh[0] + i * p * C;

    n_int = (C * N) / p; // number of interactions

    edge = (int **)calloc(N, sizeof(int *));
    edge[0] = (int *)calloc(C * N, sizeof(int));
    for (i = 1; i < N; i++)
      edge[i] = edge[i - 1] + C;
    J = (int *)calloc(C * N, sizeof(int));

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
      for (i = 0; i < n_int; i++)
        if (list[2 * i] == list[2 * i + 1])
        {
          j = (int)(Xrandom() * C * N);
          tmp = list[2 * i];
          list[2 * i] = list[j];
          list[j] = tmp;
          flag = 1;
        }
    } while (flag);

    // fill edge
    for (i = 0; i < n_int; i++)
    {
      i1 = list[2 * i];
      i2 = list[2 * i + 1];
      edge[i1][deg[i1]++] = i;
      edge[i2][deg[i2]++] = i;
    }

    // check deg
    for (i = 0; i < N; i++)
      if (deg[i] != C)
        error("in deg");
    // check self-links
    for (i = 0; i < n_int; i++)
      if (list[2 * i] == list[2 * i + 1])
        error("self-loop");

    // remove double-links
    do
    {
      flag = 0;
      for (i = 0; i < N; i++)
      {
        for (j = 0; j < C - 1; j++)
          for (k = j + 1; k < C; k++)
          {
            e1 = edge[i][j];
            e2 = edge[i][k];
            if ((list[2 * e1] == list[2 * e2] && list[2 * e1 + 1] == list[2 * e2 + 1]) ||
                (list[2 * e1] == list[2 * e2 + 1] && list[2 * e1 + 1] == list[2 * e2])) // i.e. if e1 and e2 involve the same couple of spins
            {
              do
              {
                e2 = (int)(Xrandom() * n_int);
              } while (list[2 * e1] == list[2 * e2] ||
                       list[2 * e1] == list[2 * e2 + 1] ||
                       list[2 * e1 + 1] == list[2 * e2] ||
                       list[2 * e1 + 1] == list[2 * e2 + 1]); // this means that interaction e2 must not involve spins in interaction e1 (RC)
              i1 = list[2 * e1];
              i2 = list[2 * e2];
              list[2 * e1] = i2;
              list[2 * e2] = i1;
              l = 0;
              while (edge[i1][l] != e1)
                l++;
              if (l >= C)
                error("primo l");
              edge[i1][l] = e2;
              l = 0;
              while (edge[i2][l] != e2)
                l++;
              if (l >= C)
                error("secondo l");
              edge[i2][l] = e1;
              flag = 1;
            }
          }
      }
    } while (flag);
  }
  else
  {
    int L = (int)(pow(N, 1. / type) + 0.001);
    if (fabs(pow(N, 1. / type) - ((double)L)) > 0.000000001)
    {
      cout << "N=" << N << " value is not allowed for d= " << type << " square lattice!" << endl;
      exit(1);
    }
    int interactingSite;
    for (int i = 0; i < N; i++)
    {
      for (int j = 0; j < type; j++)
      {
        interactingSite = (i - (i % (int)pow(L, j + 1))) + ((int)(i + pow(L, j))) % ((int)pow(L, j + 1));
        if (i == 3)
        {
          cout << j << " " << i << " " << interactingSite << endl;
        }
        if (interactingSite > i)
        {
          list.push_back(i);
          list.push_back(interactingSite);
        }
        if (L > 2)
        {
          interactingSite = (i - (i % (int)pow(L, j + 1))) + ((int)(i - pow(L, j) + pow(L, j + 1))) % ((int)pow(L, j + 1));
          if (i == 3)
          {
            cout << j << " " << i << " " << interactingSite << endl;
          }
          if (interactingSite > i)
          {
            list.push_back(i);
            list.push_back(interactingSite);
          }
          if (interactingSite > i)
          {
            list.push_back(i);
            list.push_back(interactingSite);
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

  if (type < 0)
  {
    outputFile << N << " " << p << " " << C << " " << seed << "\n";
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
  return 0;
}
