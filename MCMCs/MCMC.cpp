#include <iostream>
#include <fstream>
#include <cassert>
#include <time.h>
#include <sys/utsname.h>

#include "../Generic/random.h"

#include "../Generic/fileSystemUtil.h"

#include "../MCtrajs/MCdyn_classi/interaction.h"

#include "../MCtrajs/MCdyn_classi/Initialization/initializeGraph.h"
#include "../MCtrajs/MCdyn_classi/Initialization/initializeReferenceConfigurations.h"

#include "../MCtrajs/MCdyn_classi/MCUtilities.h"
#include "../MCtrajs/MCdyn_classi/thermodynamicUtilities.h"

#define p 2 // as of now, the dynamics generation (and hence the code) only works for p=2 (RC)

#define stories 1000
#define MCeq 10
#define MCmeas 4
#define MCprintEv 100
#define printEv false

using namespace std;

vector<int> s_in, s;

int main(int argc, char **argv)
{

  // TO IMPLEMENT- double planted RRG: type -3

  // START of input management

  if ((argc < 2) || (atoi(argv[1]) == 0) || atoi(argv[1]) < -2 || (argc != 11 && argc != 13 && argc != 14 && argc != 16))
  {
    cout << "Probable desired usage: ";
    cout << "d(-2:ER -1:RRG k>0:k-dim sqLatt) N MC beta Hext\nC structureID fracPosJ graphID" << endl;
    cout << "flipped(if 1 flips desired reference configurations)" << endl;
    cout << "(requiredBetaOfSExtraction requiredQif)[if not present, FM(all +) conf.is considered]" << endl;
    cout << "(randomFieldType(1 : Bernoulli, 2 : Gaussian), realization, sigma) " << endl;
    cout << "If d is a lattice, C and structureID are required but ignored." << endl;
    exit(1);
  }

  int d = std::stoi(argv[1]); // should include check on d value
  int N = atoi(argv[2]);
  int MC = atoi(argv[3]);
  double beta = atof(argv[4]);
  double Hext = atof(argv[5]);

  int C = 0;
  int structureID = -1;
  if (d < 0)
  {
    C = atoi(argv[6]);
    structureID = atoi(argv[7]);
  }
  else
  {
    C = 2 * d;
  }

  double fracPosJ = atof(argv[8]);
  int graphID = atoi(argv[9]);
  int flipped = atoi(argv[10]);

  vector<vector<vector<rInteraction>>> Graph;

  string admittibleGraph, folder;
  pair<string, string> info; // 1° entry: long info (for human read), 2°entry: short info (for machines read)

  if (getGraphFromGraphAndStructureID_ForCluster(d, admittibleGraph, structureID, graphID, p, C, N, fracPosJ))
    folder = admittibleGraph + "/";
  else
    return 1;

  int randomFieldType = 0, fieldStructureRealization = 0;
  double sigma = 1.;

  vector<double> randomField(N);
  if (argc == 14 || argc == 16)
  {
    randomFieldType = atoi(argv[argc - 3]);
    fieldStructureRealization = atoi(argv[argc - 2]);
    sigma = atof(argv[argc - 1]);
  }
  else
    randomField.assign(N, 0.);

  if (!initializeGraph(folder, Graph, p, C, N, fracPosJ, randomField, randomFieldType, fieldStructureRealization, sigma))
  {
    cout << "Error in the graph initialization" << endl;
    return 1;
  }

  string graphType;
  if (d < 0)
  {
    if (d == -2)
      graphType = "ER";
    else if (d == -1)
      graphType = "RRG";
    graphType += "/p" + to_string(p) + "C" + to_string(C);
  }
  else if (d == 1)
  {
    graphType = "1dChain";
  }
  else
  {
    graphType = to_string(d) + "dSqLatt";
  }

  graphType += +"/N" + to_string(N) + "/fPosJ" + to_string(fracPosJ).substr(0, 4);

  struct utsname ugnm;
  uname(&ugnm);

  init_random(0, 0);
  long sstart = time(NULL);

  info.first += "Simulation on " + graphType + "\n";

  info.first += (string)("Simulation run on: ") + ugnm.nodename + ", with seed " + to_string(sstart) + ", started at " + getCurrentDateTime() + "\n\n";
  info.second += (string)("1 2 ") + ugnm.nodename + " " + to_string(sstart) + " " + getCurrentDateTime() + "\n";

  double requiredBetaOfSExtraction = 0.;
  int requiredIndex = 0;
  if (argc == 13 || argc == 16)
  {
    std::string input(argv[11]);
    requiredIndex = atoi(argv[12]);

    if (input == "inf")
    {
      requiredBetaOfSExtraction = 100.; // just a large number
    }
    else
    {
      requiredBetaOfSExtraction = std::stod(input);
    }
    if (requiredBetaOfSExtraction < 0.0)
    {
      std::cout << "requiredBetaOfSExtraction value is not allowed!" << std::endl;
      return 1;
    }

    if (!initializeSingleConfigurationFromParTemp(folder, N, s_in, info, requiredBetaOfSExtraction, requiredIndex))
    {
      cout << "Error in the start/end configurations initialization" << endl;
      return 1;
    }
  }
  else
  {
    s_in.assign(N, 1);
  }

  if (flipped == 1)
  {
    for (int i = 0; i < N; i++)
      s_in[i] = -s_in[i];
  }

  char buffer[200];
  sprintf(buffer, "betExtr_%.2g/%d__%.4g_%.4g_%d_%d", requiredBetaOfSExtraction, requiredIndex, beta, Hext, MC, stories);

  // folder = makeFolderNameFromBuffer(folder+"/DataForPathsMC/", string(buffer));   //Comment if on cluster
  if (argc == 14 || argc == 16)
  {
    folder = makeFolderNameFromBuffer_ForCluster(folder + "MCMC/stdMCMC/", string(buffer) + "_sigma" + to_string(sigma), sstart); // For Cluster
  }
  else
  {
    folder = makeFolderNameFromBuffer_ForCluster(folder + "MCMC/stdMCMC/", string(buffer), sstart); // For Cluster
  }

  createFolder(folder);
  cout << "Simulation is in folder " << folder << endl;

  info.first += "mcTotal=" + to_string(MC) + " MCeq=" + to_string(MCeq) + " MCmeas=" + to_string(MCmeas) + " MCprint=" + to_string(MCprintEv) + "\n\n";
  info.second += "30 " + to_string(MC) + " " + to_string(MCeq) + " " + to_string(MCmeas) + " " + to_string(MCprintEv) + "\n";

  string nomefile;

  if (argc == 16 || argc == 18)
  {
    info.second += "210 ";
  }
  else
  {
    info.second += "21 ";
  }
  info.second += to_string(N) + " " + to_string((int)p) + " " + to_string(C);
  info.second += " " + to_string(graphID) + " " + to_string(beta);

  info.second += " " + to_string(fracPosJ) + " " + to_string(Hext);
  if (argc == 16 || argc == 18)
  {
    info.second += " " + to_string(randomFieldType) + " 0. " + to_string(sigma) + " " + to_string(fieldStructureRealization); // randomFieldType, mean, sigma, realization
  }
  info.second += "\n";

  cout << graphType << " simulation with N=" << N << " p=" << p << " C=" << C << " graphID=" << graphID << " beta=" << beta;
  cout << " fracPosJ=" << fracPosJ << " h_ext=" << Hext << endl
       << endl;

  // MC run
  nomefile = folder + "/infoLong.dat";
  ofstream detFile(nomefile);
  detFile << info.first;
  detFile.close();

  nomefile = folder + "/info.dat";
  detFile.open(nomefile);
  detFile << info.second;
  detFile.close();

  vector<int> permanenceAtQ(N + 1, 0);
  int nMeasures = 0;
  double energy = 0;
  vector<int> nQinAtFinalTime(N + 1, 0);
  for (int story = 0; story < stories; story++)
  {
    s = s_in;
    for (long mc = 0; mc < MC; mc++)
    {
      // just measuring the magnetization
      if (mc % MCmeas == 0)
      {
        if (mc > MCeq)
        {
          int q_in = 0;

          for (int i = 0; i < N; i++)
          {
            q_in += s_in[i] * s[i];
          }
          permanenceAtQ[(q_in + N) / 2]++;
          energy += energy_Graph(s, N, Graph, Hext, randomField);
          nMeasures++;
        }
      }

      if (printEv && (mc % MCprintEv == 0))
      {
        nomefile = folder + "conf_." + to_string(mc);
        // print_conf(&Strajs, T, nomefile);
      }

      // MC move
      MCSweep_withGraph(s, N, Graph, beta, Hext, randomField);
      // end
    }
    int q_in = 0;

    for (int i = 0; i < N; i++)
    {
      q_in += s_in[i] * s[i];
    }
    nQinAtFinalTime[(q_in + N) / 2]++;
  }

  nomefile = folder + "/av.dat";
  detFile.open(nomefile);
  int perm = 0;
  int nAtFinalTime = 0;
  for (int i = 0; i < N + 1; i++)
  {
    perm = permanenceAtQ[i];
    nAtFinalTime = nQinAtFinalTime[i];
    if (perm+nAtFinalTime > 0)
      detFile << -N + 2 * i << " " << perm / ((double)nMeasures)<<" " << nAtFinalTime / ((double)stories) << endl;
  }
  detFile.close();
  nomefile = folder + "/ener.dat";
  detFile.open(nomefile);
  detFile << (energy / N / nMeasures);
  detFile.close();

  // END of the proper MC simulation

  return 0;
}
