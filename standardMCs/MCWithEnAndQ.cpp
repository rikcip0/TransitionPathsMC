#include <iostream>
#include <fstream>
#include <cassert>
#include <time.h>
#include <sys/utsname.h>

#define initRefConfCode 55

#include "../Generic/random.h"

#include "../McTrajs/MCdyn_classi/generic/fileSystemUtil.h"
#include "../McTrajs/MCdyn_classi/interaction.h"
#include "../McTrajs/MCdyn_classi/initialization/initializeGraph.h"
#include "../McTrajs/MCdyn_classi/thermodynamicUtilities.h"
#include "../McTrajs/MCdyn_classi/Initialization/initializeReferenceConfigurations.h"

#define p 2
#define MC 2000
#define MCeq 100
#define MCmeas 5
#define MCprint 10 // deve essere multiplo di MCmeas

int main(int argc, char *argv[])
{
  if ((argc < 2) || (atoi(argv[1]) == 0) || atoi(argv[1]) < -2 || (argc != 10 && argc != 12))
  {
    cout << "Probable desired usage: ";
    cout << "d(-2:ER -1:RRG k>0:k-dim sqLatt) N beta Hext C structureID fracPosJ graphID (requiredBetaOfSExtraction s_index) [if not present, FM(all +) conf. is considered] nStories" << endl;
    cout << "If d is a lattice, C and structureID are required but ignored." << endl;
    exit(1);
  }

  int d = std::stoi(argv[1]); // should include check on d value
  int N = atoi(argv[2]);
  double beta = atof(argv[3]);
  double Hext = atof(argv[4]);
  int C = 0;
  int structureID = -1;
  if (d < 0)
  {
    C = atoi(argv[5]);
    structureID = atoi(argv[6]);
  }
  else
  {
    C = 2 * d;
  }

  int nStories = atoi(argv[11]);
  double fracPosJ = atof(argv[7]);
  int graphID = atoi(argv[8]);

  vector<vector<vector<rInteraction>>> Graph;

  string admittibleGraph, folder;
  if (getGraphFromGraphAndStructureID_ForCluster(d, admittibleGraph, structureID, graphID, p, C, N, fracPosJ))
    folder = admittibleGraph + "/";
  else
    return 1;

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

  if (!initializeGraph(folder, Graph, p, C, N, fracPosJ))
  {
    cout << "Error in the graph initialization" << endl;
    return 1;
  }

  char buffer[200];
  string parametersSpec;
  struct utsname ugnm;
  uname(&ugnm);
  init_random(0, 0);
  long sstart = time(NULL);

  sprintf(buffer, "%.2g_%.3g", beta, Hext);

  pair<string, string> info; // 1° entry: long info (for human read), 2°entry: short info (for machines read)
  info.first += "Simulation on" + graphType + "\n";
  info.first += (string)("Simulation run on: ") + ugnm.nodename + ", with seed " + to_string(sstart) + ", started at " + getCurrentDateTime() + "\n\n";
  info.second += (string)("16 0 ") + to_string(d) + " " + ugnm.nodename + " " + to_string(sstart) + " " + getCurrentDateTime() + "\n";

  info.first += "mcTotal=" + to_string(MC) + " MCeq=" + to_string(MCeq) + " MCmeas=" + to_string(MCmeas) + "\n\n";
  info.second += "31 " + to_string(MC) + " " + to_string(MCeq) + " " + to_string(MCmeas) + "\n";

  string nomefile;

  info.second += "23 " + to_string(N) + " " + to_string((int)p) + " " + to_string(C);
  info.second += " " + to_string(graphID) + " "
                                            " " +
                 to_string(beta);
  info.second += " " + to_string(fracPosJ) + " " + to_string(Hext) + "\n";

  cout << graphType << " simulation with N=" << N << " p=" << p << " graphID=" << graphID << " beta=" << beta;
  cout << " h_ext=" << Hext << endl
       << endl;

  vector<int> *s, refS(N, 0);
  s = (vector<int> *)calloc(1, sizeof(vector<int>)); // for easiness of a later implementation including more s's
  for (int i = 0; i < 1; i++)
    s[i].assign(N, 0);

  if (argc == 12)
  {
    std::string betaOfSExtractionInput(argv[9]);
    double requiredBetaOfSExtraction = atof(argv[9]);
    int sIndex = atoi(argv[10]);
    if (betaOfSExtractionInput == "inf")
    {
      requiredBetaOfSExtraction = 100.; // just a large number
    }
    else
    {
      requiredBetaOfSExtraction = std::stod(betaOfSExtractionInput);
    }
    if (requiredBetaOfSExtraction < 0.0)
    {
      std::cout << "requiredBetaOfSExtraction value is not allowed!" << std::endl;
      return 1;
    }
    if (!initializeSingleConfigurationFromParTemp(folder, N, refS, info, requiredBetaOfSExtraction, sIndex))
    {
      return 1;
    }
  }
  else
  {
    refS.assign(N, 1); // i.e. refS is all ups.
    info.first += "Reference configuration is FM conf (all +)";
    info.second += "56";
  }

  // folder = makeFolderNameFromBuffer(folder+"/DataForPathsMC/", string(buffer));   //Comment if on cluster
  folder = makeFolderNameFromBuffer_ForCluster(folder + "StandardMCs/MCWithEnergyAndOverlap/", string(buffer), sstart); // For Cluster

  createFolder(folder);
  cout << "Simulation is in folder " << folder << endl;

  // END of initialization of graph, reference configuration

  // START of the proper MC simulation

  // MC run
  cout << folder << endl;
  nomefile = folder + "/infoLong.dat";
  ofstream fileO(nomefile);
  fileO << info.first;
  fileO.close();

  nomefile = folder + "/info.dat";
  fileO.open(nomefile);
  fileO << info.second;
  fileO.close();

  nomefile = folder + "/thermCheck.txt";
  fileO.open(nomefile);

  int nMeasures = nStories * MC / MCmeas;
  double energy = 0;
  long int qWithRefS = 0;

  for (int k = 0; k < nStories; k++)
  {
    s[0] = refS;
    for (int i = 1; i <= MC; i++)
    {
      MCSweep_withGraph(s[0], N, Graph, beta);

      if (!(i % MCmeas))
      {
        long int tempQWithRefS = 0;
        for (int j = 0; j < N; j++)
          tempQWithRefS += refS[j] * s[0][j];

        if (!(i % MCprint))
        {
          double tempH = energy_Graph(s[0], N, Graph);
          fileO << k <<" "<< i << " " << tempH << " " << tempQWithRefS << endl;
          energy += tempH;
          qWithRefS += tempQWithRefS;
        }
        else
        {
          energy += energy_Graph(s[0], N, Graph);
          qWithRefS += tempQWithRefS;
        }
      }
    }
  }

  fileO.close();

  cout << "H = " << energy / (nMeasures) << endl;
  cout << "Q = " << qWithRefS / (double)(nMeasures) << endl;

  nomefile = folder + "/TIbeta.txt";
  fileO.open(nomefile);

  if (!fileO.is_open())
  {
    std::cerr << "Error opening the file: " << nomefile << std::endl;
    return 1;
  }

  fileO << beta << " " << energy / (double)(nMeasures) << " " << qWithRefS / (double)(nMeasures) << endl;

  fileO.close();

  return EXIT_SUCCESS;
}