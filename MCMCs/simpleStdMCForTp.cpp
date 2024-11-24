#include <iostream>
#include <fstream>
#include <cassert>
#include <time.h>
#include <sys/utsname.h>

#ifdef FIXEDEXT
#define extremesFixed true
#else
#define extremesFixed false
#endif

#define initRefConfCode 55

#include "../Generic/random.h"

#include "../McTrajs/MCdyn_classi/generic/fileSystemUtil_Trajs.h"
#include "../McTrajs/MCdyn_classi/interaction.h"
#include "../McTrajs/MCdyn_classi/initialization/initializeGraph.h"
#include "../McTrajs/MCdyn_classi/thermodynamicUtilities.h"
#include "../McTrajs/MCdyn_classi/Initialization/initializeReferenceConfigurations.h"

#define p 2
#define MC 300000
#define MCeq 10000
#define MCmeas 10
#define MCprint 2000 // deve essere multiplo di MCmeas

int main(int argc, char *argv[])
{
  if ((argc < 2) || (atoi(argv[1]) == 0) || atoi(argv[1]) < -2 || (argc != 11 && argc != 13 && argc != 14 && argc != 16))
  {
    cout << "Probable desired usage: ";
    cout << "d(-2:ER -1:RRG k>0:k-dim sqLatt) N beta Hext hout Q* C structureID fracPosJ graphID (requiredBetaOfSExtraction s_index) [if not present, FM(all +) conf. is considered] \n (randomFieldType(1: Bernoulli, 2:Gaussian), realization, sigma)" << endl;
    cout << "If d is a lattice, C and structureID are required but ignored." << endl;
    exit(1);
  }

  int d = std::stoi(argv[1]); // should include check on d value
  int N = atoi(argv[2]);
  double beta = atof(argv[3]);
  double Hext = atof(argv[4]);
  double hout = atof(argv[5]);

  int Qstar = atoi(argv[6]);
  /*
  if (abs(Qstar) % 2 != N % 2 || abs(Qstar) > N)
  {
    cout << "Q* value is not allowed!" << endl;
    exit(1);
  }
  */

  int C = 0;
  int structureID = -1;
  if (d < 0)
  {
    C = atoi(argv[7]);
    structureID = atoi(argv[8]);
  }
  else
  {
    C = 2 * d;
  }

  double fracPosJ = atof(argv[9]);
  int graphID = atoi(argv[10]);

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

  vector<double> randomField(N);
  int randomFieldType = 0, fieldStructureRealization = 0;
  double sigma = 0.;
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

  char buffer[200];
  string parametersSpec;
  struct utsname ugnm;
  uname(&ugnm);
  init_random(0, 0);
  long sstart = time(NULL);

  if (!extremesFixed)
    sprintf(buffer, "inf_%.3g_%.2g_%i_%.3g", beta, Hext, Qstar, hout);
  else
    sprintf(buffer, "inf_%.3g_%.2g_inf_%i_inf", beta, Hext, Qstar);

  pair<string, string> info; // 1° entry: long info (for human read), 2°entry: short info (for machines read)
  info.first += "Simulation on" + graphType + "\n";
  info.first += (string)("Simulation run on: ") + ugnm.nodename + ", with seed " + to_string(sstart) + ", started at " + getCurrentDateTime() + "\n\n";
  info.second += (string)("15 0 ") + to_string(d) + " " + ugnm.nodename + " " + to_string(sstart) + " " + getCurrentDateTime() + "\n";

  if (extremesFixed)
  {
    info.first += ("Simulation with extremes fixed to reference configurations.\n\n");
    cout << "Simulation with extremes fixed to reference configurations.\n\n";
  }

  info.first += "mcTotal=" + to_string(MC) + " MCeq=" + to_string(MCeq) + " MCmeas=" + to_string(MCmeas) + "\n\n";
  info.second += "31 " + to_string(MC) + " " + to_string(MCeq) + " " + to_string(MCmeas) + "\n";

  string nomefile;

  if (argc == 14 || argc == 16)
  {
    info.second += "220 ";
  }
  else
  {
    info.second += "22 ";
  }

  info.second += to_string(N) + " " + to_string((int)p) + " " + to_string(C);
  info.second += " " + to_string(graphID) + " " + "inf" + " " + to_string(beta);
  if (!extremesFixed)
    info.second += " " + to_string(hout) + " " + to_string(Qstar);
  else
    info.second += " inf " + to_string(Qstar);
  info.second += " " + to_string(fracPosJ) + " " + to_string(Hext);
  if (argc == 14 || argc == 16)
  {
    info.second += " " + to_string(randomFieldType) + " 0. " + to_string(sigma) + " " + to_string(fieldStructureRealization); // randomFieldType, mean, sigma, realization
  }
  info.second += "\n";

  cout << graphType << " simulation with N=" << N << " p=" << p << " graphID=" << graphID << " T=inf beta=" << beta;
  cout << " Qstar=" << Qstar << " h_out=" << hout;
  cout << " h_ext=" << Hext << endl
       << endl;

  vector<int> *s, refS(N, 0);
  s = (vector<int> *)calloc(2, sizeof(vector<int>));
  for (int i = 0; i < 2; i++)
    s[i].assign(N, 0);

  if (argc == 13)
  {
    std::string betaOfSExtractionInput(argv[11]);
    double requiredBetaOfSExtraction = atof(argv[11]);
    int sIndex = atoi(argv[12]);
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

  if (extremesFixed)
  {
    hout = 1.;
  }

  // folder = makeFolderNameFromBuffer(folder+"/DataForPathsMC/", string(buffer));   //Comment if on cluster
  if (argc == 14 || argc == 16)
  {
    folder = makeFolderNameFromBuffer_ForCluster(folder + "DataForPathsMC/stdMCs/", string(buffer) + "_sigma" + to_string(sigma), sstart); // For Cluster
  }
  else
  {
    folder = makeFolderNameFromBuffer_ForCluster(folder + "DataForPathsMC/stdMCs/", string(buffer), sstart); // For Cluster
  }

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

  s[0] = refS;
  s[1] = refS;

  double energy = 0, energyB = 0.;

  for (int i = 1; i <= MCeq; i++)
  {
    MCSweep_withGraph(s[0], N, Graph, beta, Hext, randomField);
    MCSweep_withGraph2(s[1], refS, N, Graph, beta, Qstar, Hext, randomField);
  }

  for (int i = 1; i <= MC; i++)
  {
    MCSweep_withGraph(s[0], N, Graph, beta, Hext, randomField);
    MCSweep_withGraph2(s[1], refS, N, Graph, beta, Qstar, Hext, randomField);

    if (!(i % MCmeas))
    {
      if (!(i % MCprint))
      {
        double tempH, tempHB;
        tempH = energy_Graph(s[0], N, Graph, Hext, randomField);
        energy += tempH;

        tempHB = energy_Graph(s[1], N, Graph, Hext, randomField);
        energyB += tempHB;

        fileO << i << " " << tempH << " " << tempHB << " " << tempH - tempHB << " " << endl;
      }
      else
      {
        energy += energy_Graph(s[0], N, Graph, Hext, randomField);
        energyB += energy_Graph(s[1], N, Graph, Hext, randomField);
      }
    }
  }
  fileO.close();

  cout << "H = " << energy / MC * MCmeas << endl;
  cout << "H_B = " << energyB / MC * MCmeas << endl;
  cout << "H-H_B = " << (energy - energyB) / MC * MCmeas << endl;

  nomefile = folder + "/TIbeta.txt";
  fileO.open(nomefile);

  if (!fileO.is_open())
  {
    std::cerr << "Error opening the file: " << nomefile << std::endl;
    return 1;
  }

  fileO << beta << " " << (energy - energyB) / MC * MCmeas << endl;

  fileO.close();

  return EXIT_SUCCESS;
}