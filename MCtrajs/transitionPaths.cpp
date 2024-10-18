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

#ifndef QUENCHCONFS
#define initRefConfCode 54
#else
#define initRefConfCode 53
#endif

#ifdef INITRANDOM
#define initTrajCCode 70
#define initTrajJCode 80
#elif defined(INITANNEALING)
#define initTrajJCode 90
#ifndef SWAPPED
//define initTrajCCode 74 //to make annealing without End fixed
#define initTrajCCode 740 //to make annealing with End fixed (as if Q*=N during annealing)
#define swapConfig false
#else
#define initTrajCCode 75
#define swapConfig true
#endif
#else
#define initTrajJCode 80
#ifndef SWAPPED
#define initTrajCCode 71
#define swapConfig false
#else
#define initTrajCCode 72
#define swapConfig true
#endif
#endif

#include "../Generic/random.h"

#include "MCdyn_classi/Generic/fileSystemUtil_Trajs.h"

#include "MCdyn_classi/interaction.h"
#include "MCdyn_classi/straj.h"
#include "MCdyn_classi/field.h"

#include "MCdyn_classi/Initialization/initializeGraph.h"
#include "MCdyn_classi/Initialization/initializeTrajectories.h"
#include "MCdyn_classi/Initialization/initializeReferenceConfigurations.h"

#include "MCdyn_classi/MCUtilities.h"
#include "MCdyn_classi/thermodynamicUtilities.h"

#define p 2 // as of now, the dynamics generation (and hence the code) only works for p=2 (RC)

#define MC 1000000000

//#ifdef INITANNEALING
//#define MCeq 40000
//#else
#define MCeq 100000
//#endif

#define MCmeas 4
#define MCprint 100000
#define NpPerN 8

#define printconf false
#define MCconf 100000

#define mcForIntegratedMeasuring 10000
#define averagingMCForIntegratedMeasuring 1
#define nPointsForDistanceFromStfwdPathComputation 50

#define configurationsChoiceOption -1 // se -1: quando usato le configurazioni scelte dal parallel tempering in base all'overlap sono 1 e 2, se =1: sono la 1 e la -1, se =2 sono la -2 e la 2
using namespace std;

vector<int> s_in, s_out;

int main(int argc, char **argv)
{

  // TO IMPLEMENT- double planted RRG: type -3

  // START of input management

  if ((argc < 2) || (atoi(argv[1]) == 0) || atoi(argv[1]) < -2 || (argc != 13 && argc != 15 && argc != 16 && argc != 18))
  {
    cout << "Probable desired usage: ";
    cout << "d(-2:ER -1:RRG k>0:k-dim sqLatt) N T beta Hext hin hout Q*(if -1 is self-computed) \nC structureID fracPosJ graphID (requiredBetaOfSExtraction requiredQif)[if not present, FM(all +) conf.is considered]\n (randomFieldType(1: Bernoulli, 2:Gaussian), realization, sigma)" << endl;
    cout << "If d is a lattice, C and structureID are required but ignored." << endl;
    exit(1);
  }

  int d = std::stoi(argv[1]); // should include check on d value
  int N = atoi(argv[2]);
  double T = atof(argv[3]);
  double beta = atof(argv[4]);
  double Hext = atof(argv[5]);
  double hin = atof(argv[6]);
  double hout = atof(argv[7]);
  if (extremesFixed)
  {
    hin = 1.;
    hout = 1.;
  }

  int Qstar = atoi(argv[8]);
  if ((abs(Qstar) % 2 != N % 2 || abs(Qstar) > N) && Qstar != -1)
  {
    cout << "Q* value is not allowed!" << endl;
    exit(1);
  }

  int C = 0;
  int structureID = -1;
  if (d < 0)
  {
    C = atoi(argv[9]);
    structureID = atoi(argv[10]);
  }
  else
  {
    C = 2 * d;
  }

  double fracPosJ = atof(argv[11]);
  int graphID = atoi(argv[12]);

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
  if (argc == 16 || argc == 18)
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
  if (Qstar != -1)
  {
    info.first += (string)("Simulation run on: ") + ugnm.nodename + ", with seed " + to_string(sstart) + ", started at " + getCurrentDateTime() + "\n\n";
    info.first += "Qstar given in input\n\n";
#ifdef INITANNEALING
    info.second += (string)("100 2 ") + ugnm.nodename + " " + to_string(sstart) + " " + getCurrentDateTime() + "\n";
#else
    info.second += (string)("10 2 ") + ugnm.nodename + " " + to_string(sstart) + " " + getCurrentDateTime() + "\n";
#endif
  }
  else
  {
    info.first += (string)("Simulation run on: ") + ugnm.nodename + ", with seed " + to_string(sstart) + ", started at " + getCurrentDateTime() + "\n\n";
    info.first += "No Qstar given in input: self-overlap around final configuration will be computed\n\n";
#ifdef INITANNEALING
    info.second += (string)("110 2 ") + ugnm.nodename + " " + to_string(sstart) + " " + getCurrentDateTime() + "\n";
#else
    info.second += (string)("11 2 ") + ugnm.nodename + " " + to_string(sstart) + " " + getCurrentDateTime() + "\n";
    cout << "Simulation with extremes fixed to reference configurations.\n\n";
#endif
  }

  pair<vector<int>, vector<int>> referenceConfigurations;
  referenceConfigurations.first.assign(N, 0);
  referenceConfigurations.second.assign(N, 0);

  if (argc == 15 || argc == 18)
  {
    double requiredBetaOfSExtraction = atof(argv[13]);
    int requiredQif = atoi(argv[14]);
    if (abs(requiredQif) > N)
    {
      std::cout << "requiredQif value is not allowed!" << std::endl;
      return 1;
    }

    std::string input(argv[13]);

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

    if (!
#ifndef QUENCHCONFS
        initializeReferenceConfigurationsFromParTemp_FirstOccurrence(folder, N, referenceConfigurations, info, requiredQif, requiredBetaOfSExtraction, configurationsChoiceOption)
#else
        initializeReferenceConfigurationsSeqQuenchingFromParTemp(Graph, folder, N, referenceConfigurations, info, requiredQif, requiredBetaOfSExtraction, configurationsChoiceOption)
#endif
    )
    {
      cout << "Error in the start/end configurations initialization" << endl;
      return 1;
    }
  }
  else
  {
    if (!initializeFMConfigurations(N, referenceConfigurations, info))
    {
      cout << "Error in the start/end configurations initialization" << endl;
      return 1;
    }
  }

  s_in = referenceConfigurations.first;
  s_out = referenceConfigurations.second;

  if (Qstar == -1)
  {
    if (!computeSelfOverlap_withGraph(Qstar, s_out, Graph, beta, info, N))
    {
      cout << "Error during computation of final configuration self overlap" << endl;
      return 1;
    }
    if ((Qstar + N) % 2)
      Qstar -= 1; // decrease by 1 if Qstar and N have different parity
  }
  cout << "Using Q*= " << Qstar << endl
       << endl;

  int mutualQ = 0;
  for (int i = 0; i < N; i++)
  {
    mutualQ += s_in[i] * s_out[i];
  }

  char buffer[200];
  cout << extremesFixed << endl;
  if (extremesFixed)
    sprintf(buffer, "%.2g_%.4g_%.2g_inf_%i_inf", T, beta, Hext, Qstar);
  else
    sprintf(buffer, "%.2g_%.4g_%.2g_%.2g_%i_%.3g", T, beta, Hext, hin, Qstar, hout);

  // folder = makeFolderNameFromBuffer(folder+"/DataForPathsMC/", string(buffer));   //Comment if on cluster
  if (argc == 16 || argc == 18)
  {
    folder = makeFolderNameFromBuffer_ForCluster(folder + "DataForPathsMC/PathsMCs/", string(buffer) + "_sigma" + to_string(sigma), sstart); // For Cluster
  }
  else
  {
    folder = makeFolderNameFromBuffer_ForCluster(folder + "DataForPathsMC/PathsMCs/", string(buffer), sstart); // For Cluster
  }

  createFolder(folder);
  cout << "Simulation is in folder " << folder << endl;

  vector<straj> Strajs;

  if (!
#ifdef INITRANDOM
      initializeTrajectoriesAtRandom(N, T, Strajs, info)
#elif defined(INITANNEALING)
      callAnnealing(N, T, Strajs, beta, Hext, Graph, hin, hout, Qstar, info, referenceConfigurations, randomField, swapConfig)
#else
      initializeTrajectoriesFromRefConfs(N, T, Strajs, info, referenceConfigurations, swapConfig)
#endif
      //   initializeTrajectoriesFromRefConfs_withNoise(N, T, Strajs, info, 0.8, referenceConfigurations)
  )
  {
    cout << "Error during trajectories initialization" << endl;
    exit(1);
  }
  if (extremesFixed)
  {
    info.first += ("Simulation with extremes fixed to reference configurations.\n\n");
    cout << "Simulation with extremes fixed to reference configurations.\n\n";
  }
  // END of initialization of graph, reference configurations and trajectories.

  vector<int> Qfin = compute_Q_fin(&Strajs, s_in, s_out);

  // START of the proper MC simulation

  int Np = NpPerN * N;
  info.first += "mcTotal=" + to_string(MC) + " MCeq=" + to_string(MCeq) + " MCmeas=" + to_string(MCmeas) + " MCprint=" + to_string(MCprint) + " Np=" + to_string(Np) + "\n\n";
  info.second += "30 " + to_string(MC) + " " + to_string(MCeq) + " " + to_string(MCmeas) + " " + to_string(MCprint) + " " + to_string(Np) + "\n";

  info.first += "mcForIntegratedMeasuring=" + to_string(mcForIntegratedMeasuring) + " averagingMCForIntegratedMeasuring= " + to_string(averagingMCForIntegratedMeasuring) + " nPointsForDistanceFromStfwdPathComputation=" + to_string(nPointsForDistanceFromStfwdPathComputation) + "\n\n";
  info.second += "41 " + to_string(mcForIntegratedMeasuring) + " " + to_string(averagingMCForIntegratedMeasuring) + " " + to_string(nPointsForDistanceFromStfwdPathComputation) + "\n";

  field F(T, beta, Hext, randomField);
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
  info.second += " " + to_string(graphID) + " " + to_string(T) + " " + to_string(beta);
  if (!extremesFixed)
    info.second += " " + to_string(hin) + " " + to_string(Qstar) + " " + to_string(hout);
  else
    info.second += " inf " + to_string(Qstar) + " inf";
  info.second += " " + to_string(fracPosJ) + " " + to_string(Hext);
  if (argc == 16 || argc == 18)
  {
    info.second += " " + to_string(randomFieldType) + " 0. " + to_string(sigma) + " " + to_string(fieldStructureRealization); // randomFieldType, mean, sigma, realization
  }
  info.second += "\n";

  cout << graphType << " simulation with N=" << N << " p=" << p << " C=" << C << " graphID=" << graphID << " T=" << T << " beta=" << beta;
  cout << " h_in=" << hin << " Qstar=" << Qstar << " h_out=" << hout;
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

  nomefile = folder + "/story.dat";
  ofstream storyFile(nomefile);
  nomefile = folder + "/log.log";
  ofstream logfile(nomefile);
  nomefile = folder + "/thermCheck.dat";
  ofstream integratedMeasuringFile(nomefile);
  vector<vector<int>> ris(3, vector<int>(Np, 0));
  vector<vector<double>> risQ(3, vector<double>(Np, 0.));
  vector<double> chiQ(Np, 0);

  vector<int> mileStones = {mutualQ, 0, (int)(Qstar / 2), Qstar};
  pair<vector<vector<long int>>, vector<vector<long int>>> fracOfTrajsInCone;
  fracOfTrajsInCone.first = vector<vector<long int>>(mileStones.size(), vector<long int>(Np, 0));
  fracOfTrajsInCone.second = vector<vector<long int>>(mileStones.size(), vector<long int>(Np, 0));

  vector<double> qfin_av(3, 0);
  double chifin_av = 0.;
  double probafin_av = 0.;
  double L_av = 0.;
  int counter = 0;

  double ther_meanMeanEner, ther_meanMaxEner, ther_qDistFromStFw, ther_meanJumpsAv, ther_meanJumpsStdDev, ther_meanJumpsMin, ther_meanJumpsMax;
  string nomeAvFile, nomeTIQstarFile, nomeTIHoutFile, nomeTIBetaFile;

  nomeAvFile = folder + "/av.dat";

  nomeTIQstarFile = folder + "/TI_Qstar.dat";
  nomeTIHoutFile = folder + "/TI_hout.dat";
  nomeTIBetaFile = folder + "/TI_beta.dat";

  logfile << "#Running on host " << ugnm.nodename << endl;

  for (long mc = 0; mc < MC; mc++)
  {
    // just measuring the magnetization
    if (mc % MCmeas == 0)
    {
      ris = compute_Q_av(&Strajs, T, Np, s_in, s_out);
      if (mc > MCeq)
      {
        for (int k = 0; k < Np; k++)
        {
          for (int i = 0; i < 3; i++)
            risQ[i][k] += (double)ris[i][k] / (double)N;
          if (!extremesFixed)
            chiQ[k] += (ris[1][k] < Qstar ? exp(hout * (ris[1][k] - Qstar)) : 1); // computes
          else
            chiQ[k] += (ris[1][k] < Qstar ? 0 : 1);

          for (int i = 0; i < mileStones.size(); i++)
          {
            fracOfTrajsInCone.first[i][k] += (ris[0][k] >= mileStones[i] ? 1 : 0);
            fracOfTrajsInCone.second[i][k] += (ris[1][k] >= mileStones[i] ? 1 : 0);
          }
        }
        for (int i = 0; i < 3; i++)
          qfin_av[i] += ((double)Qfin[i] / (double)N);
        chifin_av += (Qfin[1] > Qstar ? Qfin[1] - Qstar : 0) / ((double)N);

        if (!extremesFixed)
          probafin_av += (Qfin[1] >= Qstar ? exp(2 * hout) : 1);
        else
          probafin_av += (Qfin[1] >= Qstar ? 1 : 0);
        L_av += compute_L_av(&Strajs, &Graph, T, beta, Hext, randomField);
        counter++;
      }
    }
    // print on file
    if (mc % MCprint == 0)
    {
      vector<double> j = count_jumps(&Strajs);
      vector<double> risH = compute_H_av(&Strajs, &Graph, Np, T, Hext, randomField);

      if (mc % MCmeas != 0)
        ris = compute_Q_av(&Strajs, T, Np, s_in, s_out);

      logfile << W10 << time(NULL) - sstart << W10 << mc << W10 << j[0] << W10 << j[1] << W10 << j[2] << W10;

      for (int i = 0; i < 3; i++)
        logfile << Qfin[i] << W10;
      logfile << endl;
      for (int i = 0; i < Np; i++)
      {
        storyFile << i * T / (Np - 1) << " ";
        for (int j = 0; j < 3; j++)
          storyFile << ris[j][i] << " ";
        storyFile << risH[i] << endl;
      }
      storyFile << endl;

      if (counter > 0)
      {

        ofstream fileM(nomeAvFile);
        for (int i = 0; i < Np; i++)
        {
          fileM << i * T / (Np - 1) << " ";
          for (int j = 0; j < 3; j++)
            fileM << risQ[j][i] / counter << " ";
          fileM << chiQ[i] / counter << " ";

          for (int j = 0; j < mileStones.size(); j++)
            fileM << fracOfTrajsInCone.first[j][i] / (double)counter << " ";

          for (int j = 0; j < mileStones.size(); j++)
            fileM << fracOfTrajsInCone.second[j][i] / (double)counter << " ";
          fileM << endl;
        }
        fileM << "# FINAL ";
        for (int i = 0; i < 3; i++)
          fileM << qfin_av[i] / counter << " ";
        fileM << hout << " " << chifin_av / counter << " " << Qstar << " " << probafin_av / counter << " " << beta << " " << L_av / counter << endl;
        fileM.close();

        ofstream mfinfile(nomeTIQstarFile, std::ios::app);
        mfinfile << mc << " " << hout << " ";
        for (int i = 0; i < 3; i++)
          mfinfile << qfin_av[i] / counter << " ";
        mfinfile << chifin_av / counter << endl;
        mfinfile.close();

        ofstream mfinfile2(nomeTIHoutFile, std::ios::app);
        mfinfile2 << mc << " " << Qstar << " ";
        for (int i = 0; i < 3; i++)
          mfinfile2 << qfin_av[i] / counter << " ";
        mfinfile2 << endl;
        mfinfile2.close();

        ofstream mfinfile3(nomeTIBetaFile, std::ios::app);
        mfinfile3 << mc << " " << beta << " ";
        for (int i = 0; i < 3; i++)
          mfinfile3 << qfin_av[i] / counter << " ";
        mfinfile3 << L_av / counter << endl;
        mfinfile3.close();
      }
    }
    if (printconf && (mc % MCconf == 0))
    {
      nomefile = folder + "conf_." + to_string(mc);
      print_conf(&Strajs, T, nomefile);
    }

    if (mc % mcForIntegratedMeasuring < averagingMCForIntegratedMeasuring)
    {
      if (mc % mcForIntegratedMeasuring == 0)
      {
        ther_meanMeanEner = 0.;
        ther_meanMaxEner = 0.;

        ther_meanJumpsAv = 0.;
        ther_meanJumpsStdDev = 0.;
        ther_meanJumpsMin = 0.;
        ther_meanJumpsMax = 0.;

        ther_qDistFromStFw = 0.;
      }

      double meanEner = 0., maxEner = -C * N;
      vector<double> risH = compute_H_av(&Strajs, &Graph, Np, T, Hext, randomField);
      for (int i = 0; i < Np; i++)
      {
        meanEner += risH[i];
        if (risH[i] > maxEner)
          maxEner = risH[i];
      }
      meanEner /= Np;
      ther_meanMeanEner += meanEner;
      ther_meanMaxEner += maxEner;

      double qDistFromStFw = 0.;
      vector<vector<int>> risQ = compute_Q_av(&Strajs, T, nPointsForDistanceFromStfwdPathComputation, s_in, s_out);
      for (int i = 0; i < nPointsForDistanceFromStfwdPathComputation; i++)
      {
        qDistFromStFw += fabs(risQ[0][i] + risQ[1][i] - (N + mutualQ));
      }
      qDistFromStFw /= nPointsForDistanceFromStfwdPathComputation * sqrt(2);
      ther_qDistFromStFw += qDistFromStFw;

      vector<double> jumpsStats = count_jumps(&Strajs);
      ther_meanJumpsMin += jumpsStats[0];
      ther_meanJumpsAv += jumpsStats[1];
      ther_meanJumpsStdDev += jumpsStats[2];
      ther_meanJumpsMax += jumpsStats[3];

      if (mc % mcForIntegratedMeasuring == averagingMCForIntegratedMeasuring - 1)
      {
        ther_meanMeanEner /= averagingMCForIntegratedMeasuring;
        ther_meanMaxEner /= averagingMCForIntegratedMeasuring;
        ther_qDistFromStFw /= averagingMCForIntegratedMeasuring;
        ther_meanJumpsAv /= averagingMCForIntegratedMeasuring;
        ther_meanJumpsStdDev /= averagingMCForIntegratedMeasuring;
        ther_meanJumpsMin /= averagingMCForIntegratedMeasuring;
        ther_meanJumpsMax /= averagingMCForIntegratedMeasuring;

        integratedMeasuringFile << mc / mcForIntegratedMeasuring * mcForIntegratedMeasuring;
        integratedMeasuringFile << " " << ther_meanMeanEner << " " << ther_meanMaxEner;
        integratedMeasuringFile << " " << ther_qDistFromStFw;
        integratedMeasuringFile << " " << ther_meanJumpsAv << " " << ther_meanJumpsStdDev;
        integratedMeasuringFile << " " << ther_meanJumpsMin << " " << ther_meanJumpsMax;
        integratedMeasuringFile << endl;
      }
    }
    // MC move
    for (int i = 0; i < N; i++)
    {
      int I = (int)(N * Xrandom());
      Qfin[0] -= Strajs[I].sT * s_in[I];
      Qfin[1] -= Strajs[I].sT * s_out[I];
      Qfin[2] -= Strajs[I].sT;
      Strajs[I] = F.generate_new_traj(&Graph[I], &Strajs, hin * s_in[I], Qfin[1] > Qstar ? 0 : hout * s_out[I], I);
      Qfin[0] += Strajs[I].sT * s_in[I];
      Qfin[1] += Strajs[I].sT * s_out[I];
      Qfin[2] += Strajs[I].sT;
    }
    // end
  }

  integratedMeasuringFile.close();
  storyFile.close();
  logfile.close();

  // END of the proper MC simulation

  return 0;
}
