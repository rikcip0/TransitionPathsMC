#include <iostream>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <chrono>
#include <ctime>
#include <vector>
#include <sys/stat.h>
#include <filesystem>

#ifndef FILE_SYSTEM_UTIL_H
#define FILE_SYSTEM_UTIL_H

#ifdef _WIN32
#include <direct.h>
#define CREATE_DIR(dir) mkdir(dir)
#else
#define CREATE_DIR(dir) mkdir(dir, 0777)
#endif

using std::cerr;
using std::cout;
using std::endl;
using std::string;
using std::to_string;
using std::vector;
namespace fs = std::filesystem;

std::string getCurrentDateTime()
{
  // Get the current time point
  auto now = std::chrono::system_clock::now();

  // Convert the time point to a time_t object
  std::time_t currentTime = std::chrono::system_clock::to_time_t(now);

  // Convert the time_t to a tm struct
  std::tm *localTime = std::localtime(&currentTime);

  // Extract individual components of the date and time
  int year = localTime->tm_year + 1900; // tm_year represents years since 1900
  int month = localTime->tm_mon + 1;    // tm_mon is 0-based
  int day = localTime->tm_mday;
  int hour = localTime->tm_hour;
  int minute = localTime->tm_min;
  int second = localTime->tm_sec;

  // Create a string stream to build the date-time string
  std::stringstream dateTimeStream;
  dateTimeStream << year << '-' << month << '-' << day << ' ' << hour << ':' << minute << ':' << second;

  // Return the date-time string
  return dateTimeStream.str();
}

void createOneFolder(const string &folderName)
{
  if (!(CREATE_DIR(folderName.c_str()) == 0 || errno == EEXIST))
  {
    cerr << "Error creating folder '" << folderName << "': " << (errno) << endl;
    exit(1);
  }
}

void createFolder(const string &folderPath)
{
  std::istringstream iss(folderPath);
  string folder;
  std::vector<string> folders;

  // Split the path into individual folder names
  while (std::getline(iss, folder, '/'))
  {
    if (!folder.empty())
    {
      folders.push_back(folder);
    }
  }

  // Create folders recursively
  string currentPath;
  for (const auto &folderName : folders)
  {
    currentPath += folderName;
    createOneFolder(currentPath);
    currentPath += "/";
  }
  cout << "Folder created successfully or already exists." << endl;
}

bool folderExists(const std::string &folderPath)
{
  return fs::is_directory(folderPath);
}

std::vector<std::string> getFolders(const std::string &folderPath, int depth = 1, string wordToInclude = "")
{
  std::vector<std::string> folders;

  if (!(fs::exists(folderPath) && fs::is_directory(folderPath)))
  {
    cout << "Folder " << folderPath << " does not exist!" << endl;
    return vector<string>(0);
  }

  for (const auto &entry : fs::directory_iterator(folderPath))
  {
    if (fs::is_directory(entry.path()))
    {
      folders.push_back(folderPath + "/" + entry.path().filename().string());
    }
  }
  vector<string> foldersToReturn;
  if (depth != 1)
    for (int i = 0; i < folders.size(); i++)
    {
      vector<string> subFolders = getFolders(folders[i], depth - 1, wordToInclude);
      foldersToReturn.insert(foldersToReturn.end(), subFolders.begin(), subFolders.end());
    }
  else
    for (int i = 0; i < folders.size(); i++)
      if (folders[i].find(wordToInclude) != std::string::npos)
        foldersToReturn.push_back(folders[i]);

  return foldersToReturn;
}

string makeFolderNameFromBuffer(string folder, string buffer)
{
  int i = 0;
  do
    i++;
  while (folderExists(folder + string(buffer) + "_run" + to_string(i)));
  folder += string(buffer) + "_run" + to_string(i);
  return folder;
}

string makeFolderNameFromBuffer_ForCluster(string folder, string buffer, int timeOfStart)
{
  folder += string(buffer) + "_run" + to_string(timeOfStart % 10000);
  return folder;
}

void error(string string)
{
  cout << "ERROR: " << string << endl;
  exit(EXIT_FAILURE);
}

// implementazione in corso::INIZIO
double getFFromGraphPath(string graphPath)
{
  return 0.5;
}

bool getStructuresFromPCN(vector<string> &folderPathsToFind, int p, int C, int N)
{
  string folderPath;
  vector<string> matchingStructuresFolders;
  folderPath = "Data/RRG/p" + to_string(p) + "C" + to_string(C) + "/N" + to_string(N);

  matchingStructuresFolders = getFolders(folderPath);
  if (matchingStructuresFolders.empty())
  {
    cout << "No matching structure found for p, C, N=" << p << ", " << C << ", " << N << endl;
    return false;
  }
  else
  {
    cout << "The following matching structures (" << matchingStructuresFolders.size() << ") have been found" << endl;
    for (string path : matchingStructuresFolders)
    {
      cout << path << endl;
    }
    cout << endl;

    folderPathsToFind = matchingStructuresFolders;
    return true;
  }
}

bool getGraphsFromPCNf(vector<string> &folderPathsToFind, int p, int C, int N, double f)
{
  string folderPath;
  vector<string> matchingStructuresFolders;
  vector<string> matchingGraphsFolders;
  folderPath = "Data/RRG/p" + to_string(p) + "C" + to_string(C) + "/N" + to_string(N);

  matchingStructuresFolders = getFolders(folderPath);
  if (matchingStructuresFolders.empty())
  {
    cout << "No matching structure found for p, C, N=" << p << ", " << C << ", " << N << endl;
    return false;
  }

  for (string path : matchingStructuresFolders)
  {
    vector<string> newFoundGraphs;
    folderPath = path + "/fPosJ" + to_string(f).substr(0, 4);
    if (fs::exists(folderPath) && fs::is_directory(folderPath))
    {
      newFoundGraphs = getFolders(folderPath);
      matchingGraphsFolders.insert(matchingGraphsFolders.end(), newFoundGraphs.begin(), newFoundGraphs.end());
    }
  }

  if (matchingGraphsFolders.empty())
  {
    cout << "Over all structures with given p, C and N, none shows a graph with fPosJ " << f << endl;
    return false;
  }
  else
  {
    cout << "The following matching graphs (" << matchingGraphsFolders.size() << ") have been found" << endl;
    for (string path : matchingGraphsFolders)
    {
      cout << path << endl;
    }
    cout << endl;

    folderPathsToFind = matchingGraphsFolders;
    return true;
  }
}

bool getStructuresFromStructureID(vector<string> &folderPathsToFind, int p = 0, int C = 0, int N = 0, double f = -1.)
{
  return true;
}

// returns all graphs of a certain structure.
// if f is included as argument, takes only graphs with that f.
// If p, C, N are included as arguments, also check that they match with those of the graph.
bool getGraphsFromStructureID(vector<string> &folderPathsToFind, int sID, double f = -1., int p = 0, int C = 0, int N = 0)
{
  vector<string> matchingStructuresFolders;
  vector<string> matchingGraphsFolders;

  getStructuresFromStructureID(matchingStructuresFolders, sID, p, C, N);

  if (matchingStructuresFolders.empty())
  {
    cout << "No matching structure found for ID " << sID << endl;
    return false;
  }

  for (string path : matchingStructuresFolders)
  {
    vector<string> newFoundGraphs;
    newFoundGraphs = getFolders(path);
    for (string graphPath : newFoundGraphs)
      if (f = -1. || getFFromGraphPath(graphPath) == f)
        matchingGraphsFolders.push_back(graphPath);
  }

  if (matchingGraphsFolders.empty())
  {
    if (f != -1.)
      cout << "Over all graphs from structure " << sID << " none shows a graph with fPosJ " << f << "." << endl;
    else
      cout << "A structure with ID " << sID << " has been found, but seems to contain no graph." << endl;
    return false;
  }
  else
  {

    cout << "The following matching graphs (" << matchingGraphsFolders.size() << ") have been found";
    cout << "for structure, f =" << sID << ", " << f << endl;
    for (string path : matchingGraphsFolders)
    {
      cout << path << endl;
    }
    cout << endl;

    folderPathsToFind = matchingGraphsFolders;
    return true;
  }
  return true;
}

bool getStructureFromGraphID(string &folderPathToFind, int gID, int p = 0, int C = 0, int N = 0, double f = -1.)
{
  return true;
}

bool getGraphFromGraphID(string &folderPathToFind, int graphID, int p = 0, int C = 0, int N = 0, double f = -1.)
{
  vector<string> compatibleFilePaths;
  string toSearch = "../Data/Graphs";
  compatibleFilePaths = getFolders(toSearch, 6, "graph" + to_string(graphID));
  if (compatibleFilePaths.size() > 1)
  {
    cout << "More than one graph with ID " << graphID << " found." << endl;
    return false;
  }
  else if (compatibleFilePaths.size() == 0)
  {
    cout << "No graph with ID " << graphID << " found." << endl;
    return false;
  }
  else
  {
    folderPathToFind = compatibleFilePaths[0];
    cout << "Found graph" << folderPathToFind << endl;
  }

  compatibleFilePaths.resize(0);
  if (p != 0 && C != 0)
  {
    toSearch += "/p" + to_string(p) + "C" + to_string(C);
    if (N == 0)
    {
      compatibleFilePaths = getFolders(toSearch, 4, to_string(graphID));
    }
    else
    {
      toSearch += "/N" + to_string(N);
      if (f == -1.)
        compatibleFilePaths = getFolders(toSearch, 3, to_string(graphID));
      else
      {
        vector<string> structuresFolders = getFolders(toSearch, 1);
        for (int i = 0; i < structuresFolders.size(); i++)
        {
          toSearch = structuresFolders[i] + "/fPosJ" + to_string(f).substr(0, 4);
          vector<string> foundFolders = getFolders(toSearch, 1, to_string(graphID));
          compatibleFilePaths.insert(compatibleFilePaths.end(), foundFolders.begin(), foundFolders.end());
        }
      }
    }

    for (int i = 0; i < compatibleFilePaths.size(); i++)
      cout << compatibleFilePaths[i] << endl;

    if (std::find(compatibleFilePaths.begin(), compatibleFilePaths.end(), folderPathToFind) == compatibleFilePaths.end())
    {
      cout << "Found graph does not match specified params." << endl;
      return false;
    }
  }

  return true;
}
// implementazione in corso::FINE

bool getGraphFromGraphAndStructureID_ForCluster(int type, string &folderPathToFind, int structureID, int graphID, int p, int C, int N, double f)
{
  vector<string> compatibleFilePaths;
  string toSearch;
  if (type < 0)
  {
    if (type == -3)
      toSearch = "../Data/Graphs/DPRRG";
    else if (type == -2)
      toSearch = "../Data/Graphs/ER";
    else if (type == -1)
      toSearch = "../Data/Graphs/RRG";
    folderPathToFind = toSearch + "/p" + to_string(p) + "C" + to_string(C) + "/N" + to_string(N) + "/structure" + to_string(structureID) + "/fPosJ" + to_string(f).substr(0, 4) + "/graph" + to_string(graphID);
  }
  else if (type == 1)
  {
    toSearch = "../Data/Graphs/1dChain";
    folderPathToFind = toSearch + "/N" + to_string(N) + "/fPosJ" + to_string(f).substr(0, 4) + "/graph" + to_string(graphID);
  }
  else
  {
    toSearch = "../Data/Graphs/SqLatt/" + to_string(type) + "d";
    folderPathToFind = toSearch + "/N" + to_string(N) + "/fPosJ" + to_string(f).substr(0, 4) + "/graph" + to_string(graphID);
  }

  return true;
}

bool initializeArrayFromLine(const string &fileName, int targetLine, int arraySize, int *targetArray)
{
  // Apri il file in lettura
  std::ifstream file(fileName);

  // Verifica che il file sia stato aperto correttamente
  if (!file.is_open())
  {
    cerr << "Impossibile aprire il file." << endl;
    return false;
  }

  // Variabile per contenere il contenuto della riga
  string line;

  // Vai alla riga desiderata
  for (int i = 0; i < targetLine; ++i)
  {
    if (!std::getline(file, line))
    {
      // La riga richiesta potrebbe non esistere nel file
      cerr << "Il file ha meno di " << targetLine << " righe." << endl;
      return false;
    }
  }

  // Leggi e analizza la riga desiderata
  if (std::getline(file, line))
  {
    // Usa un stringstream per estrarre i valori dalla riga
    std::istringstream iss(line);

    for (int i = 0; i < arraySize; ++i)
    {
      if (!(iss >> targetArray[i]))
      {
        // Errore nell'estrazione dei valori
        cerr << "Errore nell'estrazione dei valori dalla riga." << endl;
        return false;
      }
    }

    // Chiusura del file
    file.close();

    return true;
  }
  else
  {
    cerr << "Errore nella lettura della riga." << endl;
    return false;
  }
}

int countNumbersInRow(const std::string &row)
{
  std::istringstream iss(row);
  int count = 0;
  int num;
  while (iss >> num)
  {
    count++;
  }
  return count;
}

int countLines(const string &fileName)
{
  // Apri il file in lettura
  std::ifstream file(fileName);

  // Verifica che il file sia stato aperto correttamente
  if (!file.is_open())
  {
    cerr << "Impossibile aprire il file." << endl;
    return false;
  }

  int nLines = 0;

  std::string line;
  while (std::getline(file, line))
  {
    nLines++;
  }

  // Chiusura del file
  file.close();

  return nLines;
}

bool initializeVectorFromLine(const string &fileName, int targetLine, int arraySize, vector<int> &targetArray)
{
  targetArray.resize(arraySize);
  // Apri il file in lettura
  std::ifstream file(fileName);

  // Verifica che il file sia stato aperto correttamente
  if (!file.is_open())
  {
    cerr << "Impossibile aprire il file." << endl;
    return false;
  }

  // Variabile per contenere il contenuto della riga
  string line;

  // Vai alla riga desiderata
  for (int i = 0; i < targetLine; ++i)
  {
    if (!std::getline(file, line))
    {
      // La riga richiesta potrebbe non esistere nel file
      cerr << "Il file ha meno di " << targetLine << " righe." << endl;
      return false;
    }
  }

  // Leggi e analizza la riga desiderata
  if (std::getline(file, line))
  {
    // Usa un stringstream per estrarre i valori dalla riga
    std::istringstream iss(line);
    for (int i = 0; i < arraySize; ++i)
    {
      if (!(iss >> targetArray[i]))
      {
        // Errore nell'estrazione dei valori
        cerr << "Errore nell'estrazione dei valori dalla riga." << endl;
        return false;
      }
    }

    // Chiusura del file
    file.close();

    return true;
  }
  else
  {
    cerr << "Errore nella lettura della riga." << endl;
    return false;
  }
}

#endif