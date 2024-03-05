#include "model_parser.hpp"

Layer<> *getNetworkReference(const std::string &layerType,
                             std::map<std::string, double> &layerParams)
{
  std::map<std::string, double> origParams;
  Layer<> *layer;

  if (layerType == "linearnobias")
  {
    NoRegularizer regularizer;
    origParams["outsize"] = NAN;
    updateParams(origParams, layerParams);
    layer = new LinearNoBias(origParams["outsize"], regularizer);
  }
  // bias addition layer
  else if (layerType == "add")
  {
    layer = new Add();
  }
  else if (layerType == "relu")
  {
    layer = new LeakyReLU(0);
  }
  else if (layerType == "softmax")
  {
    layer = new Softmax();
  }
  else if (layerType == "identity")
  {
    layer = new Identity();
  }
  else
  {
    Log::Fatal << "Invalid layer type : " << layerType;
    cout << "\n";
  }

  return layer;
}

template <typename T>
void printVector(const std::vector<T> &vec)
{
  for (const auto &element : vec)
  {
    std::cout << element << " ";
  }
  std::cout << std::endl;
}

void printMap(std::map<std::string, double> params)
{
  std::map<std::string, double>::iterator itr;
  for (itr = params.begin(); itr != params.end(); ++itr)
  {
    Log::Info << itr->first << " : " << itr->second << "\n";
  }
}

void updateParams(std::map<std::string, double> &origParams,
                  std::map<std::string, double> &newParams)
{
  std::map<std::string, double>::iterator itr;
  bool error = false;
  for (itr = origParams.begin(); itr != origParams.end(); ++itr)
  {
    std::map<std::string, double>::iterator itr2 = newParams.find(itr->first);
    if (itr2 == newParams.end() && isnan(itr->second))
    {
      Log::Info << "Required parameter: " << itr->first << "\n";
      error = true;
    }
    else if (itr2 != newParams.end())
      itr->second = newParams.at(itr->first);
  }
  if (error)
    Log::Fatal << "Required parameters missing"
               << "\n";
}
