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

template <typename T>
std::ostream &operator<<(std::ostream &out, const std::vector<T> &v)
{
  if (!v.empty())
  {
    out << '[';
    std::copy(v.begin(), v.end(), std::ostream_iterator<T>(out, ", "));
    out << "\b]";
  }
  return out;
}

template <typename T>
std::ostream &
operator<<(std::ostream &out, const std::vector<std::vector<T>> &v)
{
  if (!v.empty())
  {
    out << '[';
    for (int i = 0; i < v.size(); i++)
      std::copy(
          v.at(i).begin(), v.at(i).end(), std::ostream_iterator<T>(out, ","));
    out << "\b]";
  }
  return out;
}

template <typename T1, typename T2>
std::ostream &
operator<<(std::ostream &out, const std::map<T1, T2> &m)
{
  out << "{\n";
  for (auto it : m)
  {
    out << "  {" << it.first << " : " << it.second << "}\n";
  }
  out << "\b}";
  return out;
}