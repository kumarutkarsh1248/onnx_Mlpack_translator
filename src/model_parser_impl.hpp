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
  else if (layerType == "convolution")
  {
    origParams["maps"] = NAN;
    origParams["kw"] = NAN;
    origParams["kh"] = NAN;
    origParams["dw"] = 1;
    origParams["dh"] = 1;
    origParams["padw"] = 0;
    origParams["padH"] = 0;
    origParams["paddingtype"] = 0; // None = 0 , Valid = 1, Same = 2
    updateParams(origParams, layerParams);
    std::string padding = decodePadType(origParams["paddingtype"]);
    layer = new Convolution(origParams["maps"], origParams["kw"],
                              origParams["kh"], origParams["dw"],origParams["dh"], 
                              origParams["padw"], origParams["padh"],padding);
  }
  else if (layerType == "maxpooling")
  {
    origParams["kw"] = NAN;
    origParams["kh"] = NAN;
    origParams["dw"] = NAN;
    origParams["dh"] = NAN;
    updateParams(origParams, layerParams);
    layer = new MaxPooling(origParams["kw"],
                            origParams["kh"],
                            origParams["dw"],
                            origParams["dh"]);
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
  else if (layerType == "reshape")
  {
    layer = new Identity();
  }
  else
  {
    Log::Fatal << "Invalid layer type : " << layerType;
    cout << "\n";
  }
  cout<<"********added "<<layerType<<"\n\n"<<endl;
  return layer;
}

std::string decodePadType(double val)
{
  if (val==0)
    return "None";
  else if (val==1)
    return "Valid";
  else
    return "Same";
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