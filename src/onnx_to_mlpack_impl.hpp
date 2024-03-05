#include "onnx_to_mlpack.hpp"
using namespace std;

FFN<> generateModel(GraphProto &graph)
{
  FFN<> ffn;
  size_t inputShape = graph.input(0).type().tensor_type().shape().dim(1).dim_value();
  size_t outputShape = graph.output(0).type().tensor_type().shape().dim(1).dim_value();
  vector<size_t> v = {inputShape};
  ffn.InputDimensions() = v;
  cout << "\ninput shape :: " << inputShape << " output shape ::" << outputShape << endl;

  for (auto nodeItr = std::begin(graph.node()); nodeItr != std::end(graph.node()); ++nodeItr)
  {
    string nodeType = nodeItr->op_type();
    // -----------------------merging operation ---------------
    // Each node in the graph performs an operation. However,
    // some operations in ONNX are executed in two steps, whereas
    //  mlpack accomplishes them in a single step. Therefore, we
    //  need to merge nodes that represent operations that can be
    //  performed in a single step in mlpack. Currently, we are only
    //  addressing one such case.

    // std::map<std::vector<string>, string> mergeLayers = {
    //     {{"MatMul", "Add"}, "Transformed_linear"}};
    // // Iterating through the mergerLayer
    // for (auto iter = mergeLayers.begin(); iter != mergeLayers.end(); ++iter)
    // {
    //   std::vector<std::string> mergeVector = iter->first;
    //   if (nodeItr->op_type() == mergeVector[0] && (nodeItr + 1)->op_type() == mergeVector[1])
    //   {
    //     // now nodeItr will point to Add node
    //     nodeItr += 1;
    //     nodeType = iter->second;
    //   }
    // }
    // --------------------------------------------------------
    cout << "layer:: " << nodeType << endl;
    ffn.Add(getLayer(*nodeItr, nodeType, findOutputDimension(graph, *nodeItr)));
  }
  ffn.Reset();
  return ffn;
}

Layer<> *getLayer(const NodeProto &node, string layerType, int outSize)
// ,map<string, double>& dimParams
{
  // keys => ONNX operator name
  // values => mlpack operator name
  map<string, string> operatorMap = {
      {"MatMul", "linearnobias"},
      {"Add", "add"},
      {"Relu", "relu"},
      {"Softmax", "softmax"},
      {"Identity", "identity"}};

  map<string, double> layerparams{
      {"outsize", outSize}};

  return getNetworkReference(operatorMap[layerType], layerparams);
}

int findOutputDimension(const GraphProto &graph, const NodeProto &node)
{
  int node_total_inputs = node.input().size();
  for (auto initializerItr = std::begin(graph.initializer());
       initializerItr != std::end(graph.initializer()); ++initializerItr)
  {
    if (node.input(node_total_inputs - 1) == initializerItr->name())
    {
      int total_dims = initializerItr->dims().size();
      return initializerItr->dims(total_dims - 1);
    }
  }
  return 0;
}