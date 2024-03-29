#ifndef ONNX_TO_MLPACK_hpp
#define ONNX_TO_MLPACK_hpp

#include "mlpack.hpp"
#include <iostream>
#include <string>
#include "onnx_pb.h"
#include "model_parser.hpp"

using namespace onnx;
using namespace mlpack;
using namespace std;

/**
 * Get the mlpack layer associated with the given layer type
 * instantiated with the given parameters
 *
 * @param node The ONNX node containing the layer attributes
 * @param layer The name of the ONNX operator
 * @param dimParams The map containing information about the
 * dimensions of the layer
 * @return
 */
// Layer<> *getLayer(const NodeProto &node, string layerType, int outSize);
Layer<> *getLayer(GraphProto graph, const NodeProto &node, string layerType);

/**
 * Get the input size of each layer and the output size of the last layer in
 * a vector
 *
 * @param weights The data structure containing the weights and biases
 * of the ONNX model
 * @return The number of nodes in each layer
 */
std::vector<int> findWeightDims
    // const ::google::protobuf::RepeatedPtrField => when multiple repeated message is to be hold
    (const ::google::protobuf::RepeatedPtrField<::onnx::TensorProto> &weights);

/**
 * Transfer the weights of the ONNX model to the mlpack model
 *
 * @param graph The ONNX graph containing all the weights and layer details
 * @param weightMatrix The matrix containing the mlpack model's weights
 * @return
 */
void extractWeights(GraphProto &graph, arma::mat &weightMatrix);

/**
 * more detailed
 * Get the mlpack equivalent model of a given ONNX model without
 * the transfer of weights
 *
 * @param graph The ONNX graph containing all the layer details
 * @return An mlpack FFN model corresponding to the ONNX model passed
 */
FFN<> generateModel(GraphProto &graph);

/**
 * Give the output size of the node (only for matmul)
 * in case of relu it will give the outsize = 0
 *
 * @return output size of node in int
 */
int findOutputDimension(const GraphProto &graph, const NodeProto &node);

/**
 * return the name of the input that onnx model is taking
 * with the use this input name the first node can be found
 */
string modelInput(const GraphProto &graph);

/**
 * Finds the node in a graph which takes the given input,
 * where the input is the output generated by its previous node.
 */
NodeProto nextNode(const GraphProto &graph, string prevNodeOutput);

/**
 * (will return the dimension of the input which our model can take)
 * dimension of user input
 */
vector<size_t> findModelInputDimension(GraphProto graph, string input_string);

/**
 * Returns the dimensions of the input accepted by our model.
 */
int findConvMap(GraphProto graph, NodeProto node);

#include "onnx_to_mlpack_impl.hpp"

#endif