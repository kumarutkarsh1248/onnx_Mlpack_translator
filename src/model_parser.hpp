#ifndef MODEL_PARSER_HPP
#define MODEL_PARSER_HPP

#include "mlpack.hpp"
#include <iostream>
#include <string>

using namespace mlpack;
using namespace std;

/**
 * Determine the layer type to be added to a feedforward network given a
 * string containing the type and a map containing the parameters
 *
 * @param layerType Type of layer that is to be defined
 * @param layerParams Map containing the parameters of the layer to be defined
 * @return A pointer to Layer<> object that is of the given type and is
 * initialized by the given parameters
 */
Layer<> *getNetworkReference(const std::string &layerType,
                             std::map<std::string, double> &layerParams);

/**
 * Update the values of a given stl map with that of another map
 * corresponding to the keys that are common
 *
 * Keys are of type string and values are of type double
 *
 * @param origParams The map whose values will be updated
 * @param newParams The map whose values will be used to update origParams
 */
void updateParams(std::map<std::string, double> &origParams,
                  std::map<std::string, double> &newParams);

/**
 * getLayer function calls the getNetworkRefrence function with layer
 * parameters, one of the layer parameter, parameterType is in encoded
 * form and to decode it we use this function
 * 0 : default type (padding value needs to be explicitely specified)
 * 1 : valid (no padding)
 * 2 : same ( padding should be such that there is no change in dimension)
*/
std::string decodePadType(double val);

#include "model_parser_impl.hpp"
#endif