#define MLPACK_PRINT_DEBUG
#define MLPACK_PRINT_INFO
#define MLPACK_PRINT_WARN
 
#include "onnx_pb.h"
#include "onnx_to_mlpack.hpp"
#include <fstream>
#include <iostream>
#include <memory>
#include <map>
#include <string>
#include "mlpack.hpp"

using namespace onnx;
using namespace mlpack;
using namespace ann;
using namespace std;


int main()
{
    onnx::ModelProto onnxModel;
    std::ifstream in("onnx_linear_model.onnx", std::ios_base::binary);
    onnxModel.ParseFromIstream(&in);
    in.close();

    onnx::GraphProto graph = onnxModel.graph();

    FFN<> ffn = generateModel(graph);

    return 0;
}