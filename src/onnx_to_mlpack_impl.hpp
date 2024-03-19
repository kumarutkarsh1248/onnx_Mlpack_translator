#include "onnx_to_mlpack.hpp"
using namespace std;

// FFN<> generateModel(GraphProto &graph)
// {
//     FFN<> ffn;
//     size_t inputShape = graph.input(0).type().tensor_type().shape().dim(1).dim_value();
//     size_t outputShape = graph.output(0).type().tensor_type().shape().dim(1).dim_value();

//     string input_name = graph.input(0).name();

//     vector<size_t> v = {inputShape};
//     ffn.InputDimensions() = v;
//     cout << "\ninput shape :: " << inputShape << " output shape ::" << outputShape << endl;

//     for (auto nodeItr = std::begin(graph.node()); nodeItr != std::end(graph.node()); ++nodeItr)
//     {
//         string nodeType = nodeItr->op_type();
//         ffn.Add(getLayer(*nodeItr, nodeType, findOutputDimension(graph, *nodeItr)));
//     }
//     ffn.Reset();
//     return ffn;
// }

FFN<> generateModel(GraphProto &graph)
{
    FFN<> ffn;
    string userInput = modelInput(graph);
    vector<size_t> v = findModelInputDimension(graph, userInput);
    ffn.InputDimensions() = v;
    string nodeInput = userInput;

    while (nextNode(graph, nodeInput).op_type() != "")
    {
        NodeProto node = nextNode(graph, nodeInput);
        nodeInput = node.output(0);
        string nodeType = node.op_type();
        ffn.Add(getLayer(graph, node, nodeType));
    }
    ffn.Reset();
    return ffn;
}

Layer<> *getLayer(GraphProto graph, const NodeProto &node, string layerType)
{
    // keys => ONNX operator name
    // values => mlpack operator name
    map<string, string> operatorMap = {
        {"Conv", "convolution"},
        {"Reshape", "reshape"},
        {"MaxPool", "maxpooling"},
        {"MatMul", "linearnobias"},
        {"Add", "add"},
        {"Relu", "relu"},
        {"Softmax", "softmax"},
        {"Identity", "identity"}
    };

    // ONNX layer attribute
    map<string, map<string, vector<string>>> onnxAttributes;
    onnxAttributes["Conv"] = {
        {"kernel_shape", {"kh", "kw"}},
        {"pads", {"padh", "padw"}},
        {"strides", {"dh", "dw"}}};
    onnxAttributes["MaxPool"] = {
        {"kernel_shape", {"kh", "kw"}},
        {"strides", {"dh", "dw"}}};
    onnxAttributes["MatMul"];
    onnxAttributes["LeakyRelu"] = {
        {"alpha", {"alpha"}}};
    onnxAttributes["Identity"];
    onnxAttributes["Relu"];

    map<string, vector<string>> onnxLayerAttributes = onnxAttributes[layerType];
    map<string, double> mlpackLayerAttribute;

    // special layer parameters specific to given type of layer
    if (node.op_type() == "MatMul")
    {
        mlpackLayerAttribute["outsize"] = findOutputDimension(graph, node);
    }
    if (node.op_type() == "Conv")
    {
        mlpackLayerAttribute["maps"] = findConvMap(graph, node);
    }

    for (AttributeProto attribute : node.attribute())
    {
        string attrName = attribute.name();
        // attribute of that specific layer
        vector<string> attr = onnxLayerAttributes[attrName];
        vector<string>::iterator itr;

        // checking for special cases
        if (attrName == "pads")
        {
            //[0 1 2 3] indices are top, bottom, left, right respectively
            mlpackLayerAttribute["padw"] = (int)(attribute.ints(1) + attribute.ints(3)) / 2;
            mlpackLayerAttribute["padh"] = (int)(attribute.ints(0) + attribute.ints(2)) / 2;
        }
        // else if (attrName == "auto_pad")
        // {
        //     // P = ((S-1)*W-S+F)/2
        //     // It calculates symmetric padding, meaning the same amount of padding is added on both sides.
        //     skippedAttributes.push_back("auto_pad_" + attribute.s());   //will be like auto_pad_sameupper
        // }
        else if (attrName == "auto_pad")
        {
            if (attribute.s() == "SAME_UPPER" || attribute.s() == "SAME_LOWER")
            {
                mlpackLayerAttribute["paddingType"] = 0; // same
            }
            else if (attribute.s() == "VALID")
            {
                mlpackLayerAttribute["paddingType"] = 1; // valid
            }
            else if (attribute.s() == "NOTSET")
            {
                mlpackLayerAttribute["paddingType"] = 2; // none
            }
        }

        int i = 0;
        for (itr = attr.begin(); itr < attr.end(); ++itr, ++i)
        {
            if (attribute.type() == attribute.INT)
            {
                mlpackLayerAttribute[*itr] = attribute.i();
            }
            else if (attribute.type() == attribute.INTS)
            {
                mlpackLayerAttribute[*itr] = attribute.ints()[i];
            }
            else if (attribute.type() == attribute.FLOAT)
            {
                mlpackLayerAttribute[*itr] = attribute.f();
            }
            else if (attribute.type() == attribute.FLOATS)
            {
                mlpackLayerAttribute[*itr] = attribute.floats()[i];
            }
        }
    }
    cout << "***********layer parameters****************" << endl;
    for (auto element : mlpackLayerAttribute)
    {
        cout << element.first << " " << element.second << endl;
    }
    return getNetworkReference(operatorMap[layerType], mlpackLayerAttribute);
}

// int findOutputDimension(const GraphProto &graph, const NodeProto &node)
// {
//     int node_total_inputs = node.input().size();
//     int dim;
//     for (auto initializerItr = std::begin(graph.initializer());
//          initializerItr != std::end(graph.initializer()); ++initializerItr)
//     {
//         if (node.input(node_total_inputs - 1) == initializerItr->name())
//         {
//             int total_dims = initializerItr->dims().size();
//             dim = initializerItr->dims(total_dims - 1);
//         }
//     }
//     return dim;
// }

// output dimension parameters is needed only for matmul
int findOutputDimension(const GraphProto &graph, const NodeProto &node)
{
    int dim;
    string layerOutput = node.output(0);
    NodeProto nextNod = nextNode(graph, layerOutput);
    for (auto initializer : graph.initializer())
    {
        if (initializer.name() == nextNod.input(1))
        {
            dim = initializer.dims().size() > 1 ? initializer.dims(1) : initializer.dims(0);
        }
    }
    return dim;
}

string modelInput(const GraphProto &graph)
{
    vector<string> inputNames;
    vector<string> initializerNames;
    for (auto input : graph.input())
    {
        inputNames.push_back(input.name());
    }
    for (auto initializer : graph.initializer())
    {
        initializerNames.push_back(initializer.name());
    }

    for (const auto &element : inputNames)
    {
        if (std::find(initializerNames.begin(), initializerNames.end(), element) == initializerNames.end())
        {
            return element;
        }
    }
    return "all elements found";
}

NodeProto nextNode(const GraphProto &graph, string preNodeOutput)
{
    NodeProto node;

    for (auto nodeItr = std::begin(graph.node()); nodeItr != std::end(graph.node()); ++nodeItr)
    {
        for (int i = 0; i < nodeItr->input().size(); i++)
        {
            if (preNodeOutput == nodeItr->input(i))
            {
                return *nodeItr;
            }
        }
    }
    return node;
}

vector<size_t> findModelInputDimension(GraphProto graph, string input_string)
{
    vector<size_t> dimension;
    for (auto input : graph.input())
    {
        if (input.name() == input_string)
        {
            int dim_size = input.type().tensor_type().shape().dim().size();
            for (int i = 1; i < dim_size; i++)
            {
                dimension.push_back(input.type().tensor_type().shape().dim(i).dim_value());
            }
        }
    }
    return dimension;
}

int findConvMap(GraphProto graph, NodeProto node)
{
    int maps;
    string convWeights = node.input(1);
    for (auto initializer : graph.initializer())
    {
        if (initializer.name() == convWeights)
        {
            maps = initializer.dims(0);
        }
    }
    return maps;
}