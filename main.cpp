#include <iostream>
#include <torch/script.h>
#include <vector>
#include "learch_model.h"

using namespace std;

int main(int argc, char **argv) {
    // The second argument is torch model path
    assert(argc == 2);
    std::string modelPath(argv[1]);
    std::cout << "=================================" << std::endl;
    StandardScaler standardScaler(47);
    // transform input features from vector to torch Tensor
    std::vector<double> in(47, 1);
    torch::Tensor vec = torch::from_blob(in.data(), {47}, torch::kFloat);
    std::cout << vec << std::endl;

    // first way to load model in black-box way
    std::vector<torch::jit::IValue> input;
    input.push_back(vec.reshape({1, 47}));
    torch::jit::script::Module module = torch::jit::load(modelPath);
    // predict
    torch::Tensor out = module.forward(input).toTensor();
    std::cout << out << endl;
    std::cout << "======================" << endl;
    double reward = out.item().toDouble();
    std::cout << reward << endl;
    std::cout << "======================" << endl;

    // other way to load model, to be supplemented
//    PolicyFeedforward feedforward(47, 64);
//    torch::Tensor output = feedforward.forward(vec);
//    std::cout << output << std::endl;
//    reward = output.item().toDouble();
//    std::cout << reward << std::endl;
    return 0;
}
