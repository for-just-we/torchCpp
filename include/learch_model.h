//
// Created by prophe on 2022/10/20.
//

#ifndef TORCHCPP_LEARCH_MODEL_H
#define TORCHCPP_LEARCH_MODEL_H

#include <torch/torch.h>

class StandardScaler : torch::nn::Module {
private:
    torch::Tensor mean, scale;
public:
    StandardScaler(int64_t input_dim){
        mean = register_parameter("mean", torch::ones(input_dim));
        scale = register_parameter("scale", torch::ones(input_dim));
    }

    torch::Tensor forward(torch::Tensor x) {
        return torch::div(torch::sub(x, mean), scale);
    }
};

// in learch input_dim=47, hidden_dim=64
class PolicyFeedforward : torch::nn::Module {
private:
    StandardScaler standardScaler;
    torch::nn::Sequential net;
public:
    PolicyFeedforward(int64_t input_dim, int64_t hidden_dim): standardScaler(input_dim){
        net->push_back(torch::nn::Linear(input_dim, hidden_dim));
        net->push_back(torch::nn::ReLU(torch::nn::ReLUOptions(true)));
        net->push_back(torch::nn::Linear(hidden_dim, hidden_dim));
        net->push_back(torch::nn::ReLU(torch::nn::ReLUOptions(true)));
        net->push_back(torch::nn::Linear(hidden_dim, 1));
        net = register_module("net",net);
    }

    torch::Tensor forward(torch::Tensor x) {
        return net->forward(standardScaler.forward(x));
    }
};

#endif //TORCHCPP_LEARCH_MODEL_H
