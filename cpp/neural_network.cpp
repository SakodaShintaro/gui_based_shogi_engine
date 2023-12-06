#include "neural_network.hpp"

NeuralNetworkImpl::NeuralNetworkImpl()
{
  using namespace torch::nn;
  conv1_ = register_module("conv1", Conv2d(Conv2dOptions(3, 32, 8).stride(4)));
  conv2_ = register_module("conv2", Conv2d(Conv2dOptions(32, 64, 4).stride(2)));
  conv3_ = register_module("conv3", Conv2d(Conv2dOptions(64, 64, 3).stride(1)));
  conv4_ = register_module("conv4", Conv2d(Conv2dOptions(64, 64, 3).stride(1)));
  conv5_ = register_module("conv5", Conv2d(Conv2dOptions(64, 64, 3).stride(1)));
  fc1_ = register_module("fc1", Linear(64 * 7 * 7, 512));
  fc2_ = register_module("fc2", Linear(512, 512));
  fc3_ = register_module("fc3", Linear(512, 4));
}

torch::Tensor NeuralNetworkImpl::forward(torch::Tensor input)
{
  torch::Tensor x = input;
  return x;
}
