#ifndef NEURAL_NETWORK_HPP_
#define NEURAL_NETWORK_HPP_

#include <torch/torch.h>

// 画像を入力として行動を出力するニューラルネットワーク
class NeuralNetworkImpl : public torch::nn::Module
{
public:
  NeuralNetworkImpl();

  // 画像を入力として行動を出力する
  torch::Tensor forward(torch::Tensor input);

private:
  // 畳み込み層
  torch::nn::Conv2d conv1_ = nullptr;
  torch::nn::Conv2d conv2_ = nullptr;
  torch::nn::Conv2d conv3_ = nullptr;
  torch::nn::Conv2d conv4_ = nullptr;
  torch::nn::Conv2d conv5_ = nullptr;

  // 全結合層
  torch::nn::Linear fc1_ = nullptr;
  torch::nn::Linear fc2_ = nullptr;
  torch::nn::Linear fc3_ = nullptr;
};

TORCH_MODULE(NeuralNetwork);

#endif  // NEURAL_NETWORK_HPP_
