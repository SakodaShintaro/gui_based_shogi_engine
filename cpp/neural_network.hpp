#ifndef NEURAL_NETWORK_HPP_
#define NEURAL_NETWORK_HPP_

#include <torch/torch.h>

// [上, 右, 下, 左, クリック]
enum Action { kUp, kRight, kDown, kLeft, kClick, kActionSize };

// 画像を入力として行動を出力するニューラルネットワーク
class NeuralNetworkImpl : public torch::nn::Module
{
public:
  NeuralNetworkImpl() = default;
  NeuralNetworkImpl(int64_t h, int64_t w);

  // 画像を入力として行動を出力する
  torch::Tensor forward(torch::Tensor input);

private:
  static constexpr int64_t conv_layer_num = 5;
  // 畳み込み層
  std::vector<torch::nn::Conv2d> conv_layers_{conv_layer_num, nullptr};

  // 全結合層
  torch::nn::Linear fc1_ = nullptr;
  torch::nn::Linear fc2_ = nullptr;
  torch::nn::Linear fc3_ = nullptr;
};

TORCH_MODULE(NeuralNetwork);

#endif  // NEURAL_NETWORK_HPP_
