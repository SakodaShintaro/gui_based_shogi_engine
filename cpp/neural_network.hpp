#ifndef NEURAL_NETWORK_HPP_
#define NEURAL_NETWORK_HPP_

#include <torch/torch.h>

// 画像をエンコードするニューラルネットワーク
class ImageEncoderImpl : public torch::nn::Module
{
public:
  ImageEncoderImpl() = default;
  ImageEncoderImpl(int64_t h, int64_t w);

  torch::Tensor forward(torch::Tensor image);

private:
  static constexpr int64_t conv_layer_num = 4;
  std::vector<torch::nn::Conv2d> conv_layers_{conv_layer_num, nullptr};
  torch::nn::Linear linear_ = nullptr;
};
TORCH_MODULE(ImageEncoder);

// Actor:画像を入力として行動を出力するニューラルネットワーク
class ActorImpl : public torch::nn::Module
{
public:
  ActorImpl() = default;
  ActorImpl(int64_t h, int64_t w);

  torch::Tensor forward(torch::Tensor image);

private:
  ImageEncoder image_encoder_ = nullptr;
  torch::nn::Linear linear_ = nullptr;
};
TORCH_MODULE(Actor);

// Critic:画像と行動を入力として価値を出力するニューラルネットワーク
class CriticImpl : public torch::nn::Module
{
public:
  CriticImpl() = default;
  CriticImpl(int64_t h, int64_t w);

  torch::Tensor forward(torch::Tensor image, torch::Tensor action);

private:
  ImageEncoder image_encoder_ = nullptr;
  torch::nn::Linear linear_ = nullptr;
};
TORCH_MODULE(Critic);

#endif  // NEURAL_NETWORK_HPP_
