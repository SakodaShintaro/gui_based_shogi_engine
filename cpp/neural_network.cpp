#include "neural_network.hpp"

/*
CNNの出力サイズ計算(https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)
Input: (N, C_in, H_in, W_in)
Output: (N, C_out, H_out, W_out)
where
H_out = floor((H_in + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1)
W_out = floor((W_in + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1)
ここで、簡単にするためにkernel_size = 3, stride = 1, dilation = 1で決め打つと
H_out = H_in + 2 * padding - 2
W_out = W_in + 2 * padding - 2
となる。

max_pool2dではkernel_size=K、ストライドをstride=Sとして
C' = C  (チャネル数は変わらない)
H' = (H - K) / S + 1
W' = (W - K) / S + 1
*/

NeuralNetworkImpl::NeuralNetworkImpl(int64_t h, int64_t w)
{
  using namespace torch::nn;

  const std::vector<std::pair<int64_t, int64_t>> in_out_channels = {
    {3, 32}, {32, 64}, {64, 64}, {64, 64}, {64, 64}};

  if (in_out_channels.size() != conv_layer_num) {
    std::cerr << "in_out_channels.size() != conv_layer_num" << std::endl;
    std::exit(1);
  }

  for (int64_t i = 0; i < conv_layer_num; i++) {
    const auto [in_channels, out_channels] = in_out_channels[i];

    // H,Wを変えないConv2D
    conv_layers_[i] = register_module(
      "conv" + std::to_string(i), Conv2d(Conv2dOptions(in_channels, out_channels, 3).padding(1)));

    // max_pool2dによってサイズが変わる
    h = (h - 2) / 2 + 1;
    w = (w - 2) / 2 + 1;
  }

  const int64_t last_out_ch = in_out_channels.back().second;

  fc1_ = register_module("fc1", Linear(last_out_ch * h * w, 512));
  fc2_ = register_module("fc2", Linear(512, 512));
  fc3_ = register_module("fc3", Linear(512, kActionSize));
}

torch::Tensor NeuralNetworkImpl::forward(torch::Tensor input)
{
  torch::Tensor x = input;  // [bs, 3, h, w]

  for (int64_t i = 0; i < conv_layer_num; i++) {
    x = conv_layers_[i]->forward(x);
    x = torch::relu(x);
    x = torch::max_pool2d(x, 2, 2);
  }

  x = x.flatten(1);                   // [bs, last_out_ch * h * w]
  x = torch::relu(fc1_->forward(x));  // [bs, 512]
  x = torch::relu(fc2_->forward(x));  // [bs, 512]
  x = fc3_->forward(x);               // [bs, kActionSize]
  return x;
}
