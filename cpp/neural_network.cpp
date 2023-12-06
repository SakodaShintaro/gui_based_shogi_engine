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

NeuralNetworkImpl::NeuralNetworkImpl()
{
  using namespace torch::nn;

  // H,Wを変えないConv2D
  auto MyConv2d = [](int64_t in_channels, int64_t out_channels) {
    return Conv2d(Conv2dOptions(in_channels, out_channels, 3).padding(1));
  };

  conv1_ = register_module("conv1", MyConv2d(3, 32));
  conv2_ = register_module("conv2", MyConv2d(32, 64));
  conv3_ = register_module("conv3", MyConv2d(64, 64));
  conv4_ = register_module("conv4", MyConv2d(64, 64));
  conv5_ = register_module("conv5", MyConv2d(64, 64));
  fc1_ = register_module("fc1", Linear(64 * 21 * 31, 512));
  fc2_ = register_module("fc2", Linear(512, 512));
  fc3_ = register_module("fc3", Linear(512, kActionSize));
}

torch::Tensor NeuralNetworkImpl::forward(torch::Tensor input)
{
  torch::Tensor x = input;              // [bs, 3, 691, 1016]
  x = torch::relu(conv1_->forward(x));  // [bs, 32, 691, 1016]
  x = torch::max_pool2d(x, 2, 2);       // [bs, 32, 345, 508]
  x = torch::relu(conv2_->forward(x));  // [bs, 64, 345, 508]
  x = torch::max_pool2d(x, 2, 2);       // [bs, 64, 172, 254]
  x = torch::relu(conv3_->forward(x));  // [bs, 64, 172, 254]
  x = torch::max_pool2d(x, 2, 2);       // [bs, 64, 86, 127]
  x = torch::relu(conv4_->forward(x));  // [bs, 64, 86, 127]
  x = torch::max_pool2d(x, 2, 2);       // [bs, 64, 43, 63]
  x = torch::relu(conv5_->forward(x));  // [bs, 64, 43, 63]
  x = torch::max_pool2d(x, 2, 2);       // [bs, 64, 21, 31]
  x = x.flatten(1);                     // [bs, 64 * 21 * 31]
  x = torch::relu(fc1_->forward(x));    // [bs, 512]
  x = torch::relu(fc2_->forward(x));    // [bs, 512]
  x = fc3_->forward(x);                 // [bs, kActionSize]
  return x;
}
