#include "decision_transformer.hpp"

static constexpr int64_t kInnerDim = 128;

DecisionTransformerImpl::DecisionTransformerImpl(int64_t h, int64_t w, int64_t patch_size)
{
  using namespace torch::nn;
  if (h % patch_size != 0 || w % patch_size != 0) {
    std::cerr << "h % patch_size != 0 || w % patch_size != 0" << std::endl;
    std::exit(1);
  }
  const int64_t ph = h / patch_size;
  const int64_t pw = w / patch_size;
  const int64_t num_patches = ph * pw;

  first_conv_ = register_module(
    "first_conv", Conv2d(Conv2dOptions(3, kInnerDim, patch_size).stride(patch_size)));
  image_pos_enc_ =
    register_parameter("image_pose_enc_", torch::randn({1, 1, num_patches, kInnerDim}));

  return_enc_ = register_module("return_enc", Embedding(kReturnBinNum, kInnerDim));
  action_enc_ = register_module("action_enc", Embedding(kActionSize, kInnerDim));
  reward_enc_ = register_module("reward_enc", Embedding(kRewardBinNum, kInnerDim));

  const int64_t tokens_per_step = num_patches + 3;
  positional_embedding_ = register_parameter(
    "positional_embedding", torch::randn({tokens_per_step * kInputTimestep, kInnerDim}));

  // src_mask構築
  // (image_t, return_t, action_t, reward_t)
  const int64_t seq_len = tokens_per_step * kInputTimestep;
  src_mask_ = register_parameter("src_mask", torch::zeros({seq_len, seq_len}), false);
  for (int64_t i = 0; i < seq_len; i++) {
    const int64_t ti = i / tokens_per_step;
    for (int64_t j = 0; j < seq_len; j++) {
      if (j % tokens_per_step < num_patches) {
        // 同じ画像の中は同じタイムステップのトークンを見れる
        const int64_t tj = j / tokens_per_step;
        src_mask_[i][j] = (tj <= ti);
      } else {
        // それ以外は通常のcausal mask
        src_mask_[i][j] = (j <= i);
      }
    }
  }

  TransformerEncoderLayerOptions options(kInnerDim, 8);
  options.dim_feedforward(kInnerDim * 4);
  options.dropout(0.0);
  TransformerEncoderLayer layer(options);
  const int64_t layer_num = 6;
  encoder_ = register_module("encoder", TransformerEncoder(layer, layer_num));

  predict_head_ = register_module("predict_head", Linear(kInnerDim, kActionSize));
}

torch::Tensor DecisionTransformerImpl::forward(
  torch::Tensor images, torch::Tensor returns, torch::Tensor actions, torch::Tensor rewards)
{
  //   images  : (bs, T, 3, H, W)
  //   returns : (bs, T)
  //   actions : (bs, T)
  //   rewards : (bs, T)

  // encode images
  const int64_t bs = images.size(0);
  const int64_t T = images.size(1);
  const int64_t C = images.size(2);
  const int64_t H = images.size(3);
  const int64_t W = images.size(4);
  images = images.view({-1, C, H, W});
  images = first_conv_->forward(images);         // (bs * T, kInnerDim, ph, pw)
  images = images.permute({0, 2, 3, 1});         // (bs * T, ph, pw, kInnerDim)
  images = images.view({bs, T, -1, kInnerDim});  // (bs, T, ph * pw, kInnerDim)
  images = images + image_pos_enc_;              // (bs, T, ph * pw, kInnerDim)
  images = images.flatten(2);                    // (bs, T, ph * pw * kInnerDim)

  // encode returns, actions, rewards
  returns = return_enc_->forward(returns);  // (bs, T, kInnerDim)
  actions = action_enc_->forward(actions);  // (bs, T, kInnerDim)
  rewards = reward_enc_->forward(rewards);  // (bs, T, kInnerDim)

  // concat
  torch::Tensor x =
    torch::cat({images, returns, actions, rewards}, 2);  // (bs, T, (ph * pw + 3) * kInnerDim)
  x = x.view({bs, -1, kInnerDim});                       // (bs, T * (ph * pw + 3), kInnerDim)
  x = x + positional_embedding_;                         // (bs, T * (ph * pw + 3), kInnerDim)
  x = x.permute({1, 0, 2});                              // (T * (ph * pw + 3), bs, kInnerDim)

  x = encoder_->forward(x, src_mask_);  // (T * (ph * pw + 3), bs, kInnerDim)
  x = x.permute({1, 0, 2});             // (bs, T * (ph * pw + 3), kInnerDim)

  // patch_num + 1のところから(patch_num + 3)ごとにT個取る
  const int64_t num_patches = x.size(1) / T - 3;
  const int64_t tokens_per_step = num_patches + 3;
  torch::Tensor action =
    x.slice(1, num_patches + 1, x.size(1), tokens_per_step);  // (bs, T, kInnerDim)
  action = predict_head_->forward(action);                    // (bs, T, kActionSize)
  return action;
}