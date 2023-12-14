#ifndef DECISION_TRANSFORMER_HPP_
#define DECISION_TRANSFORMER_HPP_

#include "action.hpp"

#include <torch/torch.h>

static constexpr int64_t kInputTimestep = 4;
const int64_t kReturnBinNum = 10;
const int64_t kRewardBinNum = 2;

// 実装参考) https://github.com/etaoxing/multigame-dt/blob/master/multigame_dt.py#L200

class DecisionTransformerImpl : public torch::nn::Module
{
public:
  DecisionTransformerImpl() = default;
  DecisionTransformerImpl(int64_t h, int64_t w, int64_t patch_size);

  // 参考) https://arxiv.org/abs/2205.15241
  // 1タイムステップ分 : [画像(6x6), 収益(1), 行動(1), 報酬(1)]
  //                      = [39]系列
  // それをT(= kInputTimestep)タイムステップ分入れる
  // 引数
  //   images  : (bs, T, 3, H, W)
  //   returns : (bs, T)
  //   actions : (bs, T)
  //   rewards : (bs, T)
  // 出力
  //   predicted_actions : (bs, kActionSize)
  torch::Tensor forward(
    torch::Tensor images, torch::Tensor returns, torch::Tensor actions, torch::Tensor rewards);

private:
  torch::nn::Conv2d first_conv_ = nullptr;
  torch::Tensor image_pos_enc_;
  torch::nn::Embedding return_enc_ = nullptr;
  torch::nn::Embedding action_enc_ = nullptr;
  torch::nn::Embedding reward_enc_ = nullptr;
  torch::Tensor positional_embedding_;
  torch::Tensor src_mask_;
  torch::nn::TransformerEncoder encoder_ = nullptr;
  torch::nn::Linear predict_head_ = nullptr;
};
TORCH_MODULE(DecisionTransformer);

#endif  // DECISION_TRANSFORMER_HPP_
