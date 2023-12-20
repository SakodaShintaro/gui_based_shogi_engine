#include "../grid_world.hpp"

#include <fstream>

class NeuralNetworkImpl : public torch::nn::Module
{
public:
  NeuralNetworkImpl() = default;
  NeuralNetworkImpl(int64_t h, int64_t w) : h_(h), w_(w), square_(h * w)
  {
    using namespace torch::nn;

    const std::vector<std::pair<int64_t, int64_t>> in_out_channels = {
      {2, 32}, {32, 64}, {64, 64}, {64, 64}};
    const int64_t hidden_dim = in_out_channels.back().second * h * w;

    if (in_out_channels.size() != conv_layer_num) {
      std::cerr << "in_out_channels.size() != conv_layer_num" << std::endl;
      std::exit(1);
    }

    for (int64_t i = 0; i < conv_layer_num; i++) {
      // H,Wを変えないConv2D
      const auto [in_channels, out_channels] = in_out_channels[i];
      conv_layers_[i] = register_module(
        "conv" + std::to_string(i), Conv2d(Conv2dOptions(in_channels, out_channels, 3).padding(1)));
    }

    linear_policy_ = register_module("linear_policy", Linear(hidden_dim, kActionSize));
    linear_value_ = register_module("linear_value", Linear(hidden_dim, 1));

    const int64_t s2 = square_ * square_;
    policy_embedding_ = register_module("policy_embedding", Embedding(s2, kActionSize));
    value_embedding_ = register_module("value_embedding", Embedding(s2, 1));
  }

  std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor input)
  {
    // torch::Tensor x = input;  // [2, h, w]
    // x = x.unsqueeze(0);       // [bs, 2, h, w]
    // for (int64_t i = 0; i < conv_layer_num; i++) {
    //   x = conv_layers_[i]->forward(x);
    //   x = torch::relu(x);
    // }
    // x = x.flatten(1);  // [bs, 64 * h * w]
    // torch::Tensor policy = linear_policy_->forward(x)[0];
    // torch::Tensor value = linear_value_->forward(x)[0];
    // return std::make_pair(policy, value);

    int64_t one = 0, two = 0;
    for (int64_t i = 0; i < input.size(1); i++) {
      for (int64_t j = 0; j < input.size(2); j++) {
        if (input[0][i][j].item<float>() == 1) {
          one = i * input.size(1) + j;
        }
        if (input[1][i][j].item<float>() == 1) {
          two = i * input.size(1) + j;
        }
      }
    }
    const int64_t index = one * square_ + two;
    torch::Tensor index_tensor = torch::tensor({index});
    torch::Tensor policy = policy_embedding_->forward(index_tensor)[0];
    torch::Tensor value = value_embedding_->forward(index_tensor)[0];
    return std::make_pair(policy, value);
  }

private:
  const int64_t h_;
  const int64_t w_;
  const int64_t square_;
  static constexpr int64_t conv_layer_num = 4;
  std::vector<torch::nn::Conv2d> conv_layers_{conv_layer_num, nullptr};
  torch::nn::Linear linear_policy_ = nullptr;
  torch::nn::Linear linear_value_ = nullptr;
  torch::nn::Embedding policy_embedding_ = nullptr;
  torch::nn::Embedding value_embedding_ = nullptr;
};
TORCH_MODULE(NeuralNetwork);

int main()
{
  const int64_t kGridSize = 4;
  GridWorld grid(kGridSize);

  NeuralNetwork network(kGridSize, kGridSize);
  const int64_t kTableSize = kGridSize * kGridSize * kGridSize * kGridSize;
  torch::Tensor value_table = torch::zeros({kTableSize}).requires_grad_(true);

  std::vector<torch::Tensor> parameters = network->parameters();
  parameters.push_back(value_table);
  torch::optim::SGD optimizer(parameters, 1.0);

  constexpr float kGamma = 0.9;  // 割引率

  std::deque<int64_t> is_ideal_actions;
  constexpr int64_t kWindowSize = 200;
  int64_t ideal_action_num = 0;

  std::ofstream ofs("grid_world_log.tsv");
  ofs << std::fixed << "iteration\tvalue_loss\tpolicy_loss\tsuccess\tis_ideal_action\n";
  std::cout << std::fixed;

  for (int64_t i = 0; i < 20000; i++) {
    grid.print();

    // 現在の状態を取得
    torch::Tensor state = grid.state();
    const int64_t s = grid.state_as_int();

    // 行動選択
    const auto [policy_logit, _v] = network->forward(state);
    torch::Tensor policy = torch::softmax(policy_logit, 0);
    torch::Tensor value = value_table[s];
    const int64_t action = torch::multinomial(policy, 1).item<int64_t>();

    // 最適判定
    const bool is_ideal_action = grid.is_ideal_action(static_cast<Action>(action));
    is_ideal_actions.push_back(is_ideal_action);
    ideal_action_num += is_ideal_action;
    if (is_ideal_actions.size() > kWindowSize) {
      ideal_action_num -= is_ideal_actions.front();
      is_ideal_actions.pop_front();
    }

    // 行動aを実行し、r, nsを観測
    const bool success = grid.step(static_cast<Action>(action));
    const float reward = (success ? 1.0 : -0.1);

    // 損失を計算
    const int64_t ns = grid.state_as_int();
    const float next_value = torch::max(value_table[ns]).item<float>();
    torch::Tensor td = reward + kGamma * next_value - value;
    torch::Tensor value_loss = td * td;

    torch::Tensor log_prob = torch::log_softmax(policy_logit, 0)[action];
    torch::Tensor actor_loss = -log_prob * td.detach();

    torch::Tensor loss = 1 * actor_loss + 0.1 * value_loss;

    // 更新
    optimizer.zero_grad();
    loss.backward();
    optimizer.step();

    const auto [policy_logit_after, _vv] = network->forward(state);
    torch::Tensor policy_after = torch::softmax(policy_logit_after, 0);

    for (int64_t j = 0; j < kActionSize; j++) {
      const float before = policy[j].item<float>();
      const float after = policy_after[j].item<float>();
      const float diff = after - before;
      std::cout << "action[" << j << "] = " << before << "\t" << after << "\t" << diff;
      if (j == action) {
        std::cout << " <- current_action";
      }
      std::cout << std::endl;
    }

    std::cout << "i = " << i << ", action = " << action << ", reward = " << reward
              << ", value = " << value.item<float>() << ", td = " << td.item<float>()
              << ", ideal_action_rate = " << 100.0 * ideal_action_num / kWindowSize
              << ", is_ideal = " << is_ideal_action << std::endl;

    // ログを記録
    ofs << i << "\t" << value_loss.item<float>() << "\t" << actor_loss.item<float>() << "\t"
        << success << "\t" << is_ideal_action << std::endl;
  }
}
