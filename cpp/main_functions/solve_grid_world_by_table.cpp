#include "../grid_world.hpp"

int main()
{
  const int64_t kGridSize = 3;
  GridWorld grid(kGridSize);

  // テーブルの初期化
  const int64_t kTableSize = kGridSize * kGridSize * kGridSize * kGridSize;
  torch::Tensor policy_table = torch::zeros({kTableSize, kActionSize}).requires_grad_(true);
  torch::Tensor value_table = torch::zeros({kTableSize}).requires_grad_(true);

  torch::optim::SGD optimizer({policy_table, value_table}, 1.0);

  constexpr float kGamma = 0.9;  // 割引率

  int64_t success_num = 0;

  for (int64_t i = 0; i < 10000; i++) {
    grid.print();

    // 現在の状態を取得
    const int64_t s = grid.state_as_int();

    // 行動選択
    torch::Tensor policy = torch::softmax(policy_table[s], 0);
    std::cout << policy << std::endl;
    const int64_t action = torch::multinomial(policy, 1).item<int64_t>();

    // 最適判定
    const bool is_ideal_action = grid.is_ideal_action(static_cast<Action>(action));

    // 行動aを実行し、r, nsを観測
    const bool success = grid.step(static_cast<Action>(action));
    const float reward = (success ? 1.0 : -0.1);
    success_num += success;

    std::cout << "i = " << i << ", action = " << action << ", reward = " << reward
              << ", success_num = " << success_num << ", is_ideal = " << is_ideal_action
              << std::endl;

    // 損失を計算
    const int64_t ns = grid.state_as_int();
    const float next_value = torch::max(value_table[ns]).item<float>();
    torch::Tensor td = reward + kGamma * next_value - value_table[s];
    torch::Tensor value_loss = 0.1 * td * td;

    torch::Tensor log_prob = torch::log_softmax(policy_table[s], 0)[action];
    torch::Tensor actor_loss = -log_prob * td.detach();

    // 更新
    optimizer.zero_grad();
    value_loss.backward();
    actor_loss.backward();
    optimizer.step();
  }
}
