#include "../grid_world.hpp"

int main()
{
  const int64_t kGridSize = 3;
  GridWorld grid(kGridSize);

  // Qテーブルの初期化
  const int64_t kTableSize = kGridSize * kGridSize * kGridSize * kGridSize;
  torch::Tensor q_table = torch::zeros({kTableSize, kActionSize});

  std::mt19937_64 engine(std::random_device{}());
  std::uniform_real_distribution<float> dist(0.0, 1.0);
  std::uniform_int_distribution<int64_t> dist_action(0, kActionSize - 1);
  constexpr float kEpsilon = 0.1;
  constexpr float kAlpha = 0.1;  // 学習率
  constexpr float kGamma = 0.9;  // 割引率

  int64_t success_num = 0;

  for (int64_t i = 0; i < 10000; i++) {
    grid.print();

    // 現在の状態を取得
    const int64_t s = grid.state_as_int();

    // ε-greedyに従って行動を選択
    int64_t action;
    if (dist(engine) < kEpsilon) {
      action = dist_action(engine);
    } else {
      action = torch::argmax(q_table[s], 0).item<int64_t>();
    }

    // 行動aを実行し、r, s'を観測
    const bool success = grid.step(static_cast<Action>(action));
    const float reward = (success ? 1.0 : -0.1);
    const int64_t ns = grid.state_as_int();
    success_num += success;

    std::cout << "i = " << i << ", action = " << action << ", reward = " << reward
              << ", success_num = " << success_num << std::endl;

    // Q関数を更新
    const float max_future_q = torch::max(q_table[ns]).item<float>();
    q_table[s][action] += kAlpha * (reward + kGamma * max_future_q - q_table[s][action]);
  }
}
