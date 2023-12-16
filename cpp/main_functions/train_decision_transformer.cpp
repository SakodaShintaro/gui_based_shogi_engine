#include "../action.hpp"
#include "../cv_mat_to_tensor.hpp"
#include "../decision_transformer.hpp"
#include "../gui_control.hpp"
#include "../timer.hpp"
#include "../window_size.hpp"

#include <opencv2/opencv.hpp>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <thread>

int main()
{
  Timer timer;
  timer.start();

  DecisionTransformer transformer(kWindowHeight, kWindowWidth, kPatchSize);
  const torch::Device device(torch::kCUDA);
  transformer->to(device);

  torch::optim::AdamWOptions options;
  options.set_lr(1e-4);
  options.weight_decay(0.1);
  options.betas({0.9, 0.999});
  torch::optim::AdamW optimizer(transformer->parameters(), options);

  const std::string data_root_dir = "./data/";
  const std::string data_image_dir = data_root_dir + "/play/image/";
  const std::string save_dir = data_root_dir + "/decision_transformer/";
  std::filesystem::create_directories(save_dir);
  std::ofstream ofs(save_dir + "/log.tsv");

  // データをロード
  std::vector<cv::Mat> images;
  std::vector<int64_t> actions;
  std::vector<float> rewards;
  {
    std::ifstream ifs(data_root_dir + "/play/info.tsv");
    std::string line;
    std::getline(ifs, line);  // ヘッダーを読み飛ばす
    while (std::getline(ifs, line)) {
      std::istringstream iss(line);
      std::string itr_str, action_str, reward_str;
      std::getline(iss, itr_str, '\t');
      std::getline(iss, action_str, '\t');
      std::getline(iss, reward_str, '\t');
      const int64_t itr = std::stoi(itr_str);
      const int64_t action = std::stoi(action_str);
      const float reward = std::stof(reward_str);
      const std::string image_name =
        (std::stringstream() << data_image_dir << "/input_image_" << std::setfill('0')
                             << std::setw(8) << itr << ".png")
          .str();
      cv::Mat image = cv::imread(image_name, cv::IMREAD_COLOR);
      images.push_back(image);
      actions.push_back(action);
      rewards.push_back(reward);
    }
  }

  const int64_t data_num = images.size();

  // 指数割引の累積報酬
  constexpr float gamma = 0.9f;
  std::vector<float> returns(data_num, 0.0f);
  returns.back() = rewards.back();
  for (int64_t i = data_num - 2; i >= 0; i--) {
    returns[i] = gamma * returns[i + 1] + rewards[i];
  }

  const int64_t batch_size = 64;

  auto get_data = [&](int64_t index) {
    std::vector<torch::Tensor> curr_images(kInputTimestep);
    std::vector<torch::Tensor> curr_returns(kInputTimestep);
    std::vector<torch::Tensor> curr_actions(kInputTimestep);
    std::vector<torch::Tensor> curr_rewards(kInputTimestep);
    for (int64_t j = 0; j < kInputTimestep; j++) {
      curr_images[j] = cv_mat_to_tensor(images[index - kInputTimestep + j]);
      curr_returns[j] =
        torch::tensor(returns[index - kInputTimestep + j] * kReturnScale, torch::kFloat);
      curr_actions[j] = torch::tensor(actions[index - kInputTimestep + j], torch::kLong);
      curr_rewards[j] = torch::tensor(rewards[index - kInputTimestep + j], torch::kLong);
    }
    return std::make_tuple(
      torch::stack(curr_images), torch::stack(curr_returns), torch::stack(curr_actions),
      torch::stack(curr_rewards));
  };

  // 実行してみて結果を保存する
  auto save_result = [&](const std::string & path) {
    std::ofstream ofs(path);
    ofs << std::fixed;
    torch::NoGradGuard no_grad_guard;
    transformer->eval();
    for (int64_t i = kInputTimestep; i < data_num; i += batch_size) {
      std::vector<torch::Tensor> batch_images(batch_size);
      std::vector<torch::Tensor> batch_returns(batch_size);
      std::vector<torch::Tensor> batch_actions(batch_size);
      std::vector<torch::Tensor> batch_rewards(batch_size);
      int64_t actual_num = 0;
      for (int64_t j = 0; j < batch_size && i + j < data_num; j++) {
        const int64_t index = i + j;
        const auto [curr_images, curr_returns, curr_actions, curr_rewards] = get_data(index);
        batch_images[j] = curr_images;
        batch_returns[j] = curr_returns;
        batch_actions[j] = curr_actions;
        batch_rewards[j] = curr_rewards;
        actual_num++;
      }
      batch_images.resize(actual_num);
      batch_returns.resize(actual_num);
      batch_actions.resize(actual_num);
      batch_rewards.resize(actual_num);

      const torch::Tensor images = torch::stack(batch_images).to(device);
      const torch::Tensor returns = torch::stack(batch_returns).to(device);
      const torch::Tensor actions = torch::stack(batch_actions).to(device);
      const torch::Tensor rewards = torch::stack(batch_rewards).to(device);
      const torch::Tensor policy_logits = transformer->forward(images, returns, actions, rewards);
      const torch::Tensor policy = torch::softmax(policy_logits, 2);
      for (int64_t j = 0; j < batch_size && i + j < data_num; j++) {
        ofs << std::setfill('0') << std::setw(8) << i + j << "\t";
        ofs << batch_returns[j][kInputTimestep - 1].item<float>() << "\t";
        ofs << batch_actions[j][kInputTimestep - 1].item<int64_t>() << "\t";
        ofs << batch_rewards[j][kInputTimestep - 1].item<int64_t>() << "\t";
        for (int64_t k = 0; k < kActionSize; k++) {
          ofs << policy[j][kInputTimestep - 1][k].item<float>() << "\t\n"[k == kActionSize - 1];
        }
      }
    }
    transformer->train();
  };

  save_result(save_dir + "/policy_result_before.tsv");

  std::mt19937_64 engine(std::random_device{}());
  std::uniform_int_distribution<int64_t> dist(kInputTimestep, data_num);

  constexpr int64_t kPrintInterval = 10;
  float sum_loss_interval = 0.0f;

  for (int64_t itr = 0; itr < 2000; itr++) {
    std::vector<torch::Tensor> batch_images(batch_size);
    std::vector<torch::Tensor> batch_returns(batch_size);
    std::vector<torch::Tensor> batch_actions(batch_size);
    std::vector<torch::Tensor> batch_rewards(batch_size);
    for (int64_t i = 0; i < batch_size; i++) {
      const int64_t index = dist(engine);
      const auto [curr_images, curr_returns, curr_actions, curr_rewards] = get_data(index);
      batch_images[i] = curr_images;
      batch_returns[i] = curr_returns;
      batch_actions[i] = curr_actions;
      batch_rewards[i] = curr_rewards;
    }

    const torch::Tensor images = torch::stack(batch_images).to(device);
    const torch::Tensor returns = torch::stack(batch_returns).to(device);
    const torch::Tensor actions = torch::stack(batch_actions).to(device);
    const torch::Tensor rewards = torch::stack(batch_rewards).to(device);

    // (bs, T, kActionSize)
    torch::Tensor policy_logits = transformer->forward(images, returns, actions, rewards);
    policy_logits = torch::log_softmax(policy_logits, 2);
    policy_logits = policy_logits.view({-1, kActionSize});
    // actionと交差エントロピー
    const torch::Tensor loss =
      torch::nll_loss(policy_logits, actions.flatten(), {}, torch::Reduction::Mean);

    optimizer.zero_grad();
    loss.backward();
    optimizer.step();

    sum_loss_interval += loss.item<float>();
    if ((itr + 1) % kPrintInterval == 0) {
      sum_loss_interval /= kPrintInterval;
      std::stringstream ss;
      ss << std::fixed << timer.elapsed_time() << "\t" << itr + 1 << "\t" << sum_loss_interval
         << std::endl;
      std::cout << ss.str();
      ofs << ss.str();
      sum_loss_interval = 0.0f;
    }
  }

  save_result(save_dir + "/policy_result_after.tsv");

  torch::save(transformer, save_dir + "/transformer.pt");
}
