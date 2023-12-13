#include "action.hpp"
#include "cv_mat_to_tensor.hpp"
#include "decision_transformer.hpp"
#include "gui_control.hpp"
#include "window_size.hpp"

#include <opencv2/opencv.hpp>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <thread>

int main()
{
  DecisionTransformer transformer(kWindowHeight, kWindowWidth, 25);
  const torch::Device device(torch::kCUDA);
  transformer->to(device);

  torch::optim::Adam optimizer(transformer->parameters(), torch::optim::AdamOptions(1e-4));

  const std::string data_root_dir = "./data/";
  const std::string data_image_dir = data_root_dir + "/play/image/";
  const std::string save_dir = data_root_dir + "/offline_training/";
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

  const float max_return = *std::max_element(returns.begin(), returns.end());
  const float min_return = *std::min_element(returns.begin(), returns.end());
  const float width_return = max_return - min_return;
  const float unit_return = width_return / kReturnBinNum;

  const int64_t batch_size = 32;

  // 実行してみて結果を保存する
  auto save_result = [&](const std::string & path) {
    // std::ofstream ofs(path);
    // ofs << std::fixed;
    // torch::NoGradGuard no_grad_guard;
    // transformer->eval();
    // for (int64_t i = 0; i < data_num; i += batch_size) {
    //   std::vector<torch::Tensor> batch_images(batch_size);
    //   std::vector<torch::Tensor> batch_actions(batch_size);
    //   std::vector<torch::Tensor> batch_rewards(batch_size);
    //   int64_t actual_num = 0;
    //   for (int64_t j = 0; j < batch_size && i + j < data_num; j++) {
    //     const int64_t index = i + j;
    //     batch_images[j] = cv_mat_to_tensor(images[index]);
    //     batch_actions[j] = torch::tensor(actions[index], torch::kLong);
    //     batch_rewards[j] = torch::tensor(returns[index]);
    //     actual_num++;
    //   }
    //   batch_images.resize(actual_num);
    //   batch_actions.resize(actual_num);
    //   batch_rewards.resize(actual_num);

    //   const torch::Tensor images = torch::stack(batch_images).to(device);
    //   const torch::Tensor actions = torch::stack(batch_actions).to(device);
    //   const torch::Tensor rewards = torch::stack(batch_rewards).to(device);
    //   const torch::Tensor policy_logits = transformer->forward(images);
    //   const torch::Tensor policy = torch::softmax(policy_logits, 1);
    //   for (int64_t j = 0; j < batch_size && i + j < data_num; j++) {
    //     ofs << std::setfill('0') << std::setw(8) << i + j << "\t";
    //     ofs << batch_actions[j].item<int64_t>() << "\t";
    //     ofs << batch_rewards[j].item<float>() << "\t";
    //     for (int64_t k = 0; k < kActionSize; k++) {
    //       ofs << policy[j][k].item<float>() << "\t\n"[k == kActionSize - 1];
    //     }
    //   }
    // }
    // transformer->train();
  };

  save_result(save_dir + "/policy_result_before.tsv");

  std::mt19937_64 engine(std::random_device{}());
  std::uniform_int_distribution<int64_t> dist(kInputTimestep, data_num);

  for (int64_t itr = 0; itr < 1000; itr++) {
    std::vector<torch::Tensor> batch_images(batch_size);
    std::vector<torch::Tensor> batch_returns(batch_size);
    std::vector<torch::Tensor> batch_actions(batch_size);
    std::vector<torch::Tensor> batch_rewards(batch_size);
    for (int64_t i = 0; i < batch_size; i++) {
      const int64_t index = dist(engine);
      std::vector<torch::Tensor> curr_images(kInputTimestep);
      std::vector<torch::Tensor> curr_returns(kInputTimestep);
      std::vector<torch::Tensor> curr_actions(kInputTimestep);
      std::vector<torch::Tensor> curr_rewards(kInputTimestep);
      for (int64_t j = 0; j < kInputTimestep; j++) {
        int64_t returns_int =
          static_cast<int64_t>((returns[index - kInputTimestep + j] - min_return) / unit_return);
        returns_int = std::clamp(returns_int, static_cast<int64_t>(0), kReturnBinNum - 1);
        curr_images[j] = cv_mat_to_tensor(images[index - kInputTimestep + j]);
        curr_returns[j] = torch::tensor(returns_int, torch::kLong);
        curr_actions[j] = torch::tensor(actions[index - kInputTimestep + j], torch::kLong);
        curr_rewards[j] = torch::tensor(rewards[index - kInputTimestep + j], torch::kLong);
      }
      batch_images[i] = torch::stack(curr_images);
      batch_returns[i] = torch::stack(curr_returns);
      batch_actions[i] = torch::stack(curr_actions);
      batch_rewards[i] = torch::stack(curr_rewards);
    }

    const torch::Tensor images = torch::stack(batch_images).to(device);
    const torch::Tensor returns = torch::stack(batch_returns).to(device);
    const torch::Tensor actions = torch::stack(batch_actions).to(device);
    const torch::Tensor rewards = torch::stack(batch_rewards).to(device);

    // (bs, T, kActionSize)
    torch::Tensor policy_logits = transformer->forward(images, returns, actions, rewards);
    policy_logits = policy_logits.view({-1, kActionSize});
    policy_logits = torch::log_softmax(policy_logits, 1);
    // actionと交差エントロピー
    const torch::Tensor loss =
      torch::nll_loss(policy_logits, actions.flatten(), {}, torch::Reduction::Mean);

    optimizer.zero_grad();
    loss.backward();
    optimizer.step();
    std::stringstream ss;
    ss << itr << "\t" << loss.item<float>() << std::endl;
    std::cout << ss.str();
    ofs << ss.str();
  }

  save_result(save_dir + "/policy_result_after.tsv");

  torch::save(transformer, save_dir + "/transformer.pt");
}
