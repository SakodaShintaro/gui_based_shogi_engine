#include "action.hpp"
#include "cv_mat_to_tensor.hpp"
#include "gui_control.hpp"
#include "neural_network.hpp"
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
  Actor actor = Actor(kWindowHeight, kWindowWidth);
  const torch::Device device(torch::kCUDA);
  actor->to(device);

  torch::optim::Adam optimizer(actor->parameters(), torch::optim::AdamOptions(1e-4));

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
  std::vector<float> sum_rewards(data_num, 0.0f);
  sum_rewards.back() = rewards.back();
  for (int64_t i = data_num - 2; i >= 0; i--) {
    sum_rewards[i] = gamma * sum_rewards[i + 1] + rewards[i];
  }

  const int64_t batch_size = 64;

  // 実行してみて結果を保存する
  auto save_result = [&](const std::string & path) {
    std::ofstream ofs(path);
    ofs << std::fixed;
    torch::NoGradGuard no_grad_guard;
    actor->eval();
    for (int64_t i = 0; i < data_num; i += batch_size) {
      std::vector<torch::Tensor> batch_images(batch_size);
      std::vector<torch::Tensor> batch_actions(batch_size);
      std::vector<torch::Tensor> batch_rewards(batch_size);
      int64_t actual_num = 0;
      for (int64_t j = 0; j < batch_size && i + j < data_num; j++) {
        const int64_t index = i + j;
        batch_images[j] = cv_mat_to_tensor(images[index]);
        batch_actions[j] = torch::tensor(actions[index], torch::kLong);
        batch_rewards[j] = torch::tensor(sum_rewards[index]);
        actual_num++;
      }
      batch_images.resize(actual_num);
      batch_actions.resize(actual_num);
      batch_rewards.resize(actual_num);

      const torch::Tensor images = torch::stack(batch_images).to(device);
      const torch::Tensor actions = torch::stack(batch_actions).to(device);
      const torch::Tensor rewards = torch::stack(batch_rewards).to(device);
      const torch::Tensor policy_logits = actor->forward(images);
      const torch::Tensor policy = torch::softmax(policy_logits, 1);
      for (int64_t j = 0; j < batch_size && i + j < data_num; j++) {
        ofs << std::setfill('0') << std::setw(8) << i + j << "\t";
        ofs << batch_actions[j].item<int64_t>() << "\t";
        ofs << batch_rewards[j].item<float>() << "\t";
        for (int64_t k = 0; k < kActionSize; k++) {
          ofs << policy[j][k].item<float>() << "\t\n"[k == kActionSize - 1];
        }
      }
    }
    actor->train();
  };

  save_result(save_dir + "/policy_result_before.tsv");

  std::mt19937_64 engine(std::random_device{}());
  std::uniform_int_distribution<int64_t> dist(0, data_num - 1);

  for (int64_t itr = 0; itr < 100; itr++) {
    std::vector<torch::Tensor> batch_images(batch_size);
    std::vector<torch::Tensor> batch_actions(batch_size);
    std::vector<torch::Tensor> batch_rewards(batch_size);
    for (int64_t i = 0; i < batch_size; i++) {
      const int64_t index = dist(engine);
      batch_images[i] = cv_mat_to_tensor(images[index]);
      batch_actions[i] = torch::tensor(actions[index], torch::kLong);
      batch_rewards[i] = torch::tensor(sum_rewards[index]);
    }

    const torch::Tensor images = torch::stack(batch_images).to(device);
    const torch::Tensor actions = torch::stack(batch_actions).to(device);
    const torch::Tensor rewards = torch::stack(batch_rewards).to(device);

    const torch::Tensor policy_logits = actor->forward(images);
    const torch::Tensor log_policy = torch::log_softmax(policy_logits, 1);
    const torch::Tensor action_log_prob = log_policy.gather(1, actions.unsqueeze(1));
    const torch::Tensor loss = -action_log_prob.squeeze(1) * rewards;
    const torch::Tensor loss_mean = loss.mean();

    optimizer.zero_grad();
    loss_mean.backward();
    optimizer.step();
    std::stringstream ss;
    ss << itr << "\t" << loss_mean.item<float>() << std::endl;
    std::cout << ss.str();
    ofs << ss.str();
  }

  save_result(save_dir + "/policy_result_after.tsv");

  torch::save(actor, save_dir + "/actor.pt");
}
