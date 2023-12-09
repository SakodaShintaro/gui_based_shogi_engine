#include "action.hpp"
#include "cv_mat_to_tensor.hpp"
#include "gui_control.hpp"
#include "neural_network.hpp"

#include <opencv2/opencv.hpp>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <thread>

// ウィンドウサイズ
constexpr int kWindowWidth = 800;
constexpr int kWindowHeight = 600;

// ウィンドウ左上を原点としたときの盤面の範囲
constexpr int kBoardLU_x = 0;
constexpr int kBoardLU_y = 0;
constexpr int kBoardRD_x = 800;
constexpr int kBoardRD_y = 600;

int main()
{
  Actor actor = Actor(kWindowHeight, kWindowWidth);
  const torch::Device device(torch::kCUDA);
  actor->to(device);

  torch::optim::Adam optimizer(actor->parameters(), torch::optim::AdamOptions(1e-4));

  const std::string data_root_dir = "./data/";
  const std::string data_image_dir = data_root_dir + "/image/";
  const std::string save_dir = data_root_dir + "/offline_training/";
  std::filesystem::create_directories(save_dir);
  std::ofstream ofs(save_dir + "/log.tsv");

  // データをロード
  std::vector<cv::Mat> images;
  std::vector<int64_t> actions;
  std::vector<float> rewards;
  {
    std::ifstream ifs(data_root_dir + "/info.tsv");
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
  const int64_t batch_size = 64;
  const int64_t sequence_length = 64;

  std::mt19937_64 engine(std::random_device{}());
  std::uniform_int_distribution<int64_t> dist(0, data_num - sequence_length);

  for (int64_t itr = 0; itr < 1000; itr++) {
    std::vector<int64_t> indices(batch_size);
    std::vector<torch::Tensor> batch_images(batch_size);
    std::vector<torch::Tensor> batch_actions(batch_size);
    std::vector<torch::Tensor> batch_rewards(batch_size);
    for (int64_t i = 0; i < batch_size; i++) {
      indices[i] = dist(engine);
      const int64_t index = indices[i];

      float reward_sum = 0;
      constexpr float gamma = 0.9;
      for (int64_t j = 0; j < sequence_length; j++) {
        reward_sum += std::pow(gamma, j) * rewards[index + j];
      }
      batch_images[i] = cv_mat_to_tensor(images[index]);
      batch_actions[i] = torch::tensor(actions[index], torch::kLong);
      batch_rewards[i] = torch::tensor(reward_sum);
    }

    const torch::Tensor images = torch::stack(batch_images).to(device);
    const torch::Tensor actions = torch::stack(batch_actions).to(device);
    const torch::Tensor rewards = torch::stack(batch_rewards).to(device);

    const torch::Tensor policy_logits = actor->forward(images);
    const torch::Tensor log_policy = torch::log_softmax(policy_logits, 0);
    const torch::Tensor action_log_prob = log_policy.gather(1, actions.unsqueeze(1));
    const torch::Tensor loss = -action_log_prob * rewards;
    const torch::Tensor loss_mean = loss.mean();
    optimizer.zero_grad();
    loss_mean.backward();
    optimizer.step();
    std::stringstream ss;
    ss << itr << "\t" << loss_mean.item<float>() << std::endl;
    std::cout << ss.str();
    ofs << ss.str();
  }

  torch::save(actor, save_dir + "/actor.pt");
}
