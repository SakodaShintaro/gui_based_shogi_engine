#include "cv_mat_to_tensor.hpp"
#include "gui_control.hpp"
#include "neural_network.hpp"

#include <opencv2/opencv.hpp>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <thread>

// ウィンドウ左上を原点としたときの盤面の範囲
// (10, 60) ~ (570, 500)
// constexpr int kBoardLU_x = 10;
// constexpr int kBoardLU_y = 60;
// constexpr int kBoardRD_x = 570;
// constexpr int kBoardRD_y = 500;
constexpr int kBoardLU_x = 0;
constexpr int kBoardLU_y = 0;
constexpr int kBoardRD_x = 800;
constexpr int kBoardRD_y = 600;

int main()
{
  Display * display = XOpenDisplay(nullptr);

  if (!display) {
    std::cout << "Cannot open display" << std::endl;
    return 1;
  }

  const torch::Device device(torch::kCUDA);
  const int64_t window_h = 600;
  const int64_t window_w = 800;
  NeuralNetwork nn(window_h, window_w);
  nn->to(device);

  torch::optim::Adam optimizer(nn->parameters(), torch::optim::AdamOptions(1e-4));

  int64_t counter = 0;

  const std::string save_root_dir = "./data/";
  const std::string save_image_dir = save_root_dir + "/image/";
  std::filesystem::create_directories(save_image_dir);
  std::ofstream ofs(save_root_dir + "/info.tsv");
  ofs << "counter\taction\treward" << std::endl;

  while (true) {
    auto now = std::chrono::system_clock::now();
    std::time_t end_time = std::chrono::system_clock::to_time_t(now);

    Window window = get_active_window(display);
    const Rect rect = get_window_rect(display, window);
    const std::string title = get_window_title(display, window);
    std::cerr << "\n"
              << std::ctime(&end_time) << " " << title  // time and title
              << "\trect: (" << rect.x << ", " << rect.y << ", " << rect.width << ", "
              << rect.height << ")" << std::endl;

    // Windowのタイトルでいろいろ判断する
    // TODO:
    // 成れる指し手を行ったときに「成りますか？」という小さいウィンドウが出ることに注意
    // const std::string key = "将棋所";
    const std::string key = "Siv3D App";
    const std::size_t pos = title.find(key);
    if (pos == std::string::npos) {
      std::this_thread::sleep_for(std::chrono::seconds(1));
      continue;
    }

    const cv::Mat image_before = get_screenshot(display, rect);

    // Cursor位置追加
    cv::Mat image_with_cursor = image_before.clone();
    const cv::Point curr_cursor = get_current_cursor_abs_position(display);
    const cv::Point cursor_in_image(curr_cursor.x - rect.x, curr_cursor.y - rect.y);
    cv::circle(image_with_cursor, cursor_in_image, 5, cv::Scalar(0, 0, 255), -1);

    const std::string save_name =
      (std::stringstream() << save_image_dir << "/input_image_" << std::setfill('0') << std::setw(8)
                           << counter << ".png")
        .str();
    cv::imwrite(save_name, image_with_cursor);

    torch::Tensor image_tensor = cv_mat_to_tensor(image_before);
    image_tensor = image_tensor.unsqueeze(0);
    image_tensor = image_tensor.to(device);
    torch::Tensor out = nn->forward(image_tensor);
    out = out.squeeze(0);  // [1, 5] -> [5]
    torch::Tensor softmax_score = torch::softmax(out, 0);

    // argmax取得
    torch::Tensor index_tensor = torch::multinomial(softmax_score, 1);
    const int64_t action_index = index_tensor[0].item<int64_t>();

    constexpr int kUnit = 20;
    if (action_index == kClick) {
      mouse_click(display, 1);
      std::cerr << "Click!" << std::endl;
    } else {
      cv::Point curr_cursor = get_current_cursor_abs_position(display);
      if (action_index == kUp) {
        curr_cursor.y -= kUnit;
      } else if (action_index == kRight) {
        curr_cursor.x += kUnit;
      } else if (action_index == kDown) {
        curr_cursor.y += kUnit;
      } else if (action_index == kLeft) {
        curr_cursor.x -= kUnit;
      }

      // 適正な領域内に収める
      curr_cursor.x = std::clamp(curr_cursor.x, rect.x + kBoardLU_x, rect.x + kBoardRD_x);
      curr_cursor.y = std::clamp(curr_cursor.y, rect.y + kBoardLU_y, rect.y + kBoardRD_y);

      warp_cursor(display, curr_cursor.x, curr_cursor.y);
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(250));

    const cv::Mat image_after = get_screenshot(display, rect);

    // 画像の差分量を計算
    cv::Mat diff;
    cv::absdiff(image_before, image_after, diff);
    diff.convertTo(diff, CV_32F);
    const double diff_norm = cv::norm(cv::mean(diff));

    // 更新されるように強化学習
    torch::Tensor reward = torch::tensor(diff_norm > 0).to(device);
    torch::Tensor log_policy = torch::log_softmax(out, 0)[action_index];
    torch::Tensor loss = -log_policy * reward;
    std::cerr << "reward: " << reward.item<float>() << "\tlog_policy: " << log_policy.item<float>()
              << "\tloss: " << loss.item<float>() << std::endl;
    optimizer.zero_grad();
    loss.backward();
    optimizer.step();

    ofs << counter << "\t" << action_index << "\t" << reward.item<float>() << std::endl;

    counter++;
  }

  XCloseDisplay(display);
}
