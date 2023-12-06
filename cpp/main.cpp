#include "cv_mat_to_tensor.hpp"
#include "gui_control.hpp"
#include "neural_network.hpp"

#include <opencv2/opencv.hpp>

#include <chrono>
#include <iostream>
#include <thread>

// ウィンドウ左上を原点としたときの盤面の範囲
// (10, 60) ~ (570, 500)
constexpr int kBoardLU_x = 10;
constexpr int kBoardLU_y = 60;
constexpr int kBoardRD_x = 570;
constexpr int kBoardRD_y = 500;

int main()
{
  Display * display = XOpenDisplay(nullptr);

  if (!display) {
    std::cout << "Cannot open display" << std::endl;
    return 1;
  }

  const torch::Device device(torch::kCUDA);
  NeuralNetwork nn;
  nn->to(device);

  torch::optim::Adam optimizer(nn->parameters(), torch::optim::AdamOptions(1e-4));

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
    const std::string key = "将棋所";
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

    cv::imwrite("screenshot.png", image_with_cursor);

    torch::Tensor image_tensor = cv_mat_to_tensor(image_before);
    image_tensor = image_tensor.unsqueeze(0);
    image_tensor = image_tensor.to(device);
    torch::Tensor out = nn->forward(image_tensor);
    out = out.squeeze(0);  // [1, 5] -> [5]
    torch::Tensor softmax_score = torch::softmax(out, 0);

    // argmax取得
    torch::Tensor index_tensor = torch::multinomial(softmax_score, 1);
    const int64_t index = index_tensor[0].item<int64_t>();

    constexpr int kUnit = 10;
    if (index == kClick) {
      mouse_click(display, 1);
      std::cerr << "Click!" << std::endl;
    } else {
      cv::Point curr_cursor = get_current_cursor_abs_position(display);
      if (index == kUp) {
        curr_cursor.y -= kUnit;
      } else if (index == kRight) {
        curr_cursor.x += kUnit;
      } else if (index == kDown) {
        curr_cursor.y += kUnit;
      } else if (index == kLeft) {
        curr_cursor.x -= kUnit;
      }

      // 適正な領域内に収める
      curr_cursor.x = std::clamp(curr_cursor.x, rect.x + kBoardLU_x, rect.x + kBoardRD_x);
      curr_cursor.y = std::clamp(curr_cursor.y, rect.y + kBoardLU_y, rect.y + kBoardRD_y);

      warp_cursor(display, curr_cursor.x, curr_cursor.y);
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    const cv::Mat image_after = get_screenshot(display, rect);
    cv::imwrite("screenshot2.png", image_after);

    // 画像の差分量を計算
    cv::Mat diff;
    cv::absdiff(image_before, image_after, diff);
    const double diff_norm = cv::norm(cv::sum(diff));

    // 差分を大きくするように強化学習（TODO:流石にこれではまともに動かない）
    torch::Tensor reward = torch::tensor(diff_norm).to(device);
    torch::Tensor loss = -torch::log_softmax(out, 0)[index] * reward;
    std::cerr << "loss: " << loss.item<float>() << std::endl;
    optimizer.zero_grad();
    loss.backward();
    optimizer.step();

    std::this_thread::sleep_for(std::chrono::milliseconds(500));
  }

  XCloseDisplay(display);
}
