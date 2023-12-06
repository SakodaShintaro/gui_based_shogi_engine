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

    const cv::Mat image = get_screenshot(display, rect);
    cv::imwrite("screenshot.png", image);

    torch::Tensor image_tensor = cv_mat_to_tensor(image);
    image_tensor = image_tensor.unsqueeze(0);
    image_tensor = image_tensor.to(device);
    torch::Tensor out = nn->forward(image_tensor);
    out = out.squeeze(0);  // [1, 5] -> [5]
    torch::Tensor softmax_score = torch::softmax(out, 0);
    std::cerr << "softmax_score: " << softmax_score << std::endl;

    // argmax取得
    torch::Tensor index_tensor = torch::multinomial(softmax_score, 1);
    const int64_t index = index_tensor[0].item<int64_t>();
    std::cerr << "index = " << index << std::endl;

    constexpr int kUnit = 10;
    if (index == kClick) {
      mouse_click(display, 1);
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

    std::this_thread::sleep_for(std::chrono::seconds(1));
  }

  XCloseDisplay(display);
}
