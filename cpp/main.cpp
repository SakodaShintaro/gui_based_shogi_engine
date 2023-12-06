#include "cv_mat_to_tensor.hpp"
#include "gui_control.hpp"
#include "neural_network.hpp"

#include <opencv2/opencv.hpp>

#include <chrono>
#include <iostream>
#include <random>
#include <thread>

int main()
{
  Display * display = XOpenDisplay(nullptr);

  if (!display) {
    std::cout << "Cannot open display" << std::endl;
    return 1;
  }

  std::mt19937 mt(std::random_device{}());

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

    // ランダムにカーソル位置を変える
    // 将棋所の将棋盤の範囲内でランダムにカーソルを動かす
    // ウィンドウ左上を原点としたときの盤面の範囲 x[10, 570] y[60, 500]
    std::uniform_int_distribution<> rand_x(10, 570);
    std::uniform_int_distribution<> rand_y(60, 500);
    const int rx_catch = rand_x(mt);
    const int ry_catch = rand_y(mt);
    std::cerr << "rx_catch: " << rx_catch << " ry_catch: " << ry_catch << std::endl;

    // 掴む
    warp_cursor(display, rect.x + rx_catch, rect.y + ry_catch);
    XSync(display, false);
    mouse_click(display, 1);
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    // 離す
    const int rx_release = rand_x(mt);
    const int ry_release = rand_y(mt);
    std::cerr << "rx_release: " << rx_release << " ry_release: " << ry_release << std::endl;
    warp_cursor(display, rect.x + rx_release, rect.y + ry_release);
    XSync(display, false);
    mouse_click(display, 1);

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

    std::this_thread::sleep_for(std::chrono::seconds(1));
  }

  XCloseDisplay(display);
}
