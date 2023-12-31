#include "../action.hpp"
#include "../cv_mat_to_tensor.hpp"
#include "../decision_transformer.hpp"
#include "../gui_control.hpp"
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
  Display * display = XOpenDisplay(nullptr);

  if (!display) {
    std::cout << "Cannot open display" << std::endl;
    return 1;
  }

  const std::string save_root_dir = "./data/play_transformer/";
  const std::string save_image_dir = save_root_dir + "/image/";
  std::filesystem::create_directories(save_image_dir);
  std::ofstream ofs(save_root_dir + "/info.tsv");
  ofs << "itr\taction\treward" << std::endl;

  DecisionTransformer dt(kWindowHeight, kWindowWidth, kPatchSize);
  const std::string model_path = save_root_dir + "/../decision_transformer/transformer.pt";
  if (std::filesystem::exists(model_path)) {
    torch::load(dt, model_path);
    std::cout << "load!" << std::endl;
  }
  const torch::Device device(torch::kCUDA);
  dt->to(device);

  cv::Mat next_image_without_cursor;

  //   images  : (bs, T, 3, H, W)
  //   returns : (bs, T)
  //   actions : (bs, T)
  //   rewards : (bs, T)
  torch::Tensor images = torch::zeros({1, kInputTimestep, 3, kWindowHeight, kWindowWidth});
  torch::Tensor returns = torch::zeros({1, kInputTimestep}, torch::kFloat);
  torch::Tensor actions = torch::zeros({1, kInputTimestep}, torch::kLong);
  torch::Tensor rewards = torch::zeros({1, kInputTimestep}, torch::kLong);

  images = images.to(device);
  returns = returns.to(device);
  actions = actions.to(device);
  rewards = rewards.to(device);

  for (int64_t i = 0; i < kInputTimestep; i++) {
    returns[0][i] = kReturnScale;
  }

  for (int64 itr = 0; itr < 10000; itr++) {
    auto now = std::chrono::system_clock::now();
    std::time_t end_time = std::chrono::system_clock::to_time_t(now);

    Window window = get_active_window(display);
    const Rect rect = get_window_rect(display, window);
    const std::string title = get_window_title(display, window);
    std::cerr << "\n"
              << std::ctime(&end_time) << " " << title << " " << itr << "\trect: (" << rect.x
              << ", " << rect.y << ", " << rect.width << ", " << rect.height << ")" << std::endl;

    // Windowのタイトルで判断する
    const std::string key = "Siv3D App";
    const std::size_t pos = title.find(key);
    if (pos == std::string::npos) {
      std::this_thread::sleep_for(std::chrono::seconds(1));
      itr--;
      continue;
    }

    if (itr == 0) {
      const cv::Point center_point((kBoardLU_x + kBoardRD_x) / 2, (kBoardLU_y + kBoardRD_y) / 2);
      warp_cursor(display, center_point.x + rect.x, center_point.y + rect.y);
    }

    // 画像をnextからcurrへコピーする
    const cv::Mat curr_image_without_cursor =
      (itr == 0 ? get_screenshot(display, rect) : next_image_without_cursor.clone());

    // カーソル位置書き込み
    cv::Point curr_cursor = get_current_cursor_abs_position(display);
    const cv::Point cursor_in_image(curr_cursor.x - rect.x, curr_cursor.y - rect.y);
    cv::Mat curr_image_with_cursor = curr_image_without_cursor.clone();
    cv::circle(curr_image_with_cursor, cursor_in_image, 5, cv::Scalar(0, 0, 255), -1);
    cv::circle(curr_image_with_cursor, cursor_in_image, 2, cv::Scalar(0, 255, 0), -1);

    const std::string save_name =
      (std::stringstream() << save_image_dir << "/input_image_" << std::setfill('0') << std::setw(8)
                           << itr << ".png")
        .str();
    cv::imwrite(save_name, curr_image_with_cursor);

    // 行動取得
    torch::Tensor input_tensor = cv_mat_to_tensor(curr_image_with_cursor);
    input_tensor = input_tensor.to(device);
    images[0][kInputTimestep - 1] = input_tensor;
    torch::Tensor action_tensor = dt->forward(images, returns, actions, rewards);
    torch::Tensor softmax_tensor = torch::softmax(action_tensor, 2);
    softmax_tensor = softmax_tensor[0][kInputTimestep - 1];
    const int64_t action_index = torch::multinomial(softmax_tensor, 1).item<int64_t>();

    if (action_index == kClick) {
      mouse_click(display, 1);
    } else {
      if (action_index == kUp) {
        curr_cursor.y -= kMoveUnit;
      } else if (action_index == kRight) {
        curr_cursor.x += kMoveUnit;
      } else if (action_index == kDown) {
        curr_cursor.y += kMoveUnit;
      } else if (action_index == kLeft) {
        curr_cursor.x -= kMoveUnit;
      }

      // 適正な領域内に収める
      curr_cursor.x = std::clamp(curr_cursor.x, rect.x + kBoardLU_x, rect.x + kBoardRD_x);
      curr_cursor.y = std::clamp(curr_cursor.y, rect.y + kBoardLU_y, rect.y + kBoardRD_y);

      warp_cursor(display, curr_cursor.x, curr_cursor.y);
    }

    // GUIへ反映されるのを待つ
    std::this_thread::sleep_for(std::chrono::milliseconds(80));

    next_image_without_cursor = get_screenshot(display, rect);

    // 画像の差分量を計算
    cv::Mat diff;
    cv::absdiff(next_image_without_cursor, curr_image_without_cursor, diff);
    diff.convertTo(diff, CV_32F);
    const double diff_norm = cv::norm(cv::mean(diff));
    const double reward = (diff_norm > 0);

    actions[0][kInputTimestep - 1] = action_index;
    rewards[0][kInputTimestep - 1] = reward;

    // 状態を一つずらす
    for (int64_t i = 0; i < kInputTimestep - 1; i++) {
      images[0][i] = images[0][i + 1];
      actions[0][i] = actions[0][i + 1];
      rewards[0][i] = rewards[0][i + 1];
    }

    std::stringstream ss;
    ss << std::fixed << itr << "\t" << action_index << "\t"
       << softmax_tensor[action_index].item<float>() << "\t" << reward << std::endl;
    ofs << ss.str();
    std::cout << ss.str();
  }

  XCloseDisplay(display);
}
