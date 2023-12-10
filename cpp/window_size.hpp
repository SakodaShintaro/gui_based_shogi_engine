#ifndef WINDOW_SIZE_HPP_
#define WINDOW_SIZE_HPP_

#include <opencv2/opencv.hpp>

// ウィンドウサイズ
constexpr int kWindowWidth = 800;
constexpr int kWindowHeight = 600;

// ウィンドウ左上を原点としたときの盤面の範囲
// 【将棋所】
// (10, 60) ~ (570, 500)
// constexpr int kBoardLU_x = 10;
// constexpr int kBoardLU_y = 60;
// constexpr int kBoardRD_x = 570;
// constexpr int kBoardRD_y = 500;
// 【自作App】
constexpr int kBoardLU_x = 0;
constexpr int kBoardLU_y = 0;
constexpr int kBoardRD_x = 800;
constexpr int kBoardRD_y = 600;
const cv::Point center_point((kBoardLU_x + kBoardRD_x) / 2, (kBoardLU_y + kBoardRD_y) / 2);

#endif
