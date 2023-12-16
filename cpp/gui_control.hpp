#ifndef GUI_CONTROL_HPP_
#define GUI_CONTROL_HPP_

#include <opencv2/opencv.hpp>

// clang-format off
#include <X11/Xlib.h>
#include <X11/Xutil.h>
// clang-format on

struct Rect
{
  int x;
  int y;
  int width;
  int height;
};

// ディスプレイの絶対座標について位置(x, y)にカーソルをワープさせる
void warp_cursor(Display * display, int x, int y);

// マウスの左クリックをする
void mouse_click(Display * display, int button);

// ディスプレイの絶対座標についてカーソル位置(x, y)
cv::Point get_current_cursor_abs_position(Display * display);

// Rectの範囲についてスクリーンショットを取る
cv::Mat get_screenshot(Display * display, const Rect & rect);

// アクティブになっているWindowのIDを得る
Window get_active_window(Display * display);

// IDで指定したウィンドウについて位置・サイズをRectで得る
Rect get_window_rect(Display * display, Window window);

// IDで指定したウィンドウについてタイトルを得る
std::string get_window_title(Display * display, Window window);

#endif  // GUI_CONTROL_HPP_
