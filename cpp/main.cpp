#include "cv_mat_to_tensor.hpp"
#include "neural_network.hpp"

#include <opencv2/opencv.hpp>

#include <chrono>
#include <iostream>
#include <random>
#include <thread>

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

void move_cursor(Display * display, int x, int y)
{
  std::cerr << "move_cursor (" << x << ", " << y << ")" << std::endl;
  XWarpPointer(display, None, DefaultRootWindow(display), 0, 0, 0, 0, x, y);
}

void mouse_click(Display * display, int button)
{
  std::cerr << "mouse_click" << std::endl;
  XButtonEvent event;
  event.type = ButtonPress;
  event.button = button;
  event.same_screen = True;

  XQueryPointer(
    display, RootWindow(display, DefaultScreen(display)), &event.root, &event.window, &event.x_root,
    &event.y_root, &event.x, &event.y, &event.state);

  event.subwindow = event.window;

  while (event.subwindow) {
    event.window = event.subwindow;
    XQueryPointer(
      display, event.window, &event.root, &event.subwindow, &event.x_root, &event.y_root, &event.x,
      &event.y, &event.state);
  }

  XSendEvent(display, PointerWindow, True, 0xfff, (XEvent *)&event);
  XFlush(display);

  event.type = ButtonRelease;
  event.state = 0x100;

  XSendEvent(display, PointerWindow, True, 0xfff, (XEvent *)&event);
  XFlush(display);
}

cv::Mat get_screenshot(Display * display, const Rect & rect)
{
  XImage * image = XGetImage(
    display, RootWindow(display, DefaultScreen(display)), rect.x, rect.y, rect.width, rect.height,
    AllPlanes, ZPixmap);

  cv::Mat mat(rect.height, rect.width, CV_8UC3);

  for (int y = 0; y < rect.height; y++) {
    for (int x = 0; x < rect.width; x++) {
      unsigned long pixel = XGetPixel(image, x, y);

      unsigned char blue = pixel & image->blue_mask;
      unsigned char green = (pixel & image->green_mask) >> 8;
      unsigned char red = (pixel & image->red_mask) >> 16;

      mat.at<cv::Vec3b>(y, x) = cv::Vec3b(blue, green, red);
    }
  }

  // Cursor位置
  Window root_return, child_return;
  int root_x_return, root_y_return, win_x_return, win_y_return;
  unsigned int mask_return;

  XQueryPointer(
    display, RootWindow(display, DefaultScreen(display)), &root_return, &child_return,
    &root_x_return, &root_y_return, &win_x_return, &win_y_return, &mask_return);
  std::cerr << root_x_return << " " << root_y_return << std::endl;
  cv::Point cursor(root_x_return - rect.x, root_y_return - rect.y);
  cv::circle(mat, cursor, 5, cv::Scalar(0, 0, 255), -1);
  XDestroyImage(image);

  return mat;
}

Window get_active_window(Display * display)
{
  Window focused;
  int revert;
  XGetInputFocus(display, &focused, &revert);

  XWindowAttributes attrs;
  XGetWindowAttributes(display, focused, &attrs);

  return focused;
}

Rect get_window_rect(Display * display, Window window)
{
  XWindowAttributes attrs;
  XGetWindowAttributes(display, window, &attrs);

  int x, y;
  Window child;
  XTranslateCoordinates(
    display, window, RootWindow(display, DefaultScreen(display)), 0, 0, &x, &y, &child);

  return Rect{x, y, attrs.width, attrs.height};
}

std::string get_window_title(Display * display, Window window)
{
  Atom name_atom = XInternAtom(display, "WM_NAME", False);

  Atom type;
  unsigned long bytes_after;
  int len;
  unsigned long bytes_after_return;
  unsigned char * name;
  XGetWindowProperty(
    display, window, name_atom, 0, 1024, False, AnyPropertyType, &type, &len, &bytes_after,
    &bytes_after_return, &name);

  // 文字列に変換
  if (name == nullptr) {
    return "";
  }
  std::string title((char *)name);

  // 後処理
  XFree(name);

  return title;
}

void print_tensor_info(const torch::Tensor & tensor)
{
  std::cout << "tensor_info: " << tensor.sizes() << " " << tensor.dtype() << " " << tensor.device()
            << std::endl;
}

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
    move_cursor(display, rect.x + rx_catch, rect.y + ry_catch);
    XSync(display, false);
    mouse_click(display, 1);
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    // 離す
    const int rx_release = rand_x(mt);
    const int ry_release = rand_y(mt);
    std::cerr << "rx_release: " << rx_release << " ry_release: " << ry_release << std::endl;
    move_cursor(display, rect.x + rx_release, rect.y + ry_release);
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
