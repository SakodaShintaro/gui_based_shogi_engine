#include <chrono>
#include <iostream>
#include <thread>

#include <opencv2/opencv.hpp>

// clang-format off
#include <X11/Xlib.h>
#include <X11/Xutil.h>
// clang-format on

struct Rect {
  int x;
  int y;
  int width;
  int height;
};

void move_cursor(Display *display, int x, int y) {
  std::cerr << "move_cursor (" << x << ", " << y << ")" << std::endl;
  XWarpPointer(display, None, DefaultRootWindow(display), 0, 0, 0, 0, x, y);
}

cv::Mat get_screenshot(Display *display, const Rect &rect) {
  XImage *image =
      XGetImage(display, RootWindow(display, DefaultScreen(display)), rect.x,
                rect.y, rect.width, rect.height, AllPlanes, ZPixmap);

  cv::Mat mat(rect.height, rect.width, CV_8UC4);

  for (int y = 0; y < rect.height; y++) {
    for (int x = 0; x < rect.width; x++) {

      unsigned long pixel = XGetPixel(image, x, y);

      unsigned char blue = pixel & image->blue_mask;
      unsigned char green = (pixel & image->green_mask) >> 8;
      unsigned char red = (pixel & image->red_mask) >> 16;

      mat.at<cv::Vec4b>(y, x) = cv::Vec4b(blue, green, red, 255);
    }
  }

  XDestroyImage(image);

  return mat;
}

Window get_active_window(Display *display) {
  Window focused;
  int revert;
  XGetInputFocus(display, &focused, &revert);

  XWindowAttributes attrs;
  XGetWindowAttributes(display, focused, &attrs);

  return focused;
}

Rect get_window_rect(Display *display, Window window) {
  XWindowAttributes attrs;
  XGetWindowAttributes(display, window, &attrs);

  int x, y;
  Window child;
  XTranslateCoordinates(display, window,
                        RootWindow(display, DefaultScreen(display)), 0, 0, &x,
                        &y, &child);

  return Rect{x, y, attrs.width, attrs.height};
}

std::string get_window_title(Display *display, Window window) {
  Atom name_atom = XInternAtom(display, "WM_NAME", False);

  Atom type;
  unsigned long bytes_after;
  int len;
  unsigned long bytes_after_return;
  unsigned char *name;
  XGetWindowProperty(display, window, name_atom, 0, 1024, False,
                     AnyPropertyType, &type, &len, &bytes_after,
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

int main() {
  Display *display = XOpenDisplay(nullptr);

  if (!display) {
    std::cout << "Cannot open display" << std::endl;
    return 1;
  }

  while (true) {
    auto now = std::chrono::system_clock::now();
    std::time_t end_time = std::chrono::system_clock::to_time_t(now);

    Window window = get_active_window(display);
    const Rect rect = get_window_rect(display, window);
    const std::string title = get_window_title(display, window);
    std::cerr << std::ctime(&end_time) << " " << title // time and title
              << "\trect: (" << rect.x << ", " << rect.y << ", " << rect.width
              << ", " << rect.height << ")\n" << std::endl;

    const std::string key = "将棋所";
    const std::size_t pos = title.find(key);
    if (pos == std::string::npos) {
      std::this_thread::sleep_for(std::chrono::seconds(1));
      continue;
    }

    const cv::Mat image = get_screenshot(display, rect);
    cv::imwrite("screenshot.png", image);

    std::this_thread::sleep_for(std::chrono::seconds(1));
  }

  for (int64_t i = 0; i < 5; i++) {
    move_cursor(display, 10, 10);
    XSync(display, false);
  }

  XCloseDisplay(display);
}
