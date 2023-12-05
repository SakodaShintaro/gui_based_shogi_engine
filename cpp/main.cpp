#include <chrono>
#include <iostream>
#include <thread>

#include <opencv2/opencv.hpp>

// clang-format off
#include <X11/Xlib.h>
#include <X11/Xutil.h>
// clang-format on

void move_cursor(Display *display, int x, int y) {
  std::cerr << "move_cursor (" << x << ", " << y << ")" << std::endl;
  XWarpPointer(display, None, DefaultRootWindow(display), 0, 0, 0, 0, x, y);
}

cv::Mat get_screenshot(Display *display, int x, int y, int width, int height) {
  XImage *image =
      XGetImage(display, RootWindow(display, DefaultScreen(display)), x, y,
                width, height, AllPlanes, ZPixmap);

  cv::Mat mat(height, width, CV_8UC4);

  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {

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

int main() {
  Display *display = XOpenDisplay(nullptr);

  if (!display) {
    std::cout << "Cannot open display" << std::endl;
    return 1;
  }

  for (int64_t i = 0; i < 5; i++) {
    move_cursor(display, 10, 10);

    XSync(display, false);

    std::this_thread::sleep_for(std::chrono::seconds(1));
  }

  cv::Mat image = get_screenshot(display, 0, 0, 1920, 1080);

  cv::imwrite("screenshot.png", image);

  XCloseDisplay(display);
}
