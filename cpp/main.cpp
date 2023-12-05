#include <X11/Xlib.h>
#include <chrono>
#include <iostream>
#include <thread>

void move_cursor(Display *display, int x, int y) {
  std::cerr << "move_cursor (" << x << ", " << y << ")" << std::endl;
  XWarpPointer(display, None, DefaultRootWindow(display), 0, 0, 0, 0, x, y);
}

int main() {
  Display *display = XOpenDisplay(nullptr);

  if (!display) {
    std::cout << "Cannot open display" << std::endl;
    return 1;
  }

  for (int64_t i = 0; i < 10; i++) {
    move_cursor(display, 10, 10);

    XSync(display, false);

    std::this_thread::sleep_for(std::chrono::seconds(1));
  }

  XCloseDisplay(display);
}
