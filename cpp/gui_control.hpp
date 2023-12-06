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

void warp_cursor(Display * display, int x, int y)
{
  XWarpPointer(display, None, DefaultRootWindow(display), 0, 0, 0, 0, x, y);
  XSync(display, false);
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

cv::Point get_current_cursor_abs_position(Display * display)
{
  Window root_return, child_return;
  int root_x_return, root_y_return, win_x_return, win_y_return;
  unsigned int mask_return;

  XQueryPointer(
    display, RootWindow(display, DefaultScreen(display)), &root_return, &child_return,
    &root_x_return, &root_y_return, &win_x_return, &win_y_return, &mask_return);
  return cv::Point(root_x_return, root_y_return);
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

#endif  // GUI_CONTROL_HPP_
