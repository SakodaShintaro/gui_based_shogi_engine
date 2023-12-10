#include "../cpp/window_size.hpp"

#include <Siv3D.hpp>

void Main()
{
  s3d::Circle circle;
  ColorF circle_color;
  Window::Resize(kWindowWidth, kWindowHeight);
  const int64_t radius = std::min(kWindowWidth, kWindowHeight) / 2;

  auto update = [&]() {
    // Scene::SetBackground(RandomColorF());
    Scene::SetBackground(s3d::Palette::White);
    const s3d::Size size = Window::GetState().frameBufferSize;
    const int64_t x = s3d::Random(size.x - 1);
    const int64_t y = s3d::Random(size.y - 1);
    circle = s3d::Circle{x, y, radius};
    // circle_color = RandomColorF();
    circle_color = s3d::Palette::Blue;
  };

  update();

  while (System::Update()) {
    circle.draw(circle_color);
    circle.drawFrame(2.0, Palette::Black);
    if (circle.leftClicked()) {
      update();
    }
  }
}
