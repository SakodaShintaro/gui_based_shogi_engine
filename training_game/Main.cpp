#include <Siv3D.hpp>

void Main()
{
  s3d::Circle circle;
  ColorF circle_color;

  auto update = [&]() {
    Scene::SetBackground(RandomColorF());
    const s3d::Size size = Window::GetState().frameBufferSize;
    const int64_t x = s3d::Random(size.x - 1);
    const int64_t y = s3d::Random(size.y - 1);
    circle = s3d::Circle{x, y, 50.0};
    circle_color = RandomColorF();
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
