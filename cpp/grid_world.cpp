#include "grid_world.hpp"

GridWorld::GridWorld(const int64_t grid_size)
: grid_size_(grid_size), engine_(std::random_device{}()), dist_(0, grid_size - 1)
{
  self_position_.x = dist_(engine_);
  self_position_.y = dist_(engine_);
  goal_position_.x = dist_(engine_);
  goal_position_.y = dist_(engine_);
}

void GridWorld::print() const
{
  std::cout << "\nstate" << std::endl;
  std::vector<std::vector<int64_t>> v(grid_size_, std::vector<int64_t>(grid_size_, 0));
  v[self_position_.y][self_position_.x] += 1;
  v[goal_position_.y][goal_position_.x] += 2;
  for (int64_t i = 0; i < grid_size_; i++) {
    for (int64_t j = 0; j < grid_size_; j++) {
      std::cout << v[i][j];
    }
    std::cout << std::endl;
  }
}

bool GridWorld::step(const Action action)
{
  if (action == kClick) {
    const bool same = (self_position_ == goal_position_);
    if (same) {
      goal_position_.x = dist_(engine_);
      goal_position_.y = dist_(engine_);
      return true;
    } else {
      return false;
    }
  }

  if (action == kUp) {
    self_position_.y--;
  } else if (action == kRight) {
    self_position_.x++;
  } else if (action == kDown) {
    self_position_.y++;
  } else if (action == kLeft) {
    self_position_.x--;
  }
  self_position_.x = std::clamp(self_position_.x, (int64_t)0, grid_size_ - 1);
  self_position_.y = std::clamp(self_position_.y, (int64_t)0, grid_size_ - 1);
  return false;
}

torch::Tensor GridWorld::state() const
{
  torch::Tensor state = torch::zeros({2, grid_size_, grid_size_});
  state[0][self_position_.y][self_position_.y] = 1;
  state[1][goal_position_.y][goal_position_.y] = 1;
  return state;
}

bool GridWorld::is_ideal_action(const Action action) const
{
  const bool same = (self_position_ == goal_position_);
  if (same) {
    return (action == kClick);
  } else {
    const int64_t dx = goal_position_.x - self_position_.x;
    const int64_t dy = goal_position_.y - self_position_.y;
    if (action == kUp) {
      return (dy < 0);
    } else if (action == kRight) {
      return (dx > 0);
    } else if (action == kDown) {
      return (dy > 0);
    } else if (action == kLeft) {
      return (dx < 0);
    }
  }
}
