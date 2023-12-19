#ifndef GRID_WORLD_HPP_
#define GRID_WORLD_HPP_

#include "action.hpp"

#include <torch/torch.h>

#include <random>

struct Position
{
  int64_t x, y;
  bool operator==(const Position rhs) const { return (x == rhs.x && y == rhs.y); }
};

class GridWorld
{
public:
  GridWorld(const int64_t grid_size);

  void print() const;

  bool step(const Action action);

  torch::Tensor state() const;

  bool is_ideal_action(const Action action) const;

  int64_t state_as_int() const
  {
    const int64_t index_self = self_position_.y * grid_size_ + self_position_.x;
    const int64_t index_goal = goal_position_.y * grid_size_ + goal_position_.x;
    const int64_t s = index_self * (grid_size_ * grid_size_) + index_goal;
    return s;
  }

private:
  const int64_t grid_size_;
  Position self_position_;
  Position goal_position_;
  std::mt19937_64 engine_;
  std::uniform_int_distribution<int64_t> dist_;
};

#endif  // GRID_WORLD_HPP_
