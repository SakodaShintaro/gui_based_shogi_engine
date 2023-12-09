#ifndef ACTION_HPP_
#define ACTION_HPP_

#include <cstdint>

// [上, 右, 下, 左, クリック]
enum Action { kUp, kRight, kDown, kLeft, kClick, kActionSize };

constexpr int64_t kMoveUnit = 40;

#endif
