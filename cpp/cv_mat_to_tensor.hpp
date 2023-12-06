#include <opencv2/core/core.hpp>

#include <torch/torch.h>

#include <vector>

torch::Tensor cv_mat_to_tensor(const cv::Mat & cv_img)
{
  const int height = cv_img.rows;
  const int width = cv_img.cols;
  const int channels = cv_img.channels();
  torch::Tensor tensor_img =
    torch::from_blob(cv_img.data, {height, width, channels}, torch::kByte).clone();
  tensor_img = tensor_img.to(torch::kFloat32) / 255.0;
  tensor_img = tensor_img.permute({2, 0, 1});
  return tensor_img;
}
