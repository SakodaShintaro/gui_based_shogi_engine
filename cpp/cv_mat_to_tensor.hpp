#include <opencv2/core/core.hpp>

#include <torch/torch.h>

#include <vector>

torch::Tensor cv_mat_to_tensor(const cv::Mat & cv_img)
{
  const int height = cv_img.rows;
  const int width = cv_img.cols;
  const int channels = cv_img.channels();

  std::vector<uint8_t> data(height * width * channels);

  // cv::Matデータをvectorにコピー
  memcpy(&data[0], cv_img.data, sizeof(uint8_t) * data.size());

  // torch::Tensorを作成
  torch::Tensor tensor_img = torch::from_blob(&data[0], {height, width, channels});
  tensor_img = tensor_img.permute({2, 0, 1});   // channels firstに並び替え
  tensor_img = tensor_img.to(torch::kFloat32);  // float32型に変換

  return tensor_img;
}
