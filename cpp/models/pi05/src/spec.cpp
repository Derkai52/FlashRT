#include "flashrt/cpp/models/pi05/spec.h"

#include <algorithm>

namespace flashrt {
namespace models {
namespace pi05 {

modalities::VisionPreprocessSpec vision_preprocess_spec(int num_views) {
    modalities::VisionPreprocessSpec spec;
    // 1–3: OpenPI/LIBERO names. 5: Sculptor sculptor_0911 camera keys.
    // 4 is rejected by resolve_pi05_shape / native open.
    static const char* kViews3[] = {"image", "wrist_image", "wrist_image_right"};
    static const char* kViews5[] = {
        "base_0_rgb", "base_1_rgb", "base_2_rgb", "left_wrist_0_rgb",
        "right_wrist_0_rgb"};
    if (num_views < 1) num_views = 1;
    if (num_views > 5) num_views = 5;
    if (num_views == 4) num_views = 3;
    const char* const* views = (num_views == 5) ? kViews5 : kViews3;
    spec.view_order.reserve(static_cast<std::size_t>(num_views));
    for (int i = 0; i < num_views; ++i) spec.view_order.emplace_back(views[i]);
    spec.target_width = kImageSize;
    spec.target_height = kImageSize;
    spec.output_dtype = modalities::DType::kBFloat16;
    spec.output_layout = modalities::Layout::kNHWC;
    spec.normalize.mode = modalities::NormalizeMode::kDivideShift;
    spec.normalize.divisor = 127.5f;
    spec.normalize.shift = -1.0f;
    spec.require_exact_views = true;
    return spec;
}

modalities::ActionPostprocessSpec action_postprocess_spec(
    const std::vector<float>& mean,
    const std::vector<float>& stddev,
    int chunk,
    int model_dim,
    int robot_dim) {
    modalities::ActionPostprocessSpec spec;
    spec.chunk = chunk;
    spec.model_dim = model_dim;
    spec.robot_dim = robot_dim;
    spec.schema = "eef_delta_xyz_rpy_gripper";
    spec.mean = mean;
    spec.stddev = stddev;
    spec.clip_model_input = true;
    spec.model_input_min = -1.0f;
    spec.model_input_max = 1.0f;
    return spec;
}

}  // namespace pi05
}  // namespace models
}  // namespace flashrt
