#include <neat.h>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace fs = std::filesystem;
namespace neat = simaai::neat;

constexpr int kInputW = 640;
constexpr int kInputH = 640;
constexpr int kNumClasses = 80;
constexpr int kRegMax = 16;

const std::array<const char*, 80> kCocoClasses = {
    "person",      "bicycle",      "car",           "motorcycle",    "airplane",
    "bus",         "train",        "truck",         "boat",          "traffic light",
    "fire hydrant", "stop sign",   "parking meter", "bench",         "bird",
    "cat",         "dog",          "horse",         "sheep",         "cow",
    "elephant",    "bear",         "zebra",         "giraffe",       "backpack",
    "umbrella",    "handbag",      "tie",           "suitcase",      "frisbee",
    "skis",        "snowboard",    "sports ball",   "kite",          "baseball bat",
    "baseball glove", "skateboard", "surfboard",    "tennis racket", "bottle",
    "wine glass",  "cup",          "fork",          "knife",         "spoon",
    "bowl",        "banana",       "apple",         "sandwich",      "orange",
    "broccoli",    "carrot",       "hot dog",       "pizza",         "donut",
    "cake",        "chair",        "couch",         "potted plant",  "bed",
    "dining table", "toilet",      "tv",            "laptop",        "mouse",
    "remote",      "keyboard",     "cell phone",    "microwave",     "oven",
    "toaster",     "sink",         "refrigerator",  "book",          "clock",
    "vase",        "scissors",     "teddy bear",    "hair drier",    "toothbrush"};

struct Args {
  fs::path model = "build/yolov8x-p2_opt_4o/yolov8x-p2_opt_4o_mpk.tar.gz";
  fs::path images = "test_images";
  fs::path output = "neat/cpp/results";
  float conf_threshold = 0.50F;
  float iou_threshold = 0.50F;
  int timeout_ms = 30000;
  int limit = -1;
};

struct TensorData {
  std::vector<int64_t> shape;
  std::vector<float> values;

  float at(int n, int h, int w, int c) const {
    const int height = static_cast<int>(shape.at(1));
    const int width = static_cast<int>(shape.at(2));
    const int channels = static_cast<int>(shape.at(3));
    const size_t index =
        ((static_cast<size_t>(n) * height + h) * width + w) * channels + c;
    return values.at(index);
  }
};

struct Detection {
  cv::Rect2f box;
  float score = 0.0F;
  int class_id = 0;
};

[[noreturn]] void usage(const char* argv0) {
  std::cerr << "Usage: " << argv0
            << " [--model PATH] [--images DIR] [--output DIR] [--limit N]"
               " [--conf FLOAT] [--iou FLOAT] [--timeout-ms N]\n";
  std::exit(2);
}

Args parse_args(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    const std::string key = argv[i];
    auto require_value = [&](const std::string& flag) -> std::string {
      if (++i >= argc) {
        throw std::runtime_error("Missing value for " + flag);
      }
      return argv[i];
    };

    if (key == "--model" || key == "-m") {
      args.model = require_value(key);
    } else if (key == "--images" || key == "-i") {
      args.images = require_value(key);
    } else if (key == "--output" || key == "-o") {
      args.output = require_value(key);
    } else if (key == "--limit" || key == "-l") {
      args.limit = std::stoi(require_value(key));
    } else if (key == "--conf" || key == "-ct") {
      args.conf_threshold = std::stof(require_value(key));
    } else if (key == "--iou" || key == "-it") {
      args.iou_threshold = std::stof(require_value(key));
    } else if (key == "--timeout-ms") {
      args.timeout_ms = std::stoi(require_value(key));
    } else if (key == "--help" || key == "-h") {
      usage(argv[0]);
    } else {
      throw std::runtime_error("Unknown argument: " + key);
    }
  }
  return args;
}

bool has_image_extension(const fs::path& path) {
  std::string ext = path.extension().string();
  std::transform(ext.begin(), ext.end(), ext.begin(),
                 [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
  return ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp" ||
         ext == ".tif" || ext == ".tiff";
}

std::vector<fs::path> get_image_paths(const fs::path& folder, int limit) {
  if (!fs::is_directory(folder)) {
    throw std::runtime_error("Input directory does not exist: " + folder.string());
  }

  std::vector<fs::path> paths;
  for (const auto& entry : fs::directory_iterator(folder)) {
    if (entry.is_regular_file() && has_image_extension(entry.path())) {
      paths.push_back(entry.path());
    }
  }
  std::sort(paths.begin(), paths.end());
  if (limit >= 0 && static_cast<int>(paths.size()) > limit) {
    paths.resize(static_cast<size_t>(limit));
  }
  return paths;
}

void prepare_output_dir(const fs::path& output_dir) {
  if (fs::exists(output_dir)) {
    fs::remove_all(output_dir);
  }
  fs::create_directories(output_dir);
}

cv::Mat bgr_to_rgb(const cv::Mat& img_bgr) {
  cv::Mat rgb;
  cv::cvtColor(img_bgr, rgb, cv::COLOR_BGR2RGB);
  return rgb;
}

cv::Mat make_overlay_image(const cv::Mat& img_bgr) {
  const int orig_h = img_bgr.rows;
  const int orig_w = img_bgr.cols;
  const float scale =
      std::min(kInputW / static_cast<float>(orig_w), kInputH / static_cast<float>(orig_h));
  const int new_w = static_cast<int>(std::round(orig_w * scale));
  const int new_h = static_cast<int>(std::round(orig_h * scale));

  cv::Mat resized;
  if (orig_w != new_w || orig_h != new_h) {
    cv::resize(img_bgr, resized, cv::Size(new_w, new_h), 0.0, 0.0, cv::INTER_LINEAR);
  } else {
    resized = img_bgr.clone();
  }

  const int dw = kInputW - new_w;
  const int dh = kInputH - new_h;
  const int left = dw / 2;
  const int right = dw - left;
  const int top = dh / 2;
  const int bottom = dh - top;

  cv::Mat padded_bgr;
  cv::copyMakeBorder(resized, padded_bgr, top, bottom, left, right, cv::BORDER_CONSTANT,
                     cv::Scalar(114, 114, 114));

  return padded_bgr;
}

float sigmoid(float x) {
  return 1.0F / (1.0F + std::exp(-x));
}

std::array<float, kRegMax> softmax_bins(const TensorData& tensor, int h, int w, int side) {
  std::array<float, kRegMax> out{};
  float max_value = tensor.at(0, h, w, side * kRegMax);
  for (int i = 1; i < kRegMax; ++i) {
    max_value = std::max(max_value, tensor.at(0, h, w, side * kRegMax + i));
  }

  float sum = 0.0F;
  for (int i = 0; i < kRegMax; ++i) {
    out[i] = std::exp(tensor.at(0, h, w, side * kRegMax + i) - max_value);
    sum += out[i];
  }
  for (float& value : out) {
    value /= sum;
  }
  return out;
}

float expected_distance(const TensorData& tensor, int h, int w, int side, int stride) {
  const auto probs = softmax_bins(tensor, h, w, side);
  float expected = 0.0F;
  for (int i = 0; i < kRegMax; ++i) {
    expected += probs[static_cast<size_t>(i)] * static_cast<float>(i);
  }
  return expected * static_cast<float>(stride);
}

float iou(const cv::Rect2f& a, const cv::Rect2f& b) {
  const float x1 = std::max(a.x, b.x);
  const float y1 = std::max(a.y, b.y);
  const float x2 = std::min(a.x + a.width, b.x + b.width);
  const float y2 = std::min(a.y + a.height, b.y + b.height);
  const float inter_w = std::max(0.0F, x2 - x1);
  const float inter_h = std::max(0.0F, y2 - y1);
  const float inter = inter_w * inter_h;
  const float area_a = std::max(0.0F, a.width) * std::max(0.0F, a.height);
  const float area_b = std::max(0.0F, b.width) * std::max(0.0F, b.height);
  return inter / std::max(area_a + area_b - inter, 1.0e-7F);
}

std::vector<Detection> nms(const std::vector<Detection>& detections, float iou_threshold) {
  std::vector<size_t> order(detections.size());
  std::iota(order.begin(), order.end(), size_t{0});
  std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
    if (detections[a].score == detections[b].score) {
      return a > b;
    }
    return detections[a].score > detections[b].score;
  });
  std::vector<Detection> keep;
  std::vector<bool> suppressed(detections.size(), false);
  for (size_t order_i = 0; order_i < order.size(); ++order_i) {
    const size_t i = order[order_i];
    if (suppressed[i]) {
      continue;
    }
    keep.push_back(detections[i]);
    for (size_t order_j = order_i + 1; order_j < order.size(); ++order_j) {
      const size_t j = order[order_j];
      if (!suppressed[j] && iou(detections[i].box, detections[j].box) >= iou_threshold) {
        suppressed[j] = true;
      }
    }
  }
  return keep;
}

std::vector<Detection> postprocess_yolov8x_p2_4o(const std::vector<TensorData>& outputs,
                                                 float conf_threshold,
                                                 float iou_threshold) {
  if (outputs.size() != 8) {
    throw std::runtime_error("Expected 8 output tensors, got " + std::to_string(outputs.size()));
  }

  const std::array<int, 4> strides = {4, 8, 16, 32};
  std::vector<Detection> candidates;

  for (int level = 0; level < 4; ++level) {
    const TensorData& bbox = outputs[static_cast<size_t>(level)];
    const TensorData& cls = outputs[static_cast<size_t>(level + 4)];
    const int height = static_cast<int>(bbox.shape.at(1));
    const int width = static_cast<int>(bbox.shape.at(2));
    const int stride = strides[static_cast<size_t>(level)];

    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        int class_id = 0;
        float score = sigmoid(cls.at(0, y, x, 0));
        for (int c = 1; c < kNumClasses; ++c) {
          const float class_score = sigmoid(cls.at(0, y, x, c));
          if (class_score > score) {
            score = class_score;
            class_id = c;
          }
        }
        if (score < conf_threshold) {
          continue;
        }

        const float center_x = (static_cast<float>(x) + 0.5F) * static_cast<float>(stride);
        const float center_y = (static_cast<float>(y) + 0.5F) * static_cast<float>(stride);
        const float left = expected_distance(bbox, y, x, 0, stride);
        const float top = expected_distance(bbox, y, x, 1, stride);
        const float right = expected_distance(bbox, y, x, 2, stride);
        const float bottom = expected_distance(bbox, y, x, 3, stride);

        const float x1 = std::clamp(center_x - left, 0.0F, static_cast<float>(kInputW - 1));
        const float y1 = std::clamp(center_y - top, 0.0F, static_cast<float>(kInputH - 1));
        const float x2 = std::clamp(center_x + right, 0.0F, static_cast<float>(kInputW - 1));
        const float y2 = std::clamp(center_y + bottom, 0.0F, static_cast<float>(kInputH - 1));
        candidates.push_back({cv::Rect2f(x1, y1, std::max(0.0F, x2 - x1),
                                         std::max(0.0F, y2 - y1)),
                              score, class_id});
      }
    }
  }

  return nms(candidates, iou_threshold);
}

cv::Mat draw_detections(const cv::Mat& img_bgr, const std::vector<Detection>& detections) {
  cv::Mat out = img_bgr.clone();
  const cv::Scalar color(0, 255, 0);

  for (const Detection& det : detections) {
    const int x1 = std::clamp(static_cast<int>(det.box.x), 0, out.cols - 1);
    const int y1 = std::clamp(static_cast<int>(det.box.y), 0, out.rows - 1);
    const int x2 =
        std::clamp(static_cast<int>(det.box.x + det.box.width), 0, out.cols - 1);
    const int y2 =
        std::clamp(static_cast<int>(det.box.y + det.box.height), 0, out.rows - 1);
    cv::rectangle(out, cv::Point(x1, y1), cv::Point(x2, y2), color, 2);

    const std::string class_name =
        det.class_id >= 0 && det.class_id < static_cast<int>(kCocoClasses.size())
            ? kCocoClasses[static_cast<size_t>(det.class_id)]
            : "id_" + std::to_string(det.class_id);
    char score_text[32];
    std::snprintf(score_text, sizeof(score_text), ":%.2f", det.score);
    const std::string label = class_name + score_text;

    int baseline = 0;
    const cv::Size text_size =
        cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
    const int top = std::max(0, y1 - text_size.height - baseline);
    cv::rectangle(out, cv::Point(x1, top),
                  cv::Point(std::min(out.cols - 1, x1 + text_size.width), y1), color,
                  cv::FILLED);
    cv::putText(out, label, cv::Point(x1, y1 - baseline), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
  }
  return out;
}

TensorData tensor_to_data(const neat::Tensor& tensor) {
  neat::Tensor cpu_tensor = tensor.to_cpu_if_needed().contiguous();
  if (cpu_tensor.dtype != neat::TensorDType::Float32) {
    throw std::runtime_error("Expected Float32 output tensor");
  }

  const std::vector<uint8_t> bytes = cpu_tensor.copy_dense_bytes_tight();
  TensorData out;
  out.shape = cpu_tensor.shape;
  out.values.resize(bytes.size() / sizeof(float));
  std::memcpy(out.values.data(), bytes.data(), bytes.size());

  if (out.shape.size() != 4 || out.shape.at(0) != 1) {
    throw std::runtime_error("Unexpected output tensor rank/shape");
  }
  return out;
}

std::vector<TensorData> sample_to_outputs(const neat::Sample& sample) {
  if (sample.kind == neat::SampleKind::Tensor) {
    if (!sample.tensor) {
      throw std::runtime_error("Tensor sample has no tensor payload");
    }
    return {tensor_to_data(*sample.tensor)};
  }

  if (sample.kind != neat::SampleKind::Bundle) {
    throw std::runtime_error("Unsupported output sample kind");
  }
  if (sample.fields.size() != 8) {
    throw std::runtime_error("Expected 8 output tensors, got " +
                             std::to_string(sample.fields.size()));
  }

  std::vector<neat::Sample> fields = sample.fields;
  std::sort(fields.begin(), fields.end(), [](const neat::Sample& a, const neat::Sample& b) {
    return a.output_index < b.output_index;
  });

  std::vector<TensorData> outputs;
  outputs.reserve(fields.size());
  for (const neat::Sample& field : fields) {
    if (field.kind != neat::SampleKind::Tensor || !field.tensor) {
      throw std::runtime_error("Output bundle field is not a tensor");
    }
    outputs.push_back(tensor_to_data(*field.tensor));
  }
  return outputs;
}

std::pair<int, int> input_bounds(const std::vector<fs::path>& image_paths) {
  int max_width = 0;
  int max_height = 0;
  for (const fs::path& image_path : image_paths) {
    const cv::Mat img_bgr = cv::imread(image_path.string(), cv::IMREAD_COLOR);
    if (img_bgr.empty()) {
      continue;
    }
    max_width = std::max(max_width, img_bgr.cols);
    max_height = std::max(max_height, img_bgr.rows);
  }
  if (max_width <= 0 || max_height <= 0) {
    throw std::runtime_error("Could not read any images to determine input bounds");
  }
  return {max_width, max_height};
}

neat::Model::Options model_options(int max_width, int max_height) {
  neat::Model::Options opts;
  opts.media_type = "video/x-raw";
  opts.format = "RGB";
  opts.original_width = kInputW;
  opts.original_height = kInputH;
  opts.input_max_width = max_width;
  opts.input_max_height = max_height;
  opts.input_max_depth = 3;

  opts.preproc.input_img_type = "RGB";
  opts.preproc.output_img_type = "RGB";
  opts.preproc.output_width = kInputW;
  opts.preproc.output_height = kInputH;
  opts.preproc.scaled_width = kInputW;
  opts.preproc.scaled_height = kInputH;
  opts.preproc.normalize = true;
  opts.preproc.aspect_ratio = true;
  opts.preproc.padding_type = "CENTER";
  opts.preproc.scaling_type = "BILINEAR";
  return opts;
}

neat::RunOptions run_options() {
  neat::RunOptions opts;
  opts.queue_depth = 4;
  opts.overflow_policy = neat::OverflowPolicy::Block;
  opts.output_memory = neat::OutputMemory::Auto;
  opts.advanced.copy_input = true;
  return opts;
}

neat::Run build_runner(neat::Model& model, const cv::Mat& input_rgb) {
  neat::Session session;
  session.add(model.session());
  neat::Tensor input_tensor =
      neat::Tensor::from_cv_mat(input_rgb, neat::ImageSpec::PixelFormat::RGB);
  return session.build(input_tensor, neat::RunMode::Sync, run_options());
}

int main(int argc, char** argv) {
  try {
    const Args args = parse_args(argc, argv);
    if (!fs::is_regular_file(args.model)) {
      throw std::runtime_error("Model archive not found: " + args.model.string());
    }

    const std::vector<fs::path> image_paths = get_image_paths(args.images, args.limit);
    if (image_paths.empty()) {
      throw std::runtime_error("No images found in: " + args.images.string());
    }
    prepare_output_dir(args.output);

    std::cout << "Found " << image_paths.size() << " image(s) in " << args.images << "\n";
    std::cout << "Results will be written to " << args.output << "\n";

    const auto [max_width, max_height] = input_bounds(image_paths);
    neat::Model model(args.model.string(), model_options(max_width, max_height));

    for (const fs::path& image_path : image_paths) {
      std::cout << "Processing image: " << image_path.filename().string() << "\n";
      const cv::Mat img_bgr = cv::imread(image_path.string(), cv::IMREAD_COLOR);
      if (img_bgr.empty()) {
        std::cout << "  WARNING: Could not read image, skipping: " << image_path << "\n";
        continue;
      }

      cv::Mat rgb = bgr_to_rgb(img_bgr);
      neat::Run runner = build_runner(model, rgb);
      neat::Tensor input_tensor =
          neat::Tensor::from_cv_mat(rgb, neat::ImageSpec::PixelFormat::RGB);
      try {
        if (!runner.push(input_tensor)) {
          throw std::runtime_error("NEAT pipeline rejected input: " + image_path.string());
        }

        std::optional<neat::Sample> sample = runner.pull(args.timeout_ms);
        if (!sample) {
          throw std::runtime_error("Timed out waiting for NEAT output");
        }

        const std::vector<TensorData> outputs = sample_to_outputs(*sample);
        const std::vector<Detection> detections =
            postprocess_yolov8x_p2_4o(outputs, args.conf_threshold, args.iou_threshold);

        if (detections.empty()) {
          std::cout << "  No detections above confidence threshold.\n";
        } else {
          std::cout << "  Detections: " << detections.size() << "\n";
        }

        const cv::Mat annotated = draw_detections(make_overlay_image(img_bgr), detections);
        const fs::path output_path = args.output / image_path.filename();
        if (!cv::imwrite(output_path.string(), annotated)) {
          throw std::runtime_error("Failed to write output image: " + output_path.string());
        }
        std::cout << "  Annotated image written to: " << output_path << "\n";
      } catch (...) {
        runner.close();
        throw;
      }
      runner.close();
    }

    return 0;
  } catch (const std::exception& exc) {
    std::cerr << "ERROR: " << exc.what() << "\n";
    return 1;
  }
}
