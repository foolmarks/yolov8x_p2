#include "neat.h"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

constexpr int kInputW = 640;
constexpr int kInputH = 640;
constexpr float kConfThreshold = 0.25f;
constexpr float kNmsThreshold = 0.45f;
constexpr int kTimeoutMs = 30000;

const std::array<std::string, 80> kCocoClasses = {
    "person",        "bicycle",      "car",           "motorcycle",    "airplane",
    "bus",           "train",        "truck",         "boat",          "traffic light",
    "fire hydrant",  "stop sign",    "parking meter", "bench",         "bird",
    "cat",           "dog",          "horse",         "sheep",         "cow",
    "elephant",      "bear",         "zebra",         "giraffe",       "backpack",
    "umbrella",      "handbag",      "tie",           "suitcase",      "frisbee",
    "skis",          "snowboard",    "sports ball",   "kite",          "baseball bat",
    "baseball glove", "skateboard",  "surfboard",     "tennis racket", "bottle",
    "wine glass",    "cup",          "fork",          "knife",         "spoon",
    "bowl",          "banana",       "apple",         "sandwich",      "orange",
    "broccoli",      "carrot",       "hot dog",       "pizza",         "donut",
    "cake",          "chair",        "couch",         "potted plant",  "bed",
    "dining table",  "toilet",       "tv",            "laptop",        "mouse",
    "remote",        "keyboard",     "cell phone",    "microwave",     "oven",
    "toaster",       "sink",         "refrigerator",  "book",          "clock",
    "vase",          "scissors",     "teddy bear",    "hair drier",    "toothbrush"};

struct Args {
  fs::path model;
  fs::path images;
  fs::path results;
};

struct TensorHWC {
  int h = 0;
  int w = 0;
  int c = 0;
  std::vector<float> data;
};

struct Detection {
  float x1 = 0.0f;
  float y1 = 0.0f;
  float x2 = 0.0f;
  float y2 = 0.0f;
  float score = 0.0f;
  int class_id = -1;
};

fs::path executable_path(const char* argv0) {
  fs::path p(argv0 ? argv0 : "");
  if (p.is_relative()) {
    p = fs::current_path() / p;
  }
  std::error_code ec;
  fs::path resolved = fs::weakly_canonical(p, ec);
  return ec ? p.lexically_normal() : resolved;
}

fs::path repo_root_from_executable(const char* argv0) {
  fs::path dir = executable_path(argv0).parent_path();
  for (int i = 0; i < 3 && !dir.empty(); ++i) {
    dir = dir.parent_path();
  }
  return dir.empty() ? fs::current_path() : dir;
}

bool get_arg(int argc, char** argv, const std::string& key, std::string& out) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (key == argv[i]) {
      out = argv[i + 1];
      return true;
    }
  }
  return false;
}

Args parse_args(int argc, char** argv) {
  const fs::path root = repo_root_from_executable(argv[0]);
  Args args;
  args.model = root / "build/yolov8x-p2_opt_4o/yolov8x-p2_opt_4o_mpk.tar.gz";
  args.images = root / "test_images";
  args.results = root / "neat/cpp/results";

  std::string value;
  if (get_arg(argc, argv, "--model", value)) {
    args.model = value;
  }
  if (get_arg(argc, argv, "--images", value)) {
    args.images = value;
  }
  if (get_arg(argc, argv, "--results", value)) {
    args.results = value;
  }
  return args;
}

bool is_image_path(const fs::path& path) {
  std::string ext = path.extension().string();
  for (char& c : ext) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  return ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp" ||
         ext == ".tif" || ext == ".tiff";
}

std::vector<fs::path> get_image_paths(const fs::path& folder) {
  if (!fs::is_directory(folder)) {
    throw std::runtime_error("input directory does not exist: " + folder.string());
  }

  std::vector<fs::path> paths;
  for (const auto& entry : fs::directory_iterator(folder)) {
    if (entry.is_regular_file() && is_image_path(entry.path())) {
      paths.push_back(entry.path());
    }
  }
  std::sort(paths.begin(), paths.end());
  return paths;
}

void prepare_output_dir(const fs::path& output_dir) {
  std::error_code ec;
  fs::remove_all(output_dir, ec);
  if (ec) {
    throw std::runtime_error("failed to remove output directory: " + output_dir.string());
  }
  fs::create_directories(output_dir, ec);
  if (ec) {
    throw std::runtime_error("failed to create output directory: " + output_dir.string());
  }
}

cv::Mat read_bgr(const fs::path& image_path) {
  cv::Mat image = cv::imread(image_path.string(), cv::IMREAD_COLOR);
  if (image.empty()) {
    throw std::runtime_error("failed to read image: " + image_path.string());
  }
  return image;
}

cv::Mat bgr_to_rgb_contiguous(const cv::Mat& bgr) {
  cv::Mat rgb;
  cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
  return rgb.isContinuous() ? rgb : rgb.clone();
}

cv::Mat letterbox_bgr(const cv::Mat& bgr) {
  const int src_w = bgr.cols;
  const int src_h = bgr.rows;
  const float scale =
      std::min(static_cast<float>(kInputW) / static_cast<float>(src_w),
               static_cast<float>(kInputH) / static_cast<float>(src_h));
  const int resized_w = static_cast<int>(std::round(static_cast<float>(src_w) * scale));
  const int resized_h = static_cast<int>(std::round(static_cast<float>(src_h) * scale));

  cv::Mat resized;
  if (resized_w != src_w || resized_h != src_h) {
    cv::resize(bgr, resized, cv::Size(resized_w, resized_h), 0.0, 0.0, cv::INTER_LINEAR);
  } else {
    resized = bgr;
  }

  const int pad_w = kInputW - resized_w;
  const int pad_h = kInputH - resized_h;
  const int left = pad_w / 2;
  const int right = pad_w - left;
  const int top = pad_h / 2;
  const int bottom = pad_h - top;

  cv::Mat padded;
  cv::copyMakeBorder(resized, padded, top, bottom, left, right, cv::BORDER_CONSTANT,
                     cv::Scalar(114, 114, 114));
  return padded;
}

simaai::neat::Model::Options build_model_options(int max_width, int max_height) {
  simaai::neat::Model::Options opt;
  opt.media_type = "video/x-raw";
  opt.format = "RGB";
  opt.input_max_width = max_width;
  opt.input_max_height = max_height;
  opt.input_max_depth = 3;
  opt.preproc.input_width = max_width;
  opt.preproc.input_height = max_height;
  opt.preproc.output_width = kInputW;
  opt.preproc.output_height = kInputH;
  opt.preproc.normalize = true;
  opt.preproc.aspect_ratio = true;
  opt.preproc.input_img_type = "RGB";
  opt.preproc.output_img_type = "RGB";
  opt.preproc.scaling_type = "BILINEAR";
  opt.preproc.padding_type = "CENTER";
  return opt;
}

void set_input_limits(simaai::neat::InputOptions& input, int max_width, int max_height) {
  input.media_type = "video/x-raw";
  input.format = "RGB";
  input.width = max_width;
  input.height = max_height;
  input.depth = 3;
  input.max_width = max_width;
  input.max_height = max_height;
  input.max_depth = 3;
}

simaai::neat::Run build_sync_run(const simaai::neat::Model& model, const cv::Mat& seed_rgb,
                                 int max_width, int max_height) {
  simaai::neat::InputOptions input = model.input_appsrc_options(false);
  set_input_limits(input, max_width, max_height);

  simaai::neat::Session session;
  session.add(simaai::neat::nodes::Input(input));
  session.add(model.preprocess());
  session.add(simaai::neat::nodes::groups::MLA(model));
  session.add(simaai::neat::nodes::DetessDequant(simaai::neat::DetessDequantOptions(model)));
  session.add(simaai::neat::nodes::Output());

  simaai::neat::RunOptions options;
  options.queue_depth = 1;
  options.overflow_policy = simaai::neat::OverflowPolicy::Block;
  options.output_memory = simaai::neat::OutputMemory::Owned;

  simaai::neat::Tensor seed = simaai::neat::from_cv_mat(
      seed_rgb, simaai::neat::ImageSpec::PixelFormat::RGB, true);
  return session.build(seed, simaai::neat::RunMode::Sync, options);
}

std::vector<simaai::neat::Tensor> collect_tensors(const simaai::neat::Sample& sample) {
  if (sample.kind == simaai::neat::SampleKind::Tensor) {
    if (!sample.tensor.has_value()) {
      throw std::runtime_error("tensor sample missing payload");
    }
    return {*sample.tensor};
  }

  if (sample.kind == simaai::neat::SampleKind::Bundle) {
    std::vector<simaai::neat::Tensor> tensors;
    for (const auto& field : sample.fields) {
      std::vector<simaai::neat::Tensor> child = collect_tensors(field);
      tensors.insert(tensors.end(), child.begin(), child.end());
    }
    return tensors;
  }

  throw std::runtime_error("unexpected sample kind");
}

TensorHWC tensor_to_hwc_f32(const simaai::neat::Tensor& tensor) {
  if (tensor.dtype != simaai::neat::TensorDType::Float32) {
    throw std::runtime_error("expected Float32 model output tensor");
  }

  TensorHWC out;
  if (tensor.shape.size() == 4) {
    if (tensor.shape[0] != 1) {
      throw std::runtime_error("only batch=1 model outputs are supported");
    }
    out.h = static_cast<int>(tensor.shape[1]);
    out.w = static_cast<int>(tensor.shape[2]);
    out.c = static_cast<int>(tensor.shape[3]);
  } else if (tensor.shape.size() == 3) {
    out.h = static_cast<int>(tensor.shape[0]);
    out.w = static_cast<int>(tensor.shape[1]);
    out.c = static_cast<int>(tensor.shape[2]);
  } else {
    throw std::runtime_error("unexpected model output tensor rank");
  }

  const size_t elems =
      static_cast<size_t>(out.h) * static_cast<size_t>(out.w) * static_cast<size_t>(out.c);
  const std::vector<uint8_t> bytes = tensor.copy_dense_bytes_tight();
  if (bytes.size() < elems * sizeof(float)) {
    throw std::runtime_error("model output tensor byte size is smaller than expected");
  }
  out.data.resize(elems);
  std::memcpy(out.data.data(), bytes.data(), elems * sizeof(float));
  return out;
}

std::vector<TensorHWC> collect_yolo_outputs(const simaai::neat::Sample& sample) {
  const std::vector<simaai::neat::Tensor> tensors = collect_tensors(sample);
  if (tensors.size() != 8) {
    throw std::runtime_error("expected 8 dequantized output tensors, got " +
                             std::to_string(tensors.size()));
  }

  std::vector<TensorHWC> bbox;
  std::vector<TensorHWC> cls;
  for (const auto& tensor : tensors) {
    TensorHWC hwc = tensor_to_hwc_f32(tensor);
    if (hwc.c == 64) {
      bbox.push_back(std::move(hwc));
    } else if (hwc.c == 80) {
      cls.push_back(std::move(hwc));
    }
  }

  auto by_height_desc = [](const TensorHWC& a, const TensorHWC& b) { return a.h > b.h; };
  std::sort(bbox.begin(), bbox.end(), by_height_desc);
  std::sort(cls.begin(), cls.end(), by_height_desc);
  if (bbox.size() != 4 || cls.size() != 4) {
    throw std::runtime_error("unexpected YOLO output tensor channel layout");
  }

  std::vector<TensorHWC> ordered;
  ordered.reserve(8);
  ordered.insert(ordered.end(), std::make_move_iterator(bbox.begin()),
                 std::make_move_iterator(bbox.end()));
  ordered.insert(ordered.end(), std::make_move_iterator(cls.begin()),
                 std::make_move_iterator(cls.end()));
  return ordered;
}

float sigmoid(float x) {
  return 1.0f / (1.0f + std::exp(-x));
}

float dfl_distance_16(const float* logits) {
  float max_value = -std::numeric_limits<float>::infinity();
  for (int i = 0; i < 16; ++i) {
    max_value = std::max(max_value, logits[i]);
  }

  float denom = 0.0f;
  float numer = 0.0f;
  for (int i = 0; i < 16; ++i) {
    const float e = std::exp(logits[i] - max_value);
    denom += e;
    numer += static_cast<float>(i) * e;
  }
  return denom > 0.0f ? numer / denom : 0.0f;
}

float iou_xyxy(const Detection& a, const Detection& b) {
  const float x1 = std::max(a.x1, b.x1);
  const float y1 = std::max(a.y1, b.y1);
  const float x2 = std::min(a.x2, b.x2);
  const float y2 = std::min(a.y2, b.y2);
  const float w = std::max(0.0f, x2 - x1);
  const float h = std::max(0.0f, y2 - y1);
  const float inter = w * h;
  const float area_a = std::max(0.0f, a.x2 - a.x1) * std::max(0.0f, a.y2 - a.y1);
  const float area_b = std::max(0.0f, b.x2 - b.x1) * std::max(0.0f, b.y2 - b.y1);
  const float den = area_a + area_b - inter;
  return den > 0.0f ? inter / den : 0.0f;
}

std::vector<Detection> postprocess_yolov8x_p2_4o(const std::vector<TensorHWC>& outputs) {
  if (outputs.size() != 8) {
    throw std::runtime_error("expected 8 ordered YOLO outputs");
  }

  constexpr std::array<int, 4> strides = {4, 8, 16, 32};
  std::vector<Detection> candidates;
  candidates.reserve(4000);

  for (int level = 0; level < 4; ++level) {
    const TensorHWC& bbox = outputs[static_cast<size_t>(level)];
    const TensorHWC& cls = outputs[static_cast<size_t>(level + 4)];
    if (bbox.h != cls.h || bbox.w != cls.w || bbox.c != 64 || cls.c != 80) {
      throw std::runtime_error("unexpected YOLO output shape pairing");
    }

    for (int y = 0; y < bbox.h; ++y) {
      for (int x = 0; x < bbox.w; ++x) {
        const size_t cls_base =
            (static_cast<size_t>(y) * static_cast<size_t>(cls.w) + static_cast<size_t>(x)) *
            static_cast<size_t>(cls.c);

        int class_id = -1;
        float best_score = 0.0f;
        for (int c = 0; c < cls.c; ++c) {
          const float score = sigmoid(cls.data[cls_base + static_cast<size_t>(c)]);
          if (score > best_score) {
            best_score = score;
            class_id = c;
          }
        }
        if (best_score < kConfThreshold || class_id < 0) {
          continue;
        }

        const size_t bbox_base =
            (static_cast<size_t>(y) * static_cast<size_t>(bbox.w) + static_cast<size_t>(x)) * 64U;
        const float stride = static_cast<float>(strides[static_cast<size_t>(level)]);
        const float l = dfl_distance_16(&bbox.data[bbox_base + 0U]) * stride;
        const float t = dfl_distance_16(&bbox.data[bbox_base + 16U]) * stride;
        const float r = dfl_distance_16(&bbox.data[bbox_base + 32U]) * stride;
        const float b = dfl_distance_16(&bbox.data[bbox_base + 48U]) * stride;
        const float cx = (static_cast<float>(x) + 0.5f) * stride;
        const float cy = (static_cast<float>(y) + 0.5f) * stride;

        Detection det;
        det.x1 = std::clamp(cx - l, 0.0f, static_cast<float>(kInputW - 1));
        det.y1 = std::clamp(cy - t, 0.0f, static_cast<float>(kInputH - 1));
        det.x2 = std::clamp(cx + r, 0.0f, static_cast<float>(kInputW - 1));
        det.y2 = std::clamp(cy + b, 0.0f, static_cast<float>(kInputH - 1));
        det.score = best_score;
        det.class_id = class_id;
        if (det.x2 > det.x1 && det.y2 > det.y1) {
          candidates.push_back(det);
        }
      }
    }
  }

  std::sort(candidates.begin(), candidates.end(),
            [](const Detection& a, const Detection& b) { return a.score > b.score; });

  std::vector<Detection> keep;
  keep.reserve(candidates.size());
  for (const Detection& det : candidates) {
    bool suppressed = false;
    for (const Detection& kept : keep) {
      if (iou_xyxy(det, kept) >= kNmsThreshold) {
        suppressed = true;
        break;
      }
    }
    if (!suppressed) {
      keep.push_back(det);
    }
  }
  return keep;
}

std::string label_for(int class_id) {
  if (class_id >= 0 && class_id < static_cast<int>(kCocoClasses.size())) {
    return kCocoClasses[static_cast<size_t>(class_id)];
  }
  return "id_" + std::to_string(class_id);
}

void draw_detections(cv::Mat& image, const std::vector<Detection>& detections) {
  for (const Detection& det : detections) {
    const int x1 = std::clamp(static_cast<int>(det.x1), 0, image.cols - 1);
    const int y1 = std::clamp(static_cast<int>(det.y1), 0, image.rows - 1);
    const int x2 = std::clamp(static_cast<int>(det.x2), 0, image.cols - 1);
    const int y2 = std::clamp(static_cast<int>(det.y2), 0, image.rows - 1);
    if (x2 <= x1 || y2 <= y1) {
      continue;
    }

    const cv::Scalar color(0, 255, 0);
    cv::rectangle(image, cv::Point(x1, y1), cv::Point(x2, y2), color, 2);

    const std::string text = label_for(det.class_id) + ":" + cv::format("%.2f", det.score);
    int baseline = 0;
    const cv::Size text_size =
        cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
    const int text_height = text_size.height + baseline;
    cv::rectangle(image, cv::Point(x1, std::max(0, y1 - text_height)),
                  cv::Point(std::min(image.cols - 1, x1 + text_size.width), y1), color,
                  cv::FILLED);
    cv::putText(image, text, cv::Point(x1, std::max(0, y1 - baseline)),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
  }
}

int run_one_image(simaai::neat::Run& run, const fs::path& image_path, const fs::path& output_dir) {
  const cv::Mat bgr = read_bgr(image_path);
  const cv::Mat rgb = bgr_to_rgb_contiguous(bgr);
  simaai::neat::Tensor input =
      simaai::neat::from_cv_mat(rgb, simaai::neat::ImageSpec::PixelFormat::RGB, true);

  if (!run.push(input)) {
    throw std::runtime_error("push failed for " + image_path.string() + ": " + run.last_error());
  }
  auto output = run.pull(kTimeoutMs);
  if (!output.has_value()) {
    throw std::runtime_error("timed out waiting for output for " + image_path.string() + ": " +
                             run.last_error());
  }

  const std::vector<TensorHWC> outputs = collect_yolo_outputs(*output);
  const std::vector<Detection> detections = postprocess_yolov8x_p2_4o(outputs);

  cv::Mat overlay = letterbox_bgr(bgr);
  draw_detections(overlay, detections);
  const fs::path out_path = output_dir / (image_path.stem().string() + ".jpg");
  if (!cv::imwrite(out_path.string(), overlay)) {
    throw std::runtime_error("failed to write output image: " + out_path.string());
  }
  return static_cast<int>(detections.size());
}

} // namespace

int main(int argc, char** argv) {
  std::cout.setf(std::ios::unitbuf);
  std::cerr.setf(std::ios::unitbuf);

  try {
    const Args args = parse_args(argc, argv);
    if (!fs::is_regular_file(args.model)) {
      throw std::runtime_error("model archive not found: " + args.model.string());
    }
    const std::vector<fs::path> image_paths = get_image_paths(args.images);
    if (image_paths.empty()) {
      throw std::runtime_error("no images found in: " + args.images.string());
    }
    prepare_output_dir(args.results);

    int max_width = 0;
    int max_height = 0;
    for (const fs::path& path : image_paths) {
      const cv::Mat image = read_bgr(path);
      max_width = std::max(max_width, image.cols);
      max_height = std::max(max_height, image.rows);
    }

    simaai::neat::Model model(args.model.string(), build_model_options(max_width, max_height));
    const cv::Mat seed_rgb = bgr_to_rgb_contiguous(read_bgr(image_paths.front()));
    simaai::neat::Run run = build_sync_run(model, seed_rgb, max_width, max_height);

    for (size_t i = 0; i < image_paths.size(); ++i) {
      const int detections = run_one_image(run, image_paths[i], args.results);
      std::cout << "[" << (i + 1) << "/" << image_paths.size() << "] "
                << image_paths[i].filename().string() << " -> "
                << (args.results / (image_paths[i].stem().string() + ".jpg")).string() << " ("
                << detections << " detections)\n";
    }

    run.close();
    std::cout << "wrote results to " << args.results.string() << "\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
