#define main sync_main_hidden
#include "main_sync.cpp"
#undef main

#include <chrono>

namespace {

constexpr int kPollTimeoutMs = 100;
constexpr int kMaxInFlight = 1;

simaai::neat::Run build_async_run(const simaai::neat::Model& model, const cv::Mat& seed_rgb,
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
  options.queue_depth = 4;
  options.overflow_policy = simaai::neat::OverflowPolicy::Block;
  options.output_memory = simaai::neat::OutputMemory::Owned;

  simaai::neat::Tensor seed = simaai::neat::from_cv_mat(
      seed_rgb, simaai::neat::ImageSpec::PixelFormat::RGB, true);
  return session.build(seed, simaai::neat::RunMode::Async, options);
}

simaai::neat::Tensor make_rgb_tensor(const fs::path& image_path) {
  const cv::Mat bgr = read_bgr(image_path);
  const cv::Mat rgb = bgr_to_rgb_contiguous(bgr);
  return simaai::neat::from_cv_mat(rgb, simaai::neat::ImageSpec::PixelFormat::RGB, true);
}

bool try_admit_image(simaai::neat::Run& run, const fs::path& image_path,
                     std::vector<fs::path>& in_flight) {
  simaai::neat::Tensor input = make_rgb_tensor(image_path);
  const bool accepted = run.try_push(input);
  if (accepted) {
    in_flight.push_back(image_path);
  }
  return accepted;
}

int write_result(const simaai::neat::Sample& sample, const fs::path& image_path,
                 const fs::path& output_dir) {
  const std::vector<TensorHWC> outputs = collect_yolo_outputs(sample);
  const std::vector<Detection> detections = postprocess_yolov8x_p2_4o(outputs);

  cv::Mat overlay = letterbox_bgr(read_bgr(image_path));
  draw_detections(overlay, detections);

  const fs::path out_path = output_dir / (image_path.stem().string() + ".jpg");
  if (!cv::imwrite(out_path.string(), overlay)) {
    throw std::runtime_error("failed to write output image: " + out_path.string());
  }
  return static_cast<int>(detections.size());
}

int run_async_loop(simaai::neat::Run& run, const std::vector<fs::path>& image_paths,
                   const fs::path& output_dir) {
  if (!run.can_push()) {
    throw std::runtime_error("built pipeline does not support pushed input");
  }
  if (!run.can_pull()) {
    throw std::runtime_error("built pipeline does not support pulled output");
  }
  if (!run.running()) {
    throw std::runtime_error("pipeline is not running after build");
  }

  size_t next_input = 0;
  int processed = 0;
  std::vector<fs::path> in_flight;
  auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(kTimeoutMs);

  while (processed < static_cast<int>(image_paths.size())) {
    bool made_progress = false;

    while (next_input < image_paths.size() &&
           static_cast<int>(in_flight.size()) < kMaxInFlight) {
      if (!run.running()) {
        throw std::runtime_error("pipeline stopped before push: " + run.last_error());
      }
      if (!try_admit_image(run, image_paths[next_input], in_flight)) {
        break;
      }
      ++next_input;
      made_progress = true;
    }

    auto sample = run.pull(kPollTimeoutMs);
    if (sample.has_value()) {
      if (in_flight.empty()) {
        throw std::runtime_error("received output with no matching in-flight image");
      }
      const fs::path image_path = in_flight.front();
      in_flight.erase(in_flight.begin());

      const int detections = write_result(*sample, image_path, output_dir);
      ++processed;
      made_progress = true;
      std::cout << "[" << processed << "/" << image_paths.size() << "] "
                << image_path.filename().string() << " -> "
                << (output_dir / (image_path.stem().string() + ".jpg")).string() << " ("
                << detections << " detections)\n";
    }

    if (made_progress) {
      deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(kTimeoutMs);
      continue;
    }
    if (std::chrono::steady_clock::now() > deadline) {
      throw std::runtime_error("timed out waiting for async pipeline progress: " +
                               run.last_error());
    }
  }

  return processed;
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
    simaai::neat::Run run = build_async_run(model, seed_rgb, max_width, max_height);

    const int processed = run_async_loop(run, image_paths, args.results);
    if (processed != static_cast<int>(image_paths.size())) {
      throw std::runtime_error("processed " + std::to_string(processed) + " of " +
                               std::to_string(image_paths.size()) + " images");
    }

    run.close();
    std::cout << "wrote results to " << args.results.string() << "\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
