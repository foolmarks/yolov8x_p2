//**************************************************************************
//||                        SiMa.ai CONFIDENTIAL                          ||
//||   Unpublished Copyright (c) 2022-2023 SiMa.ai, All Rights Reserved.  ||
//**************************************************************************
// NOTICE:  All information contained herein is, and remains the property of
// SiMa.ai. The intellectual and technical concepts contained herein are
// proprietary to SiMa and may be covered by U.S. and Foreign Patents,
// patents in process, and are protected by trade secret or copyright law.
//
// Dissemination of this information or reproduction of this material is
// strictly forbidden unless prior written permission is obtained from
// SiMa.ai.  Access to the source code contained herein is hereby forbidden
// to anyone except current SiMa.ai employees, managers or contractors who
// have executed Confidentiality and Non-disclosure agreements explicitly
// covering such access.
//
// The copyright notice above does not evidence any actual or intended
// publication or disclosure  of  this source code, which includes information
// that is confidential and/or proprietary, and is a trade secret, of SiMa.ai.
//
// ANY REPRODUCTION, MODIFICATION, DISTRIBUTION, PUBLIC PERFORMANCE, OR PUBLIC
// DISPLAY OF OR THROUGH USE OF THIS SOURCE CODE WITHOUT THE EXPRESS WRITTEN
// CONSENT OF SiMa.ai IS STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE
// LAWS AND INTERNATIONAL TREATIES. THE RECEIPT OR POSSESSION OF THIS SOURCE
// CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS TO
// REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE, USE, OR
// SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.
//
//**************************************************************************

#ifndef IMAGE_YUV420_H_
#define IMAGE_YUV420_H_

#include <iostream>
#include <map>
#include <memory>
#include <string>

#include <opencv2/opencv.hpp>

#include <simaai/cv_helpers.h>
#include <simaai/topk.h>

/* Check README.md */
#include "image.h"
#include "overlay_types.h"

/**
 * @brief Possible buffer names for bounding box
 * @todo Configure this
 */
#define PIPELINE_POINTS_BUF1 "a65-topk"
#define PIPELINE_POINTS_BUF2 "tracker"

namespace simaai {
namespace overlay {

using Points = simaai::cv_helpers::Points;
using Distance = simaai::cv_helpers::Distance;
using namespace simaai::overlay;
// Forward declaration
class Yuv420;

struct TrackerHeader {
  std::uint32_t frame_id;
  std::int32_t no_of_boxes;
} __attribute__((packed));

/**
 * @brief template to hold map of rendering option string to varied arg functors
 */
template <class... Args>
struct MapHolder {
  static std::map<std::string, bool (Yuv420::*)(Args...)> CallbackMap;
};

/**
 * @brief YUV420 clss for overlay
 */
class Yuv420 {
public:
  Yuv420() = delete;
  ~Yuv420() = default;

  explicit Yuv420 (OverlayImageFormat fmt, unsigned int width, unsigned int height, float stride);

  /* Getter functions */

  /**
   * @brief Get nv12 image from the class, encoder supports only nv12
   */
  void get_nv12_image (uint8_t * dst, int w, int h) ;

  /**
   * @brief Get points to draw from the input buffers, i.e vector of points
   * @return Returns a vector of Points
   */
  std::vector<Points> get_points_from_buffer (int no_of_boxes, const uint8_t * buffer, size_t bbox_offset, OverlayBufType type);

  /**
   * @brief Get labels from classification labels list
   */
  const std::string & get_label(int classification_idx);

  /**
   * @brief Get labels from classification labels list
   */
  void get_labels (const uint8_t * buf_memptr, std::vector<std::string> & labels, int no_of_boxes);

  /**
   * @brief Helper to update labels to be used for text rendering
   */
  void update_labels (const std::vector<std::string> & _labels) {
    labels = _labels;
  };

  /**
   * @brief template function to register a rendering function to a rendering-type
   */
  template <class... Args>
  void register_fn(std::string name, bool (Yuv420::*func)(Args...)) {
    MapHolder<Args...>::CallbackMap[name] = func;
  };

  /**
   * @brief API to render a specific type
   */
  template <class... Args>
  bool render (std::string & op_name, Args &&... args) {
    return execute_fn (op_name, std::forward<Args>(args)...);
  }

  /**
   * @brief API to update graphics rendring engine
   */
  bool update_renderer(unsigned char * pixels, unsigned int width, unsigned int height, float stride);

  /**
   * @brief API to flush graphics state/data
   */
  void flush();

  /**
   * @brief Execute function called to run the rendering stage
   */
  template <class... Args>
  bool execute_fn (std::string name, Args &&... args) {
    return (this->*(MapHolder<Args...>::CallbackMap[name]))(std::forward<Args>(args)...);
  };

  /**
   * @brief Helper API to draw a rectangle on the input image, based on cv::Rectangle
   * @param buf_ptr_m: is the map of memory buffer to render stage name
   * @param buffer_name: An input buffer name
   * @return true on success and false on fail
   */
  bool draw_rectangle (std::map<std::string, unsigned char *> & buf_ptr_m, std::string & buffer_name);

  /**
   * @brief Helper API to draw a rectangle on the input image, based on cv::Rectangle, also writes score
   * @param buf_ptr_m: is the map of memory buffer to render stage name
   * @param buffer_name: An input buffer name
   * @return true on success and false on fail
   */
  bool draw_rectangle_score (std::map<std::string, unsigned char *> & buf_ptr_m, std::string & buffer_name);

  /**
   * @brief Helper API to draw a rectangle from yolo nms output on the input image, based on cv::Rectangle
   * @param buf_ptr_m: is the map of memory buffer to render stage name
   * @param buffer_name: An input buffer name
   * @return true on success and false on fail
   */
  bool draw_rectangle_yolo (std::map<std::string, unsigned char *> & buf_ptr_m, std::string & buffer_name);

  /**
   * @brief Helper API to draw a rectangle and text from tracker on the input image, based on cv::Rectangle
   * @param buf_ptr_m: is the map of memory buffer to render stage name
   * @param buffer_name: An input buffer name
   * @return true on success and false on fail
   * @todo use draw_rectangel & draw_text for convinence
   */
  bool draw_rectangle_tracker (std::map<std::string, unsigned char *> & buf_ptr_m, std::string & buffer_name);

  /**
   * @brief Helper API to draw a rectangle and distance on the input image, based on cv::Rectangle
   * @param buf_ptr_m: is the map of memory buffer to render stage name
   * @param buffer_name: An input buffer name
   * @return true on success and false on fail
   * @todo use draw_rectangel & draw_text for convinence
   */
  bool draw_rectangle_distance (std::map<std::string, unsigned char *> & buf_ptr_m, std::string & buffer_name);

  /**
   * @brief Helper API to draw overlay text  on the input image, based on cv::Rectangle
   * @param buf_ptr_m: is the map of memory buffer to render stage name
   * @param buffer_name: An input buffer name
   * @return true on success and false on fail
   */
  bool draw_text (std::map<std::string, unsigned char *> & buf_ptr_m, std::string & buffer_name);

  /**
   * @brief Helper API to draw a polygon on the input image, based on cv::Rectangle
   * @param buf_ptr_m: is the map of memory buffer to render stage name
   * @param buffer_name: An input buffer name
   * @return true on success and false on fail
   */
  bool draw_polygon (std::map<std::string, unsigned char *> & buf_ptr_m, std::string & buffer_name);

  /**
   * @brief Helper API to draw a points and lines on the input image, based on cv::Rectangle
   * @param buf_ptr_m: is the map of memory buffer to render stage name
   * @param buffer_name: An input buffer name
   * @return true on success and false on fail
   */
  bool draw_points_lines (std::map<std::string, unsigned char *> & buf_ptr_m, std::string & buffer_name);

  /**
   * @brief Helper API to draw a points and lines on the input image using OpenPosePostProcess 
   * data, based on cv::Rectangle
   * @param buf_ptr_m: is the map of memory buffer to render stage name
   * @param buffer_name: An input buffer name
   * @return true on success and false on fail
   */
  bool draw_points_lines_pose(std::map<std::string, unsigned char *> & buf_ptr_m, std::string & buffer_name);
  
  /**
   * @brief Helper API to convert yuv to bgr
   */
  cv::Mat convert2bgr();

  /**
   * @brief Helper API to enable logs
   */
  void set_verbose(bool _verbose) {
    verbose = _verbose;
    cv_helpers->set_verbose(_verbose);
  }

  /**
   * @brief Helper API to enable dump data
   */
  void set_dump_data(bool _dump_data) { dump_data = _dump_data; }

  /**
   * @brief Helper API to set visible instance name
   */
  void set_name(const std::string& _name) { name = _name; }

  /**
   * @brief Helper to update frame id to be used in the next rendering
   */
  void update_frame_id(const int64_t _frame_id) { frame_id = _frame_id; };

private:
  /**
   * @brief Helper to create a dump file
   */
  bool make_dump_file(std::ofstream& file);

  /**
   * @brief Helper to dump bounding boxes coordinates to a dump file
   */
  void dump_points_to_file(const std::vector<Points>& points);

  /**
   * @brief Helper to dump distances to a dump file
   */
  void dump_distance_to_file(const std::vector<Distance>& distances);

  /**
   * @brief Helper to dump poses to a dump file
   */
  void dump_poses_to_file(const std::vector<std::vector<float>>& pose_entires,
                          const std::vector<std::vector<float>>& all_keypoints);

 private:
  cv::Mat y_, u_, v_;
  int w, h;
  float s;
  cv::Scalar color, text_color;
  int lineType;
  int y_stride, u_stride, v_stride;
  int fontFace = cv::FONT_HERSHEY_SIMPLEX;
  float fontScale = 1.0;
  int thickness = 2;
  bool bottomLeftOrigin = false;
  std::vector<std::string> labels = {"Mask", "No Mask"};
  std::vector<Points> points;
  bool verbose = false;
  bool dump_data = false;
  int64_t frame_id{-1};
  std::string name;
  std::chrono::time_point<std::chrono::steady_clock> t0;
  std::chrono::time_point<std::chrono::steady_clock> t1;
  std::unique_ptr<simaai::cv_helpers::CvHelpers> cv_helpers;
};

}
}

#endif // IMAGE_YUV420_H_
