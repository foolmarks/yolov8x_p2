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

#include <chrono>
#include <fstream>
#include <iostream>

#include <simaai/pose_serialization.h>
#include <simaai/simaailog.h>
#include <simaai/yolo_nms2.h>

/* Check README.md */
#include "image.h"
#include "yuv420.h"

namespace simaai {
namespace overlay {

/* static class member initialization */
template <class... Args>
std::map<std::string, bool (Yuv420::*)(Args...)> MapHolder<Args...>::CallbackMap;

Yuv420::Yuv420 (OverlayImageFormat fmt, unsigned int width, unsigned int height, float stride)
      :w(width),
       h(height),
       s(stride),
       thickness(1),
       lineType(8),
       color(0,255,0),
       text_color(0,255,0)
{
  simaai::cv_helpers::CvDrawConfig draw_config(color, text_color,
                                               lineType, fontFace,
                                               fontScale, thickness,
                                               bottomLeftOrigin);

  switch (fmt) {
  case OverlayImageFormat::SIMAAI_IMG_YUV420:
    cv_helpers = std::make_unique<simaai::cv_helpers::CvHelpersYUV420>(draw_config);
    break;
  case OverlayImageFormat::SIMAAI_IMG_NV12:
    cv_helpers = std::make_unique<simaai::cv_helpers::CvHelpersNV12>(draw_config);
    break;
  default:
    throw std::invalid_argument("Yuv420: unsupported OverlayImageFormat");
    break;
  }

/* CHECK::Class register function before using them from the application */
  register_fn("bbox", &simaai::overlay::Yuv420::draw_rectangle);
  register_fn("bboxs", &simaai::overlay::Yuv420::draw_rectangle_score);
  register_fn("bboxy", &simaai::overlay::Yuv420::draw_rectangle_yolo);
  register_fn("bboxt", &simaai::overlay::Yuv420::draw_rectangle_tracker);
  register_fn("bboxd", &simaai::overlay::Yuv420::draw_rectangle_distance);
  register_fn("text", &simaai::overlay::Yuv420::draw_text);
  register_fn("pose", &simaai::overlay::Yuv420::draw_points_lines_pose);
};

bool Yuv420::update_renderer(unsigned char * pixels, unsigned int width, unsigned int height, float stride) {
  return cv_helpers->update_renderer(pixels, width, height, stride);
}

void Yuv420::get_nv12_image (uint8_t * dst, int w, int h) {
  cv_helpers->get_nv12_image(dst, w, h);
}

std::vector<Points> Yuv420::get_points_from_buffer (int no_of_boxes, const uint8_t * buffer, size_t bbox_offset, OverlayBufType type) {
  const uint8_t * buf = static_cast<const uint8_t *>(buffer + bbox_offset);

  if (type == OverlayBufType::SIMAAI_BUF_TOPK) { /* Topk data protocol specific */
    for (int i = 0 ; i < no_of_boxes; i++) {
      Points p;
      p.x1 = *((unsigned int *) (buf));
      p.y1 = *((unsigned int *) (buf + 4));
      p.w = *((unsigned int *) (buf + 8));
      p.h = *((unsigned int *) (buf + 12));
      p.trackId = *((unsigned int *) (buf + 16));
      p.classId = *((unsigned int *) (buf + 20));
      if (verbose)
        fprintf(stderr, "#%d| point x1:%d, y1:%d, w:%d, h:%d class_id:%d\n", i, p.x1, p.y1, p.w, p.h, p.classId);
      points.emplace_back(p);
      buf = buf + 24;
    }
  } else if (type == OverlayBufType::SIMAAI_BUF_YOLO) { /* Yolo data protocol specific */
    for (int i = 0 ; i < no_of_boxes; i++) {
      YoloNMS::Box * bb = (YoloNMS::Box*)(buf);
      Points p;
      p.w = bb->x2 - bb->x1;
      p.h = bb->y2 - bb->y1;
      p.x1 = bb->x1 ; //- (p.w / 2);
      p.y1 = bb->y1 ; //- (p.h / 2);
      p.trackId = -1;
      p.classId = bb->class_id;
      if (verbose)
        fprintf(stderr, "#%d| point x1:%d, y1:%d, w:%d, h:%d class_id:%d\n", i, p.x1, p.y1, p.w, p.h, bb->class_id);
      points.emplace_back(p);
      buf = buf + sizeof(YoloNMS::Box);
    }
  } else if (type == OverlayBufType::SIMAAI_BUF_TRACKER) { /* Tracker data protocol specific */
    for (int i = 0 ; i < no_of_boxes; i++) {
      Points p;
      p.x1 = *((unsigned int *) (buf));
      p.y1 = *((unsigned int *) (buf + 4));
      p.w = *((unsigned int *) (buf + 8));
      p.h = *((unsigned int *) (buf + 12));
      p.trackId = *((unsigned int *) (buf + 16));
      p.classId = *((unsigned int *) (buf + 20));
      if (verbose)
        fprintf(stderr, "x1:%d, y1:%d, w:%d, h:%d, trackId:%d", p.x1, p.y1, p.w, p.h, p.trackId);
      points.emplace_back(p);
      buf = buf + 24;
    }
  }
  return points;
}

cv::Mat Yuv420::convert2bgr() {
  unsigned w = y_.cols;
  unsigned h = y_.rows;
  cv::Mat u_scaled;
  cv::resize(u_, u_scaled, cv::Size(w, h), 0, 0, cv::INTER_LINEAR);
  cv::Mat v_scaled;
  cv::resize(v_, v_scaled, cv::Size(w, h), 0, 0, cv::INTER_LINEAR);
  cv::Mat bgr;
  std::vector<cv::Mat> channels = {y_, u_scaled, v_scaled};
  cv::Mat merged;
  cv::merge(channels, merged);
  cv::cvtColor(merged, bgr, cv::COLOR_YUV2BGR);
  return bgr;
}

const std::string & Yuv420::get_label (int classification_idx) {
  return labels[classification_idx];
}

void Yuv420::get_labels (const uint8_t * buf_memptr, std::vector<std::string> & labels, int no_of_boxes) {

  for (int i = 0 ; i < no_of_boxes; i++) {
    size_t offset = i * sizeof(int32_t);
    char * ptrBbox = (char *)(buf_memptr + offset);
    int32_t classification_idx = *((int32_t *)(ptrBbox));
    labels.push_back(get_label(classification_idx));
  }

  return;
}

bool Yuv420::draw_rectangle (std::map<std::string, unsigned char *> & buf_ptr_m, std::string & buffer_name) {
  if (!buf_ptr_m[buffer_name]) {
    simaailog(SIMAAILOG_ERR, "input buffer is unavailable for %s:%p" ,buffer_name, buf_ptr_m);
    return false;
  }

  int no_of_boxes = *((int *)(buf_ptr_m[buffer_name]));
  if ((no_of_boxes == 0) || (no_of_boxes > 24))
    return false;

  simaailog(SIMAAILOG_DEBUG, "Number of bounding boxes : %d" ,no_of_boxes);

  points = get_points_from_buffer (no_of_boxes, static_cast<const uint8_t *>(buf_ptr_m[buffer_name]), 4, OverlayBufType::SIMAAI_BUF_TOPK);
  cv_helpers->draw_rectangle(points, labels);

  if (dump_data)
    dump_points_to_file(points);

  // fprintf(stderr, "Done Number of bounding boxes : %d\n" ,no_of_boxes);
  return true;
}

bool Yuv420::draw_rectangle_score (std::map<std::string, unsigned char *> & buf_ptr_m, std::string & buffer_name) {
  
  if (!buf_ptr_m[buffer_name]) {
    simaailog(SIMAAILOG_ERR, "input buffer is unavailable for %s:%p" ,buffer_name, buf_ptr_m);
    return false;
  }

  int no_of_boxes = *((int *)(buf_ptr_m[buffer_name]));
  if ((no_of_boxes == 0) || (no_of_boxes > 24))
    return false;

  simaailog(SIMAAILOG_DEBUG, "Number of bounding boxes : %d" ,no_of_boxes);

  points = get_points_from_buffer (no_of_boxes, static_cast<const uint8_t *>(buf_ptr_m[buffer_name]), 4, OverlayBufType::SIMAAI_BUF_TOPK);
  cv_helpers->draw_rectangle_score(points, labels);

  if (dump_data)
    dump_points_to_file(points);

  return true;
}

bool Yuv420::draw_rectangle_yolo (std::map<std::string, unsigned char *> & buf_ptr_m, std::string & buffer_name) {
  if (!buf_ptr_m[buffer_name]) {
    simaailog(SIMAAILOG_ERR, "input buffer is unavailable for %s:%p" ,buffer_name, buf_ptr_m);
    return false;
  }

  int no_of_boxes = *((int *)(buf_ptr_m[buffer_name]));
  if ((no_of_boxes == 0) || (no_of_boxes > 24))
    return false;

  simaailog(SIMAAILOG_DEBUG, "Number of bounding boxes : %d" ,no_of_boxes);

  points = get_points_from_buffer (no_of_boxes, static_cast<const uint8_t *>(buf_ptr_m[buffer_name]), 4, OverlayBufType::SIMAAI_BUF_YOLO);
  cv_helpers->draw_rectangle(points, labels);

  if (dump_data)
    dump_points_to_file(points);

  // fprintf(stderr, "Done Number of bounding boxes : %d\n" ,no_of_boxes);
  return true;
}

void Yuv420::flush() {
  points.clear();
}

bool Yuv420::draw_rectangle_tracker (std::map<std::string, unsigned char *> & buf_ptr_m, std::string & buffer_name) {
  if (!buf_ptr_m[buffer_name]) {
    simaailog(SIMAAILOG_ERR, "input buffer is unavailable for %s:%p" ,buffer_name, buf_ptr_m);
    return false;
  }

  uint64_t frame_id = *((uint64_t *)buf_ptr_m[buffer_name]);

  int no_of_boxes = *((int *)(buf_ptr_m[buffer_name] + 4));
  if ((no_of_boxes == 0) || (no_of_boxes > 24))
    return false;

  simaailog(SIMAAILOG_DEBUG, "Number of bounding boxes : %d" ,no_of_boxes);
  points = get_points_from_buffer (no_of_boxes, static_cast<const uint8_t *>(buf_ptr_m[buffer_name]), 8, OverlayBufType::SIMAAI_BUF_TRACKER);

  if (verbose)
    fprintf(stderr, "Number of points: %zu\n", points.size());

  cv_helpers->draw_rectangle_tracker(points, labels);

  if (dump_data)
    dump_points_to_file(points);

  return true;
}

bool Yuv420::draw_rectangle_distance (std::map<std::string, unsigned char *> & buf_ptr_m, std::string & buffer_name) {
  if (!buf_ptr_m[buffer_name]) {
    simaailog(SIMAAILOG_ERR, "input buffer is unavailable for %s:%p" ,buffer_name, buf_ptr_m);
    return false;
  }

  const TrackerHeader* const hdr = reinterpret_cast<TrackerHeader*>(buf_ptr_m[buffer_name]);
  const Distance* in_frame = reinterpret_cast<Distance*>(buf_ptr_m[buffer_name] + sizeof(*hdr));

  if (hdr->no_of_boxes == 0)
    return false;

  simaailog(SIMAAILOG_DEBUG, "Number of bounding boxes : %d", hdr->no_of_boxes);

  std::vector<Distance> distances;

  for (int i = 0; i < hdr->no_of_boxes; i++)
    distances.emplace_back(*in_frame++);

  cv_helpers->draw_rectangle_distance(distances, labels);

  if (dump_data)
    dump_distance_to_file(distances);

  return true;
}

bool Yuv420::draw_text (std::map<std::string, unsigned char *> & buf_ptr_m, std::string & buffer_name) {
  int no_of_boxes  = 0;

  if (buf_ptr_m[PIPELINE_POINTS_BUF1]) {
    no_of_boxes = *((int *)(buf_ptr_m[PIPELINE_POINTS_BUF1]));
    if ((no_of_boxes == 0) || (no_of_boxes > 24))
      return false;
  }
  else
  {
    no_of_boxes = 1;
    simaai::cv_helpers::Points point;
    point.x1 = 100;
    point.y1 = 100;
    point.h = 20;
    point.w = 0;
    point.trackId = 0;
    point.classId = 0;
    points.push_back(point);
  }

  std::vector<std::string> labels;
  labels.reserve(no_of_boxes);
  if (buf_ptr_m[buffer_name])
    get_labels(buf_ptr_m[buffer_name], labels, no_of_boxes);

  cv_helpers->draw_text(points, labels);

  if (dump_data)
    dump_points_to_file(points);

  return true;
};

bool Yuv420::draw_points_lines_pose(std::map<std::string, unsigned char *> & buf_ptr_m, std::string & buffer_name)
{
  if (!buf_ptr_m[buffer_name])
  {
    simaailog(SIMAAILOG_ERR, "input buffer is unavailable for %s:%p" ,buffer_name, buf_ptr_m);
    return false;
  }

  std::vector<std::vector<float>> pose_entries;
  std::vector<std::vector<float>> all_keypoints;

  auto offset = simaai::cv_helpers::deserialize(reinterpret_cast<void *>(buf_ptr_m[buffer_name]), pose_entries);
  if (offset < 0) {
    if (verbose)
      fprintf(stderr, "No pose entries deserialized\n");
    return false;
  }

  offset = simaai::cv_helpers::deserialize(reinterpret_cast<void *>(buf_ptr_m[buffer_name] + offset), all_keypoints);
  if (offset < 0) {
    if (verbose)
      fprintf(stderr, "No keypoints deserialized\n");
    return false;
  }

  cv_helpers->draw_points_lines_pose(pose_entries, all_keypoints);

  if (dump_data)
    dump_poses_to_file(pose_entries, all_keypoints);

  return true;
}

bool Yuv420::draw_polygon (std::map<std::string, unsigned char *> & buf_ptr_m, std::string & buffer_name) {
  return true;
}

bool Yuv420::draw_points_lines (std::map<std::string, unsigned char *> & buf_ptr_m, std::string & buffer_name) {
  return true;
}

bool Yuv420::make_dump_file(std::ofstream& file)
{
  std::ostringstream fname;
  fname << "/tmp/" << name << "-" << frame_id << ".out";

  file.open(fname.str(), std::ios::binary | std::ios::trunc);
  if (file.fail()) {
    simaailog(SIMAAILOG_ERR, "Unable to create dump file: %s", fname.str().c_str());
    return false;
  }

  return true;
}

void Yuv420::dump_points_to_file(const std::vector<Points>& points)
{
  std::ofstream dump;
  if (!make_dump_file(dump))
    return;

  for (const auto& point : points) {
    std::ostringstream record(std::ios::binary);

    record << "frame_id:" << frame_id
           << " x1:" << point.x1
           << " y1:" << point.y1
           << " w:" << point.w
           << " h:" << point.h
           << " trackId:" << point.trackId
           << " classId:" << point.classId
           << "\n";

    dump << record.str();
  }

  dump.flush();
}

void Yuv420::dump_distance_to_file(const std::vector<Distance>& distances)
{
  std::ofstream dump;
  if (!make_dump_file(dump))
    return;

  for (const auto& distance : distances) {
    std::ostringstream record(std::ios::binary);

    record << "frame_id:" << frame_id
           << " x1:" << distance.points.x1
           << " y1:" << distance.points.y1
           << " w:" << distance.points.w
           << " h:" << distance.points.h
           << " trackId:" << distance.points.trackId
           << " classId:" << distance.points.classId
           << " distance:" << distance.distance
           << "\n";

    dump << record.str();
  }

  dump.flush();
}

void Yuv420::dump_poses_to_file(const std::vector<std::vector<float>>& pose_entires,
                                const std::vector<std::vector<float>>& all_keypoints)
{
  std::ofstream dump;
  if (!make_dump_file(dump))
    return;

  auto do_dump = [](std::ofstream& dump, const auto& entries) {
    for (const auto& entry : entries) {
      for (const auto& val : entry)
        dump << val << " ";
      dump << "\n";
    }
  };

  dump << "frame_id:" << frame_id << " pose_entires\n";
  do_dump(dump, pose_entires);

  dump << "frame_id:" << frame_id << " all_keypoints\n";
  do_dump(dump, all_keypoints);

  dump.flush();
}

}
}
