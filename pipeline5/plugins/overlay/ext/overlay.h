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

#ifndef OVERLAY_H_
#define OVERLAY_H_

#include <string.h>
#include <exception>
#include <map>
#include <memory>

#include <opencv2/opencv.hpp>

#include <simaai/topk.h>
#include <simaai/simaailog.h>

#include "image.h"
#include "yuv420.h"
#include "overlay_types.h"

namespace simaai {
namespace overlay {

static std::string get_renderers() {
    return "application/vnd.simaai.tensor, format=(string){BBOX,BBOX_SCORE,"
    "BBOX_YOLO,BBOX_TRACKER,BBOX_DISTANCE,PLAIN_TEXT,POSES,TEST_FORMAT}";
}

/**
 * @brief Overlay Class, This is the important class used by the plugins/application
 */
class Overlay {
  using GstDataPtr = unsigned char *;
  using OverlayRenderConfig = std::map<int, std::pair<std::string, std::string>>;

 public:
  Overlay() = delete;

  explicit Overlay (OverlayImageFormat fmt, int width, int height)
      :format(fmt),
       w(width),
       h(height) {
  };

  ~Overlay() = default;

  /**
   * @brief Setter API for rendreing pipeline rules
   */
  void set_render_rules (std::vector<std::string> _render_rules) {
    render_rules = _render_rules;
    std::cout << "render_rules " << render_rules.size() << "\n";
    for (int i = 0; i < render_rules.size(); i++)
      std::cout << "Render rule " << render_rules[i] << "\n";

  }

  /* Helper API to get buffer name from the profile_map */
  const std::string get_buf_name (const std::string & search_type) const {
    for (auto & [key, value]: profile_map) {
      auto pair = value;
      if (value.first == search_type)
        return value.second;
    }
    return "";
  }

  /**
   * @brief Render YUV420, for more informatin check yuv420.cc
   */
  bool render_yuv420 (unsigned char * out_mem_ptr) {
    if (verbose)
      t0 = std::chrono::steady_clock::now();

    auto input = get_buf_name("input");
    if (input.empty()) {
      simaailog(SIMAAILOG_ERR, "Input buffer not found %p", out_mem_ptr);
      return false;
    }

    unsigned char * img_ptr  = static_cast<unsigned char *>(buf_ptr_m[input]);

    // Copy input image into output buffer and draw primitives on top of output buffer
    auto sz = w * h * 1.5;
    memcpy(out_mem_ptr, img_ptr, sz);

    image->update_frame_id(frame_id);

    for (int i = 0; i < render_rules.size(); i++) {
      auto buffer_name = get_buf_name(render_rules[i]);
      if (image->update_renderer(out_mem_ptr, w, h, 1.5)) {
        if (!image->render (render_rules[i], buf_ptr_m, buffer_name)) {
          image->flush();
          return true;
        }
      } else {
        simaailog(SIMAAILOG_ERR, "Critical error, unable to update rendering engine");
      }
    }

    if (render_rules.size() == 0)
    {
      image->update_renderer(out_mem_ptr, w, h, 1.5);
    }

    image->flush();

    if (verbose) {
      t1 = std::chrono::steady_clock::now();
      auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(t1-t0);
      std::cout << "Image rendering took run time in ms: " << elapsed.count()/1000.0 << "\n";
      return true;
    }
    return true;
  }

  /**
   * @brief Helper to enable logs
   */
  void set_verbose(bool _verbose) {
    verbose = _verbose;
  }

  /**
   * @brief Helper to enable dump data
   */
  void set_dump_data(bool _dump_data) {
    dump_data = _dump_data;
  }

  /**
   * @brief Helper to set visible instance name
   */
  void set_name(const std::string& _name) {
    name = _name;
  }

  /**
   * @brief rendering wrapper
   */
  bool init_render (const std::vector<std::string> & labels = {}) {
    switch (format) {
      case OverlayImageFormat::SIMAAI_IMG_YUV420:
      case OverlayImageFormat::SIMAAI_IMG_NV12:
        try {
          image = std::make_unique<simaai::overlay::Yuv420>(format, w, h, 1.5);
          image->update_labels(labels);
          image->set_verbose(verbose);
          image->set_dump_data(dump_data);
          image->set_name(name);
        } catch (const std::exception& err) {
          simaailog(SIMAAILOG_ERR, "%s error: %s", __PRETTY_FUNCTION__, err.what());
          return false;
        }
        break;
      case OverlayImageFormat::SIMAAI_IMG_YUV444:
      case OverlayImageFormat::SIMAAI_IMG_RGB:
      default:
        simaailog(SIMAAILOG_ERR, "Format not supported yet: %d", static_cast<int>(format));
        return false;
    }
    return true;
  };

  /**
   * @brief rendering wrapper
   */
  bool update_renderer (unsigned char * out_mem_ptr) {
    if (buf_ptr_m.size() <= 0) {
      simaailog(SIMAAILOG_ERR, "[OVERLAY]: memory address list is empty: %p", out_mem_ptr);
      return false;
    }

    switch(format) {
      case OverlayImageFormat::SIMAAI_IMG_YUV420:
      case OverlayImageFormat::SIMAAI_IMG_NV12:
        return render_yuv420 (out_mem_ptr);
      case OverlayImageFormat::SIMAAI_IMG_YUV444:
      case OverlayImageFormat::SIMAAI_IMG_RGB:
      default:
        simaailog(SIMAAILOG_ERR, "Format not supported yet: %d", static_cast<int>(format));
        return false;
    }
    return true;
  };

  /**
   * @brief Helper API to set rendering config
   */
  void update_render_config (const OverlayRenderConfig & profile_map_in) {
    profile_map = profile_map_in;
  }

  /**
   * @brief Helper API to set buffer memory information
   */
  void update_mem_info (const char * buf_name, const GstDataPtr memptr) {
    // fprintf(stderr, "buf_name:[%s]:[%p]\n", buf_name, memptr);
    buf_ptr_m[buf_name] = memptr;
  };

  /**
   * @brief Helper API to set frame id
   */
  void update_frame_id (const int64_t _frame_id) {
    frame_id = _frame_id;
  }

 private:
  std::map<std::string, GstDataPtr> buf_ptr_m;
  std::map<int, std::pair<std::string, std::string>> profile_map;

  std::chrono::time_point<std::chrono::steady_clock> t0;
  std::chrono::time_point<std::chrono::steady_clock> t1;

  std::vector<std::string> render_rules;
  OverlayImageFormat format;
  int w, h;
  bool verbose = false;
  bool dump_data = false;
  int64_t frame_id{-1};
  std::string name;
  std::unique_ptr<simaai::overlay::Yuv420> image;
};

}
}

#endif // OVERLAY_H_
