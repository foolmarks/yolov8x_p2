/*
 * GStreamer
 * Copyright (C) 2023 SiMa.ai
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Library General Public
 * License as published by the Free Software Foundation; either
 * version 2 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Library General Public License for more details.
 *
 * You should have received a copy of the GNU Library General Public
 * License along with this library; if not, write to the
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * Boston, MA 02111-1307, USA.
 */

#ifndef GST_SIMAAI_OVERLAY_COMMON_H_
#define GST_SIMAAI_OVERLAY_COMMON_H_

#include <map>
#include <memory>
#include <string>

#include <gst/gst.h>

#include "ext/overlay.h"
#include "ext/overlay_renderer.h"

/* labels to be used for classification */
#define DEFAULT_LABELS_FILE "labels.txt"

/* Default rendering information metadata string */
#define DEFAULT_OVERLAY_INFO "input::allegrodec,bbox::a65-topk"

/* Default rendering-pipeline information */
#define DEFAULT_OVERLAY_RENDER_RULES "bbox"

/* Default maximum supported pipeline stages for the graphics pipeline */
#define OVERLAY_MAX_SUPPORTED_PIPELINE_STAGES 4

#define DEFAULT_IN_FORMAT "YUV420P"
#define DEFAULT_OUT_FORMAT "NV12"

#define DEFAULT_MEM_TARGET 0

/* Min amount of buffers to allocate in output buffer pool */
#define MIN_POOL_SIZE 2

/**
 * @brief Private member structure for GstSimaaOverlay2Private instances
 */
typedef struct _GstSimaaiOverlay2Private
{
  GstBufferPool *pool; /**< Buffer pool */
  gint64 out_buffer_id; /**< Output buffer-id, retruned from simaaimemlib */
  GstBufferList * list; /**< Aggregate list of input buffers */
  gint64 frame_id;/**< Input frame-id extracted from the incoming buffer */
  guint64 timestamp; /*< Placeholder for timestamp from incoming buffer*/
  GString *stream_id;
  gint image_width; /**< Output width size */
  gint image_height; /**< Output height size */
  gboolean is_text_overlay; /**< Check to validated is labels.txt is provided when text overlay is enabled in the graphsic pipeline */
  GString *labels_file; /**< classification labels string file */
  GString *out_buffer_name; /*< Output buffer name - currently hard-coded to "overlay"*/
  gint mem_target; /*< Memory target*/
  std::unique_ptr<simaai::overlay::Overlay> overlay; /**< Overlay object to the external overlay library-like interfaces */
  std::map<int, std::pair<std::string, std::string>> profile_map; /**< The map of render-info */
  std::vector<std::string> labels; /**< classification labels string */
  std::chrono::time_point<std::chrono::steady_clock> t0; /**< begin point of execution time measurement */
  std::chrono::time_point<std::chrono::steady_clock> t1; /**< end point of execution time measurement */
  gboolean dump_data; /**< True to dump buffer metadata */

  simaai::overlay::OverlayImageFormat in_img_format;

  int framerate_numerator;
  int framerate_denominator;

  std::vector<std::string> render_rules;

  gint mem_type;
  gint mem_flag;
} GstSimaaiOverlay2Private;

#endif // GST_SIMAAI_OVERLAY_COMMON_H_
