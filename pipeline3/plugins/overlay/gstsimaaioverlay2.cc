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

/**
 * @file gstsimaaioverlay2.cc
 * @brief Gstreamer plugin for Overlay, [bbox, text, polygons, points]
 * @author SiMa.Ai\TM
 * @bug Currently no known bugs
 * @todo Advanced CAPS negotiation technique.
 * @todo Advanced parameter setting from the command line
 */

/**
 * SECTION: element-simaai_overlay2
 *
 * Plugin to overlay the inferences on the input/source image.
 *
 * <refsect2>
 * <title> Example Launch line </title>
 * |[
 * ..input
 * ! simaai-overlay2 name=overlay width=1280 height=720 format=2
 *   render-info="input::allegrodec,bbox::a65-topk" render-rules="bbox" !
 *   allegroenc2 config="something.json"
 * ]|
 * |[
 * ..input
 * ! simaai-overlay2 name=overlay width=1280 height=720 format=2
 * render-info="input::allegrodec,bbox::a65-topk,text::a65-transforms"
 * render-rules="bbox|text" ! allegroenc2 config="something.json"
 * ]|
 * </refsect2>
 */

#include <cstdio>
#include <inttypes.h>
#include <stdlib.h>

#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <string>

#include <gst/gst.h>
#include <gst/base/gstaggregator.h>

#include <gstsimaaiallocator.h>
#include <gstsimaaibufferpool.h>
#include <simaai/trace/pipeline_tp.h>

#include "gstsimaaioverlay2.h"
#include "overlay_renderer.h"

#include <simaai/trace/pipeline_new_tp.h>

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#define DEFAULT_INPUT_WIDTH (1280)
#define DEFAULT_INPUT_HEIGHT (720)
#define DEFAULT_FRAMERATE_NUM (30)
#define DEFAULT_FRAMERATE_DEN (1)

/* For 'extern' definitions refer to gstoverlaycommon.cc */
extern void gst_simaai_overlay2_install_properties(GObjectClass * gobj_class);
extern void gst_simaai_overlay2_set_property (GObject * obj, guint prop_id,
                                              const GValue * value, GParamSpec * pspec);
extern void gst_simaai_overlay2_get_property (GObject * obj, guint prop_id,
                                              GValue * value, GParamSpec * pspec);
extern simaai::overlay::OverlayImageFormat str_format_to_enum(const char * const str);
extern const char * enum_format_to_string(const simaai::overlay::OverlayImageFormat value);

/* Forward definitions */
static gboolean gst_simaai_overlay2_add2list (GstSimaaiOverlay2 * self, GValue * value);
static GstStateChangeReturn gst_simaai_overlay2_change_state (GstElement * element,
                                                              GstStateChange transition);
static void gst_simaai_overlay2_finalize (GObject * object);
static gboolean run_overlay2 (GstSimaaiOverlay2 * self, GstBuffer * outbuf);
static gboolean gst_simaai_overlay2_check_for_text_render (GstSimaaiOverlay2 * self);
static GstFlowReturn gst_simaai_overlay2_aggregate (GstAggregator * aggregator, gboolean timeout);
static void gst_simaai_overlay2_class_init (GstSimaaiOverlay2Class * klass);
static int32_t dump_intermediate_buffer(GstSimaaiOverlay2 * self,
                                        void * vaddr,
                                        gint64 id,
                                        gsize s,
                                        char * name) __attribute__((deprecated("The dumping API would "
                                                                                "be deprecated please use "
                                                                                "filesink")));

/**
 * @brief OutputImageFormat definition
 */
enum OutImageFormat {
  YUV420 = 1,
  RGB,
};

GST_DEBUG_CATEGORY_STATIC(gst_simaai_overlay2_debug);
#define GST_CAT_DEFAULT gst_simaai_overlay2_debug

#define gst_simaai_overlay2_parent_class parent_class
G_DEFINE_TYPE (GstSimaaiOverlay2, gst_simaai_overlay2, GST_TYPE_AGGREGATOR);

/* From https://stackoverflow.com/questions/216823/how-to-trim-an-stdstring */
/* ------------------------------------------------------------------------ */
static inline std::string &ltrim(std::string &s) {
  s.erase(s.begin(), std::find_if(s.begin(), s.end(),
                                  std::not1(std::ptr_fun<int, int>(std::isspace))));
  return s;
}

static inline std::string &rtrim(std::string &s) {
  s.erase(std::find_if(s.rbegin(), s.rend(),
                       std::not1(std::ptr_fun<int, int>(std::isspace))).base(), s.end());
  return s;
}

static inline std::string &trim(std::string &s) {
  return ltrim(rtrim(s));
}
/* ------------------------------------------------------------------------ */

static void gst_simaai_overlay2_load_classification (GstSimaaiOverlay2 * self) {
  GST_DEBUG_OBJECT(self, "Loading labels from = %s", self->priv->labels_file->str);
  std::string line;
  std::ifstream fptr(self->priv->labels_file->str);

  if (!fptr)
    GST_ERROR("Unable to load the classification file: %s", self->priv->labels_file->str);

  while (std::getline(fptr, line)) {
    trim(line);
    self->priv->labels.push_back(line);
  }

  GST_DEBUG_OBJECT(self, "Loaded %zu labels", self->priv->labels.size());
}

/**
 * @brief Helper API to check text rendering
 */
static gboolean gst_simaai_overlay2_check_for_text_render (GstSimaaiOverlay2 * self)
{
  for (auto & i: self->priv->profile_map) {
    auto pair = i.second;
    if (strcmp (pair.first.c_str(), "text") == 0) {
      if (strlen((const char *)self->priv->labels_file) > 0) {
        gst_simaai_overlay2_load_classification(self);
        return TRUE;
      }
    }
    if (strcmp (pair.first.c_str(), "bboxt") == 0
        || strcmp (pair.first.c_str(), "bboxd") == 0
        || strcmp (pair.first.c_str(), "bboxy") == 0
        || strcmp (pair.first.c_str(), "bbox") == 0
        || strcmp (pair.first.c_str(), "bboxs") == 0) {
      gst_simaai_overlay2_load_classification(self);
    }
  }

  return TRUE;
}

/**
 * @brief Helper API to get stride
 */
static gfloat get_stride (simaai::overlay::OverlayImageFormat fmt) {
  switch(fmt) {
    case simaai::overlay::OverlayImageFormat::SIMAAI_IMG_YUV420:
    case simaai::overlay::OverlayImageFormat::SIMAAI_IMG_NV12:
      return 1.5;
    case simaai::overlay::OverlayImageFormat::SIMAAI_IMG_YUV444:
    case simaai::overlay::OverlayImageFormat::SIMAAI_IMG_RGB:
    default:
      GST_ERROR("Can't get stride! Unsupported Image Format!");
      return -1;
  }
}

/**
 * @brief Helper API to check buffer list to contain allowable input
 */
static gboolean gst_simaai_overlay2_accept_buffer_list (GstSimaaiOverlay2 * self)
{
  auto get_buf_name = [self](const std::string& search_type) {
    for (const auto & [key, value]: self->priv->profile_map) {
      if (value.first == search_type)
        return value.second;
    }
    return std::string();
  };

  auto input = get_buf_name("input");
  if (input.empty())
    return FALSE;

  guint no_of_bufs = gst_buffer_list_length(self->priv->list);

  for (guint i = 0; i < no_of_bufs; ++i) {
    GstBuffer *buf = gst_buffer_list_get(self->priv->list, i);
    if (buf == NULL)
      continue;
    GstCustomMeta *meta = gst_buffer_get_custom_meta(buf, SIMAAI_META_STR);
    if (meta == NULL)
      continue;
    GstStructure *s = gst_custom_meta_get_structure(meta);
    if (s == NULL)
      continue;
    const gchar *buf_name = gst_structure_get_string(s, "buffer-name");
    if (buf_name == NULL)
      continue;
    if (!strcmp(input.c_str(), buf_name))
      return TRUE;
  }

  return FALSE;
}

/**
 * @brief Helper API to append buffers to input processing list
 */
static gboolean gst_simaai_overlay2_add2list (GstSimaaiOverlay2 * self, GValue * value)
{
  gint64 buf_id = 0;
  gint64 frame_id = 0;
  gint64 in_buf_offset = 0;

  GstAggregatorPad * pad = (GstAggregatorPad *) g_value_get_object (value);
  GstBuffer * buf = gst_aggregator_pad_peek_buffer (pad);

  if (buf) {
    GstCustomMeta * meta;
    GstStructure * s;
    gchar * buf_name;

    buf = gst_aggregator_pad_pop_buffer(pad);
    meta = gst_buffer_get_custom_meta(buf, SIMAAI_META_STR);
    if (meta != NULL) {
      s = gst_custom_meta_get_structure(meta);
      if (s == NULL) {
        gst_buffer_unref(buf);
        GST_OBJECT_UNLOCK (self);
        return GST_FLOW_ERROR;
      } else {
        if ((gst_structure_get_int64(s, "buffer-id", &buf_id) == TRUE) &&
            (gst_structure_get_int64(s, "frame-id", &frame_id) == TRUE) &&
            (gst_structure_get_int64(s, "buffer-offset", &in_buf_offset) == TRUE)) {

          g_string_assign(self->priv->stream_id, gst_structure_get_string(s, "stream-id"));
          buf_name = (gchar *)gst_structure_get_string(s, "buffer-name");
          gst_structure_get_uint64(s, "timestamp", &self->priv->timestamp);
          self->priv->frame_id = frame_id;
          gst_buffer_list_add(self->priv->list, buf);
          GST_DEBUG_OBJECT(self, "OVERLAY: Copied metadata, [%s]:[%ld]:[%ld]:[%s], buffer list length: %d",
                           (const char *) buf_name, frame_id, in_buf_offset, self->priv->stream_id->str, gst_buffer_list_length (self->priv->list));
        }
      }
    } else {
      GST_ERROR_OBJECT(self, "Please check readme to use metadata information, meta not found");
      return FALSE;
    }
  } else {
    GST_ERROR_OBJECT(self, "[CRITICAL] input buffer is NULL");
    return FALSE;
  }

  return TRUE;
}

/**
 * @brief The aggregate callback registered with the base gobject_class.
 * When data is queued on all the attached sinkpads this API is scheduled,
 * This means this synchronized with the upstream data available on the sinkpads
 * https://gstreamer.freedesktop.org/documentation/base/gstaggregator.html?gi-language=c
 */
static GstFlowReturn gst_simaai_overlay2_aggregate (GstAggregator * aggregator, gboolean timeout)
{
  GstIterator *iter;
  gboolean done_iterating = FALSE;
  GstMapInfo map;

  char buffer_name[MAX_NODE_NAME];

  GstSimaaiOverlay2 *self = GST_SIMAAI_OVERLAY2 (aggregator);

  auto clean_buffer_list = [](GstBufferList * list) {
    guint no_of_inbufs = gst_buffer_list_length(list);
    for (guint i = 0; i < no_of_inbufs ; ++i)
      gst_buffer_unref(gst_buffer_list_get(list, i));
    gst_buffer_list_remove(list, 0 , no_of_inbufs);
  };

  iter = gst_element_iterate_sink_pads (GST_ELEMENT (self));
  while (!done_iterating) {
    GValue value = { 0, };
    GstAggregatorPad *pad;
    GstBuffer *buf;

    switch (gst_iterator_next (iter, &value)) {
      case GST_ITERATOR_OK:
        if (!gst_simaai_overlay2_add2list(self, &value)) {
          gst_iterator_free (iter);
          return GST_FLOW_ERROR;
        }
        break;
      case GST_ITERATOR_RESYNC:
        gst_iterator_resync (iter);
        break;
      case GST_ITERATOR_ERROR:
        GST_WARNING_OBJECT (self, "Sinkpads iteration error");
        done_iterating = TRUE;
        gst_aggregator_pad_drop_buffer (pad);
        break;
      case GST_ITERATOR_DONE:
        done_iterating = TRUE;
        break;
    }
  }

  gst_iterator_free (iter);

  // fprintf(stderr, "Curent overlay frame_id: %ld", self->priv->frame_id);

  // Multi stream support: accept only allowed buffers
  if (gst_simaai_overlay2_accept_buffer_list(self) != TRUE) {
    GST_DEBUG_OBJECT(self, "Drop buffer list");

    clean_buffer_list(self->priv->list);

    /* The aggregator must send out a buffer to preserve the pipeline execution
     * flow. Therefore send a dummy buffer that will be dropped in the next
     * filter plugin.
     */
    GstBuffer * dummy_buf = gst_buffer_new();
    GST_BUFFER_FLAG_SET (dummy_buf, GST_BUFFER_FLAG_DECODE_ONLY);
    GST_BUFFER_FLAG_SET (dummy_buf, GST_BUFFER_FLAG_DROPPABLE);
    gst_aggregator_finish_buffer (aggregator, dummy_buf);

    return GST_FLOW_OK;
  } else {
    GST_DEBUG_OBJECT(self, "Accept buffer list");
  }

  GstBuffer * outbuf;
  GstFlowReturn ret = gst_buffer_pool_acquire_buffer(self->priv->pool, &outbuf, NULL);

  if (G_LIKELY (ret == GST_FLOW_OK)) {
    GST_DEBUG_OBJECT (self, "Acquired a buffer from pool %p", outbuf);
  } else {
    GST_ERROR_OBJECT (self, "Failed to allocate buffer");
    clean_buffer_list(self->priv->list);
    return GST_FLOW_ERROR;
  }

  // Run overlay2 here
  if (run_overlay2(self, outbuf) != TRUE) {
    GST_ERROR("Unable to run overlay, drop and continue");
    gst_buffer_unref(outbuf);
    return GST_FLOW_ERROR;
  }

  GstCustomMeta * meta = gst_buffer_add_custom_meta(outbuf, SIMAAI_META_STR);
  if (meta == NULL) {
    GST_ERROR("Unable to add metadata info to the buffer");
    gst_buffer_unref(outbuf);
    return GST_FLOW_ERROR;
  }
  GstStructure *s = gst_custom_meta_get_structure (meta);
  if (s != NULL) {
    gst_structure_set (s,
                        "buffer-id", G_TYPE_INT64, self->priv->out_buffer_id,
                        "buffer-name", G_TYPE_STRING, self->priv->out_buffer_name->str,
                        "buffer-offset", G_TYPE_INT64, (gint64)0 ,
                        "frame-id", G_TYPE_INT64, self->priv->frame_id,
                        "stream-id", G_TYPE_STRING, self->priv->stream_id->str,
                        "timestamp", G_TYPE_UINT64, self->priv->timestamp,
                        NULL);
  }

  GST_DEBUG_OBJECT(self, "Adding metadata: buffer_name=%s frame_id=%ld stream-id=%s",
                          self->priv->out_buffer_name->str,
                          self->priv->frame_id, self->priv->stream_id->str);

  gfloat stride = get_stride(self->priv->in_img_format);
  GST_DEBUG_OBJECT(self, ">>> out_sz=%f w=%d h=%d stride=%f",
                         self->priv->image_width * self->priv->image_height * stride,
                         self->priv->image_width, self->priv->image_height,
                         stride);

  gst_aggregator_finish_buffer (aggregator, outbuf);

  return GST_FLOW_OK;
}

/*
 * @brief Helper function to determine memory type for allocation
*/
static GstSimaaiMemoryFlags get_simamem_target (int mem_target) {
    switch(mem_target) {
    case 0:
        return GST_SIMAAI_MEMORY_TARGET_GENERIC;
    case 1:
        return GST_SIMAAI_MEMORY_TARGET_EV74;
    case 2:
        return GST_SIMAAI_MEMORY_TARGET_DMS0;
    case 4:
        return GST_SIMAAI_MEMORY_TARGET_DMS0;
    default:
        return GST_SIMAAI_MEMORY_TARGET_GENERIC;
    }
}

/**
 * @brief Helper API to allocate memory from gstsimaaiallocator->simamemlib
 * @todo: Current support is for YUV420/NV12/RGB only
 */
static gboolean gst_simaai_overlay2_allocate_out_memory (GstSimaaiOverlay2 * self)
{
  gsize sz = self->priv->image_width * self->priv->image_height 
      * get_stride(self->priv->in_img_format);
  GstSimaaiMemoryFlags mem_target = get_simamem_target(self->priv->mem_target);

  if (mem_target < (GstSimaaiMemoryFlags)self->priv->mem_type) {
    mem_target = (GstSimaaiMemoryFlags)self->priv->mem_type;
  }

  constexpr gsize no_of_bufs = MIN_POOL_SIZE;

  self->priv->pool = gst_simaai_allocate_buffer_pool(GST_OBJECT (self),
                                                      gst_simaai_memory_get_segment_allocator(),
                                                      sz,
                                                      no_of_bufs,
                                                      no_of_bufs,
                                                      (GstMemoryFlags)(mem_target | GST_SIMAAI_MEMORY_FLAG_CACHED));

  GST_DEBUG_OBJECT (self, "Allocated %zu buffers with size = %zu [w=%d h=%d "
                          "stride=%f]",
                          no_of_bufs, sz,
                          self->priv->image_width,
                          self->priv->image_height,
                          get_stride(self->priv->in_img_format));

  return TRUE;
}

/**
 * @brief Helper API to validate profile string
 * @todo: Handle undefined behavior
 * @todo: Graceful exit, if wrong information
 */
static gboolean gst_simaai_overlay2_validate_profile (GstSimaaiOverlay2 * self) {
  if (self->priv->profile_map.size() == 0) {
    GST_ERROR("Please check README, to work with profiles for SiMa's overlay plugin");
    return FALSE;
  }

#if 0
  if (!self->priv->overlay->is_overlay_supported()) {
    GST_ERROR_OBJECT(self, "The overlay type is not supported yet");
    return FALSE;
  }
#endif

  if (!gst_simaai_overlay2_check_for_text_render (self)) {
    GST_ERROR_OBJECT(self, "For text overlay, labels.txt should be provided");
    return FALSE;
  }

  return TRUE;
}

const char * get_color_format_string(const simaai::overlay::OverlayImageFormat &format)
{
  const char *ret = NULL;

  switch (format)
  {
  case simaai::overlay::OverlayImageFormat::SIMAAI_IMG_NV12:
    ret = "NV12";
    break;
  case simaai::overlay::OverlayImageFormat::SIMAAI_IMG_YUV420:
    ret = "I420";
    break;
  case simaai::overlay::OverlayImageFormat::SIMAAI_IMG_YUV444:
  case simaai::overlay::OverlayImageFormat::SIMAAI_IMG_RGB:
  default:
    ret = "";
    break;
  }

  return ret;
}

GstFlowReturn 
gst_simaai_overlay2_update_src_caps(GstAggregator *  aggregator,
                                    GstCaps       *  caps,
                                    GstCaps       ** ret)
{
  GstSimaaiOverlay2 * self = GST_SIMAAI_OVERLAY2(aggregator);

  if (caps == NULL)
  {
    GST_ERROR_OBJECT(self, "Not negotiated, input caps is NULL");
    return GST_FLOW_NOT_NEGOTIATED;
  }

  GST_DEBUG_OBJECT(self, "input caps: %s", gst_caps_to_string(caps));

  // Try to specify caps according to the private fields(set prop)
  GstCaps *custom_caps = gst_caps_new_simple("video/x-raw",
                                              "width", G_TYPE_INT, self->priv->image_width,
                                              "height", G_TYPE_INT, self->priv->image_height,
                                              "format", G_TYPE_STRING, get_color_format_string(self->priv->in_img_format),
                                              "framerate", GST_TYPE_FRACTION, self->priv->framerate_numerator, self->priv->framerate_denominator,
                                              NULL);

  GST_DEBUG_OBJECT(self, "Try to set src caps: %s", gst_caps_to_string(custom_caps));
  if (gst_caps_is_subset (custom_caps, caps) == TRUE) {
    GST_DEBUG_OBJECT(self, "Src caps was set successful");
    *ret = custom_caps;
    return GST_FLOW_OK;
  }

  GST_ERROR_OBJECT(self, "Not negotiated, due to the fact that caps '%s' is not subset of '%s'",
                    gst_caps_to_string(custom_caps),
                    gst_caps_to_string(caps));

  return GST_FLOW_NOT_NEGOTIATED;
}

gboolean parse_sink_event_caps_video_x_raw (GstSimaaiOverlay2 * const self,
                                            const GstStructure * const structure)
{
  gboolean ret = TRUE;

  // Get input image format
  const char * in_img_format = gst_structure_get_string(structure, "format");
  if (in_img_format == NULL) {
    GST_ERROR_OBJECT(self, "Input format is NULL!");
    ret = FALSE;
  } else {
    self->priv->in_img_format = str_format_to_enum(in_img_format);
    if (self->priv->in_img_format == simaai::overlay::OverlayImageFormat::SIMAAI_IMG_UNKNOWN) {
      GST_ERROR_OBJECT(self, "Unsupported input image format!");
      ret = FALSE;
    }
  }

  // Get input framerate
  if (gst_structure_get_fraction(structure, "framerate", &self->priv->framerate_numerator,
                                                         &self->priv->framerate_denominator) == FALSE) {
    self->priv->framerate_numerator = DEFAULT_FRAMERATE_NUM;
    self->priv->framerate_denominator = DEFAULT_FRAMERATE_DEN;
    GST_INFO_OBJECT(self, "Framerate set as default: %d/%d",
                    self->priv->framerate_numerator,
                    self->priv->framerate_denominator);
  }

  // Get input width and height
  if (gst_structure_get_int(structure, "width", &self->priv->image_width) == FALSE) {
    GST_ERROR_OBJECT(self, "Image width didn't set!");
    ret = FALSE;
  }
  if (gst_structure_get_int(structure, "height", &self->priv->image_height) == FALSE) {
    GST_ERROR_OBJECT(self, "Image height didn't set!");
    ret = FALSE;
  }

  return ret;
}

static void remove_spaces_in_front_and_back(std::string &str)
{
  auto front_iter = str.begin();
  while (*front_iter == ' ')
    str.erase(front_iter);

  auto back_iter = str.end() - 1;
  while (*back_iter == ' ')
    str.erase(back_iter--);
}

static std::vector<std::string> split_string(std::string source, const char ch)
{
  std::vector<std::string> result;

  int start = 0;
  int end = 0;
  std::string each_str;

  end = source.find(ch, start);
  while (end != std::string::npos)
  {
    int len = end - start;
    each_str = std::move(source.substr(start, len));
    remove_spaces_in_front_and_back(each_str);

    result.push_back(each_str);
    start = end + 1;

    end = source.find(ch, start);
  }

  each_str = std::move(source.substr(start));
  remove_spaces_in_front_and_back(each_str);
  result.push_back(std::move(each_str));

  return result;
}

gboolean parse_sink_event_video_simaai_tensors( GstSimaaiOverlay2 * const self,
                                                const GstStructure * const structure)
{
  // key:   groupkeypoints format
  // value: overlay format
  static const std::map<std::string, std::string> format_convertation {
    {"BBOX",          "bbox"},
    {"BBOX_SCORE",    "bboxs"},
    {"BBOX_YOLO",     "bboxy"},
    {"BBOX_TRACKER",  "bboxt"},
    {"BBOX_DISTANCE", "bboxd"},
    {"PLAIN_TEXT",    "text"},
    {"POSES",         "pose"},
  };

  const char * application_format = gst_structure_get_string(structure, "format");

  // FIXME: it is necessary to determine which separator will be used in the caps
  // In the gst-lines used "|"
  std::vector<std::string> source_formats = split_string(application_format, ',');
  for (const auto &iter : source_formats) {
    if (format_convertation.contains(iter)) {
      std::string format = format_convertation.at(iter);
      self->priv->render_rules.push_back(format);
    }
  }

  if (self->priv->render_rules.empty())
    return FALSE;

  for (const auto &iter : self->priv->render_rules) {
    GST_DEBUG_OBJECT(self, "Adding app format to render rules: %s", iter.c_str());
  }

  return TRUE;
}

gboolean
gst_simaai_overlay2_sink_event (GstAggregator * aggregator,
            GstAggregatorPad * aggregator_pad,
            GstEvent * event)
{
  GstCaps * caps;
  GstStructure *structure = NULL;

  GstSimaaiOverlay2 * self = GST_SIMAAI_OVERLAY2(aggregator);

  if (GST_EVENT_TYPE(event) == GST_EVENT_CAPS) {
    gst_event_parse_caps(event, &caps);
    GST_DEBUG_OBJECT(self, "sink_pad = %s, caps = %s", gst_pad_get_name(aggregator_pad), gst_caps_to_string(caps));

    structure = gst_caps_get_structure(caps, 0);
    const gchar* media_type = gst_structure_get_name(structure);

    if (strcmp(media_type, "video/x-raw") == 0) {
      if (parse_sink_event_caps_video_x_raw(self, structure) == FALSE) {
        GST_ERROR_OBJECT(self, "[video/x-raw]: If we reached here, there is something really wrong.");
        return FALSE;
      }
    } else if (strcmp(media_type, "application/vnd.simaai.tensor") == 0) {
      if (parse_sink_event_video_simaai_tensors(self, structure) == FALSE) {
        GST_ERROR_OBJECT(self, "[application/vnd.simaai.tensor]: If we reached here, there is something really wrong.");
        return FALSE;
      }
    }
  }

  return GST_AGGREGATOR_CLASS (parent_class)->sink_event (aggregator, aggregator_pad, event);
}

gboolean gst_simaai_overlay2_negotiated_src_caps (GstAggregator * aggregator,
                                                  GstCaps       * caps)
{
  GstSimaaiOverlay2 * self = GST_SIMAAI_OVERLAY2(aggregator);
  GstElement *element = GST_ELEMENT(aggregator);

  if (!gst_simaai_overlay2_allocate_out_memory(self)) {
    GST_ERROR ("Allocation failed, simaai_mem");
    return FALSE;
  }

  self->priv->overlay = std::make_unique<simaai::overlay::Overlay>(self->priv->in_img_format, self->priv->image_width, self->priv->image_height);
  self->priv->overlay->update_render_config(self->priv->profile_map);
  self->priv->overlay->set_render_rules(self->priv->render_rules);

  if (!gst_simaai_overlay2_validate_profile(self)) {
    GST_ERROR ("Overlay type not supported yet");
    return FALSE;
  }

  self->priv->overlay->set_verbose(!self->silent);
  self->priv->overlay->set_dump_data(self->priv->dump_data);

  /* Store a name of the element */
  gchar *ename = gst_element_get_name(element);
  if (ename != NULL)
    self->priv->overlay->set_name(ename);
  else
    self->priv->overlay->set_name("");

  g_free(ename);

  if (!self->priv->overlay->init_render(self->priv->labels)) {
    GST_ERROR("Unable to update labels information");
    return FALSE;
  }

  return TRUE;
}

static gboolean gst_simaai_overlay2_propose_allocation (GstAggregator * self,
                    GstAggregatorPad * pad,
                    GstQuery * decide_query,
                    GstQuery * query)
{
  GstSimaaiOverlay2 * overlay = GST_SIMAAI_OVERLAY2(self);
  GST_DEBUG_OBJECT(overlay, "propose_allocation called");

  GstStructure *allocation_meta = gst_simaai_allocation_query_create_meta(GST_SIMAAI_MEMORY_TARGET_GENERIC, GST_SIMAAI_MEMORY_FLAG_DEFAULT);

  gst_simaai_allocation_query_add_meta(query, allocation_meta);   

  return TRUE;
}

static gboolean gst_simaai_overlay2_decide_allocation(GstAggregator * self, GstQuery * query)
{
  GstSimaaiOverlay2 * overlay = GST_SIMAAI_OVERLAY2(self);
  GST_DEBUG_OBJECT(overlay, "decide_allocation called");

  GstSimaaiMemoryFlags mem_type;
  GstSimaaiMemoryFlags mem_flag;

  if (!gst_simaai_allocation_query_parse(query, &mem_type, &mem_flag)) {
    GST_WARNING_OBJECT(self, "Can't find allocation meta!");
  } else {
    overlay->priv->mem_type = (gint)mem_type;
    overlay->priv->mem_flag = (gint)mem_flag;
  }

  GST_DEBUG_OBJECT(overlay, "Memory flags to allocate: [ %s ] [ %s ]",
    gst_simaai_allocation_query_sima_mem_type_to_str(mem_type),
    gst_simaai_allocation_query_sima_mem_flag_to_str(mem_flag));
  
  return TRUE;
}

/**
 * @brief Called to perform state change.
 */
static GstStateChangeReturn
gst_simaai_overlay2_change_state (GstElement * element,
                                  GstStateChange transition)

{
  GstSimaaiOverlay2 * self = GST_SIMAAI_OVERLAY2 (element);
  GstAggregator * aggregator = (GstAggregator*) (element);
  GstStateChangeReturn ret;

  ret = GST_ELEMENT_CLASS(parent_class)-> change_state(element, transition);
  if (ret == GST_STATE_CHANGE_FAILURE) {
    return ret;
  }

  switch(transition) {
    case GST_STATE_CHANGE_READY_TO_NULL:
      if (self->priv->pool != NULL) {
        gst_buffer_pool_set_active(self->priv->pool, FALSE);
        gst_object_unref(self->priv->pool);
        self->priv->pool = NULL;
      }
      break;
    default:
      break;
  }

  return ret;
}

/**
 * @brief Helper API to dump buffers
 * @todo We don't need this, as we may use ``filesink``
 */
static int32_t dump_intermediate_buffer(GstSimaaiOverlay2 * self, void * vaddr,
                                        gint64 id, gsize s, char * name)
{
  FILE *ofp;
  int32_t res = 0;

  char full_opath[256];
  snprintf(full_opath, sizeof(full_opath) - 1, "/tmp/%s-%ld.yuv", name, id);

  ofp = fopen(full_opath, "w");
  if(ofp == NULL) {
    return res = -1;
  }

  size_t wosz = fwrite(vaddr, 1, s, ofp);
  if(s != wosz) {
    return res = -3;
  }

  fclose(ofp);
  return res;
}

/**
 * @brief Helper API to update meminfo to the overlay library-like impl
 */
static gboolean gst_simaai_overlay2_update_meminfo (GstSimaaiOverlay2 * self , GstBuffer * buf, GstMapInfo * meminfo)
{
  GstCustomMeta * meta;
  GstStructure *s;
  gchar * buf_name;

  meta = gst_buffer_get_custom_meta(buf, SIMAAI_META_STR);

  if (!meta) {
    GST_ERROR("SiMaMetaInformation not found");
    return FALSE;
  }

  s = gst_custom_meta_get_structure(meta);
  if (!s) {
    GST_ERROR("SiMaMetaInformation structure not found");
    return FALSE;
  }

  buf_name = (gchar *)gst_structure_get_string(s, "buffer-name");
  
  self->priv->overlay->update_mem_info(buf_name, (unsigned char *)meminfo->data);
  
  return TRUE;
}

/**
 * @brief Run-API, the worker like function which handles the rendring and this is run for each input GstBuffer
 */
static gboolean run_overlay2 (GstSimaaiOverlay2 * self, GstBuffer * outbuf)
{

  self->priv->t0 = std::chrono::steady_clock::now();
  gchar* plugin_id = gst_element_get_name(GST_ELEMENT(self));
  if (self->transmit) {
    tracepoint_pipeline_a65_start(self->priv->frame_id, plugin_id, self->priv->stream_id->str);
  }

  GstMemory *mem;

  gsize no_of_inbufs = gst_buffer_list_length(self->priv->list);
  if (no_of_inbufs <= 0) {
    GST_ERROR("No input buffers in the list, :%ld", no_of_inbufs);
    return FALSE;
  }

  GstBuffer * buf[no_of_inbufs];
  GstMapInfo meminfo[OVERLAY_MAX_MEMINFO];

  for (int i = 0; i < no_of_inbufs; i++) {
    buf[i] = gst_buffer_list_get(self->priv->list, i);
    gst_buffer_map(buf[i], &meminfo[i], GST_MAP_READ);

    

    /* Enable this incase of problems, to debug */
#if 0
    GST_DEBUG_OBJECT(self, "Buffer addr:%p, memaddr: %p", buf[i], &meminfo[i]);
#endif
    if (!gst_simaai_overlay2_update_meminfo(self, buf[i], &meminfo[i])) {
      GST_ERROR("Failed to run overlay, exiting");
      return FALSE;
    }
  }

  GstMapInfo out_map;

  gst_buffer_map(outbuf, &out_map, GST_MAP_WRITE);
  if ((out_map.data == NULL) || (out_map.size == 0)) {
    GST_ERROR("Unable to output buffer map error");
    return FALSE;
  }

  self->priv->out_buffer_id = gst_simaai_segment_memory_get_phys_addr(out_map.memory);
  self->priv->overlay->update_frame_id(self->priv->frame_id);

  if (!self->priv->overlay->update_renderer((unsigned char *)out_map.data)) {
    GST_ERROR("Rendering overlay failed");
    gst_buffer_unmap(outbuf, &out_map);
    return FALSE;
  }

  gst_buffer_unmap(outbuf, &out_map);

  if (self->transmit) {
    tracepoint_pipeline_a65_end(self->priv->frame_id, plugin_id, self->priv->stream_id->str);
  }
  g_free(plugin_id);

  self->priv->t1 = std::chrono::steady_clock::now();
  auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(self->priv->t1 - self->priv->t0);
  auto duration = elapsed.count() / 1000.0 ;
  if (!self->silent) {
    std::cout << "Overlay frame_cnt[" << self->priv->frame_id
	      << "], run time in ms: " << elapsed.count()/1000.0 << "\n";
  }

  for (int i = 0; i < no_of_inbufs ; i++)  {
    gst_buffer_unmap(buf[i], &meminfo[i]);
    gst_buffer_unref(gst_buffer_list_get(self->priv->list, i));
  }

  gst_buffer_list_remove(self->priv->list, 0 , no_of_inbufs);


  return TRUE;
}

/**
 * @brief Glib Class init
 */
static void gst_simaai_overlay2_class_init (GstSimaaiOverlay2Class * klass)
{
  GObjectClass *gobj_class = G_OBJECT_CLASS(klass);
  GstElementClass *gstelement_class = (GstElementClass *) klass;
  GstAggregatorClass *base_aggregator_class = (GstAggregatorClass *) klass;

  static GstStaticPadTemplate _src_template =
      GST_STATIC_PAD_TEMPLATE("src", GST_PAD_SRC, GST_PAD_ALWAYS,
      GST_STATIC_CAPS("video/x-raw, format=(string){I420,NV12}, "
      "width=(int)[1,4096], height=(int)[1,4096]"));

  static GstStaticPadTemplate _sink_img_src_template =
      GST_STATIC_PAD_TEMPLATE("sink_in_img_src", GST_PAD_SINK, GST_PAD_REQUEST,
      GST_STATIC_CAPS("video/x-raw, format=(string){I420,NV12}, "
      "width=(int)[1,4096], height=(int)[1,4096]"));

  GstCaps *sink_caps = gst_caps_from_string(simaai::overlay::get_renderers().c_str());
  static GstStaticPadTemplate _sink_data_template = 
      GST_STATIC_PAD_TEMPLATE("sink_application_data", GST_PAD_SINK, 
                              GST_PAD_REQUEST,sink_caps);

  gst_element_class_add_static_pad_template_with_gtype (gstelement_class,
                                                        &_src_template, 
                                                        GST_TYPE_AGGREGATOR_PAD);

  gst_element_class_add_static_pad_template_with_gtype (gstelement_class,
                                                        &_sink_img_src_template, GST_TYPE_AGGREGATOR_PAD);

  gst_element_class_add_static_pad_template_with_gtype (gstelement_class,
                                                        &_sink_data_template, GST_TYPE_AGGREGATOR_PAD);

  gst_element_class_set_static_metadata (gstelement_class, "Overlay2",
                                         "Overlay",
                                         "Overlay data on input image",
                                         "SiMa.Ai Technologies");

  gobj_class->set_property = gst_simaai_overlay2_set_property;
  gobj_class->get_property = gst_simaai_overlay2_get_property;
  gobj_class->finalize = gst_simaai_overlay2_finalize;

  base_aggregator_class->aggregate =
      GST_DEBUG_FUNCPTR (gst_simaai_overlay2_aggregate);
  base_aggregator_class->sink_event =
      GST_DEBUG_FUNCPTR (gst_simaai_overlay2_sink_event);
  base_aggregator_class->negotiated_src_caps =
      GST_DEBUG_FUNCPTR (gst_simaai_overlay2_negotiated_src_caps);

  base_aggregator_class->update_src_caps =
      GST_DEBUG_FUNCPTR (gst_simaai_overlay2_update_src_caps);

  base_aggregator_class->decide_allocation = 
      GST_DEBUG_FUNCPTR (gst_simaai_overlay2_decide_allocation);

  base_aggregator_class->propose_allocation =
      GST_DEBUG_FUNCPTR (gst_simaai_overlay2_propose_allocation);

  gstelement_class->change_state =
      GST_DEBUG_FUNCPTR (gst_simaai_overlay2_change_state);

  gst_simaai_overlay2_install_properties(gobj_class);

  gst_element_class_set_static_metadata (gstelement_class,
                                         "SiMa.AI Overlay Plugin",
                                         "simaai-overlay2",
                                         "Overlay Plugin, support for bbox, text, points, polygons",
                                         "SiMa.AI");

  GST_DEBUG_CATEGORY_INIT (GST_CAT_DEFAULT,
			   "simaai-overlay2", 0, "SiMaAi Overlay");
}

/**
 * @brief Subclass initialization
 */
static void gst_simaai_overlay2_init (GstSimaaiOverlay2 * self)
{
  GstAggregator *agg = GST_AGGREGATOR (self);
  //gst_segment_init (&GST_AGGREGATOR_PAD (agg->srcpad)->segment,
  //                  GST_FORMAT_TIME);

  gst_simaai_segment_memory_init_once();

  self->priv = new GstSimaaiOverlay2Private;
  self->priv->list = gst_buffer_list_new();
  self->priv->frame_id = -1;
  self->priv->timestamp = 0;
  self->priv->stream_id = g_string_new("stream-id-unknown");

  self->priv->is_text_overlay = FALSE;
  self->priv->labels_file = g_string_new(DEFAULT_LABELS_FILE);

  self->priv->out_buffer_name = g_string_new("overlay");

  self->silent = DEFAULT_SILENT;
  self->transmit = DEFAULT_TRANSMIT;

  self->priv->mem_target = 0;
  self->priv->pool = NULL;

  self->priv->dump_data = FALSE;

  self->priv->mem_type = (gint)GST_SIMAAI_MEMORY_TARGET_GENERIC;
  self->priv->mem_type = (gint)GST_SIMAAI_MEMORY_FLAG_DEFAULT;
}

/**
 * @brief Finalize/Cleanup overlay2 callback
 */
static void
gst_simaai_overlay2_finalize (GObject * object)
{
  GstSimaaiOverlay2 * self = GST_SIMAAI_OVERLAY2 (object);
  g_string_free(self->priv->labels_file, TRUE);
  g_string_free(self->priv->out_buffer_name, TRUE);
  g_string_free(self->priv->stream_id, TRUE);
  gst_buffer_list_unref(self->priv->list);
  delete self->priv;
  G_OBJECT_CLASS (parent_class)->finalize (object);
}

/**
 * @brief plugin init
 */
static gboolean
plugin_init (GstPlugin * plugin)
{
  if (!gst_element_register(plugin, "simaai-overlay2", GST_RANK_NONE,
                            GST_TYPE_SIMAAI_OVERLAY2)) {
    GST_ERROR("Unable to register simaai overlay2 plugin");
    return FALSE;
  }

  return TRUE;
}

/**
 * @brief Gobject definition
 */
GST_PLUGIN_DEFINE(
    GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    PLUGIN_NAME_LOWER,
    "GStreamer SiMa.ai Overlay2 Plugin",
    plugin_init,
    VERSION,
    GST_LICENSE,
    GST_PACKAGE_NAME,
    GST_PACKAGE_ORIGIN);
