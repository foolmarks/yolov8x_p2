/*
 * GStreamer
 * Copyright (C) 2023-2025 SiMa.ai
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
 * @file gstsimaaiprocessmla.cc
 * @brief Gstreamer plugin to run mla
 * @author SiMa.Ai\TM
 * @bug Currently no known bugs
 * @todo Advanced CAPS negotiation technique.
 * @todo Cleanup serialized stream call flow.
 */

/**
 * SECTION: element-simaaiprocessmla
 *
 * Plugin to run MLA in any pipeline
 *
 * <refsect2>
 * <title> Example Launch line </title>
 * |[
 * ..input
 * ! simaaiprocessmla name=mla config="mla.json" ! fakesink
 * ]|
 * </refsect2>
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

//#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>

#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <gst/gst.h>

#include <gstsimaaiallocator.h>
#include <gstsimaaibufferpool.h>
#include <simaai/trace/pipeline_tp.h>
#include <gstsimaaicaps.h>

#include "gstsimaaiprocessmla.h"
#include <dispatcherfactory.hh>
#include <simaai/nlohmann/json.hpp>
#include <simaai/trace/pipeline_new_tp.h>
#include <simaai/trace/remote_core_tp.h>
#include <utils_string.h>

/**
 * @brief Macro for debug mode.
 */
#ifndef DBG
#define DBG (!self->silent)
#endif

/**
 * @brief Macro for debug message.
 */
#define SILENT_GST_DEBUG(self, ...) do { \
    if (DBG) { \
      GST_DEBUG_OBJECT (self, __VA_ARGS__); \
    } \
  } while (0)

/**
 * @brief Flag to print minimized log.
 */
#define DEFAULT_SILENT TRUE
#define DEFAULT_TRANSMIT FALSE
#define PLUGIN_CPU_TYPE "MLA"

/**
 * @brief keypoints properties
 */
enum {
  PROP_0,
  PROP_TRANSMIT,
  PROP_MULTIPIPELINE,
  PROP_CONFIG_PATH,
  PROP_NO_OF_BUFS,
  PROP_DUMP_DATA,
  PROP_SILENT,
  PROP_LAST,
};

enum SEG_NAME_STATE {
  UNDEFINED = 0,
  NO_NAME,
  HAS_NAME
};

/**
 * @brief Private member structure for GstSimaaiTopk2 instances
 */
struct _GstSimaaiProcessMLA_Private
{
  gboolean dump_data;
  std::string node_name;

  GstBufferPool *pool; /**< Buffer pool */
  int no_of_obufs; /**< number of output memory chuncks to allocate */
  size_t out_size; /**< size of 1 memory chunk to allocate */

  simaaidispatcher::DispatcherBase *dispatcher;
  simaaidispatcher::DispatcherFactory::HWType MLA_dispatcher_type;

  std::string model_path;
  void *model_handle;
  gint32 batch_size, batch_model;
  /// @brief state of input segment name. Is is specified, or not
  SEG_NAME_STATE in_segment_name_state;
  std::string input_seg_name;
  gint32 timeout;
  std::string config_path;
  nlohmann::json config;
  guintptr out_buffer_id; /**< physicall addres of current output buffer */
  gint64 in_buf_id;

  gint64 frame_id; /**< Input frame id placeholder */
  guint64 timestamp;
  std::string stream_id;
  gint64 in_pcie_buf_id;
  gboolean is_pcie;

  std::chrono::time_point<std::chrono::steady_clock> t0;
  std::chrono::time_point<std::chrono::steady_clock> t1;

  std::vector<std::string> segment_names;
  std::vector<size_t> segment_sizes;

  /// Time point pair to store the kernel start and end time measured in dispatcher
  std::pair<TimePoint, TimePoint> tp;
  GstSimaaiMemoryFlags mem_type;
  GstSimaaiMemoryFlags mem_flag;

  GstSimaaiCaps *simaai_caps;
};

GST_DEBUG_CATEGORY_STATIC(gst_simaai_process_mla_debug);
#define GST_CAT_DEFAULT gst_simaai_process_mla_debug

#define gst_simaai_process_mla_parent_class parent_class
G_DEFINE_TYPE(GstSimaaiProcessMLA, gst_simaai_process_mla, GST_TYPE_BASE_TRANSFORM);

/* All statics go in here */
static void gst_simaai_process_mla_set_property (GObject * obj, guint prop_id,
                                           const GValue * value, GParamSpec * pspec);
static void gst_simaai_process_mla_get_property (GObject * obj, guint prop_id,
                                           GValue * value, GParamSpec * pspec);

static gboolean run_process_mla (GstSimaaiProcessMLA * self, GstBuffer * inbuf, GstBuffer * outbuf);
static gboolean gst_simaai_process_mla_extract_meta_info(GstSimaaiProcessMLA * self,
                                                            GstBuffer * inbuf);
static GstCaps * gst_simaai_process_mla_transform_caps (
  GstBaseTransform * trans, GstPadDirection direction, GstCaps * caps, GstCaps * filter);

static gboolean gst_simaai_process_mla_propose_allocation (
  GstBaseTransform * trans, GstQuery * decide_query, GstQuery * query);

static gboolean gst_simaai_process_mla_decide_allocation (
  GstBaseTransform * trans, GstQuery * query);

/**
 * @brief helper to print time of code exectution
 */
static inline std::chrono::microseconds
print_exec_time(const std::chrono::time_point<std::chrono::steady_clock>& start,
                const std::chrono::time_point<std::chrono::steady_clock>& end,
                const std::string& msg)
{
  auto elapsed = 
      std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  GST_DEBUG("%s : %f ms", msg.c_str(), elapsed.count()/1000.0);
  return elapsed;
}

/**
 * @brief Setter for process mla properties.
 */
static void gst_simaai_process_mla_set_property (GObject * object,
                                           guint prop_id,
                                           const GValue * value,
                                           GParamSpec * pspec)
{
  GstSimaaiProcessMLA * self = GST_SIMAAI_PROCESS_MLA (object);
  gboolean multipipeline;

  switch(prop_id) {
    case PROP_TRANSMIT:
      self->transmit = g_value_get_boolean (value);
      GST_DEBUG_OBJECT(self, "Set transmit = %d", self->transmit);
      break;
    case PROP_MULTIPIPELINE:
      multipipeline = g_value_get_boolean (value);
      GST_DEBUG_OBJECT(self, "Set multi-pipeline = %d", multipipeline);
      self->priv->MLA_dispatcher_type = (multipipeline == TRUE) ?
          simaaidispatcher::DispatcherFactory::MLASHM :
          simaaidispatcher::DispatcherFactory::MLA;
      break;
    case PROP_SILENT:
      self->silent = g_value_get_boolean (value);
      GST_DEBUG_OBJECT (self, "Set silent = %d", self->silent);
      break;
    case PROP_DUMP_DATA:
      self->priv->dump_data = g_value_get_boolean (value);
      GST_DEBUG_OBJECT (self, "Set dump_data = %d", self->priv->dump_data);
      break;
    case PROP_CONFIG_PATH:
      self->priv->config_path = std::string(g_value_get_string(value));
      GST_DEBUG_OBJECT(self, "Config file path argument was changed to %s", 
                       self->priv->config_path.c_str());
      break;
    case PROP_NO_OF_BUFS:
      self->priv->no_of_obufs = g_value_get_ulong(value);
      GST_DEBUG_OBJECT(self, "Number of buffers argument was changed to %d", 
                       self->priv->no_of_obufs);
      break;
    default:
      GST_DEBUG_OBJECT(self, "Default case warning");
      G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
      break;
  }
}

/**
 * @brief Getter for process mla properties.
 */
static void gst_simaai_process_mla_get_property (GObject * object,
                                           guint prop_id,
                                           GValue * value,
                                           GParamSpec * pspec)
{
  GstSimaaiProcessMLA * self = GST_SIMAAI_PROCESS_MLA (object);

  switch(prop_id) {
    case PROP_TRANSMIT:
      g_value_set_boolean (value, self->transmit);
      break;
    case PROP_MULTIPIPELINE:
      if (self->priv->MLA_dispatcher_type ==
          simaaidispatcher::DispatcherFactory::MLASHM)
        g_value_set_boolean (value, TRUE);
      else
        g_value_set_boolean (value, FALSE);
      break;
    case PROP_SILENT:
      g_value_set_boolean(value, self->silent);
      break;
    case PROP_DUMP_DATA:
      g_value_set_boolean(value, self->priv->dump_data);
      break;
    case PROP_CONFIG_PATH:
      g_value_set_string(value, self->priv->config_path.c_str());
      break;
    case PROP_NO_OF_BUFS:
      g_value_set_ulong(value, self->priv->no_of_obufs);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
      break;
  }
}

/**
 * @brief helper function to get memory target
 * @todo move to libgstsimautils.so
 */
static GstSimaaiMemoryFlags get_mem_target (sima_cpu_e cpu) {
    switch(cpu) {
    case SIMA_CPU_EVXX:
        return GST_SIMAAI_MEMORY_TARGET_EV74;
    case SIMA_CPU_MLA:
        return GST_SIMAAI_MEMORY_TARGET_DMS0;
    case SIMA_CPU_MOSAIC:
        return GST_SIMAAI_MEMORY_TARGET_DMS0;
    case SIMA_CPU_A65:
    case SIMA_CPU_DEV:
        return GST_SIMAAI_MEMORY_TARGET_GENERIC;
    default:
        return GST_SIMAAI_MEMORY_TARGET_GENERIC;
    }
}

/**
 * @brief helper function to allocate output memory
 */
static gboolean gst_simaai_process_mla_allocate_memory(GstSimaaiProcessMLA * self)
{
  if (self->priv->pool) {
    gst_buffer_pool_set_active(self->priv->pool, FALSE);
    gst_object_unref(self->priv->pool);
    self->priv->pool = NULL;
  }

  if (self->priv->no_of_obufs < MIN_POOL_SIZE)
    self->priv->no_of_obufs = MIN_POOL_SIZE;

  sima_cpu_e cpu = SIMA_CPU_MLA;
  int no_of_segments = self->priv->config["outputs"].size();
  GST_DEBUG_OBJECT(self, "Allocating %d segments", no_of_segments);
  GstSimaaiMemoryFlags mem_target = get_mem_target(cpu);

  mem_target = 
      (mem_target < self->priv->mem_type) ? self->priv->mem_type : mem_target;

  GstMemoryFlags flags = (GstMemoryFlags)(mem_target | GST_SIMAAI_MEMORY_FLAG_CACHED);
  GstAllocator *allocator = gst_simaai_memory_get_segment_allocator();
  
  std::vector<char*> cstr;
  cstr.reserve(no_of_segments + 1);

  for (auto &it : self->priv->segment_names) {
       cstr.push_back(&it[0]);
  }

  cstr.push_back(nullptr);

  self->priv->pool = gst_simaai_allocate_buffer_pool2((GstObject*) self,
                                               allocator,
                                               MIN_POOL_SIZE,
                                               self->priv->no_of_obufs,
                                               flags,
                                               no_of_segments,
                                               self->priv->segment_sizes.data(),
                                               const_cast<const char**>(cstr.data()));                                                    
  GST_DEBUG_OBJECT (self, "Output buffer pool: %d buffers of size %ld",
                    self->priv->no_of_obufs, self->priv->out_size);
  return TRUE;
}

/**
 * @brief Callback called when element stops processing
 *
 * @param trans the gobject for the base class GstBaseTransform
 * @return TRUE on success or FALSE on Failure
 */
static gboolean
gst_simaai_process_mla_stop (GstBaseTransform *trans)
{
  GstSimaaiProcessMLA *self = GST_SIMAAI_PROCESS_MLA(trans);
  gst_simaai_free_buffer_pool(self->priv->pool);
  self->priv->pool = NULL;
  return TRUE;
}

bool parse_json_from_file(GstSimaaiProcessMLA * plugin,
                          const std::string config_file_path, 
                          nlohmann::json &json) 
{
  try {
    //parse file to nlohmann::json objects
    std::ifstream input_file(config_file_path.c_str());
    if (!input_file) 
      throw std::runtime_error( std::string("Error opening file ") +
                                config_file_path );
    std::ostringstream string_stream;
    string_stream << input_file.rdbuf();

    json = nlohmann::json::parse(string_stream.str());

    return true;

  } catch (std::exception & ex) {
    GST_ERROR_OBJECT(plugin, "Unable to parse config file: %s", ex.what());
    return false;
  }
}

bool parse_output_segments(GstSimaaiProcessMLA *self)
{
  if (self->priv->config.contains("outputs")) {
    size_t no_of_segments = self->priv->config["outputs"].size();
    if (no_of_segments == 0) {
      GST_ERROR_OBJECT(self, "No output segments provied in the config file!");
    } else {
      self->priv->segment_names.reserve(no_of_segments);
      self->priv->segment_sizes.reserve(no_of_segments);
    }
    
    for (auto &it : self->priv->config["outputs"].items()) {
        self->priv->segment_names.push_back(it.value()["name"]);
        self->priv->segment_sizes.push_back(it.value()["size"]);
        self->priv->out_size += static_cast<size_t>(it.value()["size"]);
        GST_DEBUG_OBJECT (self, "Adding output segment %s with size %ld",
                          static_cast<std::string>(it.value()["name"]).c_str(),
                          static_cast<size_t>(it.value()["size"]));
      }
    }

  return true;
}

/**
 * @brief Callback called when element starts processing
 *
 * @param trans the gobject for the base class GstBaseTransform
 * @return TRUE on success or FALSE on Failure
 */
static gboolean
gst_simaai_process_mla_start (GstBaseTransform * trans)
{
  GstSimaaiProcessMLA *self = GST_SIMAAI_PROCESS_MLA(trans);

  GstCaps * sink_caps = gst_pad_get_allowed_caps(trans->sinkpad);
  GstCaps * src_caps = gst_pad_get_allowed_caps(trans->srcpad);
  GST_INFO_OBJECT(self, "Allowed caps on sinkpad = %" GST_PTR_FORMAT, sink_caps);
  GST_INFO_OBJECT(self, "Allowed caps on srcpad = %" GST_PTR_FORMAT, src_caps);

  auto dispatcher_type = self->priv->MLA_dispatcher_type;
  self->priv->dispatcher =
      simaaidispatcher::DispatcherFactory::getDispatcher(dispatcher_type);

  //get node name
  self->priv->node_name = std::string(gst_element_get_name(trans));

  //parse config file
  nlohmann::json json;
  if (!parse_json_from_file(self, self->priv->config_path, json))
    return FALSE;

  self->priv->config = json["simaai__params"];

  self->priv->model_path = self->priv->config["model_path"];
  self->priv->model_handle = self->priv->dispatcher->load(
                               self->priv->model_path.c_str());
  self->priv->batch_size = self->priv->config["batch_size"];
  if (self->priv->batch_size != 1)
    self->priv->batch_model = self->priv->config["batch_sz_model"];

  if (!parse_output_segments(self)) {
    GST_ERROR_OBJECT(self, "Failed to get output segment information from config!");
    return FALSE;
  }

  //allocate output memory
  if (!gst_simaai_process_mla_allocate_memory(self)) {
    GST_ERROR_OBJECT(self, "Unable to allocate memory");
    return FALSE;
  }

  return TRUE;
}

/**
 * @brief Finalize/Cleanup process mla callback
 */
static void
gst_simaai_process_mla_finalize (GObject * object)
{
  GstSimaaiProcessMLA * process_mla = GST_SIMAAI_PROCESS_MLA (object);

  if (process_mla->priv->dispatcher) {
    if (process_mla->priv->model_handle)
      process_mla->priv->dispatcher->release(process_mla->priv->model_handle);
    delete process_mla->priv->dispatcher;
  }

  gst_simaai_caps_free(process_mla->priv->simaai_caps);

  delete process_mla->priv;

  G_OBJECT_CLASS (parent_class)->finalize (object);
}

static gboolean
gst_simaai_process_mla_extract_meta_info (GstSimaaiProcessMLA * self,
                                          GstBuffer * inbuf)
{
  GstCustomMeta * meta;
  GstStructure * s;
  gchar * buf_name;
  gint64 buf_id = 0, frame_id = 0, in_buf_off = 0, pcie_buf_id = 0;
  guint64 timestamp = 0;

  meta = gst_buffer_get_custom_meta(inbuf, SIMAAI_META_STR);
  if (meta != NULL) {
    s = gst_custom_meta_get_structure(meta);
    if (s == NULL) {
      gst_buffer_unref(inbuf);
      return FALSE;
    } else {
      if ((gst_structure_get_int64(s, "pcie-buffer-id", &pcie_buf_id) == TRUE)){
        self->priv->in_pcie_buf_id = pcie_buf_id;
        self->priv->is_pcie = TRUE;
        GST_INFO_OBJECT(self, "Is PCIe. pcie-buffer-id = %ld", pcie_buf_id);
      }
      if ((gst_structure_get_int64(s, "buffer-id", &buf_id) == TRUE) &&
          (gst_structure_get_int64(s, "frame-id", &frame_id) == TRUE) &&
          (gst_structure_get_int64(s, "buffer-offset", &in_buf_off) == TRUE) &&
          (gst_structure_get_uint64(s, "timestamp", &timestamp) == TRUE)) {
        self->priv->stream_id = gst_structure_get_string(s, "stream-id");
        self->priv->timestamp = timestamp;
        buf_name = (gchar *)gst_structure_get_string(s, "buffer-name");
        self->priv->frame_id = frame_id;
        self->priv->in_buf_id = buf_id;
      } else {
        return FALSE;
      }
    }
  }

  return TRUE;
}

/**
 * @brief Virtual call to do transformation of input buffer to an output buffer, 
 * this is not inplace
 *
 * @param trans the gobject for the base class GstBaseTransform
 * @param inbuf input buffer to work on
 * @param outbuf output buffer to push
 * @return GST_FLOW_OK on success or GST_FLOW_ERROR on failure
 */
static GstFlowReturn
gst_simaai_process_mla_transform (GstBaseTransform *trans, GstBuffer *inbuf,
                                     GstBuffer *outbuf)
{
  GstSimaaiProcessMLA *self = GST_SIMAAI_PROCESS_MLA (trans);

  if (gst_simaai_process_mla_extract_meta_info(self, inbuf) != TRUE) {
    GST_ERROR_OBJECT(self, "Failed to extract meta-info from input buffer");
    return GST_FLOW_ERROR;
  }

  GST_DEBUG_OBJECT(self, "MLA frame_cnt[%ld]", self->priv->frame_id);

  if (!self->silent)
    self->priv->t0 = std::chrono::steady_clock::now();
  
  if (run_process_mla (self, inbuf, outbuf) != TRUE) {
    GST_ERROR_OBJECT(self, "Failed to run MLA for frame %ld", 
                            self->priv->frame_id);
    return GST_FLOW_ERROR;
  }

  if (!self->silent) {
    self->priv->t1 = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
                                                 self->priv->t1-self->priv->t0);
    GST_DEBUG_OBJECT(self, "MLA frame_cnt[%ld], run time in ms: %lf",
                      self->priv->frame_id,
                      elapsed.count()/1000.00);
  }

  // TO DO: KPI needs to be updated here

  // Update metadata
  GstCustomMeta * meta = gst_buffer_add_custom_meta(outbuf, SIMAAI_META_STR);
  if (meta == NULL) {
    GST_ERROR_OBJECT(self, "Unable to add metadata to the buffer");
    return GST_FLOW_ERROR;
  }
  GstStructure *s = gst_custom_meta_get_structure (meta);
  if (s != NULL) {
    gst_structure_set (s,
                       "buffer-id", G_TYPE_INT64, self->priv->out_buffer_id,
                       "buffer-name", G_TYPE_STRING, self->priv->node_name.c_str(),
                       "buffer-offset", G_TYPE_INT64, (gint64)0,
                       "frame-id", G_TYPE_INT64, self->priv->frame_id, 
                       "stream-id", G_TYPE_STRING, self->priv->stream_id.c_str(),
                       "timestamp", G_TYPE_UINT64, self->priv->timestamp,
                       NULL);
    if (self->priv->is_pcie) {
      GST_INFO_OBJECT (self, "Adding SiMa sPCIe metadata to buffer");
      gst_structure_set(s, "pcie-buffer-id", G_TYPE_INT64, 
                        self->priv->in_pcie_buf_id, NULL);
    }
  }
  return GST_FLOW_OK;
}

/**
 * @brief Called when class init is scheduled used to initialize the output
 *        buffer to be used by transform
 *
 * @param trans is a gobject of type GstBaseTransform
 * @param input input buffer
 * @param outbuf output allocated buffer
 */
static GstFlowReturn
gst_simaai_process_mla_prepare_output_buffer (GstBaseTransform *trans, 
                                              GstBuffer *input,
                                              GstBuffer **outbuf)
{
  GstSimaaiProcessMLA *self = GST_SIMAAI_PROCESS_MLA(trans);
  GstFlowReturn ret = 
      gst_buffer_pool_acquire_buffer(self->priv->pool, outbuf, NULL);

  if (G_LIKELY (ret == GST_FLOW_OK))
    GST_DEBUG_OBJECT (self, "Acquired a buffer from pool %p", *outbuf);
  else
    GST_WARNING_OBJECT (self, "Failed to allocate buffer");

  return ret;
}

static GstCaps *
gst_simaai_process_mla_transform_caps(GstBaseTransform *trans,
  GstPadDirection direction, GstCaps *caps, GstCaps *filter)
{
  GstSimaaiProcessMLA *processmla = GST_SIMAAI_PROCESS_MLA(trans);
  GstCaps* result;

  if (direction == GST_PAD_SINK)
    result = processmla->priv->simaai_caps->src_caps;
  else
    result = processmla->priv->simaai_caps->sink_caps;

  if (filter)
      return gst_caps_intersect(result, filter);
  else
      return gst_caps_ref(result);
}

static GstCaps *
gst_simaai_process_mla_fixate_caps(GstBaseTransform *trans,
  GstPadDirection direction, GstCaps *caps, GstCaps *othercaps)
{
  GstSimaaiProcessMLA *processmla = GST_SIMAAI_PROCESS_MLA(trans);

  return gst_simaai_caps_fixate_src_caps(GST_ELEMENT(processmla),
    processmla->priv->simaai_caps, caps);
}

static gboolean
gst_simaai_process_mla_accept_caps(GstBaseTransform *trans,
  GstPadDirection direction, GstCaps *caps)
{
  return TRUE;
}

static gboolean
gst_simaai_process_mla_query(GstBaseTransform *trans, GstPadDirection direction,
  GstQuery *query)
{
  GstSimaaiProcessMLA *processmla = GST_SIMAAI_PROCESS_MLA(trans);

  switch (GST_QUERY_TYPE(query)) {
    case GST_QUERY_CAPS:
      return gst_simaai_caps_query(GST_ELEMENT(processmla),
        processmla->priv->simaai_caps, query, direction);
  }

  return GST_BASE_TRANSFORM_CLASS(parent_class)->query(trans, direction, query);
}

static gboolean
gst_simaai_process_mla_sink_eventfunc(GstBaseTransform *trans, GstEvent *event)
{
  GstSimaaiProcessMLA *processmla = GST_SIMAAI_PROCESS_MLA(trans);

  switch (GST_EVENT_TYPE(event)) {
    case GST_EVENT_CAPS:
      if (!gst_simaai_caps_negotiate(GST_ELEMENT(processmla),
        processmla->priv->simaai_caps)) {
        GST_ERROR_OBJECT(processmla, "<%s>: Error negotiating caps", G_STRFUNC);
        return FALSE;
      }

      return gst_simaai_caps_process_sink_caps(GST_ELEMENT(processmla),
        processmla->priv->simaai_caps, event);
  }

  return GST_BASE_TRANSFORM_CLASS(parent_class)->sink_event(trans, event);
}

static GstStateChangeReturn
gst_simaai_process_mla_change_state(GstElement *element,
  GstStateChange transition)
{
  GstSimaaiProcessMLA *processmla = GST_SIMAAI_PROCESS_MLA(element);
  GstStateChangeReturn retval;

  retval = GST_ELEMENT_CLASS(parent_class)->change_state(element, transition);
  if (retval == GST_STATE_CHANGE_FAILURE) {
    return retval;
  }

  switch(transition) {
    case GST_STATE_CHANGE_NULL_TO_READY:
      if (!gst_simaai_caps_parse_config(GST_ELEMENT(processmla),
        processmla->priv->simaai_caps, processmla->priv->config_path.c_str())) {
        GST_ERROR_OBJECT(processmla, "<%s>: Error parsing config", G_STRFUNC);
        return GST_STATE_CHANGE_FAILURE;
      }
      break;
    default:
      break;
  }

  return retval;
}

static gboolean gst_simaai_process_mla_propose_allocation (
  GstBaseTransform * trans, GstQuery * decide_query, GstQuery * query)
{
  GstSimaaiProcessMLA *self = GST_SIMAAI_PROCESS_MLA (trans);
  GST_DEBUG_OBJECT(self, "propose_allocation called");

  GstStructure *allocation_meta = 
      gst_simaai_allocation_query_create_meta(GST_SIMAAI_MEMORY_TARGET_EV74, 
                                              GST_SIMAAI_MEMORY_FLAG_CACHED);

  gst_simaai_allocation_query_add_meta(query, allocation_meta); 

  return TRUE;
}

static gboolean gst_simaai_process_mla_decide_allocation (
  GstBaseTransform * trans, GstQuery * query)
{
  GstSimaaiProcessMLA *self = GST_SIMAAI_PROCESS_MLA (trans);
  GST_DEBUG_OBJECT(self, "decide allocation called");

  GstSimaaiMemoryFlags mem_type;
  GstSimaaiMemoryFlags mem_flag;

  if (!gst_simaai_allocation_query_parse(query, &mem_type, &mem_flag)) {
    GST_WARNING_OBJECT(self, "Can't find allocation meta!");
  } else {
    if (self->priv->mem_type != mem_type || self->priv->mem_flag != mem_flag) {
      self->priv->mem_type = mem_type;
      self->priv->mem_flag = mem_flag;

      gst_simaai_process_mla_allocate_memory(self);
    }
  };

  GST_DEBUG_OBJECT(self, "Memory flags to allocate: [ %s ] [ %s ]",
    gst_simaai_allocation_query_sima_mem_type_to_str(self->priv->mem_type),
    gst_simaai_allocation_query_sima_mem_flag_to_str(self->priv->mem_flag));

  

  return TRUE;
}

/**
 * @brief Callback to process mla init
 */
static void
gst_simaai_process_mla_class_init (GstSimaaiProcessMLAClass * klass)
{
  GObjectClass *gobject_class;
  GstElementClass *gstelement_class;
  GstBaseTransformClass *gstbasetransform_class;
  GstPadTemplate *sink_pad_template, *src_pad_template;

  gobject_class = G_OBJECT_CLASS(klass);
  gstelement_class = GST_ELEMENT_CLASS(klass);
  gstbasetransform_class = GST_BASE_TRANSFORM_CLASS(klass);

  GST_DEBUG_CATEGORY_INIT(gst_simaai_process_mla_debug, "simaaiprocessmla", 0,
                          "SiMa.ai Process MLA"
                          );

  gobject_class->finalize = gst_simaai_process_mla_finalize;
  gobject_class->set_property = gst_simaai_process_mla_set_property;
  gobject_class->get_property = gst_simaai_process_mla_get_property;

  gst_element_class_set_static_metadata(gstelement_class,
                                        "SiMa.ai Process MLA element",
                                        "Transform",
                                        "Executing MLA",
                                        "SiMa.ai"
                                        );
  /**
   * GstSimaaiProcessMLA::PROPERTIES:
   *
   * The properties for simaaiprocessmla plugin.
   */
  g_object_class_install_property(gobject_class, PROP_CONFIG_PATH,
                                  g_param_spec_string("config",
                                                      "Config Path",
                                                      "Path to configuration file",
                                                      DEFAULT_LM_FILE,
                                                      (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
  /* This property is used to enable/disable kpi message transmission to the GstBus */
  g_object_class_install_property (gobject_class, PROP_TRANSMIT,
                                   g_param_spec_boolean ("transmit",
                                                         "Transmit",
                                                         "Transmit KPI Message",
                                                         DEFAULT_TRANSMIT,
                                                         (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
  g_object_class_install_property (gobject_class, PROP_MULTIPIPELINE,
                                   g_param_spec_boolean ("multi-pipeline",
                                                         "Multiple pipelines",
                                                         "Use MLA dispatcher that allows running several separate "
                                                         "pipelines (as separate processes) on the device",
                                                         FALSE,
                                                         (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
  g_object_class_install_property(gobject_class, PROP_NO_OF_BUFS,
                                  g_param_spec_ulong("num-buffers", 
                                                     "Number Of Buffers",
                                                     "Number of buffers to be allocated of size of buffer",
                                                     1, G_MAXUINT, 
                                                     DEFAULT_NUM_BUFFERS,
                                                     (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
  g_object_class_install_property(gobject_class, PROP_DUMP_DATA,
                                  g_param_spec_boolean ("dump-data",
                                                        "DumpData",
                                                        "Save binary outputs to /tmp/[node-name]",
                                                        FALSE,
                                                        (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
  g_object_class_install_property(gobject_class, PROP_SILENT,
                                  g_param_spec_boolean ("silent",
                                                        "Silent",
                                                        "Produce verbose output",
                                                        FALSE,
                                                        (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  sink_pad_template = gst_pad_template_new(PAD_TEMPLATE_NAME_SINK, GST_PAD_SINK,
    GST_PAD_ALWAYS, gst_caps_new_any());
  if (!sink_pad_template) {
    GST_ERROR_OBJECT(gstbasetransform_class, "<%s>: Error creating a sink pad "
        "template", G_STRFUNC);
    return;
  }
  gst_element_class_add_pad_template(gstelement_class, sink_pad_template);

  src_pad_template = gst_pad_template_new(PAD_TEMPLATE_NAME_SRC, GST_PAD_SRC,
    GST_PAD_ALWAYS, gst_caps_new_any());
  if (!src_pad_template) {
    GST_ERROR_OBJECT(gstbasetransform_class, "<%s>: Error creating a source pad"
        " template", G_STRFUNC);
    return;
  }
  gst_element_class_add_pad_template(gstelement_class, src_pad_template);

  gstbasetransform_class->start = 
    GST_DEBUG_FUNCPTR(gst_simaai_process_mla_start);
  gstbasetransform_class->stop = 
    GST_DEBUG_FUNCPTR(gst_simaai_process_mla_stop);
  gstbasetransform_class->prepare_output_buffer =
    GST_DEBUG_FUNCPTR(gst_simaai_process_mla_prepare_output_buffer);
  gstbasetransform_class->transform = 
    GST_DEBUG_FUNCPTR(gst_simaai_process_mla_transform);
  gstbasetransform_class->transform_caps =
    GST_DEBUG_FUNCPTR(gst_simaai_process_mla_transform_caps);
  gstbasetransform_class->fixate_caps =
    GST_DEBUG_FUNCPTR(gst_simaai_process_mla_fixate_caps);
  gstbasetransform_class->accept_caps =
    GST_DEBUG_FUNCPTR(gst_simaai_process_mla_accept_caps);
  gstbasetransform_class->query =
    GST_DEBUG_FUNCPTR(gst_simaai_process_mla_query);
  gstbasetransform_class->sink_event =
    GST_DEBUG_FUNCPTR(gst_simaai_process_mla_sink_eventfunc);
  gstbasetransform_class->propose_allocation = 
    GST_DEBUG_FUNCPTR(gst_simaai_process_mla_propose_allocation);
  gstbasetransform_class->decide_allocation = 
    GST_DEBUG_FUNCPTR(gst_simaai_process_mla_decide_allocation);

  gstelement_class->change_state =
    GST_DEBUG_FUNCPTR(gst_simaai_process_mla_change_state);

  static const gchar *tags[] = { NULL };
  gst_meta_register_custom ("GstSimaMeta", tags, NULL, NULL, NULL);
}

/**
 * @brief Helper to dump intermediate data
 * @todo move to utils
 */
static int32_t
dump_output_buffer(GstSimaaiProcessMLA * self, void * vaddr)
{
  FILE *ofp;
  size_t wosz;
  void *ovaddr;
  int32_t res = 0, i;
  char *dumpaddr;
  char full_opath[256];

  snprintf(full_opath, sizeof(full_opath) - 1, "/tmp/%s-%ld.out",
           self->priv->node_name.c_str(),
           self->priv->frame_id);

  ofp = fopen(full_opath, "w");
  if(ofp == NULL) {
    res = -1;
    goto err_ofp;
  }

  wosz = fwrite(vaddr , 1, self->priv->out_size, ofp);
  if(self->priv->out_size != wosz) {
    res = -3;
    goto err_owrite;
  }

err_owrite:
err_omap:
  fclose(ofp);
err_ofp:
  return res;
}

std::chrono::system_clock::time_point get_boot_time()
{
  std::ifstream uptime_file("/proc/uptime");
  double uptime_seconds = 0.0;
  uptime_file >> uptime_seconds;

  auto now = std::chrono::system_clock::now();
  return now - std::chrono::duration_cast<std::chrono::system_clock::duration>(std::chrono::duration<double>(uptime_seconds));
}

std::chrono::system_clock::time_point steady_to_system(
    std::chrono::steady_clock::time_point steady_tp,
    std::chrono::system_clock::time_point boot_time = get_boot_time()
)
{
  auto now_steady = std::chrono::steady_clock::now();
  auto delta = steady_tp - now_steady;
  auto now_system = std::chrono::system_clock::now();

  return now_system + delta;
}

/**
 * @brief The entry point function to running process_mla
 */
static gboolean
run_process_mla(GstSimaaiProcessMLA *self, GstBuffer *inbuf, GstBuffer * outbuf)
{
  simaaidispatcher::JobMLA job;
  GstMapInfo in_meminfo, out_meminfo;
  int retval;

  if (self->transmit) {
    tracepoint_pipeline_mla_start(self->priv->frame_id, (char *)self->priv->node_name.c_str(), (char *)self->priv->stream_id.c_str());
  }
  self->priv->t0 = std::chrono::steady_clock::now();

  job.path = self->priv->model_path;
  job.handle = self->priv->model_handle;
  job.batchSize = self->priv->batch_size;
  if (job.batchSize != 1)
    job.batchModel = self->priv->batch_model;
  job.timeout = std::chrono::seconds(self->priv->timeout);

  std::string combined_id = self->priv->node_name + self->priv->stream_id;
  uint64_t request_id = ((uint64_t)str_to_uint32_hash(combined_id.c_str()) << 32) | self->priv->frame_id;
  job.requestID = request_id;

  if (!gst_buffer_map(inbuf, &in_meminfo, GST_MAP_READ)) {
    GST_ERROR_OBJECT(self, "Input buffer map failed");
    return FALSE;
  }

  if (!gst_buffer_map(outbuf, &out_meminfo, GST_MAP_WRITE)) {
    GST_ERROR_OBJECT(self, "Output buffer map failed");
    return FALSE;
  }

  nlohmann::json & cfg = self->priv->config;
  if (self->priv->in_segment_name_state == SEG_NAME_STATE::UNDEFINED) {
    self->priv->in_segment_name_state = cfg.contains("input_segment_name") ? 
        SEG_NAME_STATE::HAS_NAME : SEG_NAME_STATE::NO_NAME;
    
    if (self->priv->in_segment_name_state == SEG_NAME_STATE::HAS_NAME) {
      self->priv->input_seg_name = cfg["input_segment_name"];
    }
  }
  
  // NULL == parent segment == full buffer
  const gchar *seg_name =
      (self->priv->in_segment_name_state == SEG_NAME_STATE::HAS_NAME) ?
          self->priv->input_seg_name.c_str() : NULL;

  if ((job.buffers["ifm0"] =
      (simaai_memory_t *)gst_simaai_memory_get_segment(in_meminfo.memory,
      seg_name)) == nullptr) {
        GST_ERROR_OBJECT(self, "Attach to the input memory chunk failed. Either "
                               "segment with requested name is not in input memory, "
                               "or memory was allocated without using segment allocator");
    return FALSE;
  }

  if ((job.buffers["ofm0"] =
      (simaai_memory_t *)gst_simaai_memory_get_segment(out_meminfo.memory,
      NULL)) == nullptr) {
    GST_ERROR_OBJECT(self, "Attach to the output memory chunk failed");
    return FALSE;
  }
  retval = self->priv->dispatcher->run(job, self->priv->tp);
  if (retval != 0) {
    GST_ERROR_OBJECT(self, "Dispatcher returned error: %d", retval);
    return FALSE;
  }

  if (self->transmit) {
    auto boot_time = get_boot_time().time_since_epoch();
    auto start_time = boot_time + self->priv->tp.first;
    uint64_t kernel_start = std::chrono::duration_cast<std::chrono::microseconds>(start_time.time_since_epoch()).count();
    tracepoint_mla_kernel_start(kernel_start, request_id);

    auto end_time = boot_time + self->priv->tp.second;
    uint64_t kernel_end = std::chrono::duration_cast<std::chrono::microseconds>(end_time.time_since_epoch()).count();
    tracepoint_mla_kernel_end(kernel_end, request_id);

    tracepoint_pipeline_mla_end(self->priv->frame_id, (char *)self->priv->node_name.c_str(), (char *)self->priv->stream_id.c_str());
  }

  self->priv->t1 = std::chrono::steady_clock::now();
  auto elapsed = 
      std::chrono::duration_cast<std::chrono::microseconds>(self->priv->t1 - 
                                                            self->priv->t0);
  auto duration = elapsed.count() / 1000.0 ;
  auto kernel_rt = 
      std::chrono::duration_cast<std::chrono::microseconds>(self->priv->tp.second - 
                                                            self->priv->tp.first);
  auto kernel_duration = kernel_rt.count() / 1000.0 ;
  GST_DEBUG_OBJECT(self, "MLA model  %s run time is :  %f ms", 
                          job.path.c_str(), kernel_duration);

  if (self->priv->dump_data) {
    retval = dump_output_buffer(self, out_meminfo.data);
    if (retval < 0) {
      GST_INFO_OBJECT(self, "Error(%d) while dumping frame with ID: %ld",
        retval, self->priv->frame_id);
      gst_buffer_unmap(outbuf, &out_meminfo);
      gst_buffer_unmap(inbuf, &in_meminfo);
      return FALSE;
    }
  }
  gst_buffer_unmap(outbuf, &out_meminfo);
  gst_buffer_unmap(inbuf, &in_meminfo);

  return TRUE;
}

/**
 * @brief klass init for process_mla
 */
static void
gst_simaai_process_mla_init (GstSimaaiProcessMLA * self)
{
  gst_simaai_segment_memory_init_once();
  self->silent = DEFAULT_SILENT;
  self->transmit = DEFAULT_TRANSMIT;
  self->priv = new GstSimaaiProcessMLA_Private;
  self->priv->pool = NULL;
  self->priv->MLA_dispatcher_type = simaaidispatcher::DispatcherFactory::MLA;
  self->priv->in_segment_name_state = SEG_NAME_STATE::UNDEFINED;
  self->priv->in_pcie_buf_id = 0;
  self->priv->is_pcie = FALSE;
  self->priv->timeout = 60;
  self->priv->frame_id = -1;
  self->priv->dump_data = false;
  self->priv->no_of_obufs = MIN_POOL_SIZE;
  self->priv->timestamp = 0;
  self->priv->dispatcher = nullptr;
  self->priv->model_handle = nullptr;

  self->priv->mem_type = GST_SIMAAI_MEMORY_TARGET_EV74;
  self->priv->mem_flag = GST_SIMAAI_MEMORY_FLAG_CACHED;

  self->priv->out_size = 0;

  self->priv->simaai_caps = gst_simaai_caps_init();
}

static gboolean
plugin_init (GstPlugin * plugin)
{
  if (!gst_element_register(plugin, "simaaiprocessmla", GST_RANK_NONE,
                            GST_TYPE_SIMAAI_PROCESS_MLA)) {
    GST_ERROR("Unable to register process mla plugin");
    return FALSE;
  }

  return TRUE;
}

GST_PLUGIN_DEFINE(
    GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    PLUGIN_NAME_LOWER,
    "GStreamer SiMa.ai MLA Plugin",
    plugin_init,
    VERSION,
    GST_LICENSE,
    GST_PACKAGE_NAME,
    GST_PACKAGE_ORIGIN
);
