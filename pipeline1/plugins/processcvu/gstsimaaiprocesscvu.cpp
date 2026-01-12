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
 * @file gstsimaaiprocesscvu.cc
 * @brief Gstreamer plugin to run a graph on CVU(EVXX)
 * @author SiMa.Ai\TM
 * @bug Currently no known bugs
 */

/**
 * SECTION: element-simaaiprocesscvu
 *
 * Plugin to run a graph node on the cvu
 *
 * <refsect2>
 * <title> Example Launch line </title>
 * |[
 * simasrc location="input" node-name="allegrodec" blocksize="size_of_input"
 * ! simaaiprocesscvu name=cvu config="evxx_reid_preproc_p.json" source-node-name="input_image_name" buffers-list="a65-topk,test"
 * ! fakesink
 * ]|
 * </refsect2>
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include <inttypes.h>
#include <stdlib.h>
#include <stdint.h>

#include <algorithm>
#include <cctype>
#include <cstring>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>
#include <fstream>

#include <gst/gst.h>
#include <gst/base/gstaggregator.h>

#include <gstsimaaiallocator.h>
#include <gstsimaaibufferpool.h>
#include <simaai/parser_types.h>
#include <simaai/parser.h>
#include <simaai/simaai_memory.h>
#include <simaai/simaailog.h>
#include <simaai/trace/pipeline_tp.h>
#include <simaai/nlohmann/json.hpp>
#include <gstsimaaicaps.h>

/* dispatcher related header files */
#include <dispatcherfactory.hh>
#include <configManager.h>

#include "gstsimaaiprocesscvu.h"
#include "nlohmann_helpers.h"
#include <simaai/trace/pipeline_new_tp.h>
#include <utils_string.h>

/**
 * @brief Flag to print minimized log.
 */
#define DEFAULT_SILENT TRUE
#define DEFAULT_TRANSMIT FALSE
#define PLUGIN_CPU_TYPE "EV74"

/**
 * @brief cvu properties
 */
enum {
  PROP_0,
  PROP_CONF_F,
  PROP_SILENT,
  PROP_TRANSMIT,
  PROP_NO_OF_BUFS,
  PROP_DUMP_DATA,
  PROP_UNKNONW,
};

enum {
  SIMA_CPU_A65,
  SIMA_CPU_EVXX,
  SIMA_CPU_MLA,
  SIMA_CPU_TVM,
  SIMA_CPU_NUM
};

/* All statics go in here */
static void gst_simaai_processcvu_set_property (GObject * obj, 
                                                guint prop_id,
                                                const GValue * value, 
                                                GParamSpec * pspec);
static void gst_simaai_processcvu_get_property (GObject * object,
                                                guint prop_id,
                                                GValue * value,
                                                GParamSpec * pspec);

static gboolean run_processcvu (GstSimaaiProcesscvu * self, 
                                simaaidispatcher::JobEVXX & job);
static GstStateChangeReturn gst_simaai_processcvu_change_state (GstElement * element,
                                                           GstStateChange transition);
static void
gst_simaai_processcvu_child_proxy_init (gpointer g_iface, gpointer iface_data);

static gboolean gst_simaai_processcvu_propose_allocation(
  GstAggregator * self, GstAggregatorPad * pad, GstQuery * decide_query, GstQuery * query);

static gboolean gst_simaai_processcvu_decide_allocation(GstAggregator * self, GstQuery * query);

GST_DEBUG_CATEGORY_STATIC(gst_simaai_processcvu_debug);
#define GST_CAT_DEFAULT gst_simaai_processcvu_debug

/**
 * @brief structure wich holds data about graph input/output memories
 */
struct GraphMemory {
  /// name of memory
  std::string memory_name;
  /// name of dispatcher input/output
  std::string dispatcher_name;
  /// size of memory
  size_t size;
};

/**
 * @brief Private member structure for GstSimaaiProcesscvu instances
 */
struct _GstSimaaiProcesscvuPrivate
{
  /// Buffer pool
  GstBufferPool *pool;
  /// Allocation params
  GstSimaaiMemoryFlags mem_type;
  GstSimaaiMemoryFlags mem_flag;

  /// Current output buffer
  GstBuffer *outbuf;
  /// Output simaai-memlib buffer id (phys_addr)
  gint64 out_buffer_id;
  /// size of buffer pool
  uint32_t num_of_out_buf;
  /// @brief map of buffers, that contains memories used by current graph.
  /// @param string name of buffer
  /// @param vector<GraphMemory> buffer memories
  std::map<std::string, std::vector<GraphMemory>> graph_buffers;

  /// CPU of current plugin
  int target_cpu;
  /// CPU of next plugin
  int next_cpu;
  /// Size of output memory chunk
  int output_size;

  gint64 run_count;
  /// Input frame id placeholder
  gint64 frame_id;
  /// Metadata fields that are not used internally, but are needed to be passed osn/
  gint64 in_pcie_buf_id;
  gboolean is_pcie;
  std::string stream_id;
  guint64 timestamp; 

  /// Aggregator input buffers
  GstBufferList *list;

  /// Flag to create binary output dumps
  bool dump_data;

  /// JSON configuration file
  std::string config_file_path;
  /// Plugin instance node name
  std::string node_name;

  /// Graph config manager
  std::unique_ptr <ConfigManager> config_manager;
  /// Map of graph mempories where key is name of memory, and value is a pair
  /// where first is size and second is type of buffer 
  std::map <std::string, std::pair<unsigned int, enum bufferType>> cm_memories;
  /// Dispatcher handle
  simaaidispatcher::DispatcherBase * dispatcher;

  /// A map of inputs <buffer name, buffer index in GstBufferList>
  std::map<std::string, guint> buf_name_idx_map;

  /// Start and end point of time measurement
  std::chrono::time_point<std::chrono::steady_clock> t0, t1;

  std::mutex event_mtx;
  /// Time point pair to store the kernel start and end time measured in dispatcher
  std::pair<TimePoint, TimePoint> tp;

  GstSimaaiCaps *simaai_caps;
};

#define gst_simaai_processcvu_parent_class parent_class
G_DEFINE_TYPE_WITH_CODE ( GstSimaaiProcesscvu, 
                          gst_simaai_processcvu, 
                          GST_TYPE_AGGREGATOR, 
                          G_IMPLEMENT_INTERFACE(GST_TYPE_CHILD_PROXY,
                                                gst_simaai_processcvu_child_proxy_init));

/**
 * @brief Helper API to dump output buffer from CVU
 */
int32_t cvu_dump_output_buffer (GstSimaaiProcesscvu * self, uint32_t id)
{
  FILE *ofp;
  char full_opath[256];
  
  snprintf( full_opath, 
            sizeof(full_opath) - 1,
            "/tmp/%s-%03d.out",
            self->priv->node_name.c_str(),
            id);

  ofp = fopen(full_opath, "w");
  if(ofp == NULL)
    return -1;

  GstMapInfo map;
  gst_buffer_map(self->priv->outbuf, &map, GST_MAP_READ);
  fwrite((char *)(map.data), 1, map.size, ofp);
  gst_buffer_unmap(self->priv->outbuf, &map);
  
  fclose(ofp);

  return 0;
}

/**
 * @brief Helper API to add input buffers from aggreagte to list after 
 *        parsing the Meta
 */
static gboolean gst_simaai_processcvu_add2list (GstSimaaiProcesscvu * self, 
                                                GValue * value)
{
  gint64 buf_id = 0;
  gint64 frame_id = 0;
  gint64 in_buf_offset = 0;
  gint64 in_pcie_buf_id = 0;

  GstAggregatorPad * pad = (GstAggregatorPad *) g_value_get_object (value);
  GstBuffer * buf = gst_aggregator_pad_peek_buffer (pad);

  if (buf) {
    GstCustomMeta * meta;
    GstStructure * s;
    std::string buf_name;
    gchar * stream_id;
    guint64 timestamp = 0;

    buf = gst_aggregator_pad_pop_buffer(pad);
    meta = gst_buffer_get_custom_meta(buf, SIMAAI_META_STR);
    if (meta != NULL) {
      s = gst_custom_meta_get_structure(meta);
      if (s == NULL) {
        gst_buffer_unref(buf);
        GST_OBJECT_UNLOCK (self);
        return GST_FLOW_ERROR;
      } else {
        if ((gst_structure_get_int64(s, "pcie-buffer-id", &in_pcie_buf_id) == TRUE))
        {
          self->priv->in_pcie_buf_id = in_pcie_buf_id;
          self->priv->is_pcie = TRUE;
          GST_DEBUG_OBJECT(self, 
                            "pcie-buffer-id = %ld", 
                            in_pcie_buf_id);
        }
        if ((gst_structure_get_int64(s, "buffer-id", &buf_id) == TRUE) &&
            (gst_structure_get_int64(s, "frame-id", &frame_id) == TRUE) &&
            (gst_structure_get_int64(s, "buffer-offset", &in_buf_offset) == TRUE) &&
            (gst_structure_get_uint64(s, "timestamp", &timestamp) == TRUE))
        {
          self->priv->stream_id = gst_structure_get_string(s, "stream-id");
          buf_name = (char *)gst_structure_get_string(s, "buffer-name");
          self->priv->frame_id = frame_id;
          self->priv->timestamp = timestamp;
          gst_buffer_list_add(self->priv->list, buf);

          self->priv->buf_name_idx_map[buf_name] = gst_buffer_list_length(self->priv->list)-1;

          GST_DEBUG_OBJECT(self, 
                            "Copied metadata, [%s]:[%ld]:[%ld],"
                            " buffer list length: %d,"
                            " stream-id: %s, timestamp: %ld",
                            buf_name.c_str(), 
                            frame_id, 
                            in_buf_offset, 
                            gst_buffer_list_length(self->priv->list),
                            self->priv->stream_id.c_str(),
                            self->priv->timestamp);
        }
      }
    } else {
      GST_ERROR_OBJECT (self, "Please check readme to use metadata information,"
                        " meta not found");
      return FALSE;
    }
  } else {
    GST_ERROR_OBJECT (self, "[CRITICAL] input buffer is NULL");
    return FALSE;
  }

  return TRUE;
}

/**
 * @brief Helper API to update output metadata information
 */
static gboolean gst_simaai_processcvu_buffer_update_metainfo(GstSimaaiProcesscvu * self, 
                                                             GstBuffer * buffer)
{
  GstCustomMeta * meta = gst_buffer_add_custom_meta(buffer, SIMAAI_META_STR);
  if (meta == NULL) {
    GST_ERROR_OBJECT (self, "Unable to add metadata info to the buffer");
    return FALSE;
  }

  GstStructure *s = gst_custom_meta_get_structure (meta); 

  if (s != NULL) {
    GST_DEBUG_OBJECT(self, 
                      "[%s]Adding SiMa metadata to buffer: "
                      "buffer-id=%ld buffer-name=%s "
                      "buffer-offset-%d frame-id=%ld",
                      self->priv->node_name.c_str(), 
                      self->priv->out_buffer_id,
                      self->priv->node_name.c_str(), 
                      0, 
                      self->priv->frame_id);
    gst_structure_set (s,
                       "buffer-id", G_TYPE_INT64, self->priv->out_buffer_id,
                       "buffer-name", G_TYPE_STRING, self->priv->node_name.c_str(),
                       "buffer-offset", G_TYPE_INT64, (gint64)0,
                       "frame-id", G_TYPE_INT64, self->priv->frame_id,
                       "stream-id", G_TYPE_STRING, self->priv->stream_id.c_str(),
                       "timestamp", G_TYPE_UINT64, self->priv->timestamp,
                       NULL);
    
    // Update PCIe related metadata if exists
    if (self->priv->is_pcie)
    {
      GST_DEBUG_OBJECT (self, "[%s]Adding SiMa sPCIe metadata to buffer", 
                        self->priv->node_name.c_str());
      gst_structure_set(s, "pcie-buffer-id", G_TYPE_INT64, 
                        self->priv->in_pcie_buf_id, NULL);
    }
  }

  return TRUE;
}

/// @brief Helper function to delete spaces on the start and in the end of string
/// @param str
void strip(std::string& str)
{
  if (str.empty())
    return;

  auto start_it = str.begin();
  auto end_it = str.rbegin();
  //delete spaces in the begining
  while (std::isspace(*start_it))
    if (++start_it == str.end())
      break;

  //delete spaces in the end
  while (std::isspace(*end_it))
    if (++end_it == str.rend())
      break;

  int start_pos = start_it - str.begin();
  int end_pos = end_it.base() - str.begin();
  str = start_pos <= end_pos ? std::string(start_it, end_it.base()) : "";
}

/// @brief Helper function to split `s` to substrings and remove redundant spaces
/// @param s string to split
/// @param delimiter character to split on
/// @return vector of splited strings without spaces
std::vector<std::string> split_string(const std::string &s, char delimiter)
{
  std::vector<std::string> res;
  std::string token;
  std::istringstream tokenStream(s);
  while(std::getline(tokenStream, token, delimiter)){
    strip(token);
    if (!token.empty())
      res.push_back(token);
  }
  return res;
}

/**
 * @brief Helper API to create EVXX job for particular plugin
 */
bool gst_simaai_processcvu_configure_job (GstSimaaiProcesscvu * self,
                                          simaaidispatcher::JobEVXX & job)
{
  // configure a job
  job.graphID = self->priv->config_manager->getPipelineConfig().graphId;
  job.cm = self->priv->config_manager.get();
  job.timeout = std::chrono::seconds(60);

  std::string combined_id = self->priv->node_name + self->priv->stream_id;
  job.requestID = ((uint64_t)str_to_uint32_hash(combined_id.c_str()) << 32) | self->priv->frame_id;

  // add memories
  GstBuffer * buffer;
  GstMemory * buffer_mem;
  // add input memories
  for (auto& [ json_bufname, buf_memories ] : self->priv->graph_buffers) {
    std::string buf_name;

    // if buffer is output buffer - continue. Only inputs updated here
    if (json_bufname == self->priv->node_name)
      continue;

    // parse the valid buffer names provided for this buffer
    std::vector<std::string> valid_buf_names = split_string(json_bufname, ',');
  
    // find buffer which name correspond to one of valid names provided by json
    for (auto & name : valid_buf_names) {
      auto & buf_name_idx_map = self->priv->buf_name_idx_map;
      if (buf_name_idx_map.find(name) != buf_name_idx_map.end()) {
        buf_name = name;
        buffer = gst_buffer_list_get (self->priv->list,
                                      self->priv->buf_name_idx_map[buf_name]);
      }
    }

    if (buf_name.empty()) {
      GST_ERROR_OBJECT(self, "No input buffer, that correspond valid names: %s",
                              json_bufname.c_str());
      return false;
    }

    buffer_mem = gst_buffer_peek_memory(buffer, 0);
    if (!buffer_mem) {
      GST_ERROR_OBJECT (self, "Can not peak a memory from buffer %s", 
                        buf_name.c_str());
      return false;
    }

    for (auto & memory : buf_memories) {
      simaai_memory_t * seg_ptr = nullptr;
      seg_ptr = (simaai_memory_t *) gst_simaai_memory_get_segment(buffer_mem,
                                          (gchar*)(memory.memory_name.c_str()));
      if (seg_ptr) {
        job.buffers[memory.dispatcher_name] = seg_ptr;
      } else {
        GST_ERROR_OBJECT (self, "Failed to get memory with name %s from buffer %s. "
                                "Either segment with requested name is not in "
                                "input memory, or memory was allocated without "
                                "using segment allocator",
                                memory.memory_name.c_str(), buf_name.c_str());
        return false;
      }
      GST_DEBUG_OBJECT (self, "Added memory with name %s as dispatcher input %s "
                              "from buffer %s (addr: %p)",
                              memory.memory_name.c_str(),
                              memory.dispatcher_name.c_str(), 
                              buf_name.c_str(),
                              seg_ptr);
    }
  }

  // add output buffer to job
  buffer_mem = gst_buffer_peek_memory(self->priv->outbuf, 0);
  auto & buf_memories = self->priv->graph_buffers[self->priv->node_name];
  if (!buffer_mem) {
    GST_ERROR_OBJECT(self, "Can not peak a memory from output buffer");
    return false;
  }

  for (auto & memory : buf_memories) {        
    simaai_memory_t * seg_ptr = nullptr;
      seg_ptr = (simaai_memory_t *) gst_simaai_memory_get_segment(buffer_mem,
                                          (gchar*)(memory.memory_name.c_str()));
      if (seg_ptr) {
        GST_DEBUG_OBJECT (self, "Segment %s addr: %p", 
                          memory.dispatcher_name.c_str(), seg_ptr);
        job.buffers[memory.dispatcher_name] = seg_ptr;
      }
      GST_DEBUG_OBJECT (self, "Added memory with name %s as dispatcher output %s "
                              "(addr: %p)",
                              memory.memory_name.c_str(),
                              memory.dispatcher_name.c_str(), 
                              seg_ptr);
  }

  return true;
}

/**
 * @brief Aggregate callback registered to be called when buffers are ready to 
 *        be used by the plugin from different sources
 * @return GST_FLOW_OK on success, or GST_FLOW_ERROR on failure
 */
static GstFlowReturn gst_simaai_processcvu_aggregate (GstAggregator * aggregator, 
                                                      gboolean timeout)
{
  GstIterator *iter;

  gboolean done_iterating = FALSE;
  GstMapInfo map;
  gint64 buf_id = 0;
  gint64 frame_id = 0, in_buf_offset = 0;

  GstSimaaiProcesscvu *self = GST_SIMAAI_PROCESSCVU (aggregator);
  
  auto clean_buffer_list = [](GstBufferList * list) {
    guint no_of_inbufs = gst_buffer_list_length(list);
    for (guint i = 0; i < no_of_inbufs ; ++i)
      gst_buffer_unref(gst_buffer_list_get(list, i));
    gst_buffer_list_remove(list, 0 , no_of_inbufs);
  };

  self->priv->buf_name_idx_map.clear();

  iter = gst_element_iterate_sink_pads (GST_ELEMENT (self));
  while (!done_iterating) {
    GValue value = { 0, };
    GstAggregatorPad *pad;
    GstBuffer *buf;

    switch (gst_iterator_next (iter, &value)) {
      case GST_ITERATOR_OK:
        if (!gst_simaai_processcvu_add2list(self, &value)) {
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

  GstFlowReturn ret = gst_buffer_pool_acquire_buffer(self->priv->pool,
                                                     &self->priv->outbuf, NULL);

  if (G_LIKELY (ret == GST_FLOW_OK)) {
    GST_DEBUG_OBJECT (self, "Output buffer from pool: %p", self->priv->outbuf);
    /// get phys memory of buffer as buffer ID
    self->priv->out_buffer_id = gst_simaai_segment_memory_get_phys_addr(
      gst_buffer_peek_memory(self->priv->outbuf, 0));
  } else {
    GST_ERROR_OBJECT (self, "Failed to allocate buffer");
    clean_buffer_list(self->priv->list);
    return GST_FLOW_ERROR;
  }

  simaaidispatcher::JobEVXX job;
  if (!gst_simaai_processcvu_configure_job(self, job)) {
    GST_ERROR_OBJECT (self, "Failed to configure job");
    clean_buffer_list(self->priv->list);
    return GST_FLOW_ERROR;
  }

  /* Run processcvu here */
  if (run_processcvu(self, job) != TRUE) {
    GST_ERROR_OBJECT (self, "Unable to run processcvu, drop and continue");
    return GST_FLOW_ERROR;
  }

  if (!gst_simaai_processcvu_buffer_update_metainfo(self, self->priv->outbuf)) {
    GST_ERROR_OBJECT (self, "Unable to run processcvu, drop and continue");
    return GST_FLOW_ERROR;
  }

  /* Clear input buffer list */
  clean_buffer_list(self->priv->list);

  gst_aggregator_finish_buffer (aggregator, self->priv->outbuf);

  return GST_FLOW_OK;
}

/**
 * @brief Setter for processcvu properties.
 */
static void gst_simaai_processcvu_set_property (GObject * object,
                                                guint prop_id,
                                                const GValue * value,
                                                GParamSpec * pspec)
{
  GstSimaaiProcesscvu * self = GST_SIMAAI_PROCESSCVU (object);

  std::string bool_res = "";

  switch(prop_id) {
    case PROP_CONF_F:
      self->priv->config_file_path = g_value_get_string(value);
      GST_DEBUG_OBJECT(self, "ConfigFile argument was changed to %s", 
                        self->priv->config_file_path.c_str());
      break;
    case PROP_SILENT:
      self->silent = g_value_get_boolean (value);
      bool_res = (self->silent) ? "TRUE" : "FALSE";
      GST_DEBUG_OBJECT (self, "Silent argument was changed to %s", 
                        bool_res.c_str());
      break;
    case PROP_TRANSMIT:
      self->transmit = g_value_get_boolean (value);
      GST_DEBUG_OBJECT(self, "Set transmit = %d", self->transmit);
      break;
    case PROP_NO_OF_BUFS:
      self->priv->num_of_out_buf = g_value_get_ulong(value);
      GST_DEBUG_OBJECT(self, "NumberOfBuffers argument was changed to %d", 
                        self->priv->num_of_out_buf);
      break;
    case PROP_DUMP_DATA:
      self->priv->dump_data = g_value_get_boolean (value);
      bool_res = (self->priv->dump_data) ? "TRUE" : "FALSE";
      GST_DEBUG_OBJECT (self, "DumpData argument was changed to = %s", 
                        bool_res.c_str());
      break;
    default:
      GST_DEBUG_OBJECT(self, "Default case warning");
      G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
      break;
  }
}

/**
 * @brief Getter for processcvu properties.
 */
static void gst_simaai_processcvu_get_property (GObject * object,
                                           guint prop_id,
                                           GValue * value,
                                           GParamSpec * pspec)
{
  GstSimaaiProcesscvu * self = GST_SIMAAI_PROCESSCVU (object);

  switch(prop_id) {
    case PROP_CONF_F:
      g_value_set_string(value, self->priv->config_file_path.c_str());
      break;
    case PROP_SILENT:
      g_value_set_boolean (value, self->silent);
      break;
    case PROP_TRANSMIT:
      g_value_set_boolean (value, self->transmit);
      break;
    case PROP_NO_OF_BUFS:
      g_value_set_ulong(value, self->priv->num_of_out_buf);
      break;
    case PROP_DUMP_DATA:
      g_value_set_boolean(value, self->priv->dump_data);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
      break;
  }
}

static std::string cm2pluginHW[] = {
  [SIMA_CPU_A65] = "APU",
  [SIMA_CPU_EVXX] = "CVU",
  [SIMA_CPU_MLA] = "MLA",
};

static gint32 string2pluginCPU(std::string & in)
{
  for (int i = 0; i < sizeof(cm2pluginHW)/sizeof(cm2pluginHW[0]); i++)
    if (cm2pluginHW[i] == in)
      return i;
  return SIMA_CPU_EVXX;
}

/**
 * @brief helper function to get memory target
 * @return CPU type on success, -1 on error
 */
static GstSimaaiMemoryFlags get_mem_target (GstSimaaiProcesscvu * self) {
  if (self->priv->target_cpu < SIMA_CPU_A65 || 
      self->priv->next_cpu < SIMA_CPU_A65 ||
      self->priv->target_cpu > SIMA_CPU_NUM || 
      self->priv->next_cpu > SIMA_CPU_NUM )
    return GST_SIMAAI_MEMORY_FLAG_DEFAULT;
  
  if (self->priv->mem_type == GST_SIMAAI_MEMORY_TARGET_EV74) {
    return GST_SIMAAI_MEMORY_TARGET_EV74;
  }


  //memory decision map
  static const GstSimaaiMemoryFlags targets[SIMA_CPU_NUM][SIMA_CPU_NUM] = {
    { GST_SIMAAI_MEMORY_TARGET_DMS1, GST_SIMAAI_MEMORY_TARGET_EV74, GST_SIMAAI_MEMORY_TARGET_DMS2, GST_SIMAAI_MEMORY_TARGET_DMS3, },
    { GST_SIMAAI_MEMORY_TARGET_EV74, GST_SIMAAI_MEMORY_TARGET_EV74, GST_SIMAAI_MEMORY_TARGET_EV74, GST_SIMAAI_MEMORY_TARGET_EV74, },
    { GST_SIMAAI_MEMORY_TARGET_DMS2, GST_SIMAAI_MEMORY_TARGET_EV74, GST_SIMAAI_MEMORY_TARGET_DMS3, GST_SIMAAI_MEMORY_TARGET_DMS1, },
    { GST_SIMAAI_MEMORY_TARGET_DMS3, GST_SIMAAI_MEMORY_TARGET_EV74, GST_SIMAAI_MEMORY_TARGET_DMS1, GST_SIMAAI_MEMORY_TARGET_DMS2, },
  };

  const std::map<GstSimaaiMemoryFlags, std::string> mem_types  = {
      { GST_SIMAAI_MEMORY_TARGET_GENERIC, "GST_SIMAAI_MEMORY_TARGET_GENERIC" },
      { GST_SIMAAI_MEMORY_TARGET_OCM,     "GST_SIMAAI_MEMORY_TARGET_OCM"     },
      { GST_SIMAAI_MEMORY_TARGET_DMS0,    "GST_SIMAAI_MEMORY_TARGET_DMS0"    },
      { GST_SIMAAI_MEMORY_TARGET_DMS1,    "GST_SIMAAI_MEMORY_TARGET_DMS1"    },
      { GST_SIMAAI_MEMORY_TARGET_DMS2,    "GST_SIMAAI_MEMORY_TARGET_DMS2"    },
      { GST_SIMAAI_MEMORY_TARGET_DMS3,    "GST_SIMAAI_MEMORY_TARGET_DMS3"    },
      { GST_SIMAAI_MEMORY_TARGET_EV74,    "GST_SIMAAI_MEMORY_TARGET_EV74"    } 
    };

  //decide memory type
  auto result = targets[self->priv->target_cpu][self->priv->next_cpu];

  GST_DEBUG_OBJECT (self, 
                    "Decided memory: %s based on: next_cpu: %s target_cpu: %s",
                    mem_types.find(result)->second.c_str(),
                    (cm2pluginHW[self->priv->next_cpu]).c_str(),
                    (cm2pluginHW[self->priv->target_cpu]).c_str());


  return result;
}

/**
 * @brief helper function to free allocated output memory
 */
static void gst_simaai_processcvu_free_memory(GstSimaaiProcesscvu * self)
{
  if (self->priv->pool != nullptr)
    if (gst_simaai_free_buffer_pool(self->priv->pool));
      self->priv->pool = nullptr;
}

/**
 * @brief helper function to allocate output memory
 */
static gboolean gst_simaai_processcvu_allocate_memory(GstSimaaiProcesscvu * self)
{
  // if buffer pool is already allocated - deactivate and free
  if (self->priv->pool)
    gst_simaai_free_buffer_pool(self->priv->pool);

  auto & output_memories = self->priv->graph_buffers[self->priv->node_name];
  int number_of_segments = output_memories.size();
  // array of segments sizes
  gsize * segment_sizes = (gsize *) calloc(number_of_segments, sizeof(gsize));
  // arrary of segments names
  gchar ** segment_names = (gchar **)calloc(number_of_segments, sizeof(gchar *));

  if (segment_sizes == nullptr || segment_names == nullptr) {
    GST_ERROR_OBJECT(self, "Failed to allocate intermediate segments buffers");
    return FALSE;
  }
  
  // fill in all memory sizes and names
  int i = 0;
  for (auto & memory : output_memories) {
    segment_sizes[i] = memory.size;
    segment_names[i++] = (gchar *)(memory.dispatcher_name.c_str());
  }

  GstMemoryFlags flags = static_cast<GstMemoryFlags>(get_mem_target(self)
                                             | GST_SIMAAI_MEMORY_FLAG_CACHED);

  self->priv->pool = 
      gst_simaai_allocate_buffer_pool2((GstObject*) self, 
                                        gst_simaai_memory_get_segment_allocator(), 
                                        MIN_POOL_SIZE, 
                                        self->priv->num_of_out_buf,
                                        flags, number_of_segments,
                                        segment_sizes, 
                                        const_cast<const char**>(segment_names));

  free (segment_sizes);
  free (segment_names);

  if (self->priv->pool == nullptr) {
    GST_ERROR_OBJECT (self, "Failed to allocate buffer pool");
    return FALSE;
  }

  GST_DEBUG_OBJECT (self, "Output buffer pool: %d buffers of size %d",
                      self->priv->num_of_out_buf, self->priv->output_size);

  return TRUE;
}

/**
 * @brief helper function to initialize config manager
 */
static gboolean gst_simaai_processcvu_init_cm(GstSimaaiProcesscvu * self)
{
  if (self->priv->config_file_path.empty())
    return FALSE;

  try {
    self->priv->config_manager.reset(new ConfigManager(self->priv->config_file_path));
  } catch (std::exception & ex) {
    GST_ERROR_OBJECT(self, "Error allocating CM: %s", ex.what());
    return FALSE;
  }

  auto * graph_config = self->priv->config_manager->getConfigStruct();
  if (graph_config == nullptr) {
    GST_ERROR_OBJECT(self, "Error getting config structure. "
                           "Please, check syslogs for more information");
    return FALSE;
  }

  struct PipelineConfig conf = self->priv->config_manager->getPipelineConfig();

  self->priv->target_cpu = string2pluginCPU(conf.currentCpu);
  self->priv->next_cpu = string2pluginCPU(conf.nextCpu);

  // get buffers map
  self->priv->cm_memories = self->priv->config_manager->getBuffers();
  self->priv->output_size = 0;

  // validate if dispatcher memories are right and copy sizes
  for (auto& [ buffer_name, memories_info ] : self->priv->graph_buffers) {
    // get type of buffer
    bufferType type_of_buffer = 
        buffer_name == self->priv->node_name ? 
        bufferType::BUFFER_TYPE_OUTPUT : bufferType::BUFFER_TYPE_INPUT;
    for (auto & memory : memories_info) {
      // if no such memory in cm config - error
      auto cm_memory_iter = self->priv->cm_memories.find(memory.dispatcher_name);
      if (cm_memory_iter == self->priv->cm_memories.end()) {
        GST_ERROR_OBJECT (self, "Can not find memory with name \'%s\' in "
                                "ConfigManager buffers",
                                memory.dispatcher_name.c_str());
        return FALSE; 
      }

      auto & cm_memory_info = cm_memory_iter->second;
      
      if (type_of_buffer == bufferType::BUFFER_TYPE_OUTPUT) {
        // if output memory is not output/internal, according to CM - error
        if (cm_memory_info.second == bufferType::BUFFER_TYPE_INPUT) {
          GST_ERROR_OBJECT (self, "Output memory name \'%s\' is not an output "
                                  "name provided by ConfigManager",
                                  memory.dispatcher_name.c_str());
          return FALSE;
        }
      } else {
        // if input memory is not an input, according to CM - error
        if (cm_memory_info.second != bufferType::BUFFER_TYPE_INPUT) {
          GST_ERROR_OBJECT (self, "Input memory name \'%s\' is not an input "
                                  "name provided by ConfigManager",
                                  memory.dispatcher_name.c_str());
          return FALSE;
        }
      }

      // if memory is valid - copy size
      memory.size = cm_memory_info.first;

      if (type_of_buffer == bufferType::BUFFER_TYPE_OUTPUT)
        self->priv->output_size += memory.size;
    }
  }

  GST_DEBUG_OBJECT(self, "Total output size - %d", self->priv->output_size);

  return TRUE;
}

/**
 * @brief Called to parse from json order of input and output memories.
 */
bool gst_simaai_processcvu_parse_buffers_memories(GstSimaaiProcesscvu * self)
{
  nlohmann::json json;
  if (!parse_json_from_file(self, self->priv->config_file_path, json))
    return false;
  if (!json.contains("input_buffers") || !json.contains("output_memory_order"))
    return false;

  nlohmann::json & input_buffers(json["input_buffers"]);
  nlohmann::json & output_memories(json["output_memory_order"]);

  std::string buffer_name;
  GraphMemory tmp_mem;
 
  // add input buffers
  for (auto &input_it : input_buffers.items()) {
    auto & input = input_it.value();
    buffer_name = input["name"];

    auto & memories = self->priv->graph_buffers[buffer_name];
    GST_DEBUG_OBJECT (self, "input '%s' memories size: %ld", 
                      buffer_name.c_str(), input["memories"].size());
    memories.reserve(input["memories"].size());
    
    for (auto &mem_it : input["memories"].items()) {
      auto & mem = mem_it.value();
      if (!mem.contains("segment_name") || !mem.contains("graph_input_name")) {
        GST_ERROR_OBJECT (self, "Failed to parse buffers %s memory '%s'",
                          buffer_name.c_str(), to_string(mem).c_str());
        return false;
      }

      tmp_mem.memory_name = mem["segment_name"];
      tmp_mem.dispatcher_name = mem["graph_input_name"];

      GST_DEBUG_OBJECT (self, "Added input memory \'%s\' with name \'%s\'", 
                        tmp_mem.dispatcher_name.c_str(), 
                        tmp_mem.memory_name.c_str());
      memories.push_back(tmp_mem);
    }
  }

  // add output buffers
  auto & out_memories = self->priv->graph_buffers[self->priv->node_name];
  out_memories.reserve(output_memories.size());
  for (auto &mem : output_memories.items()) {
    tmp_mem.memory_name = mem.value();
    tmp_mem.dispatcher_name = mem.value();

    GST_DEBUG_OBJECT(self, "Added output memory %s", tmp_mem.memory_name.c_str());
    out_memories.push_back(tmp_mem);
  }

  return true;
}

static GstCaps *
gst_simaai_processcvu_fixate_src_caps(GstAggregator *aggregator, GstCaps *caps)
{
  GstSimaaiProcesscvu *processcvu = GST_SIMAAI_PROCESSCVU(aggregator);

  return gst_simaai_caps_fixate_src_caps(GST_ELEMENT(processcvu),
    processcvu->priv->simaai_caps, caps);
}

static gboolean
gst_simaai_processcvu_negotiate(GstAggregator *aggregator)
{
  GstSimaaiProcesscvu *processcvu = GST_SIMAAI_PROCESSCVU(aggregator);

  return gst_simaai_caps_negotiate(GST_ELEMENT(processcvu),
    processcvu->priv->simaai_caps);
}

static gboolean
gst_simaai_processcvu_sink_query(GstAggregator *aggregator,
    GstAggregatorPad *aggregator_pad, GstQuery *query)
{
  GstSimaaiProcesscvu *processcvu = GST_SIMAAI_PROCESSCVU(aggregator);

  switch (GST_QUERY_TYPE(query)) {
    case GST_QUERY_CAPS:
      return gst_simaai_caps_query(GST_ELEMENT(processcvu),
        processcvu->priv->simaai_caps, query, GST_PAD_SINK);
  }

  return GST_AGGREGATOR_CLASS(parent_class)->sink_query(aggregator,
      aggregator_pad, query);
}

static gboolean
gst_simaai_processcvu_src_query(GstAggregator *aggregator, GstQuery *query)
{
  GstSimaaiProcesscvu *processcvu = GST_SIMAAI_PROCESSCVU(aggregator);

  switch (GST_QUERY_TYPE(query)) {
    case GST_QUERY_CAPS:
      return gst_simaai_caps_query(GST_ELEMENT(processcvu),
        processcvu->priv->simaai_caps, query, GST_PAD_SRC);
  }

  return GST_AGGREGATOR_CLASS(parent_class)->src_query(aggregator, query);
}

/**
 * @brief Called to perform state change.
 */
static GstStateChangeReturn
gst_simaai_processcvu_change_state (GstElement * element,
                                    GstStateChange transition)
{
  GstSimaaiProcesscvu * self = GST_SIMAAI_PROCESSCVU (element);
  GstStateChangeReturn ret;

  GstEvent *stream_start_event;

  GstIterator *it = gst_element_iterate_src_pads(GST_ELEMENT_CAST(element));
  GValue item = G_VALUE_INIT;
  GstIteratorResult itret;

  ret = GST_ELEMENT_CLASS(parent_class)-> change_state(element, transition);
  if (ret == GST_STATE_CHANGE_FAILURE) {
    return ret;
  }

  switch(transition) {
  case GST_STATE_CHANGE_NULL_TO_READY:
    //get name property
    self->priv->node_name = std::string(gst_element_get_name(element));

    if (!gst_simaai_caps_parse_config(GST_ELEMENT(self),
      self->priv->simaai_caps, self->priv->config_file_path.c_str())) {
      GST_ERROR_OBJECT(self, "<%s>: Error parsing config", G_STRFUNC);
      return GST_STATE_CHANGE_FAILURE;
    }

    GST_DEBUG_OBJECT(self, "GST_STATE_CHANGE_NULL_TO_READY");
    break;
  case GST_STATE_CHANGE_READY_TO_PAUSED:
    GST_DEBUG_OBJECT(self, "GST_STATE_CHANGE_READY_TO_PAUSED");

    stream_start_event = gst_event_new_stream_start(self->priv->stream_id.c_str());

    it = gst_element_iterate_src_pads(GST_ELEMENT_CAST (element));
    while ((itret = gst_iterator_next(it, &item)) == GST_ITERATOR_OK) {
      GstPad *pad = (GstPad *) g_value_get_object(&item);
      gst_object_ref(pad);

      if (!gst_pad_push_event(pad, gst_event_ref(stream_start_event)))
        return GST_STATE_CHANGE_FAILURE;

      g_value_unset(&item);
      gst_object_unref(pad);
    }

    gst_event_unref(stream_start_event);
    gst_iterator_free (it);

    if (!gst_simaai_processcvu_parse_buffers_memories(self))
      return GST_STATE_CHANGE_FAILURE;

    self->priv->dispatcher = 
        simaaidispatcher::DispatcherFactory::getDispatcher(
            simaaidispatcher::DispatcherFactory::EVXX);
    if (self->priv->dispatcher == nullptr) {
      GST_ERROR_OBJECT (self, "Unable to get dispatcher");
      return GST_STATE_CHANGE_FAILURE;
    }

    break;
  case GST_STATE_CHANGE_READY_TO_NULL:
    gst_simaai_processcvu_free_memory(self);
    GST_DEBUG_OBJECT(self, "GST_STATE_CHANGE_READY_TO_NULL");
    break;
  default:
    break;
  }

  return ret;
}

/**
 * @brief Callback for event on the sinkpad
 */
static gboolean
gst_simaai_processcvu_sink_event (GstAggregator * agg, GstAggregatorPad * bpad, GstEvent * event)
{
  GstSimaaiProcesscvu *self = GST_SIMAAI_PROCESSCVU (agg);
  
  if (GST_EVENT_TYPE(event) == GST_EVENT_CAPS) {
    const std::lock_guard<std::mutex> guard(self->priv->event_mtx);

    if (!gst_simaai_caps_process_sink_caps(GST_ELEMENT(self),
      self->priv->simaai_caps, event)) {
      GST_ERROR_OBJECT(self, "<%s>: Error processing sink caps", G_STRFUNC);
      return FALSE;
    }

    if (!gst_simaai_processcvu_init_cm(self))
      return FALSE;

    if (!gst_simaai_processcvu_allocate_memory(self)) {
      GST_ERROR_OBJECT (self, "Unable to allocate memory");
      return FALSE;
    }
    GST_DEBUG_OBJECT( self, "[SINK CAPS EVENT] finished cm update and reallocation");

    return TRUE;
  }
  
  return GST_AGGREGATOR_CLASS (parent_class)->sink_event (agg, bpad, event);
}

/**
 * @brief Callback to request new sink pad, when a new link is created from the child proxy
 */
static GstPad *
gst_simaai_processcvu_request_new_pad (GstElement * element, GstPadTemplate * templ,
                                  const gchar * req_name, const GstCaps * caps)
{
  GstPad *newpad;

  newpad = (GstPad *)
           GST_ELEMENT_CLASS (parent_class)->request_new_pad (element,
                                                              templ, req_name, caps);

  if (newpad == NULL)
    goto could_not_create;

  gst_child_proxy_child_added (GST_CHILD_PROXY (element), G_OBJECT (newpad),
                               GST_OBJECT_NAME (newpad));

  return newpad;

could_not_create:
  {
    GST_DEBUG_OBJECT (element, "could not create/add pad");
    return NULL;
  }
}

static gboolean gst_simaai_processcvu_propose_allocation (GstAggregator * self,
                    GstAggregatorPad * pad,
                    GstQuery * decide_query,
                    GstQuery * query)
{
  GstSimaaiProcesscvu * simaaiprocesscvu = GST_SIMAAI_PROCESSCVU(self);
  GST_DEBUG_OBJECT(simaaiprocesscvu, "propose_allocation called");

  GstStructure *allocation_meta = gst_simaai_allocation_query_create_meta(GST_SIMAAI_MEMORY_TARGET_EV74, GST_SIMAAI_MEMORY_FLAG_CACHED);

  gst_simaai_allocation_query_add_meta(query, allocation_meta); 

  return TRUE;
}

static gboolean gst_simaai_processcvu_decide_allocation(GstAggregator * self, GstQuery * query)
{
  GstSimaaiProcesscvu * simaaiprocesscvu = GST_SIMAAI_PROCESSCVU(self);
  GST_DEBUG_OBJECT(simaaiprocesscvu, "decide_allocation called");

  GstSimaaiMemoryFlags mem_type;
  GstSimaaiMemoryFlags mem_flag;

  if (!gst_simaai_allocation_query_parse(query, &mem_type, &mem_flag)) {
    GST_WARNING_OBJECT(self, "Can't find allocation meta!");
  } else {
    simaaiprocesscvu->priv->mem_type = mem_type;
    simaaiprocesscvu->priv->mem_flag = mem_flag;
  };

  GST_DEBUG_OBJECT(simaaiprocesscvu, "Memory flags to allocate: [ %s ] [ %s ]",
    gst_simaai_allocation_query_sima_mem_type_to_str(simaaiprocesscvu->priv->mem_type),
    gst_simaai_allocation_query_sima_mem_flag_to_str(simaaiprocesscvu->priv->mem_flag));

  return TRUE;
}

/**
 * @brief Finalize/Cleanup processcvu callback
 */
static void
gst_simaai_processcvu_finalize (GObject * object)
{
  GstSimaaiProcesscvu * self = GST_SIMAAI_PROCESSCVU (object);

  /* Clean up current input buffer list */
  guint buf_len = gst_buffer_list_length(self->priv->list);
  if (buf_len) {
    for (guint i = 0; i < buf_len ; i++) {
      gst_buffer_unref(gst_buffer_list_get(self->priv->list, i));
    }
    gst_buffer_list_remove(self->priv->list, 0 , buf_len);
  }
  gst_buffer_list_unref(self->priv->list);

  gst_simaai_caps_free(self->priv->simaai_caps);

  delete self->priv;
  G_OBJECT_CLASS (parent_class)->finalize (object);
}

/**
 * @brief Release pads on finalize or EOS
 */
static void
gst_simaai_processcvu_release_pad (GstElement * element, GstPad * pad)
{
  GstSimaaiProcesscvu *processcvu;

  processcvu = GST_SIMAAI_PROCESSCVU (element);

  GST_DEBUG_OBJECT (processcvu, "release pad %s:%s", GST_DEBUG_PAD_NAME (pad));

  gst_child_proxy_child_removed (GST_CHILD_PROXY (processcvu), G_OBJECT (pad),
                                 GST_OBJECT_NAME (pad));

  GST_ELEMENT_CLASS (parent_class)->release_pad (element, pad);
}

/**
 * @brief Callback to processcvu init
 */
static void
gst_simaai_processcvu_class_init (GstSimaaiProcesscvuClass * klass)
{
  GObjectClass *gobj_class = G_OBJECT_CLASS(klass);
  GstElementClass *gstelement_class = (GstElementClass *) klass;
  GstAggregatorClass *base_aggregator_class = (GstAggregatorClass *) klass;

  gstelement_class->request_new_pad =
      GST_DEBUG_FUNCPTR (gst_simaai_processcvu_request_new_pad);

  gstelement_class->release_pad =
      GST_DEBUG_FUNCPTR (gst_simaai_processcvu_release_pad);

  base_aggregator_class->decide_allocation = 
      GST_DEBUG_FUNCPTR (gst_simaai_processcvu_decide_allocation);
  
  base_aggregator_class->propose_allocation =
      GST_DEBUG_FUNCPTR (gst_simaai_processcvu_propose_allocation);

  GstPadTemplate *sink_pad_template =
      gst_pad_template_new_with_gtype(PAD_TEMPLATE_NAME_SINK, GST_PAD_SINK,
      GST_PAD_REQUEST, gst_caps_new_any(), GST_TYPE_AGGREGATOR_PAD);
  if (!sink_pad_template) {
    GST_ERROR_OBJECT(base_aggregator_class, "<%s>: Error creating a sink pad "
        "template", G_STRFUNC);
    return;
  }
  gst_element_class_add_pad_template(gstelement_class, sink_pad_template);

  GstPadTemplate *src_pad_template =
      gst_pad_template_new_with_gtype(PAD_TEMPLATE_NAME_SRC, GST_PAD_SRC,
      GST_PAD_ALWAYS, gst_caps_new_any(), GST_TYPE_AGGREGATOR_PAD);
  if (!src_pad_template) {
    GST_ERROR_OBJECT(base_aggregator_class, "<%s>: Error creating a source pad "
        "template", G_STRFUNC);
    return;
  }
  gst_element_class_add_pad_template(gstelement_class, src_pad_template);

  gst_element_class_set_static_metadata(gstelement_class, 
                                        "SiMa.ai ProcessCVU element",
                                        "Transformer/Aggregator",
                                        "Perform algorithms on CVU",
                                        "SiMa.Ai");

  gobj_class->finalize = gst_simaai_processcvu_finalize;
  gobj_class->set_property = gst_simaai_processcvu_set_property;
  gobj_class->get_property = gst_simaai_processcvu_get_property;


  base_aggregator_class->aggregate =
      GST_DEBUG_FUNCPTR (gst_simaai_processcvu_aggregate);

  base_aggregator_class->sink_event =
      GST_DEBUG_FUNCPTR (gst_simaai_processcvu_sink_event);
  base_aggregator_class->sink_query =
      GST_DEBUG_FUNCPTR(gst_simaai_processcvu_sink_query);

  base_aggregator_class->src_query =
      GST_DEBUG_FUNCPTR(gst_simaai_processcvu_src_query);

  base_aggregator_class->fixate_src_caps =
      GST_DEBUG_FUNCPTR(gst_simaai_processcvu_fixate_src_caps);
  base_aggregator_class->negotiate =
      GST_DEBUG_FUNCPTR(gst_simaai_processcvu_negotiate);

  gstelement_class->change_state =
      GST_DEBUG_FUNCPTR (gst_simaai_processcvu_change_state);

  g_object_class_install_property (gobj_class, PROP_CONF_F,
                                   g_param_spec_string ("config",
                                                        "ConfigFile",
                                                        "Config JSON to be used",
                                                        DEFAULT_CONFIG_FILE,
                                                        (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  /* This property is used to allocate output buffers memory */ 
  g_object_class_install_property(gobj_class, PROP_NO_OF_BUFS,
                                  g_param_spec_ulong("num-buffers", 
                                                     "Number Of Buffers",
                                                     "Number of buffers to be allocated of size of buffer",
                                                     1, G_MAXUINT, 
                                                     DEFAULT_NUM_BUFFERS,
                                                     (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
  /* This property is used by cvu to enable/disable dumping of output buffers */
  g_object_class_install_property (gobj_class, PROP_DUMP_DATA,
                                   g_param_spec_boolean ("dump-data",
                                                         "Dump Data",
                                                         "Saves output buffers in binary dumps "
                                                         "in /tmp/{node-name}-:03{frame_id}.out",
                                                         DEFAULT_SILENT,
                                                         (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
  /* This property is used by cvu to enable/disable debugging messages */
  g_object_class_install_property (gobj_class, PROP_SILENT,
                                   g_param_spec_boolean ("silent",
                                                         "Silent",
                                                         "Produce verbose output",
                                                         DEFAULT_SILENT,
                                                         (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  /* This property is used to enable/disable kpi message transmission to the GstBus */
  g_object_class_install_property (gobj_class, PROP_TRANSMIT,
                                   g_param_spec_boolean ("transmit",
                                                         "Transmit",
                                                         "Transmit KPI Message",
                                                         DEFAULT_TRANSMIT,
                                                         (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  gst_element_class_set_static_metadata (gstelement_class,
                                         "SiMa.AI Process Cvu Plugin",
                                         "processcvu_agg",
                                         "SiMa.AI Process Cvu Plugin",
                                         "SiMa.AI");

  GST_DEBUG_CATEGORY_INIT (GST_CAT_DEFAULT,
                           "simaaiprocesscvu", 0, "ProcessCvu");

}

void gst_simaai_processcvu_print_dispatcher_error(GstSimaaiProcesscvu * self,
                                                  int res)
{
  std::string error_message;
  switch (res)
  {
  case EAGAIN:
    error_message = "queue is full";
    break;
  case EBADMSG:
    error_message = "job is lost in dispatcher or in mailbox driver";
    break;
  case EBADRQC:
    error_message = "job structure doesn't meet all requirements of EV";
    break;
  case ECANCELED:
    error_message = "operator thrown an exception";
    break;
  case ENOENT:
    error_message = "APU shared library file doesn't exist";      
    break;
  case ETIMEDOUT:
    error_message = "job is interrupted because of timeout";      
    break;
  case EBADFD:
    error_message = "issue with mailbox driver file descriptor";      
    break;
  case ENOSYS:
    error_message = "function is not supported or not found in .so file";      
    break;
  case EINVAL:
    error_message = "invalid argument";      
    break;
  default:
    error_message = "unexpected error";      
    break;
  }
  GST_ERROR_OBJECT (self, "Dispatcher run failed | code %d | message: %s", 
                          res,
                          error_message.c_str());
}

/**
 * @brief Helper API to run the graph on the CVU
 */
gboolean 
run_processcvu (GstSimaaiProcesscvu * self, simaaidispatcher::JobEVXX & job)
{
  if (!self->silent) {
    self->priv->t0 = std::chrono::steady_clock::now();
  }

  if (self->transmit) {
    tracepoint_pipeline_cvu_start(self->priv->frame_id, (char *)self->priv->node_name.c_str(), (char *)self->priv->stream_id.c_str());
  }

  int res = self->priv->dispatcher->run(job, self->priv->tp);
  if (res) {
    gst_simaai_processcvu_print_dispatcher_error(self, res);
    return FALSE;
  }

  if (self->transmit) {
    tracepoint_pipeline_cvu_end(self->priv->frame_id, (char *)self->priv->node_name.c_str(), (char *)self->priv->stream_id.c_str());
  }

  if (!self->silent) {
    self->priv->t1 = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(self->priv->t1 - self->priv->t0);
    auto duration = elapsed.count() / 1000.0 ;

    self->priv->run_count++;

    GST_DEBUG_OBJECT(self, "run count[%ld], frame_id[%ld], run time in ms: %f", 
                     self->priv->run_count, self->priv->frame_id, duration);
  }

  if (self->priv->dump_data) {
    if (cvu_dump_output_buffer (self, self->priv->frame_id) !=0 ) {
      GST_ERROR_OBJECT (self, "Dumping of buffers failed");
      return FALSE;
    }
  }

  return TRUE;
}

/**
 * @brief klass init for processcvu
 */
static void
gst_simaai_processcvu_init (GstSimaaiProcesscvu * self)
{
  GstAggregator *agg = GST_AGGREGATOR (self);
  gst_segment_init (&GST_AGGREGATOR_PAD (agg->srcpad)->segment,
                    GST_FORMAT_TIME);

  self->silent = DEFAULT_SILENT;
  self->transmit = DEFAULT_TRANSMIT;
  gst_simaai_segment_memory_init_once();
  self->priv = new GstSimaaiProcesscvuPrivate;
  self->priv->config_file_path = DEFAULT_CONFIG_FILE;

  self->priv->pool = nullptr;
  self->priv->mem_type = GST_SIMAAI_MEMORY_TARGET_EV74;
  self->priv->mem_flag = GST_SIMAAI_MEMORY_FLAG_CACHED;
  self->priv->out_buffer_id = 0;
  self->priv->num_of_out_buf = DEFAULT_NUM_BUFFERS;
  self->priv->target_cpu = SIMA_CPU_EVXX;
  self->priv->next_cpu = SIMA_CPU_EVXX;
  self->priv->frame_id = -1;
  self->priv->in_pcie_buf_id = 0;
  self->priv->is_pcie = FALSE;
  self->priv->output_size = 0;
  self->priv->dump_data = false;

  self->priv->list = gst_buffer_list_new();
  self->priv->run_count = 0;

  self->priv->mem_type = GST_SIMAAI_MEMORY_TARGET_EV74;
  self->priv->mem_flag = GST_SIMAAI_MEMORY_FLAG_CACHED;

  self->priv->simaai_caps = gst_simaai_caps_init();
}

/* GstChildProxy implementation */
static GObject *
gst_simaai_processcvu_child_proxy_get_child_by_index (GstChildProxy * child_proxy,
                                                 guint index)
{
  GstSimaaiProcesscvu *simaai_processcvu = GST_SIMAAI_PROCESSCVU (child_proxy);
  GObject *obj = NULL;

  GST_OBJECT_LOCK (simaai_processcvu);
  obj = (GObject *)g_list_nth_data (GST_ELEMENT_CAST (simaai_processcvu)->sinkpads, index);
  if (obj)
    gst_object_ref (obj);
  GST_OBJECT_UNLOCK (simaai_processcvu);

  return obj;
}

/**
 * @brief helper function to get number of children
 */
static guint
gst_simaai_processcvu_child_proxy_get_children_count (GstChildProxy * child_proxy)
{
  guint count = 0;
  GstSimaaiProcesscvu *simaai_processcvu = GST_SIMAAI_PROCESSCVU (child_proxy);

  GST_OBJECT_LOCK (simaai_processcvu);
  count = GST_ELEMENT_CAST (simaai_processcvu)->numsinkpads;
  GST_OBJECT_UNLOCK (simaai_processcvu);
  GST_INFO_OBJECT (simaai_processcvu, "Children Count: %d", count);

  return count;
}

static void
gst_simaai_processcvu_child_proxy_init (gpointer g_iface, gpointer iface_data)
{
  GstChildProxyInterface *iface = (GstChildProxyInterface *)g_iface;

  iface->get_child_by_index = gst_simaai_processcvu_child_proxy_get_child_by_index;
  iface->get_children_count = gst_simaai_processcvu_child_proxy_get_children_count;
}

static gboolean
plugin_init (GstPlugin * plugin)
{
  if (!gst_element_register(plugin, "simaaiprocesscvu", GST_RANK_NONE,
                            GST_TYPE_SIMAAI_PROCESSCVU)) {
    GST_ERROR("Unable to register process simaaiprocesscvu plugin");
    return FALSE;
  }

  return TRUE;
}

GST_PLUGIN_DEFINE(
    GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    PLUGIN_NAME_LOWER,
    "GStreamer SiMa.ai Process CVU Plugin",
    plugin_init,
    VERSION,
    GST_LICENSE,
    GST_PACKAGE_NAME,
    GST_PACKAGE_ORIGIN
     );
