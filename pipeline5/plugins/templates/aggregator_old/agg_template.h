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

#ifndef AGG_TEMPLATE_H
#define AGG_TEMPLATE_H

#include <gst/base/gstaggregator.h>
#include <gstsimaaiallocator.h>
#include <gstsimaaibufferpool.h>
#include <simaai/sgp_types.h>

#include <cstdint>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <simaai/nlohmann/json.hpp>
#include <span>
#include <string>
#include <vector>

struct Metadata
{
  std::string name;
  std::string stream_id;
  std::uint64_t timestamp;
};

class Input {
 private:
 
  Metadata m_meta;
  std::uint8_t *m_raw_data;
  std::size_t m_raw_data_size;
 
 public:

 Metadata& getMetadata() noexcept
 {
  return m_meta;
 };

 const Metadata& getMetadata() const noexcept
 {
  return m_meta;
 };

 std::size_t getDataSize() const noexcept
 {
  return m_raw_data_size;
 };

 std::span<std::uint8_t> getData() noexcept
 {
  return std::span<std::uint8_t>(m_raw_data, getDataSize());
 };
  
  //std::span<uint8_t> data;  // data in GstMemory

  Input(Metadata meta, std::uint8_t *data, std::size_t data_size)
      : m_meta(meta), m_raw_data(data), m_raw_data_size(data_size) {}
};

class UserContext {
  nlohmann::json *parser;
  nlohmann::json user_context;

 public:
  UserContext(nlohmann::json *json);
  ~UserContext();
  void run(std::vector<Input> &input, std::span<uint8_t> output);

  static const char * getSrcPadTemplateCaps();
  static const char * getSinkPadTemplateCaps();

  const char * getSrcCaps();
  const char * getSinkPadCaps();

  bool fixateSrcCaps(GstCaps **caps);
  bool parseSinkCaps(GstCaps * caps);
};

#define CONCAT_NX(A, B) A##B
#define CONCAT_NAMES(A, B) CONCAT_NX(A, B)

#define STRINGIFY_NX(x) #x
#define STRINGIFY(x) STRINGIFY_NX(x)

#define PLUGIN_STR_L STRINGIFY(PLUGIN_NAME_LOWER)
#define PLUGIN_STR_C STRINGIFY(PLUGIN_NAME_CAMEL)

#define GSP CONCAT_NAMES(GstSimaai, PLUGIN_NAME_CAMEL)
#define GS_CLASS(x) CONCAT_NAMES(GSP, x)
#define GSPrivate GS_CLASS(Private)
#define GSClass GS_CLASS(Class)

#define GSF CONCAT_NAMES(gst_simaai_, PLUGIN_NAME_LOWER)
#define GS_FUNC(x) CONCAT_NAMES(GSF, x)

#define GST_GET_TYPE (GS_FUNC(_get_type)())

#define GST_SIMAAI_AGG_TEMPLATE(obj) \
  (G_TYPE_CHECK_INSTANCE_CAST((obj), GST_GET_TYPE, GSP))

#define GSP_CLASS_CLASS(x) (G_TYPE_CHECK_CLASS_CAST((x), GST_GET_TYPE, GSClass))

#define GST_SIMAAI_AGG_TEMPLATE_GET_CLASS(obj) \
  (G_TYPE_INSTANCE_GET_CLASS((obj), GST_GET_TYPE, GSClass))

#define DEFAULT_MAX_BUFS 5
#define DEFAULT_SILENT TRUE
#define DEFAULT_TRANSMIT FALSE
#define PLUGIN_CPU_TYPE "APU"
#define DEFAULT_CONFIG_FILE "/mnt/host/" PLUGIN_STR_L ".json"
#define SIMAAI_META_STR "GstSimaMeta"
#define MIN_POOL_SIZE 2

G_BEGIN_DECLS

struct GSPrivate {
  GString *config_file;                    //  JSON configuration file
  GstBufferPool *out_pool; // Aggregator output buffer pool
  GstBuffer *out_buf; // Current output buffer
  gint64 out_buffer_id;  //  output simaai-memlib buffer id (phys_addr)
  gint64 frame_id;            // Input frame id placeholder
  gint64 in_pcie_buf_id = 0;  // PCIe buffer ID to passthrough into next plugin
  gboolean is_pcie = FALSE;
  GstBufferList *list;  // Aggregator input buffers
  nlohmann::json json;
  UserContext *user_context{nullptr};

  gint flags;
  gint64 run_count;

  std::chrono::time_point<std::chrono::steady_clock> t0;
  std::chrono::time_point<std::chrono::steady_clock> t1;

  GString *stream_id;
  guint64 timestamp;
  gint alloc_type;
};

struct GSP {
  GstAggregator parent;
  gboolean silent;
  gboolean transmit;
  GSPrivate *priv;
};

struct GSClass {
  GstAggregatorClass parent_class;
};

GType GS_FUNC(_get_type)(void);

G_END_DECLS

enum {
  PROP_0,
  PROP_CONF_F,
  PROP_SILENT,
  PROP_TRANSMIT,
  PROP_ALLOC_TYPE,
  PROP_UNKNONW,
};

static void GS_FUNC(_set_property)(GObject *obj, guint prop_id,
                                   const GValue *value, GParamSpec *pspec);

static void GS_FUNC(_get_property)(GObject *obj, guint prop_id, GValue *value,
                                   GParamSpec *pspec);

static gboolean CONCAT_NAMES(run_, PLUGIN_NAME_LOWER)(GSP *self);

static GstStateChangeReturn GS_FUNC(_change_state)(GstElement *element,
                                                   GstStateChange transition);

static void GS_FUNC(_child_proxy_init)(gpointer g_iface, gpointer);

G_DEFINE_TYPE_WITH_CODE(GSP, GSF, GST_TYPE_AGGREGATOR,
                        G_IMPLEMENT_INTERFACE(GST_TYPE_CHILD_PROXY,
                                              GS_FUNC(_child_proxy_init)));

// GstBufferPool allocation wrapper
static GstBufferPool *allocate_gst_buffer_pool(GstObject *obj,
                                               gint alloc_type,
                                               guint buf_size,
                                               guint min_buffers,
                                               guint max_buffers,
                                               GstMemoryFlags flags) {
  
  GstAllocator *allocator;

  if (alloc_type == 1) {
    GST_DEBUG_OBJECT(obj, "Creating SiMa sima-allocator (no segment API)");
    allocator = gst_simaai_memory_get_allocator();
  } else if (alloc_type == 2) {
    GST_DEBUG_OBJECT(obj, "Creating SiMa sima-segment-allocator");
    allocator = gst_simaai_memory_get_segment_allocator();
  } else {
    GST_ERROR_OBJECT(obj, "Invalid SiMa allocator type!");
    return NULL;
  }

  if (allocator == NULL) {
    GST_ERROR_OBJECT(obj, "Failed to get SiMa alloctor!");
    return NULL;
  }

  if (alloc_type == 2) {
    return gst_simaai_allocate_buffer_pool(obj,
                                             allocator,
                                             buf_size,
                                             min_buffers,
                                             max_buffers,
                                             flags);
  } else {
    GstBufferPool *pool = gst_buffer_pool_new ();
    if (pool == NULL) {
      GST_ERROR_OBJECT (obj, "gst_buffer_pool_new failed");
      return NULL;
    }

    GstStructure *config = gst_buffer_pool_get_config (pool);
    if (config == NULL) {
      GST_ERROR_OBJECT (obj, "gst_buffer_pool_get_config failed");
      gst_object_unref (pool);
      return NULL;
    }

    gst_buffer_pool_config_set_params (config, NULL, buf_size, min_buffers, max_buffers);

    GstAllocationParams params;
    gst_allocation_params_init (&params);
    params.flags = flags;

    gst_buffer_pool_config_set_allocator (config, allocator, &params);

    gboolean res = gst_buffer_pool_set_config (pool, config);
    if (res == FALSE) {
      GST_ERROR_OBJECT (obj, "gst_buffer_pool_set_config failed");
      gst_object_unref (pool);
      return NULL;
    }

    res = gst_buffer_pool_set_active (pool, TRUE);
    if (res == FALSE) {
      GST_ERROR_OBJECT (obj, "gst_buffer_pool_set_active failed");
      gst_object_unref (pool);
      return NULL;
    }

    return pool;
  }
}

static gboolean free_gst_buffer_pool(GstBufferPool *pool) {
  g_return_val_if_fail (pool != NULL, FALSE);
  g_return_val_if_fail (gst_buffer_pool_set_active (pool, FALSE), FALSE);
  gst_object_unref (pool);

  return TRUE;
}

static gboolean GS_FUNC(_add2list)(GSP *self, GValue *value) {
  gint64 buf_id = 0;
  gint64 in_buf_offset = 0;

  GstAggregatorPad *pad = (GstAggregatorPad *)g_value_get_object(value);
  GstBuffer *buf = gst_aggregator_pad_peek_buffer(pad);

  if (buf) {
    buf = gst_aggregator_pad_pop_buffer(pad);

    GstCustomMeta *meta = gst_buffer_get_custom_meta(buf, SIMAAI_META_STR);
    if (meta != nullptr) {
      GstStructure *s = gst_custom_meta_get_structure(meta);
      if (s == nullptr) {
        gst_buffer_unref(buf);
        GST_OBJECT_UNLOCK(self);
        return GST_FLOW_ERROR;
      } else {
        gint64 in_pcie_buf_id = 0;
        if (gst_structure_get_int64(s, "pcie-buffer-id", &in_pcie_buf_id) ==
            TRUE) {
          self->priv->is_pcie = TRUE;
          self->priv->in_pcie_buf_id = in_pcie_buf_id;
        }

        if ((gst_structure_get_int64(s, "buffer-id", &buf_id) == TRUE) &&
            (gst_structure_get_int64(s, "frame-id", &self->priv->frame_id) ==
             TRUE) &&
            (gst_structure_get_int64(s, "buffer-offset", &in_buf_offset) ==
             TRUE)) {
          g_string_assign(self->priv->stream_id, gst_structure_get_string(s, "stream-id"));
          gchar *buf_name = (gchar *)gst_structure_get_string(s, "buffer-name");
          gst_structure_get_uint64(s, "timestamp", &self->priv->timestamp);
          gst_buffer_list_add(self->priv->list, buf);
          GST_DEBUG_OBJECT(self,
                           PLUGIN_STR_C
                           ": Copied metadata, "
                           "[%s][%s]:[%ld]:[%ld], buffer list length: %d, timestamp: %ld",
                           (const char *)self->priv->stream_id->str,
                           (const char *)buf_name, self->priv->frame_id,
                           in_buf_offset,
                           gst_buffer_list_length(self->priv->list), self->priv->timestamp);
        }
      }
    } else {
      GST_ERROR(
          "Please check readme to use metadata information, meta not found");
      return FALSE;
    }
  } else {
    GST_ERROR("[CRITICAL] input buffer is NULL");
    return FALSE;
  }

  return TRUE;
}

static GstFlowReturn GS_FUNC(_aggregate)(GstAggregator *aggregator, gboolean) {
  GSP *self = GST_SIMAAI_AGG_TEMPLATE(aggregator);
  // GSP *testagg = self;

  auto clean_buffer_list = [](GstBufferList *list) {
    guint no_of_inbufs = gst_buffer_list_length(list);
    for (guint i = 0; i < no_of_inbufs ; ++i) {
      gst_buffer_unref(gst_buffer_list_get(list, i));
    }
    gst_buffer_list_remove(list, 0 , no_of_inbufs);
  };

  {
    GstIterator *iter = gst_element_iterate_sink_pads(GST_ELEMENT(self));

    gboolean done_iterating = FALSE;
    while (!done_iterating) {
      GValue value = {};
      GstAggregatorPad *pad = nullptr;

      switch (gst_iterator_next(iter, &value)) {
        case GST_ITERATOR_OK:
          if (!GS_FUNC(_add2list)(self, &value)) {
            gst_iterator_free(iter);
            return GST_FLOW_ERROR;
          }
          break;
        case GST_ITERATOR_RESYNC:
          gst_iterator_resync(iter);
          break;
        case GST_ITERATOR_ERROR:
          GST_WARNING_OBJECT(self, "Sinkpads iteration error");
          done_iterating = TRUE;
          gst_aggregator_pad_drop_buffer(pad);
          break;
        case GST_ITERATOR_DONE:
          done_iterating = TRUE;
          break;
      }
    }

    gst_iterator_free(iter);
  }

  GstFlowReturn ret = gst_buffer_pool_acquire_buffer(self->priv->out_pool,
                                                     &self->priv->out_buf, NULL);

  if (G_LIKELY (ret == GST_FLOW_OK)) {
    GST_DEBUG_OBJECT (self, "Output buffer from pool: %p", self->priv->out_buf);
  } else {
    GST_ERROR_OBJECT (self, "Failed to allocate buffer");
    clean_buffer_list(self->priv->list);
    return GST_FLOW_ERROR;
  }

  // Run here
  if (CONCAT_NAMES(run_, PLUGIN_NAME_LOWER)(self) != TRUE) {
    GST_ERROR("Unable to run " PLUGIN_STR_L ", drop and continue");
    clean_buffer_list(self->priv->list);
    gst_buffer_unref(self->priv->out_buf);
    return GST_FLOW_ERROR;
  }

  // Update metadata
  GstCustomMeta *meta = gst_buffer_add_custom_meta(self->priv->out_buf, SIMAAI_META_STR);
  if (meta == nullptr) {
    GST_ERROR("Unable to add metadata info to the buffer");
    clean_buffer_list(self->priv->list);
    gst_buffer_unref(self->priv->out_buf);
    return GST_FLOW_ERROR;
  }

  GstStructure *s = gst_custom_meta_get_structure(meta);
  if (s != nullptr) {
    std::string node_name = self->priv->json["node_name"];
    gst_structure_set(s,
                      "buffer-id", G_TYPE_INT64, self->priv->out_buffer_id,
                      "buffer-name", G_TYPE_STRING, node_name.c_str(),
                      "buffer-offset", G_TYPE_INT64, (gint64)0,
                      "frame-id", G_TYPE_INT64, self->priv->frame_id,
                      "stream-id", G_TYPE_STRING, self->priv->stream_id->str,
                      "timestamp", G_TYPE_UINT64, self->priv->timestamp,
                      nullptr);

    if (self->priv->is_pcie) {
      gst_structure_set(s, "pcie-buffer-id", G_TYPE_INT64,
                        self->priv->in_pcie_buf_id, nullptr);
    }
  }

  clean_buffer_list(self->priv->list);

  // Push the provided output buffer downstream
  gst_aggregator_finish_buffer(aggregator, self->priv->out_buf);

  /* We just check finish_frame return FLOW_OK */
  return GST_FLOW_OK;
}

static void GS_FUNC(_set_property)(GObject *object, guint prop_id,
                                   const GValue *value, GParamSpec *pspec) {
  GSP *self = GST_SIMAAI_AGG_TEMPLATE(object);

  switch (prop_id) {
    case PROP_CONF_F:
      g_string_assign(self->priv->config_file, g_value_get_string(value));
      GST_DEBUG_OBJECT(self, "Config argument was changed to %s",
                       self->priv->config_file->str);
      break;
    case PROP_SILENT:
      self->silent = g_value_get_boolean(value);
      GST_DEBUG_OBJECT(self, "Set silent = %d", self->silent);
      break;
    case PROP_TRANSMIT:
      self->transmit = g_value_get_boolean(value);
      GST_DEBUG_OBJECT(self, "Set transmit = %d", self->transmit);
      break;
    case PROP_ALLOC_TYPE:
      self->priv->alloc_type = g_value_get_int(value);
      GST_DEBUG_OBJECT(self, "Set alloc_type = %d", self->priv->alloc_type);
      break;
    default:
      GST_DEBUG_OBJECT(self, "Default case warning");
      G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
      break;
  }
}

static void GS_FUNC(_get_property)(GObject *object, guint prop_id,
                                   GValue *value, GParamSpec *pspec) {
  GSP *self = GST_SIMAAI_AGG_TEMPLATE(object);

  switch (prop_id) {
    case PROP_CONF_F:
      g_value_set_string(value, (const gchar *)self->priv->config_file->str);
      break;
    case PROP_SILENT:
      g_value_set_boolean(value, self->silent);
      break;
    case PROP_TRANSMIT:
      g_value_set_boolean(value, self->transmit);
      break;
    case PROP_ALLOC_TYPE:
      g_value_set_boolean(value, self->priv->alloc_type);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
      break;
  }
}

static constexpr auto get_mem_target(sima_cpu_e cpu) noexcept {
  switch (cpu) {
    case SIMA_CPU_EVXX:
      return GST_SIMAAI_MEMORY_TARGET_EV74;
    case SIMA_CPU_MLA:
      return GST_SIMAAI_MEMORY_TARGET_DMS0;
    case SIMA_CPU_MOSAIC:
      return GST_SIMAAI_MEMORY_TARGET_DMS0;
    case SIMA_CPU_A65:
    case SIMA_CPU_DEV:
    default:
      return GST_SIMAAI_MEMORY_TARGET_GENERIC;
  }
}

static gboolean GS_FUNC(_allocate_memory)(GSP *self) {
  gboolean ret = TRUE;
  auto mem_target = get_mem_target(self->priv->json["memory"]["next_cpu"]);
  auto mem_flags = GST_SIMAAI_MEMORY_FLAG_DEFAULT;
  auto no_of_bufs = self->priv->json["system"]["out_buf_queue"].get<size_t>();
  auto out_sz = self->priv->json["buffers"]["output"]["size"].get<size_t>();

  g_assert(out_sz > 0);

  if (no_of_bufs < MIN_POOL_SIZE)
    no_of_bufs = MIN_POOL_SIZE;

  if (!(mem_target & GST_SIMAAI_MEMORY_TARGET_EV74))
    mem_flags = GST_SIMAAI_MEMORY_FLAG_CACHED;

  self->priv->out_pool = allocate_gst_buffer_pool(GST_OBJECT (self),
                                                  self->priv->alloc_type,
                                                  out_sz,
                                                  no_of_bufs,
                                                  no_of_bufs,
                                                  (GstMemoryFlags)(mem_target | mem_flags));

  if (G_UNLIKELY (self->priv->out_pool == NULL)) {
    GST_ERROR_OBJECT (self, "Failed to allocate buffer pool");
    ret = FALSE;
  } else {
    GST_DEBUG_OBJECT (self, "Allocated %zu buffers of size %zu in memory %#x",
                      no_of_bufs, out_sz, mem_target);
  }

  return ret;
}

static GstStateChangeReturn GS_FUNC(_change_state)(GstElement *element,
                                                   GstStateChange transition) {
  GSP *self = GST_SIMAAI_AGG_TEMPLATE(element);
  GstStateChangeReturn ret;

  ret = GST_ELEMENT_CLASS(GS_FUNC(_parent_class))
            ->change_state(element, transition);

  if (ret == GST_STATE_CHANGE_FAILURE) {
    return ret;
  }

  switch (transition) {
    case GST_STATE_CHANGE_PLAYING_TO_PAUSED:
      break;
    case GST_STATE_CHANGE_PAUSED_TO_READY:
      break;
    case GST_STATE_CHANGE_NULL_TO_READY: {
      GST_INFO_OBJECT(self, "Dispatcher init with config :%s",
                      self->priv->config_file->str);
      std::ifstream file(self->priv->config_file->str);
      if (file) {
        file >> self->priv->json;
        file.close();
      }

      if (!GS_FUNC(_allocate_memory)(self)) {
        GST_ERROR("Unable to allocate memory");
        ret = GST_STATE_CHANGE_FAILURE;
      }

      self->priv->user_context = new UserContext(&self->priv->json);
    } break;
    case GST_STATE_CHANGE_READY_TO_NULL: {
      if (free_gst_buffer_pool(self->priv->out_pool)) {
        self->priv->out_pool = NULL;
      }
      delete self->priv->user_context;
      self->priv->user_context = nullptr;
    } break;
    default:
      break;
  }

  return ret;
}

static gboolean GS_FUNC(_sink_event) (GstAggregator * agg, GstAggregatorPad * bpad, GstEvent * event)
{
  GSP *self = GST_SIMAAI_AGG_TEMPLATE (agg);
  
  if (GST_EVENT_TYPE(event) == GST_EVENT_CAPS) {
    GstPad *otherpad;
    GstCaps *temp, *caps, *filt, *tcaps;

    gst_event_parse_caps(event, &caps);
    
    if (!self->priv->user_context->parseSinkCaps(caps))
      return FALSE;

    GST_DEBUG_OBJECT( self, "[SINK CAPS EVENT] caps: \"%s\"", 
                      gst_caps_to_string(caps));
    return TRUE;
  }
  
  return GST_AGGREGATOR_CLASS (GS_FUNC(_parent_class))->sink_event (agg, bpad, event);
}

static GstCaps * GS_FUNC(_fixate_src_caps) (GstAggregator * agg, 
                                       GstCaps * downstream_caps)
{
  GSP *self = GST_SIMAAI_AGG_TEMPLATE (agg);

  GstCaps * user_caps = gst_caps_from_string(
                              self->priv->user_context->getSrcCaps());
  GstCaps * caps_check = gst_caps_intersect_full( user_caps, 
                                                  downstream_caps, 
                                                  GST_CAPS_INTERSECT_FIRST);

  if (!gst_caps_is_always_compatible(caps_check, user_caps)) {
    g_message( "SRC caps are not compatible with downstream plugin."
                    " SRC caps: %s | Downstream caps: %s",
                    gst_caps_to_string(user_caps), 
                    gst_caps_to_string(downstream_caps));
    return FALSE;
  }
  self->priv->user_context->fixateSrcCaps(&user_caps);
  GST_DEBUG_OBJECT(self, "Fixated src caps: %s", gst_caps_to_string(user_caps));

  return user_caps;
}

static gboolean
GS_FUNC(_negotiated_src_caps) (GstAggregator * agg, GstCaps * downstream_caps)
{
  return TRUE;
}

static gboolean GS_FUNC(_sink_query)(GstAggregator *agg, GstAggregatorPad *bpad,
                                     GstQuery *query) {
  return GST_AGGREGATOR_CLASS(GS_FUNC(_parent_class))
      ->sink_query(agg, bpad, query);
}

static GstPad *GS_FUNC(_request_new_pad)(GstElement *element,
                                         GstPadTemplate *templ,
                                         const gchar *req_name,
                                         const GstCaps *caps) {
  GstPad *newpad = (GstPad *)GST_ELEMENT_CLASS(GS_FUNC(_parent_class))
                       ->request_new_pad(element, templ, req_name, caps);

  if (newpad == nullptr) goto could_not_create;

  gst_child_proxy_child_added(GST_CHILD_PROXY(element), G_OBJECT(newpad),
                              GST_OBJECT_NAME(newpad));

  return newpad;

could_not_create : {
  GST_DEBUG_OBJECT(element, "could not create/add pad");
  return NULL;
}
}

static void GS_FUNC(_finalize)(GObject *object) {
  GSP *self = GST_SIMAAI_AGG_TEMPLATE(object);

  g_string_free(self->priv->stream_id, TRUE);
  g_string_free(self->priv->config_file, TRUE);

  /* Clean up current input buffer list */
  guint buf_len = gst_buffer_list_length(self->priv->list);
  if (buf_len) {
    for (guint i = 0; i < buf_len ; i++) {
      gst_buffer_unref(gst_buffer_list_get(self->priv->list, i));
    }
    gst_buffer_list_remove(self->priv->list, 0, buf_len);
  }
  gst_buffer_list_unref(self->priv->list);
  delete self->priv;

  G_OBJECT_CLASS(GS_FUNC(_parent_class))->finalize(object);
}

static void GS_FUNC(_release_pad)(GstElement *element, GstPad *pad) {
  GSP *PLUGIN_NAME_LOWER = GST_SIMAAI_AGG_TEMPLATE(element);

  GST_DEBUG_OBJECT(PLUGIN_NAME_LOWER, "release pad %s:%s",
                   GST_DEBUG_PAD_NAME(pad));

  gst_child_proxy_child_removed(GST_CHILD_PROXY(PLUGIN_NAME_LOWER),
                                G_OBJECT(pad), GST_OBJECT_NAME(pad));

  GST_ELEMENT_CLASS(GS_FUNC(_parent_class))->release_pad(element, pad);
}

static void GS_FUNC(_class_init)(GSClass *klass) {
  GObjectClass *gobj_class = G_OBJECT_CLASS(klass);
  GstElementClass *gstelement_class = (GstElementClass *)klass;
  GstAggregatorClass *base_aggregator_class = (GstAggregatorClass *)klass;

  gstelement_class->request_new_pad =
      GST_DEBUG_FUNCPTR(GS_FUNC(_request_new_pad));

  gstelement_class->release_pad = GST_DEBUG_FUNCPTR(GS_FUNC(_release_pad));

  GstCaps * src_template_caps = gst_caps_from_string(UserContext::getSrcPadTemplateCaps());
  GstCaps * sink_template_caps = gst_caps_from_string(UserContext::getSinkPadTemplateCaps());

  static GstStaticPadTemplate _src_template = GST_STATIC_PAD_TEMPLATE(
      "src", GST_PAD_SRC, GST_PAD_ALWAYS, src_template_caps);

  static GstStaticPadTemplate _sink_template = GST_STATIC_PAD_TEMPLATE(
      "sink_%u", GST_PAD_SINK, GST_PAD_REQUEST, sink_template_caps);

  gst_element_class_add_static_pad_template_with_gtype(
      gstelement_class, &_src_template, GST_TYPE_AGGREGATOR_PAD);

  gst_element_class_add_static_pad_template_with_gtype(
      gstelement_class, &_sink_template, GST_TYPE_AGGREGATOR_PAD);

  gst_element_class_set_static_metadata(gstelement_class, "Aggregator",
                                        PLUGIN_STR_C, "Combine N buffers",
                                        "SiMa.Ai Inc");

  gobj_class->finalize = GS_FUNC(_finalize);
  gobj_class->set_property = GS_FUNC(_set_property);
  gobj_class->get_property = GS_FUNC(_get_property);
  base_aggregator_class->aggregate = GST_DEBUG_FUNCPTR(GS_FUNC(_aggregate));
  base_aggregator_class->sink_query = GST_DEBUG_FUNCPTR(GS_FUNC(_sink_query));
  base_aggregator_class->sink_event = GST_DEBUG_FUNCPTR(GS_FUNC(_sink_event));
  base_aggregator_class->negotiated_src_caps = GST_DEBUG_FUNCPTR(GS_FUNC(_negotiated_src_caps));
  base_aggregator_class->fixate_src_caps = GST_DEBUG_FUNCPTR(GS_FUNC(_fixate_src_caps));
  gstelement_class->change_state = GST_DEBUG_FUNCPTR(GS_FUNC(_change_state));

  g_object_class_install_property(
      gobj_class, PROP_CONF_F,
      g_param_spec_string(
          "config", "ConfigFile", "Config JSON to be used", DEFAULT_CONFIG_FILE,
          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property(
      gobj_class, PROP_SILENT,
      g_param_spec_boolean(
          "silent", "Silent", "Produce verbose output", DEFAULT_SILENT,
          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property(
      gobj_class, PROP_TRANSMIT,
      g_param_spec_boolean(
          "transmit", "Transmit", "Transmit KPI Messages", DEFAULT_TRANSMIT,
          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property(
      gobj_class, PROP_ALLOC_TYPE,
      g_param_spec_int(
          "sima-allocator-type", "Type of SiMa allocator to be used", "1 - no segment API, 2 - segment API support", 1,
          2, 1,
          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  gst_element_class_set_static_metadata(
      gstelement_class, "SiMa.AI " PLUGIN_STR_C " Plugin", PLUGIN_STR_L,
      "SiMa.Ai " PLUGIN_STR_C " plugin", "SiMa.AI");

  GST_DEBUG_CATEGORY_INIT(GST_CAT_DEFAULT, PLUGIN_STR_L, 0, PLUGIN_STR_C);
}

void dump_output_buffer(GSP *self, const char *vaddr, gsize sz) {
  std::string dir = "/tmp/";
  std::string node_name = self->priv->json["node_name"];
  std::string f_id = std::to_string(self->priv->frame_id);

  std::ofstream file(dir + node_name + "-" + f_id + ".out", std::ios::binary);
  if (file) {
    file.write(vaddr, sz);
    file.close();
  }
}

gboolean CONCAT_NAMES(run_, PLUGIN_NAME_LOWER)(GSP *self) {
    
  self->priv->t0 = std::chrono::steady_clock::now();
  

  // --- prepare input buffers
  guint buf_len = gst_buffer_list_length(self->priv->list);
  if (buf_len == 0) {
    GST_ERROR("Buffer list is empty");
    return FALSE;
  }

  std::vector<GstBuffer *> buf(buf_len);
  std::vector<GstMapInfo> meminfo(buf_len);
  std::vector<Input> input;

  auto unmap_input_buffers = [](std::vector<GstBuffer*>& buf,
                                std::vector<GstMapInfo>& meminfo) {
    for (guint i = 0; i < buf.size(); ++i) {
      gst_buffer_unmap(buf[i], &meminfo[i]);
    }
  };

  for (std::size_t i = 0; i < buf_len; ++i) {
    buf[i] = gst_buffer_list_get(self->priv->list, i);
    gst_buffer_map(buf[i], &meminfo[i], GST_MAP_READ);

    GstCustomMeta *meta = gst_buffer_get_custom_meta(buf[i], SIMAAI_META_STR);
    if (meta != nullptr) {
      GstStructure *structure = gst_custom_meta_get_structure(meta);
      if (structure != nullptr) {
        std::string buf_name(
            (char *)gst_structure_get_string(structure, "buffer-name"));
        std::string stream_id(
            (char *)gst_structure_get_string(structure, "stream-id"));
        
        std::uint64_t timestamp = 0;
        gst_structure_get_uint64(structure, "timestamp", &timestamp);

        Metadata meta{buf_name, stream_id, timestamp};

        std::size_t buf_size = meminfo[i].size;

        for (std::size_t j = 0; j < self->priv->json["buffers"]["input"].size();
             ++j) {
          std::string json_buf_name =
              self->priv->json["buffers"]["input"][i]["name"];
          if (json_buf_name == buf_name) {
            buf_size = self->priv->json["buffers"]["input"][i]["size"];
            break;
          }
        }

        input.emplace_back(meta, meminfo[i].data, buf_size);
      }
    }
  }

  // --- prepare output buffers
  GstMapInfo out_meminfo;
  if (!gst_buffer_map(self->priv->out_buf, &out_meminfo, GST_MAP_WRITE)) {
    GST_ERROR_OBJECT(self, "Failed to map output buffer");
    unmap_input_buffers(buf, meminfo);
    return FALSE;
  }
  if (self->priv->alloc_type == 1)
    self->priv->out_buffer_id = 
        gst_simaai_memory_get_phys_addr(out_meminfo.memory);
  else if (self->priv->alloc_type == 2)
    self->priv->out_buffer_id = 
        gst_simaai_segment_memory_get_phys_addr(out_meminfo.memory);
  else
    GST_WARNING_OBJECT (self, "Unknown allocator type. "
                              "Can not get buffer-id for output buffer");

  // PAYLOAD
  self->priv->user_context->run(input, {out_meminfo.data, out_meminfo.size});
  // ----

  gst_buffer_unmap(self->priv->out_buf, &out_meminfo);
  unmap_input_buffers(buf, meminfo);

  if (self->priv->json["system"]["dump_data"] != 0) {
    if (!gst_buffer_map(self->priv->out_buf, &out_meminfo, GST_MAP_READ)) {
      GST_ERROR_OBJECT(self, "Failed to map output buffer for dumping");
      return FALSE;
    }
    dump_output_buffer(self, (const char *)out_meminfo.data, out_meminfo.size);
    gst_buffer_unmap(self->priv->out_buf, &out_meminfo);
  }

  self->priv->t1 = std::chrono::steady_clock::now();
  auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(self->priv->t1 - self->priv->t0);
  auto duration = elapsed.count() / 1000.0 ;

  if (!self->silent) {
    
    self->priv->run_count++;
    std::cout << PLUGIN_STR_L " run_count[" << self->priv->run_count
              << "], run time in ms: " << elapsed.count() / 1000.0 << std::endl;
  }

  if (self->transmit) { // transmit is enabled

    uint64_t kernel_start = 0;
    uint64_t kernel_end = 0;
    gchar* plugin_id = gst_element_get_name(GST_ELEMENT(self));
    GstMessage *message = gst_message_new_application(GST_OBJECT(self), gst_structure_new("kpi",
     "plugin_start", G_TYPE_UINT64,  std::chrono::duration_cast<std::chrono::microseconds>(self->priv->t0.time_since_epoch()).count(), 
     "plugin_end", G_TYPE_UINT64, std::chrono::duration_cast<std::chrono::microseconds>(self->priv->t1.time_since_epoch()).count(),
     "duration", G_TYPE_DOUBLE, duration,
     "kernel_start", G_TYPE_UINT64, kernel_start,
     "kernel_end" , G_TYPE_UINT64, kernel_end,
     "frame_id", G_TYPE_INT64, self->priv->frame_id,
     "plugin_id", G_TYPE_STRING, plugin_id,
     "plugin_type", G_TYPE_STRING, PLUGIN_CPU_TYPE,
     "stream_id", G_TYPE_STRING, self->priv->stream_id->str,
    NULL));
    g_free(plugin_id);

    if(!message){
      GST_ERROR_OBJECT(self, "Failed to create a message to emit.\n");
      return FALSE;
    }
    if(!gst_element_post_message(GST_ELEMENT(self), message)){
      GST_DEBUG_OBJECT(self, "Unable to transmit message from pluginId: [%s]", plugin_id);
    }
  }

  return TRUE;
}

static void GS_FUNC(_init)(GSP *self) {
  GstAggregator *agg = GST_AGGREGATOR(self);
  gst_segment_init(&GST_AGGREGATOR_PAD(agg->srcpad)->segment, GST_FORMAT_TIME);

  self->silent = DEFAULT_SILENT;
  self->transmit = DEFAULT_TRANSMIT;

  gst_simaai_memory_init_once();
  gst_simaai_segment_memory_init_once();

  self->priv = new GSPrivate;

  self->priv->config_file = g_string_new(DEFAULT_CONFIG_FILE);

  self->priv->out_pool = NULL;
  self->priv->list = gst_buffer_list_new();
  self->priv->frame_id = 0;
  self->priv->run_count = 0;

  self->priv->stream_id = g_string_new("stream-id-unknown");
  self->priv->timestamp = 0;

  self->priv->alloc_type = 1;
}

/* GstChildProxy implementation */
static GObject *GS_FUNC(_child_proxy_get_child_by_index)(
    GstChildProxy *child_proxy, guint index) {
  GSP *PLUGIN_NAME_LOWER = GST_SIMAAI_AGG_TEMPLATE(child_proxy);
  GObject *obj = NULL;

  GST_OBJECT_LOCK(PLUGIN_NAME_LOWER);

  obj = (GObject *)g_list_nth_data(
      GST_ELEMENT_CAST(PLUGIN_NAME_LOWER)->sinkpads, index);

  if (obj) {
    gst_object_ref(obj);
  }

  GST_OBJECT_UNLOCK(PLUGIN_NAME_LOWER);

  return obj;
}

static guint GS_FUNC(_child_proxy_get_children_count)(
    GstChildProxy *child_proxy) {
  guint count = 0;
  GSP *PLUGIN_NAME_LOWER = GST_SIMAAI_AGG_TEMPLATE(child_proxy);

  GST_OBJECT_LOCK(PLUGIN_NAME_LOWER);
  count = GST_ELEMENT_CAST(PLUGIN_NAME_LOWER)->numsinkpads;
  GST_OBJECT_UNLOCK(PLUGIN_NAME_LOWER);
  GST_INFO_OBJECT(PLUGIN_NAME_LOWER, "Children Count: %d", count);

  return count;
}

static void GS_FUNC(_child_proxy_init)(gpointer g_iface, gpointer) {
  GstChildProxyInterface *iface = (GstChildProxyInterface *)g_iface;

  iface->get_child_by_index = GS_FUNC(_child_proxy_get_child_by_index);
  iface->get_children_count = GS_FUNC(_child_proxy_get_children_count);
}

static gboolean plugin_init(GstPlugin *plugin) {
  if (!gst_element_register(plugin, PLUGIN_STR_L, GST_RANK_NONE,
                            GST_GET_TYPE)) {
    GST_ERROR("Unable to register process " PLUGIN_STR_L " plugin");
    return FALSE;
  }

  return TRUE;
}

GST_PLUGIN_DEFINE(GST_VERSION_MAJOR, GST_VERSION_MINOR, PLUGIN_NAME_LOWER,
                  "GStreamer SiMa.ai " PLUGIN_STR_C " Plugin", plugin_init,
                  VERSION, GST_LICENSE, GST_PACKAGE_NAME, GST_PACKAGE_ORIGIN);

#endif  // AGG_TEMPLATE_H
