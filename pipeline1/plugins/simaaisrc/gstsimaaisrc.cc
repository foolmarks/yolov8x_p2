/*
 * GStreamer
 * Copyright (C) 2022 SiMa.ai
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
 * @file gstsimaaisrc.cc
 * @brief Gstreamer plugin to run a pipeline using custom data, or just a frame or collection of frames
 * @author SiMa.Ai\TM
 * @bug Currently no known bugs
 * @todo Support for multi-frame run
 */

/** 
 * SECTION: element-simaaisrc
 *
 * Source plugin to run pipeline with a single frame
 *
 * <refsect2>
 * <title> Example Launch line </title>
 * |[
 * gst-launch-1.0 simaaisrc location="/tmp/inp_300.yuv" node-name="allegrodec" blocksize=1382400 node-name="allegrodec" !  \
 * process2 config="evxx_cnet_preproc_face.json" ! fakesink dump=true
 * ]|
 *
 * </refsect2>
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <inttypes.h>
#include <stdio.h>

#include <iostream>
#include <chrono>
#include <vector>
#include <string>

#include <glib.h>
#include <gst/gst.h>
#include <string.h>

#include <gstsimaaiallocator.h>
#include <gstsimaaibufferpool.h>
#include <simaai/simaai_memory.h>

#include "gstsimaaisrc.h"

// Default amount of buffers to allocate in output buffer pool
#define MIN_POOL_SIZE   2
#define MAX_POOL_SIZE   2
#define MAX_MEM_SEGMENT 16


struct SegmentInfo
{
  std::string name;
  std::string path;
  size_t      size;
};

struct _GstSimaaiSrcPrivate
{
  GstBufferPool *pool; ///< Buffer pool
  guint pool_size; ///< Amount of buffers to allocate in output buffer pool
  gchar *location; ///< filename of where the input data is available
  gchar *node_name; ///< Output node name
  GString *stream_id; ///< Stream Id or Camera Id
  gint64 frame_id; ///< The sequence id incremented for every iteration
  int index;
  int initial_index;  // Added for looping
  int max_index;    // Added to store total files
  gboolean loop;      // Added for looping
  gsize mem_size;
  int mem_target;
  gint64 delay;
  gchar *segments; ///< comma separated key=value string used to specify memory segments and the associated files
  std::vector<SegmentInfo> all_segments;
};

#define UNUSED(x) (void)(x)

GST_DEBUG_CATEGORY_STATIC (gst_simaaisrc_debug);
#define GST_CAT_DEFAULT gst_simaaisrc_debug

/**
 * @brief the capabilities of the outputs 
 */
static GstStaticPadTemplate srctemplate = GST_STATIC_PAD_TEMPLATE ("src",
    GST_PAD_SRC, GST_PAD_ALWAYS, GST_STATIC_CAPS_ANY);

/**
 * @brief simaaisrc properties
 */
enum
{
  PROP_0,
  PROP_LOCATION,
  PROP_INDEX,
  PROP_NODE_NAME,
  PROP_MEM_TARGET,
  PROP_DELAY,
  PROP_STREAM_ID,
  PROP_POOL_SIZE,
  PROP_LOOP,
  PROP_SEGMENTS,
  PROP_LAST
};

#define gst_simaaisrc_parent_class parent_class
G_DEFINE_TYPE (GstSimaaiSrc, gst_simaaisrc, GST_TYPE_BASE_SRC);

static void gst_simaaisrc_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec);
static void gst_simaaisrc_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec);
static void gst_simaaisrc_class_finalize (GObject * object);
static gchar *
gst_multi_simaaisrc_get_filename (GstSimaaiSrc * self);

static gboolean gst_simaaisrc_start (GstBaseSrc * basesrc);
static gboolean gst_simaaisrc_stop (GstBaseSrc * basesrc);
static GstFlowReturn gst_simaaisrc_create (GstBaseSrc * basesrc, guint64 offset,
    guint size, GstBuffer ** out_buf);
static gboolean gst_simaai_src_decide_allocation (GstBaseSrc * src, GstQuery * query);


glong get_file_size(const gchar * path);

static gboolean
free_simaai_memory_buffer_pool (GstBufferPool * pool)
{
  g_return_val_if_fail (pool != NULL, FALSE);
  g_return_val_if_fail (gst_buffer_pool_set_active (pool, FALSE), FALSE);
  gst_object_unref (pool);

  return TRUE;
}

/**
 * @brief initialize the class
 * @param klass, gobject class defintion for the plugin of type GstSimaaiSrcClass
 */
static void
gst_simaaisrc_class_init (GstSimaaiSrcClass * klass)
{
  GObjectClass *gobject_class = G_OBJECT_CLASS (klass);
  GstElementClass *gstelement_class = GST_ELEMENT_CLASS (klass);
  GstBaseSrcClass *gstbasesrc_class = GST_BASE_SRC_CLASS (klass);

  gobject_class->set_property = gst_simaaisrc_set_property;
  gobject_class->get_property = gst_simaaisrc_get_property;
  gobject_class->finalize = gst_simaaisrc_class_finalize;

  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&srctemplate));

  g_object_class_install_property (gobject_class, PROP_LOCATION,
                                   g_param_spec_string ("location", "Location",
                                                        "location", NULL,
                                                        GParamFlags(G_PARAM_READWRITE |
                                                                    G_PARAM_STATIC_STRINGS |
                                                                    GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class, PROP_INDEX,
      g_param_spec_int ("index", "File Index",
          "Index to use with location property to create file names.  The "
          "index is incremented by one for each buffer read.",
          0, INT_MAX, DEFAULT_INDEX,
                        GParamFlags(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_DELAY,
      g_param_spec_int64 ("delay", "Delay Sleep",
          "Sleep Timer"
          "Sleep Time",
          0, INT64_MAX, DEFAULT_INDEX,
                        GParamFlags(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
  
  g_object_class_install_property (gobject_class, PROP_MEM_TARGET,
      g_param_spec_int ("mem-target", "Memory Target",
                        "Memory target for simamemlib",
                        0, 2, 0,
                        GParamFlags(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
  
  g_object_class_install_property (gobject_class, PROP_NODE_NAME,
                                   g_param_spec_string ("node-name", "NodeName",
                                                        "Node Name", NULL,
                                                        GParamFlags(G_PARAM_READWRITE |
                                                                    G_PARAM_STATIC_STRINGS |
                                                                    GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class, PROP_STREAM_ID,
                                   g_param_spec_string ("stream-id", "StreamId",
                                                        "Stream Id", "unknown-stream-id",
                                                        GParamFlags(G_PARAM_READWRITE |
                                                                    G_PARAM_STATIC_STRINGS |
                                                                    GST_PARAM_MUTABLE_READY)));
  
  g_object_class_install_property (gobject_class, PROP_POOL_SIZE,
                                   g_param_spec_uint ("pool-size", "BufferPoolSize",
                                                      "Amount of buffers to allocate in output buffer pool",
                                                      MIN_POOL_SIZE, MAX_POOL_SIZE, MAX_POOL_SIZE,
                                                      GParamFlags(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_LOOP,
                                g_param_spec_boolean ("loop", "Loop",
                                "Loop back to the initial index after reaching the last file",
                                false,  // Default value
                                GParamFlags(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_SEGMENTS,
                                   g_param_spec_string ("segments", "MemorySegments",
                                                      "Comma separated key=value pairs where the key is the segment name and value is the path to associated file",
                                                      NULL,
                                                      GParamFlags(G_PARAM_READWRITE |
                                                                    G_PARAM_STATIC_STRINGS |
                                                                    GST_PARAM_MUTABLE_READY)));

  
  static const gchar *tags[] = { NULL };
  gst_meta_register_custom ("GstSimaMeta", tags, NULL, NULL, NULL);

  gst_element_class_set_static_metadata (gstelement_class,
      "SimaCustomSrc", "Custom Source",
      "Read image and push", "SiMa.Ai");

  gstbasesrc_class->start = gst_simaaisrc_start;
  gstbasesrc_class->stop = gst_simaaisrc_stop;
  gstbasesrc_class->create = gst_simaaisrc_create;
  gstbasesrc_class->decide_allocation = gst_simaai_src_decide_allocation;
  
  GST_DEBUG_CATEGORY_INIT (GST_CAT_DEFAULT,
      "simaaisrc", 0, "SiMa.Ai src");
}

/**
 * @brief initialize simaaisrc element 
 * @param self, gobject instance of type GstSimaaiSrc
 */
static void
gst_simaaisrc_init (GstSimaaiSrc * self)
{
  GstBaseSrc *basesrc = GST_BASE_SRC (self);

  gst_simaai_segment_memory_init_once();
  
  gst_base_src_set_format (basesrc, GST_FORMAT_TIME);
  gst_base_src_set_async (basesrc, FALSE);

  self->priv = new GstSimaaiSrcPrivate;

  self->priv->location = NULL;
  self->priv->node_name = NULL;
  self->priv->stream_id = g_string_new("unknown-stream-id");
  self->priv->pool = NULL;
  self->priv->pool_size = MAX_POOL_SIZE;
  self->priv->frame_id = 1;
  self->priv->mem_size = 0;
  self->priv->index = 0;
  self->priv->initial_index = 0;  // Initialize initial_index
  self->priv->loop = false;       // Initialize loop to false
  self->priv->delay = 0;
  self->priv->segments = NULL;

  self->priv->all_segments.reserve(MAX_MEM_SEGMENT);
}

/**
 * @brief set property callback for gstreamer
 * @param object, gobject instance
 * @param prop_id property id as maintained by gstreamer
 * @param value, value of the property
 * @param paramter specifiers
 */
static void
gst_simaaisrc_set_property (GObject * object, guint prop_id, const GValue * value,
    GParamSpec * pspec)
{
  GstSimaaiSrc *self = GST_SIMAAISRC (object);

  switch (prop_id) {
    case PROP_LOCATION:
      self->priv->location = g_strdup (g_value_get_string(value));
      break;
    case PROP_INDEX:
      self->priv->index = g_value_get_int (value);
      self->priv->initial_index = self->priv->index;  // Store initial index
      break;
    case PROP_DELAY:
      self->priv->delay = g_value_get_int64 (value);
      break;
    case PROP_MEM_TARGET:
      self->priv->mem_target = g_value_get_int (value);
      break;
    case PROP_NODE_NAME:
      self->priv->node_name = g_strdup (g_value_get_string(value));
      break;
    case PROP_STREAM_ID:
      self->priv->stream_id->str = g_strdup (g_value_get_string(value));
      break;
    case PROP_POOL_SIZE:
      self->priv->pool_size = g_value_get_uint (value);
      break;
    case PROP_LOOP:
      self->priv->loop = g_value_get_boolean (value);
      break;
    case PROP_SEGMENTS:
      self->priv->segments = g_strdup(g_value_get_string(value));
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

/**
 * @brief get property callback for gstreamer
 * @param object, gobject instance
 * @param prop_id property id maitained by gstreamer
 * @param value, value of the property
 * @param paramter specifiers
 */
static void
gst_simaaisrc_get_property (GObject * object, guint prop_id, GValue * value,
    GParamSpec * pspec)
{
  GstSimaaiSrc *self = GST_SIMAAISRC (object);

  switch (prop_id) {
    case PROP_LOCATION:
      g_value_set_string (value, self->priv->location);
      break;
    case PROP_NODE_NAME:
      g_value_set_string (value, self->priv->node_name);
      break;
    case PROP_INDEX:
      g_value_set_int (value, self->priv->index);
      break;
    case PROP_DELAY:
      g_value_set_int64 (value, self->priv->delay);
      break;
    case PROP_MEM_TARGET:
      g_value_set_int (value, self->priv->mem_target);
      break;
    case PROP_STREAM_ID:
      g_value_set_string (value, self->priv->stream_id->str);
      break;
    case PROP_POOL_SIZE:
      g_value_set_uint (value, self->priv->pool_size);
      break;
    case PROP_LOOP:
      g_value_set_boolean (value, self->priv->loop);
      break;
    case PROP_SEGMENTS:
      g_value_set_string(value, self->priv->segments);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

/**
 * @brief finalize the object
 * @param gobject to be finalized
 */
static void
gst_simaaisrc_class_finalize (GObject * object)
{
  GstSimaaiSrc *self = GST_SIMAAISRC (object);
  int8_t * data_h;

  g_free (self->priv->location);
  g_free (self->priv->node_name);
  g_string_free (self->priv->stream_id, TRUE);
  g_free (self->priv);

  G_OBJECT_CLASS (parent_class)->finalize (object);
}

/*
* Function to parse string that contains a comma-separated pairs with
* the segment name and file path associated with that data
* For example: hm_tensor=/data/hm_tensor.in,paf_tensor=/data/paf_tensor.in 
*/
gboolean gst_simaaisrc_parse_segments(GstSimaaiSrc * self)
{
  guint rank = 0;
  guint64 val;
  gchar *segment_string;
  gchar ** pipeline_str;

  if (self->priv->segments == NULL)
    return FALSE;

  segment_string = g_strdup(self->priv->segments);
  g_strstrip(segment_string);

  pipeline_str = g_strsplit(segment_string, ",", MAX_MEM_SEGMENT);
  gint num_of_pairs = g_strv_length(pipeline_str);
  if (num_of_pairs > MAX_MEM_SEGMENT) {
    GST_ERROR("Unsupported number of segments,"
              "max profiles supported in this version is %d", MAX_MEM_SEGMENT);
    g_strfreev(pipeline_str);
    g_free(segment_string);
    return FALSE;
  }

  self->priv->all_segments.reserve(num_of_pairs);

  for (int i = 0; i < num_of_pairs; i++) {
    if (pipeline_str[i] != NULL) {
      g_message("pipeline_str[%d] = %s", i, pipeline_str[i]);
      gchar *pair_str = g_strdup(pipeline_str[i]);
      gchar **pair = g_strsplit(pair_str, "=", 2);
      gint num_of_tokens = g_strv_length(pipeline_str);
      if (num_of_tokens != 2) {
        GST_ERROR("Can't parse a pair: %s", pair_str);
        g_free(pair_str);
        g_strfreev(pair);

        return FALSE;
      }

      std::string name(pair[0]);
      std::string location(pair[1]);
      size_t size = get_file_size(location.c_str());

      SegmentInfo info = {name, location, size};

      self->priv->all_segments.push_back(info);
      
      GST_DEBUG_OBJECT(self, "Updating segment info: name: %s location %s size %lu",
        name.c_str(),
        location.c_str(),
        size);

      g_free(pair_str);
      g_strfreev(pair);
    }
  }

  g_strfreev(pipeline_str);
  g_free(segment_string);
  return TRUE;
}

static glong gst_simaaisrc_get_alloc_size (GstSimaaiSrc * self) {
  gchar * filename = gst_multi_simaaisrc_get_filename (self);
  g_message("Filename memalloc = %s", filename);
  FILE * fp = fopen(filename, "r");
  if (fp == NULL) {
    printf("File Not Found!\n");
    return -1;
  }
  
  fseek(fp, 0L, SEEK_END);
  long int res = ftell(fp);
  fclose(fp);
  return res;
}

/**
 * @brief helper function to get memory target
 * @todo move to libgstsimautils.so
 */
static GstSimaaiMemoryFlags get_simamem_target (int mem_target) {
    switch(mem_target) {
    case 0:
        return GST_SIMAAI_MEMORY_TARGET_GENERIC;
    case 1:
        return GST_SIMAAI_MEMORY_TARGET_EV74;
      case 2:
        return GST_SIMAAI_MEMORY_TARGET_DMS0;
    default:
        return GST_SIMAAI_MEMORY_TARGET_GENERIC;
    }
}

glong get_file_size(const gchar * path)
{
  FILE * fp = fopen(path, "r");
  if (fp == NULL) {
    printf("File Not Found!\n");
    return -1;
  }
  
  fseek(fp, 0L, SEEK_END);
  long int res = ftell(fp);
  fclose(fp);
  return res;
}

static gboolean contains_format_specifier(const char* str)
{
  while (*str) {
    if (*str == '%' && *(str + 1)) {
      char next = *(str + 1);
      if (strchr("diufFeEgGxXoscpaAn%", next)) {
        return TRUE;
      }
    }
    str++;
  }

  return FALSE;
}

/**
 * @brief start simaaisrc, called when state changed null to ready
 * @param basesrc object of type GstBaseSrc
 */
static gboolean
gst_simaaisrc_start (GstBaseSrc * basesrc)
{
  GstSimaaiSrc *self = GST_SIMAAISRC (basesrc);

  if (self->priv->segments && self->priv->location) {
    GST_ERROR("Can't use both: 'segments' and 'location' properties at the same time!"
              " You should use only one of these properties at the same time.");
    return FALSE;
  }

  gsize mem_size = gst_simaaisrc_get_alloc_size(self);

  if (mem_size <= 0)
    return FALSE;

  self->priv->mem_size = mem_size;
  GstMemoryFlags flags = static_cast<GstMemoryFlags>(get_simamem_target(self->priv->mem_target)
                                                     | GST_SIMAAI_MEMORY_FLAG_CACHED);

  if (self->priv->segments != NULL) {
    GST_DEBUG_OBJECT(self, "Allocating segments instead of entire buffer."
                           "Property 'location' will be ignored.");
    if (!gst_simaaisrc_parse_segments(self)) {
      GST_ERROR("Failed to parse 'segments' property!");
      return FALSE;
    } else {
      
      std::vector<char*> cstrings;
      std::vector<gsize> segment_sizes;
      cstrings.reserve(self->priv->all_segments.size() + 1);
      segment_sizes.reserve(self->priv->all_segments.size());

      self->priv->mem_size = 0;
      for (auto &it : self->priv->all_segments) {
        
        cstrings.push_back(&it.name[0]);
        segment_sizes.push_back(it.size);

        self->priv->mem_size += it.size;

        GST_DEBUG_OBJECT(self, "Adding segment: %s %lu", it.name.c_str(), it.size);
      }

      cstrings.push_back(nullptr);

      // Create a pool with the given segments
      self->priv->pool = gst_simaai_allocate_buffer_pool2(GST_OBJECT(self),
                                                          gst_simaai_memory_get_segment_allocator(),
                                                          self->priv->pool_size,
                                                          self->priv->pool_size,
                                                          flags,
                                                          self->priv->all_segments.size(),
                                                          segment_sizes.data(),
                                                          const_cast<const char**>(cstrings.data()));
    }
  } else {
    self->priv->pool = gst_simaai_allocate_buffer_pool(GST_OBJECT (self),
                                                      gst_simaai_memory_get_segment_allocator(),
                                                      self->priv->mem_size,
                                                      self->priv->pool_size,
                                                      self->priv->pool_size,
                                                      flags);
  }

  if (self->priv->pool == NULL)
    return FALSE;

  GST_DEBUG_OBJECT (self, "Output buffer pool: %u buffers of size %zu, target 0x%08" PRIx32,
                    self->priv->pool_size, self->priv->mem_size, flags);
  self->priv->max_index = self->priv->initial_index - 1;
  int idx = self->priv->initial_index;

  // Check series input or single input
  if (self->priv->location && !contains_format_specifier(self->priv->location)) {
    GST_INFO_OBJECT(self, "A single input file is used: %s", self->priv->location);
    return TRUE;
  }

  if (!self->priv->loop)
    return TRUE;

  // Check total files if loop=true
  while (TRUE) {
    gchar *filename = g_strdup_printf(self->priv->location, idx);
    if (g_file_test(filename, G_FILE_TEST_EXISTS)) {
      self->priv->max_index = idx;
      idx++;
      g_free(filename);
    } else {
      g_free(filename);
      break;
    }
  }
  if (self->priv->max_index < self->priv->initial_index) {
    GST_ERROR_OBJECT(self, "No files found matching the location pattern.");
    return FALSE;
  }
  GST_INFO_OBJECT(self, "Max index determined: %d", self->priv->max_index);
  return TRUE;
}

static gboolean
gst_simaaisrc_stop (GstBaseSrc * basesrc)
{
  GstSimaaiSrc * self = GST_SIMAAISRC (basesrc);

  if (!free_simaai_memory_buffer_pool(self->priv->pool))
    return FALSE;

  self->priv->pool = NULL;

  return TRUE;
}

static gchar *
gst_multi_simaaisrc_get_filename (GstSimaaiSrc * self)
{
  gchar *filename;

  if (self->priv->index != 0) {
    GST_DEBUG ("%d", self->priv->index);
    return g_strdup_printf (self->priv->location, self->priv->index);
  }
  return self->priv->location;
}

/* msleep(): Sleep for the requested number of milliseconds. */
static int msleep(long msec)
{
    struct timespec ts;
    int res;

    if (msec < 0)
    {
        errno = EINVAL;
        return -1;
    }

    ts.tv_sec = msec / 1000;
    ts.tv_nsec = (msec % 1000) * 1000000;

    do {
        res = nanosleep(&ts, &ts);
    } while (res && errno == EINTR);

    return res;
}

/**
 * @brief Create a buffer containing the subscribed data
 * @param[in] basesrc object of type GstBaseSrc
 * @param[in] offset buffer offset to be used
 * @param[in] size of the output buffer
 * @param[out] out_buf to be allocated
 */
static GstFlowReturn
gst_simaaisrc_create (GstBaseSrc * basesrc, guint64 offset, guint size,
    GstBuffer ** out_buf)
{
  GstSimaaiSrc *self = GST_SIMAAISRC (basesrc);

  size_t _offset = 0;

  UNUSED (offset);
  UNUSED (size);

  gsize bytes_ready = 0;
  if (self->priv->delay > 0)
    msleep(self->priv->delay);

  GstBuffer * buffer;
  GstFlowReturn ret = gst_buffer_pool_acquire_buffer(self->priv->pool, &buffer, NULL);

  if (G_LIKELY (ret == GST_FLOW_OK)) {
    GST_DEBUG_OBJECT (self, "Output buffer from pool: %p", buffer);
  } else {
    GST_ERROR_OBJECT (self, "Failed to allocate buffer");
    return ret;
  }

  GstMapInfo map;
  GstMemory *mem = gst_buffer_peek_memory(buffer, 0);
  gst_memory_map(mem, &map, GST_MAP_WRITE);

  if (map.data != NULL) {
    GST_DEBUG_OBJECT(self, "Mapping output buffer with address %p and size %zu", map.data, map.size);

    guint offset = 0;

    if (self->priv->segments != NULL) {
      for (auto segment_info: self->priv->all_segments) {
        FILE * fp;
        fp = fopen(segment_info.path.c_str(), "r");

        if (fp == NULL) {
          GST_ERROR_OBJECT(self, "Unable to read data from %s", segment_info.path.c_str());

          gst_memory_unmap(mem, &map);
          gst_buffer_unref(buffer);
          return GST_FLOW_ERROR;
        }

        gsize sz = fread((void *)(map.data) + offset, 1, segment_info.size, fp);
        if (sz != segment_info.size) {
          GST_ERROR_OBJECT(self, "expected size %ld, but read: %ld", segment_info.size, sz);
          gst_memory_unmap(mem, &map);
          gst_buffer_unref(buffer);
          return GST_FLOW_ERROR;
        }
        fclose(fp);
        offset += segment_info.size;
    }
    } else {
      gchar * filename = gst_multi_simaaisrc_get_filename (self);
      FILE * fp;
      fp = fopen(filename, "r");

      if (fp == NULL) {
        GST_ERROR_OBJECT(self, "Unable to read data from %s", filename);
        gst_memory_unmap(mem, &map);
        gst_buffer_unref(buffer);
        return GST_FLOW_ERROR;
      }

      gsize sz = fread((void *)(map.data), 1, map.size, fp);
      if (sz != self->priv->mem_size) {
        GST_ERROR_OBJECT(self, "expected size %ld, but read: %ld", map.size, sz);
        gst_memory_unmap(mem, &map);
        gst_buffer_unref(buffer);
        return GST_FLOW_ERROR;
      }
      fclose(fp);
    }
  }
  else {
    g_message("data_h is NULL");
    gst_memory_unmap(mem, &map);
    gst_buffer_unref(buffer);
    return GST_FLOW_ERROR;
  }

  /* The output buffer custom metadata fields */
  gint64 buf_id = 0;
  gint64 frame_id = 0;
  gint64 buf_offset = 0;
  gchar * buf_name = NULL;

  GstClock *clock = GST_ELEMENT_CLOCK(self);
  GstClockTime current_time = 0;
  if (!clock) {
    GST_WARNING_OBJECT(self, "Can't get GstClock for %s", self->priv->node_name);
  } else {
    current_time = gst_clock_get_time(clock);
  }

  GstCustomMeta * meta = gst_buffer_add_custom_meta(buffer, "GstSimaMeta");
  if (meta == NULL) {
    GST_ERROR ("SIMAAISRC:Unable to add metadata info to the buffer");
    //gst_buffer_unmap(buffer, &map);
    gst_memory_unmap(mem, &map);
    gst_buffer_unref(buffer);
    return GST_FLOW_ERROR;
  }

  GstStructure *s = gst_custom_meta_get_structure (meta);
  if (s != NULL) {
    buf_id = gst_simaai_memory_get_phys_addr(map.memory);
    buf_name = self->priv->node_name;
    buf_offset = 0;
    frame_id = self->priv->frame_id++;
    gst_structure_set (s,
                       "buffer-id", G_TYPE_INT64, buf_id,
                       "buffer-name", G_TYPE_STRING, buf_name,
                       "buffer-offset", G_TYPE_INT64, buf_offset,
                       "frame-id", G_TYPE_INT64, frame_id,
                       "stream-id", G_TYPE_STRING, self->priv->stream_id->str,
                       "timestamp", G_TYPE_UINT64, current_time, NULL);
  } else {
       GST_ERROR("SIMAAISRC: Unable to add metadata to the buffer");
       gst_memory_unmap(mem, &map);
       gst_buffer_unref(buffer);
       return GST_FLOW_ERROR;
  }
  
  gst_memory_unmap(mem, &map);
  
  *out_buf = buffer;
  if (self->priv->index != 0)
    self->priv->index++;

  // Check for looping
  if (self->priv->loop) {
    if (self->priv->index > self->priv->max_index) {
      self->priv->index = self->priv->initial_index;
    }
  }

  /* Print out output buffer's custom metadata */
  GST_DEBUG_OBJECT(self, "buffer-name[%s] buffer-offset[%ld] buffer-id[%ld] frame-id[%ld]",
      buf_name, buf_offset, buf_id, frame_id);

  return GST_FLOW_OK;
}

static gboolean gst_simaai_src_decide_allocation (GstBaseSrc * src, GstQuery * query)
{
  GstSimaaiSrc *self = GST_SIMAAISRC (src);
  GST_DEBUG_OBJECT(self, "decide_allocation called");

  GstSimaaiMemoryFlags mem_type;
  GstSimaaiMemoryFlags mem_flag;

  if (!gst_simaai_allocation_query_parse(query, &mem_type, &mem_flag)) {
    GST_WARNING_OBJECT(self, "Can't find allocation meta!");
  };

  GST_DEBUG_OBJECT(self, "Memory flags to allocate: [ %s ] [ %s ]",
    gst_simaai_allocation_query_sima_mem_type_to_str(mem_type),
    gst_simaai_allocation_query_sima_mem_flag_to_str(mem_flag));

  return TRUE;
}

static gboolean
plugin_init (GstPlugin *plugin)
{
  if (!gst_element_register(plugin, "simaaisrc", GST_RANK_NONE,
                            GST_TYPE_SIMAAISRC)) {
    GST_ERROR("Unable to register simasrc plugin");
    return FALSE;
  }

  return TRUE;
}

GST_PLUGIN_DEFINE (
  GST_VERSION_MAJOR,
  GST_VERSION_MINOR,
  PLUGIN_NAME_LOWER,
  "SiMa Custom Src",
  plugin_init,
  VERSION,
  "unknown",
  "GStreamer",
  "SiMa.ai"
)
