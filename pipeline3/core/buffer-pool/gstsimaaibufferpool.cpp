//**************************************************************************
//||                        SiMa.ai CONFIDENTIAL                          ||
//||   Unpublished Copyright (c) 2022-2024 SiMa.ai, All Rights Reserved.  ||
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

#include "gstsimaaibufferpool.h"

GST_DEBUG_CATEGORY_STATIC (gst_simaai_buffer_pool_debug);
#define GST_CAT_DEFAULT gst_simaai_buffer_pool_debug

#define gst_simaai_buffer_pool_parent_class parent_class
G_DEFINE_TYPE_WITH_CODE (GstSimaaiBufferPool, gst_simaai_buffer_pool, GST_TYPE_BUFFER_POOL,
                         GST_DEBUG_CATEGORY_INIT( GST_CAT_DEFAULT, "simaai-buffer-pool", 0,
                            "Simaai Buffer Pool"));

#define GST_SIMAAI_BUFFER_POOL_LOCK(pool)   (g_rec_mutex_lock(&pool->priv->rec_lock))
#define GST_SIMAAI_BUFFER_POOL_UNLOCK(pool) (g_rec_mutex_unlock(&pool->priv->rec_lock))

/*
 * @brief virtuall function to allocate buffer in buffer pool
 */
static GstFlowReturn gst_simaai_buffer_pool_alloc_buffer (GstBufferPool * pool, 
                                                          GstBuffer ** buffer,
                                                          GstBufferPoolAcquireParams * params)
{
  GstSimaaiBufferPool *buffer_pool = GST_SIMAAI_BUFFER_POOL(pool);

  GstAllocator * allocator;
  GstAllocationParams allocation_params;
  GstStructure * config = gst_buffer_pool_get_config(pool);

  gst_buffer_pool_config_get_allocator(config, &allocator, &allocation_params);

  GstSimaaiAllocationParams sima_params;
  sima_params.parent = allocation_params;
  sima_params.num_of_segments = buffer_pool->number_of_segments;
  gsize total_size = 0;
  for (int i = 0; i < buffer_pool->number_of_segments; i++) {
    sima_params.segments[i] = buffer_pool->segments[i];
    total_size += buffer_pool->segments[i].size;
  }
    

  //GstMemory *mem = gst_allocator_alloc(allocator, 1024, (GstAllocationParams *)(&sima_params));
  //g_assert_true(mem != NULL);

  GST_DEBUG_OBJECT(buffer_pool, "GstSimaaiBufferPool: allocating a buffer with size %zu", total_size);
  *buffer = gst_buffer_new_allocate(allocator, 
                                    total_size, 
                                    (GstAllocationParams *)(&sima_params));

  if (!*buffer)
    return GST_FLOW_ERROR;

  return GST_FLOW_OK;
}

/*
 * @brief virtuall function to initialize GstSimaaiBufferPool class
 */
static void gst_simaai_buffer_pool_class_init (GstSimaaiBufferPoolClass *klass)
{
  GstBufferPoolClass *parent_klass = GST_BUFFER_POOL_CLASS (klass);

  parent_klass->alloc_buffer = gst_simaai_buffer_pool_alloc_buffer;
}

/*
 * @brief virtuall function to initialize GstSimaaiBufferPool
 */
static void gst_simaai_buffer_pool_init (GstSimaaiBufferPool *pool)
{
  pool->number_of_segments = 0;
  for (int i = 0; i < MAX_ALLOCATION_SEGMENTS; i++) {
    pool->segments[i].size = 0;
    pool->segments[i].name = NULL;
  }
}

/**
 * @brief Creates a new GstSimaaiBufferPool instance.
 *
 * @return (transfer full): a new GstSimaaiBufferPool instance
 */
GstSimaaiBufferPool * gst_simaai_buffer_pool_new (void)
{
  GstSimaaiBufferPool *result;

  result = (GstSimaaiBufferPool *) g_object_new (GST_TYPE_SIMAAI_BUFFER_POOL, NULL);
  GST_DEBUG_OBJECT (result, "created new buffer pool");

  // Clear floating flag 
  gst_object_ref_sink (result);

  return result;
}

GstBufferPool *gst_simaai_allocate_buffer_pool2(GstObject *object,
                                                GstAllocator *allocator,
                                                guint min_buffers,
                                                guint max_buffers,
                                                GstMemoryFlags flags,
                                                gsize number_of_segments,
                                                const gsize seg_sizes[],
                                                const gchar * seg_names[])
{
  GstSimaaiBufferPool *pool = gst_simaai_buffer_pool_new ();
  GstBufferPool * pool_parent = (GstBufferPool *) pool;

  if (pool == NULL) {
    GST_ERROR_OBJECT (object, "gst_buffer_pool_new failed");
    return NULL;
  }

  GstStructure *config = gst_buffer_pool_get_config (pool_parent);
  if (config == NULL) {
    GST_ERROR_OBJECT (object, "gst_buffer_pool_get_config failed");
    gst_object_unref (pool);
    return NULL;
  }

  guint buffer_size = 0;
  GstAllocationParams params;
  gst_allocation_params_init (&params);
  params.flags = flags;
  
  pool->number_of_segments = number_of_segments;
  for (int i = 0; i < pool->number_of_segments; i++) {
    pool->segments[i].size = seg_sizes[i];
    pool->segments[i].name = seg_names[i];
    buffer_size += seg_sizes[i];
    GST_DEBUG_OBJECT(object, "buffer_pool: %s with size %zu", pool->segments[i].name, pool->segments[i].size);
  }

  gst_buffer_pool_config_set_params(config, 
                                    NULL, 
                                    buffer_size, 
                                    min_buffers, 
                                    max_buffers);

  gst_buffer_pool_config_set_allocator (config, 
                                        allocator, 
                                        (GstAllocationParams *)(&params));

  gboolean res = gst_buffer_pool_set_config (pool_parent, config);
  if (res == FALSE) {
    GST_ERROR_OBJECT (object, "gst_buffer_pool_set_config failed");
    gst_object_unref (pool);
    return NULL;
  }

  res = gst_buffer_pool_set_active (pool_parent, TRUE);
  if (res == FALSE) {
    GST_ERROR_OBJECT (object, "gst_buffer_pool_set_active failed");
    gst_object_unref (pool);
    return NULL;
  }

  return pool_parent;
}

GstBufferPool *gst_simaai_allocate_buffer_pool(GstObject *object,
                                               GstAllocator *allocator,
                                               guint buffer_size,
                                               guint min_buffers,
                                               guint max_buffers,
                                               GstMemoryFlags flags)
{
  GstSimaaiBufferPool *pool = gst_simaai_buffer_pool_new ();
  GstBufferPool * pool_parent = (GstBufferPool *) pool;

  if (pool == NULL) {
    GST_ERROR_OBJECT (object, "gst_buffer_pool_new failed");
    return NULL;
  }

  GstStructure *config = gst_buffer_pool_get_config (pool_parent);
  if (config == NULL) {
    GST_ERROR_OBJECT (object, "gst_buffer_pool_get_config failed");
    gst_object_unref (pool);
    return NULL;
  }

  GstAllocationParams params;
  gst_allocation_params_init (&params);
  params.flags = flags;

  gst_buffer_pool_config_set_params(config, 
                                    NULL, 
                                    buffer_size, 
                                    min_buffers, 
                                    max_buffers);

  gst_buffer_pool_config_set_allocator (config, 
                                        allocator, 
                                        (GstAllocationParams *)(&params));

  pool->segments[0].size = buffer_size;
  pool->segments[0].name = "parent";
  pool->number_of_segments = 1;
  

  gboolean res = gst_buffer_pool_set_config (pool_parent, config);
  if (res == FALSE) {
    GST_ERROR_OBJECT (object, "gst_buffer_pool_set_config failed");
    gst_object_unref (pool);
    return NULL;
  }

  res = gst_buffer_pool_set_active (pool_parent, TRUE);
  if (res == FALSE) {
    GST_ERROR_OBJECT (object, "gst_buffer_pool_set_active failed");
    gst_object_unref (pool);
    return NULL;
  }

  return pool_parent;
}

gboolean gst_simaai_free_buffer_pool(GstBufferPool *pool)
{
  g_return_val_if_fail (pool != NULL, FALSE);
  g_return_val_if_fail (gst_buffer_pool_set_active (pool, FALSE), FALSE);
  gst_object_unref (pool);

  return TRUE;
}
