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

#ifndef GST_SIMAAI_BUFFER_POOL_H
#define GST_SIMAAI_BUFFER_POOL_H

#include <gst/gst.h>
#include <gst/gstbufferpool.h>
#include <gst/gstmemory.h>

#include "gstsimaaiallocator.h"

G_BEGIN_DECLS

#define GST_TYPE_SIMAAI_BUFFER_POOL (gst_simaai_buffer_pool_get_type())
G_DECLARE_FINAL_TYPE (GstSimaaiBufferPool, gst_simaai_buffer_pool,
                      GST, SIMAAI_BUFFER_POOL, GstBufferPool)

//GType gst_simaai_buffer_pool_get_type(void);

#define GST_IS_SIMAAI_BUFFER_POOL(obj)              (G_TYPE_CHECK_INSTANCE_TYPE ((obj), GST_TYPE_SIMAAI_BUFFER_POOL))
#define GST_IS_SIMAAI_BUFFER_POOL_CLASS(klass)      (G_TYPE_CHECK_CLASS_TYPE ((klass), GST_TYPE_SIMAAI_BUFFER_POOL))
#define GST_SIMAAI_BUFFER_POOL_GET_CLASS(obj)       (G_TYPE_INSTANCE_GET_CLASS ((obj), GST_TYPE_SIMAAI_BUFFER_POOL, GstSimaaiBufferPoolClass))
#define GST_SIMAAI_BUFFER_POOL(obj)                 (G_TYPE_CHECK_INSTANCE_CAST ((obj), GST_TYPE_SIMAAI_BUFFER_POOL, GstSimaaiBufferPool))
#define GST_SIMAAI_BUFFER_POOL_CLASS(klass)         (G_TYPE_CHECK_CLASS_CAST ((klass), GST_TYPE_SIMAAI_BUFFER_POOL, GstSimaaiBufferPoolClass))
#define GST_SIMAAI_BUFFER_POOL_CAST(obj)            ((GstSimaaiBufferPool *)(obj))

struct _GstSimaaiBufferPool 
{
  GstBufferPool parent;
  
  /*< public >*/
  /// @brief segments data to allocate
  segment_t segments[MAX_ALLOCATION_SEGMENTS];
  
  /// @brief Number of memories to allocate
  int number_of_segments;
};

/**
 * @brief Creates a new GstSimaaiBufferPool instance.
 *
 * @return (transfer full): a new GstSimaaiBufferPool instance
 */
GstSimaaiBufferPool * gst_simaai_buffer_pool_new (void);

/**
 * gst_simaai_allocate_buffer_pool:
 * @object: the #GstObject parent structure or NULL if none
 * @allocator: the #GstAllocator to allocate memory for buffers in the pool
 * @min_buffers: the minimum amount of buffers to allocate
 * @max_buffers: the maximum amount of buffers to allocate or 0 for unlimited
 * @flags: the #GstMemoryFlags to control allocation
 * @number_of_segments: number of segments in each buffer
 * @segment_sizes: size of each segment
 * @segment_names: name of each segment
 *
 * Allocates a #GstBufferPool with a #GstAllocator provided.
 *
 * Returns: The allocated and activated #GstBufferPool or NULL if failed.
 */
GstBufferPool *gst_simaai_allocate_buffer_pool2(GstObject *object,
                                                GstAllocator *allocator,
                                                guint min_buffers,
                                                guint max_buffers,
                                                GstMemoryFlags flags,
                                                gsize number_of_segments,
                                                const gsize seg_sizes[],
                                                const gchar * seg_names[]);


/**
 * gst_simaai_allocate_buffer_pool:
 * @object: the #GstObject parent structure or NULL if none
 * @allocator: the #GstAllocator to allocate memory for buffers in the pool
 * @buffer_size: the size of each buffer, not including prefix and padding
 * @min_buffers: the minimum amount of buffers to allocate
 * @max_buffers: the maximum amount of buffers to allocate or 0 for unlimited
 * @flags: the #GstMemoryFlags to control allocation
 *
 * Allocates a #GstBufferPool with a #GstAllocator provided.
 *
 * Returns: The allocated and activated #GstBufferPool or NULL if failed.
 */
GstBufferPool *gst_simaai_allocate_buffer_pool(GstObject *object,
                                               GstAllocator *allocator,
                                               guint buffer_size,
                                               guint min_buffers,
                                               guint max_buffers,
                                               GstMemoryFlags flags);

/**
 * gst_simaai_free_buffer_pool:
 * @pool: the #GstBufferPool to free
 *
 * Deactivates and releases the #GstBufferPool.
 *
 * Returns: TRUE if the #GstBufferPool has been deactivated and released.
 */
gboolean gst_simaai_free_buffer_pool(GstBufferPool *pool);

G_END_DECLS

#endif /* GST_SIMAAI_BUFFER_POOL_H */
