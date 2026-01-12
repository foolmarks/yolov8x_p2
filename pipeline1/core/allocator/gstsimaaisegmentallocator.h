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

#ifndef GSTSIMAAISEGMENTALLOCATOR_H_
#define GSTSIMAAISEGMENTALLOCATOR_H_

#include <simaai/simaai_memory.h>

#include <gst/gst.h>
#include <gst/gstallocator.h>
#include <gst/gstmemory.h>

#include "gstsimaaiallocator_common.h"

G_BEGIN_DECLS

#define GST_TYPE_SIMAAI_SEGMENT_ALLOCATOR (gst_simaai_buffer_memory_segment_allocator_get_type())
GType gst_simaai_allocator_get_type(void);

#define GST_IS_SIMAAI_SEGMENT_ALLOCATOR(obj)              (G_TYPE_CHECK_INSTANCE_TYPE ((obj), GST_TYPE_SIMAAI_SEGMENT_ALLOCATOR))
#define GST_IS_SIMAAI_SEGMENT_ALLOCATOR_CLASS(klass)      (G_TYPE_CHECK_CLASS_TYPE ((klass), GST_TYPE_SIMAAI_SEGMENT_ALLOCATOR))
#define GST_SIMAAI_SEGMENT_ALLOCATOR_GET_CLASS(obj)       (G_TYPE_INSTANCE_GET_CLASS ((obj), GST_TYPE_SIMAAI_SEGMENT_ALLOCATOR, GstSimaaiAllocatorClass))
#define GST_SIMAAI_SEGMENT_ALLOCATOR(obj)                 (G_TYPE_CHECK_INSTANCE_CAST ((obj), GST_TYPE_SIMAAI_SEGMENT_ALLOCATOR, GstSimaaiAllocator))
#define GST_SIMAAI_SEGMENT_ALLOCATOR_CLASS(klass)         (G_TYPE_CHECK_CLASS_CAST ((klass), GST_TYPE_SIMAAI_SEGMENT_ALLOCATOR, GstSimaaiAllocatorClass))
#define GST_SIMAAI_SEGMENT_ALLOCATOR_CAST(obj)            ((GstSimaaiSegmentAllocator *)(obj))

typedef struct
{
  GstAllocator parent;
} SimaAiSegmentAllocator;

typedef struct
{
  GstAllocatorClass parent_class;
} SimaAiSegmentAllocatorClass;

GstAllocator *
gst_simaai_memory_get_segment_allocator(void);

void
simaai_target_mem_free(GstMemory * mem);

/**
 * GST_ALLOCATOR_SIMAAI_SEGMENT:
 *
 * The allocator name for the SimaAi memory allocator
 */
#define GST_ALLOCATOR_SIMAAI_SEGMENT    "SimaaiSegmentMemory"

#define GST_TYPE_SIMAAI_SEGMENT_ALLOCATOR2 (gst_simaai_segment_allocator2_get_type())
G_DECLARE_FINAL_TYPE (GstSimaaiSegmentAllocator, gst_simaai_segment_allocator2,
                      GST, SIMAAI_SEGMENT_ALLOCATOR2, GstAllocator)

struct _GstSimaaiSegmentAllocator
{
  GstAllocator parent;
};

typedef struct _GstSimaaiAllocationParams GstSimaaiAllocationParams;

/**
 * MAX_ALLOCATION_SEGMENTS:
 *
 * The maximum amount of Simaai Memory segments could be allocated.
 */
#define MAX_ALLOCATION_SEGMENTS 16

/**
 * segment_t:
 *
 * A structure containing information about specific segment.
 * Using C-style declaration to conform with old dispatcher.c
 * TODO: use C++ style declaration after migration all the pipelines to new dispatcher
 */
typedef struct {
  gsize size;
  const gchar *name;
} segment_t;

/**
 * GstSimaaiAllocationParams:
 *
 * A structure containing Simaai Memory segments allocation parameters.
 * Extends the GstAllocationParams.
 */
struct _GstSimaaiAllocationParams {
  GstAllocationParams parent;
  segment_t segments[MAX_ALLOCATION_SEGMENTS];
  gsize num_of_segments;
};

/**
 * gst_simaai_segment_memory_init_once:
 *
 * Initialize the Simaai Memory allocator. The allocator init is thread safe.
 */
void gst_simaai_segment_memory_init_once (void);

/*
 * gst_simaai_memory_get_segment_allocator:
 *
 * Find a previously registered Simaai Memory allocator.
 *
 * Returns: a pointer to registered Simaai Memory allocator.
 */
GstAllocator * gst_simaai_memory_get_segment_allocator (void);

/**
 * gst_simaai_segment_memory_get_phys_addr:
 * @mem: a #GstMemory
 *
 * Returns: Simaai memory physical address that is backing @mem, or 0 if none.
 */
guintptr gst_simaai_segment_memory_get_phys_addr (const GstMemory * memory);

/**
 * gst_simaai_memory_allocation_params_init:
 * @params: a #GstSimaaiAllocationParams
 *
 * Initialize the Simaai Memory allocation parameters.
 */
void gst_simaai_memory_allocation_params_init (GstSimaaiAllocationParams * params);

/**
 * gst_simaai_memory_allocation_params_add_segment:
 * @param: a #GstSimaaiAllocationParams to add the new segment to
 * @size: a size of the Simaai memory segment
 * @name: a name of the Simaai memory segment
 *
 * Add Simaai Memory segment to the Simaai Memory allocation parameters.
 *
 * Returns: TRUE or FALSE.
 */
gboolean gst_simaai_memory_allocation_params_add_segment (GstSimaaiAllocationParams * params,
                                                          const gsize size,
                                                          const gchar * name);

/**
 * gst_simaai_memory_get_segment:
 * @memory: a #GstMemory
 * @name: a name of the Simaai memory segment to get.
 *        If NULL then to get the parent segment.
 *
 * Get the pointer to the Simaai memory segment by a name.
 *
 * Returns: a pointer to the Simaai memory segment.
 */
void * gst_simaai_memory_get_segment (const GstMemory * memory, const gchar * name);

G_END_DECLS

#endif
