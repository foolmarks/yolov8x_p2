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

#include <gst/gst.h>
#include <inttypes.h>
#include <simaai/simaai_memory.h>

#include <cstddef>
#include <mutex>
#include <new>
#include <string>
#include <vector>

#include "gstsimaaisegmentallocator.h"

GST_DEBUG_CATEGORY_STATIC (gst_simaai_segment_allocator2_debug);
#define GST_CAT_DEFAULT gst_simaai_segment_allocator2_debug

#define gst_simaai_segment_allocator2_parent_class parent_class
G_DEFINE_TYPE_WITH_CODE (GstSimaaiSegmentAllocator, gst_simaai_segment_allocator2, GST_TYPE_ALLOCATOR,
                         GST_DEBUG_CATEGORY_INIT (GST_CAT_DEFAULT, "simaai-allocator-segment", 0,
                           "Simaai Segment Memory Allocator"));

/**
 * @brief SiMa memory segment handle
 */
struct segment {
  simaai_memory_t *memory;
  std::string name;
};

/**
 * @brief Custom memory structure. Extends the GstMemory
 */
struct GstSimaaiSegmentMemory {
  GstMemory mem;
  std::vector<segment> segments; ///< SiMa memory segment handles used by simamemlib
  simaai_memory_t **alloc_segments;
  gpointer vaddr;          ///< SiMa memory block virtual address after mapping
  std::mutex map_mutex;    ///< Concurrent map/unmap protection
};

#define GST_SIMAAI_SEGMENT_MEMORY_CAST(mem)     ((GstSimaaiSegmentMemory *)(mem))

#define GST_SIMAAI_ALLOCATION_PARAMS_CAST(params) ((GstSimaaiAllocationParams *)(params))

/**
 * @brief Map simaai memory buffer
 *
 * The memory buffer is mapped once and the same virtual address (stored in the
 * custom memory structure) can be reused in the given Linux process context
 * until the memory is destroyed.
 */
static gpointer
mem_map_full (GstMemory *memory, GstMapInfo *info, gsize maxsize)
{
  GstSimaaiSegmentMemory *mem = (GstSimaaiSegmentMemory *)memory;
  const std::lock_guard<std::mutex> guard(mem->map_mutex);

  GST_DEBUG_OBJECT(memory->allocator, "Map virt memory: %p with size=%zu", mem->vaddr, maxsize);

  if ((info->flags & GST_MAP_READ) &&
      (GST_MEMORY_FLAG_IS_SET(memory, GST_SIMAAI_MEMORY_FLAG_CACHED))) {
    GST_DEBUG_OBJECT(memory->allocator, "simaai_memory_invalidate_cache at %p",
                     mem->vaddr);
    simaai_memory_invalidate_cache(mem->segments[0].memory);
  }

  return mem->vaddr;
}

/**
 * @brief Unmap simaai memory buffer
 *
 * The memory will be unmapped when the simaai memory buffer is freed.
 * Once mapped the same virtual address (stored in the custom memory structure)
 * can be reused in the given Linux process context until the memory is destroyed.
 */
static void
mem_unmap_full (GstMemory *memory, GstMapInfo *info)
{
  GstSimaaiSegmentMemory *mem = (GstSimaaiSegmentMemory *)memory;
  const std::lock_guard<std::mutex> guard(mem->map_mutex);

  if ((info->flags & GST_MAP_WRITE) &&
      (GST_MEMORY_FLAG_IS_SET(memory, GST_SIMAAI_MEMORY_FLAG_CACHED))) {
    GST_DEBUG_OBJECT(memory->allocator, "simaai_memory_flush_cache at %p",
                     mem->vaddr);
    simaai_memory_flush_cache(mem->segments[0].memory);
  }

  GST_DEBUG_OBJECT(memory->allocator, "Unmap virt memory: %p with size=%zu", mem->vaddr, info->size);
}

static GstMemory*
gst_simaai_segment_allocator2_alloc (GstAllocator *allocator, gsize size,
                             GstAllocationParams *params)
{
  (void)size;

  GstSimaaiAllocationParams *alloc_params = GST_SIMAAI_ALLOCATION_PARAMS_CAST(params);

  if (alloc_params->num_of_segments < 1 || alloc_params->num_of_segments > MAX_ALLOCATION_SEGMENTS) {
    GST_ERROR_OBJECT (allocator, "ERROR: bad num_of_segments value: %zu",
                      alloc_params->num_of_segments);
    return NULL;
  }

  gsize total_size = 0;
  for (size_t i = 0; i < alloc_params->num_of_segments; i++)
    total_size += alloc_params->segments[i].size;

  gsize maxsize = total_size + params->prefix + params->padding;

  GstSimaaiSegmentMemory *mem = new(std::nothrow) GstSimaaiSegmentMemory;
  if (mem == nullptr) {
    GST_ERROR_OBJECT (allocator, "ERROR: allocating GstSimaaiSegmentMemory");
    return NULL;
  }

  gst_memory_init (GST_MEMORY_CAST (mem), params->flags, allocator, nullptr,
                   maxsize, params->align, params->prefix, total_size);

  // Set memory target region.
  // If several flags are set then the last one is applied.
  int target{SIMAAI_MEM_TARGET_GENERIC};

  if (GST_MEMORY_FLAG_IS_SET (GST_MEMORY_CAST(mem), GST_SIMAAI_MEMORY_TARGET_OCM))
    target = SIMAAI_MEM_TARGET_OCM;

  if (GST_MEMORY_FLAG_IS_SET (GST_MEMORY_CAST(mem), GST_SIMAAI_MEMORY_TARGET_DMS0))
    target = SIMAAI_MEM_TARGET_DMS0;

  if (GST_MEMORY_FLAG_IS_SET (GST_MEMORY_CAST(mem), GST_SIMAAI_MEMORY_TARGET_DMS1))
    target = SIMAAI_MEM_TARGET_DMS1;

  if (GST_MEMORY_FLAG_IS_SET (GST_MEMORY_CAST(mem), GST_SIMAAI_MEMORY_TARGET_DMS2))
    target = SIMAAI_MEM_TARGET_DMS2;

  if (GST_MEMORY_FLAG_IS_SET (GST_MEMORY_CAST(mem), GST_SIMAAI_MEMORY_TARGET_DMS3))
    target = SIMAAI_MEM_TARGET_DMS3;

  if (GST_MEMORY_FLAG_IS_SET (GST_MEMORY_CAST(mem), GST_SIMAAI_MEMORY_TARGET_EV74))
    target = SIMAAI_MEM_TARGET_EV74;

  // Set memory region flags
  // If several flags are set then the last one is applied.
  int flags{SIMAAI_MEM_FLAG_DEFAULT};

  if (GST_MEMORY_FLAG_IS_SET (GST_MEMORY_CAST(mem), GST_SIMAAI_MEMORY_FLAG_CACHED))
    flags = SIMAAI_MEM_FLAG_CACHED;

  if (GST_MEMORY_FLAG_IS_SET (GST_MEMORY_CAST(mem), GST_SIMAAI_MEMORY_FLAG_RDONLY))
    flags = SIMAAI_MEM_FLAG_RDONLY;

  // Allocate simaai memory segments
  uint32_t segments[MAX_ALLOCATION_SEGMENTS];
  memset(segments, 0, sizeof(segments));

  for (size_t i = 0; i < alloc_params->num_of_segments; i++)
    segments[i] = alloc_params->segments[i].size;

  mem->alloc_segments = simaai_memory_alloc_segments_flags(segments,
                                                           alloc_params->num_of_segments,
                                                           target, flags);

  if (!mem->alloc_segments) {
    GST_ERROR_OBJECT (allocator, "ERROR: allocating segments of memory");
    delete mem;
    return NULL;
  }

  for (size_t i = 0; i < alloc_params->num_of_segments; i++) {
    segment s;
    s.memory = mem->alloc_segments[i];
    s.name = alloc_params->segments[i].name;
    mem->segments.push_back(s);
  }

  // Map the memory of the parent segment once to reuse the same virtual address
  // until the memory is destroyed
  mem->vaddr = simaai_memory_map(mem->segments[0].memory);
  if (!mem->vaddr) {
    GST_ERROR_OBJECT (allocator, "ERROR: mapping contiguous memory");
    simaai_memory_free_segments(mem->alloc_segments, alloc_params->num_of_segments);
    delete mem;
    return NULL;
  }

  for (size_t i = 0; i < mem->segments.size(); i++) {
    GST_DEBUG_OBJECT (allocator, "Allocate memory segment:%zu phys:0x%" PRIx64 " target:0x%08x flags:0x%08x size:%zu",
                      i, simaai_memory_get_phys(mem->segments[i].memory), target, flags, simaai_memory_get_size(mem->segments[i].memory));
  }

  return GST_MEMORY_CAST (mem);
}

static void
gst_simaai_segment_allocator2_free (GstAllocator *allocator, GstMemory *memory)
{
  GstSimaaiSegmentMemory *mem = (GstSimaaiSegmentMemory *)memory;

  GST_DEBUG_OBJECT (allocator, "Free simaai phys memory: 0x%" PRIx64,
                    simaai_memory_get_phys(mem->segments[0].memory));

  simaai_memory_unmap(mem->segments[0].memory);
  simaai_memory_free_segments(mem->alloc_segments, mem->segments.size());
  delete mem;
}

static void gst_simaai_segment_allocator2_finalize(GObject * object)
{
  GST_DEBUG_OBJECT(object, "gst_simaai_segment_allocator2_finalize");

  G_OBJECT_CLASS (parent_class)->finalize (object);
}


static void
gst_simaai_segment_allocator2_class_init (GstSimaaiSegmentAllocatorClass *klass)
{
  GstAllocatorClass *allocator_class = GST_ALLOCATOR_CLASS (klass);
  GObjectClass *gobject_class = G_OBJECT_CLASS(klass);

  allocator_class->alloc = gst_simaai_segment_allocator2_alloc;
  allocator_class->free = gst_simaai_segment_allocator2_free;
  gobject_class->finalize = gst_simaai_segment_allocator2_finalize;
}

static void
gst_simaai_segment_allocator2_init (GstSimaaiSegmentAllocator *allocator)
{
  GstAllocator *alloc = GST_ALLOCATOR (allocator);

  alloc->mem_type = GST_ALLOCATOR_SIMAAI_SEGMENT;
  alloc->mem_map_full = (GstMemoryMapFullFunction) mem_map_full;
  alloc->mem_unmap_full = (GstMemoryUnmapFullFunction) mem_unmap_full;

  GST_OBJECT_FLAG_SET (allocator, GST_ALLOCATOR_FLAG_CUSTOM_ALLOC);
}

void
gst_simaai_segment_memory_init_once (void)
{
  static gsize _init = 0;
  GstAllocator *alloc;

  if (g_once_init_enter (&_init)) {
    alloc = (GstAllocator *)g_object_new (GST_TYPE_SIMAAI_SEGMENT_ALLOCATOR2, NULL);

    /* Clear floating flag */
    gst_object_ref_sink (alloc);

    gst_allocator_register (GST_ALLOCATOR_SIMAAI_SEGMENT, (GstAllocator *)gst_object_ref(alloc));
    GST_DEBUG_OBJECT(alloc, "Initialized %s allocator", GST_ALLOCATOR_SIMAAI_SEGMENT);
    g_once_init_leave (&_init, 1);
  }
}

GstAllocator *
gst_simaai_memory_get_segment_allocator (void)
{
  return gst_allocator_find (GST_ALLOCATOR_SIMAAI_SEGMENT);
}

guintptr
gst_simaai_segment_memory_get_phys_addr (const GstMemory * memory)
{
  if (memory != NULL && memory->allocator != NULL && GST_IS_SIMAAI_SEGMENT_ALLOCATOR2(memory->allocator))
    return simaai_memory_get_phys(GST_SIMAAI_SEGMENT_MEMORY_CAST(memory)->segments[0].memory);
  else
    return 0;
}

void
gst_simaai_memory_allocation_params_init (GstSimaaiAllocationParams * params)
{
  g_return_if_fail (params != NULL);

  memset (params, 0, sizeof (*params));
}

gboolean
gst_simaai_memory_allocation_params_add_segment (GstSimaaiAllocationParams * params,
                                                 const gsize size,
                                                 const gchar * name)
{
  g_return_val_if_fail (params != NULL, FALSE);
  g_return_val_if_fail (params->num_of_segments < MAX_ALLOCATION_SEGMENTS, FALSE);

  params->segments[params->num_of_segments].size = size;
  params->segments[params->num_of_segments].name = name;
  params->num_of_segments++;

  return TRUE;
}

void *
gst_simaai_memory_get_segment (const GstMemory * memory, const gchar * name)
{
  g_return_val_if_fail (GST_IS_SIMAAI_SEGMENT_ALLOCATOR2(memory->allocator), NULL);

  GstSimaaiSegmentMemory *mem = GST_SIMAAI_SEGMENT_MEMORY_CAST(memory);
  simaai_memory_t *memory_t = NULL;

  g_return_val_if_fail (mem != NULL, NULL);
  g_return_val_if_fail (mem->segments.size() > 0, NULL);

  if (name == NULL) {
    memory_t = mem->segments[0].memory;
  } else {
    for (const auto& s : mem->segments) {
      if (s.name == name) {
        memory_t = s.memory;
        break;
      }
    }
  }

  return memory_t;
}
