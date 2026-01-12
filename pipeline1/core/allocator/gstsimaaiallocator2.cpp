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

#include <mutex>

#include "gstsimaaiallocator_common.h"
#include "gstsimaaiallocator_old.h"

GST_DEBUG_CATEGORY_STATIC (gst_simaai_allocator2_debug);
#define GST_CAT_DEFAULT gst_simaai_allocator2_debug

#define gst_simaai_allocator2_parent_class parent_class
G_DEFINE_TYPE_WITH_CODE (GstSimaaiAllocator, gst_simaai_allocator2, GST_TYPE_ALLOCATOR,
                         GST_DEBUG_CATEGORY_INIT (GST_CAT_DEFAULT, "simaai-allocator", 0,
                           "Simaai Memory Allocator"));

/**
 * @brief Custom memory structure. Extends the GstMemory
 */
typedef struct _GstSimaaiMemory {
  GstMemory mem;
  simaai_memory_t *memory; ///< SiMa memory handle used by simamemlib
  gpointer vaddr;          ///< SiMa memory block virtual address after mapping
  std::mutex map_mutex;    ///< Concurrent map/unmap protection
} GstSimaaiMemory;

#define GST_SIMAAI_MEMORY_CAST(mem)     ((GstSimaaiMemory *)(mem))

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
  GstSimaaiMemory *mem = (GstSimaaiMemory *)memory;
  const std::lock_guard<std::mutex> guard(mem->map_mutex);

  GST_DEBUG_OBJECT(memory->allocator, "Map virt memory: %p", mem->vaddr);

  if ((info->flags & GST_MAP_READ) &&
      (GST_MEMORY_FLAG_IS_SET(memory, GST_SIMAAI_MEMORY_FLAG_CACHED))) {
    GST_DEBUG_OBJECT(memory->allocator, "simaai_memory_invalidate_cache at %p",
                     mem->vaddr);
    simaai_memory_invalidate_cache(mem->memory);
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
  GstSimaaiMemory *mem = (GstSimaaiMemory *)memory;
  const std::lock_guard<std::mutex> guard(mem->map_mutex);

  if ((info->flags & GST_MAP_WRITE) &&
      (GST_MEMORY_FLAG_IS_SET(memory, GST_SIMAAI_MEMORY_FLAG_CACHED))) {
    GST_DEBUG_OBJECT(memory->allocator, "simaai_memory_flush_cache at %p",
                     mem->vaddr);
    simaai_memory_flush_cache(mem->memory);
  }

  GST_DEBUG_OBJECT(memory->allocator, "Unmap virt memory: %p", mem->vaddr);
}

static GstMemory*
gst_simaai_allocator2_alloc (GstAllocator *allocator, gsize size,
                             GstAllocationParams *params)
{
  GstSimaaiMemory *mem = (GstSimaaiMemory *)g_slice_alloc0(sizeof(GstSimaaiMemory));
  gsize maxsize = size + params->prefix + params->padding;

  gst_memory_init (GST_MEMORY_CAST (mem), params->flags, allocator, nullptr,
                   maxsize, params->align, params->prefix, size);

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

  // Allocate simaai memory region
  mem->memory = simaai_memory_alloc_flags(size, target, flags);
  if (!mem->memory) {
    GST_ERROR_OBJECT (allocator, "ERROR: allocating contiguous memory");
    g_slice_free1(sizeof(GstSimaaiMemory), mem);
    return NULL;
  }

  // Map the memory once to reuse the same virtual address
  // until the memory is destroyed
  mem->vaddr = simaai_memory_map(mem->memory);
  if (!mem->vaddr) {
    GST_ERROR_OBJECT (allocator, "ERROR: mapping contiguous memory");
    simaai_memory_free(mem->memory);
    g_slice_free1(sizeof(GstSimaaiMemory), mem);
    return NULL;
  }

  GST_DEBUG_OBJECT (allocator, "Allocate simaai phys memory: 0x%" PRIx64 " target: 0x%08x flags: 0x%08x",
                    simaai_memory_get_phys(mem->memory), target, flags);

  return GST_MEMORY_CAST (mem);
}

static void
gst_simaai_allocator2_free (GstAllocator *allocator, GstMemory *memory)
{
  GstSimaaiMemory *mem = (GstSimaaiMemory *)memory;

  GST_DEBUG_OBJECT (allocator, "Free simaai phys memory: 0x%" PRIx64,
                    simaai_memory_get_phys(mem->memory));

  simaai_memory_unmap(mem->memory);
  simaai_memory_free(mem->memory);
  g_slice_free1(sizeof(GstSimaaiMemory), mem);
}

static void
gst_simaai_allocator2_class_init (GstSimaaiAllocatorClass *klass)
{
  GstAllocatorClass *allocator_class = GST_ALLOCATOR_CLASS (klass);

  allocator_class->alloc = gst_simaai_allocator2_alloc;
  allocator_class->free = gst_simaai_allocator2_free;
}

static void
gst_simaai_allocator2_init (GstSimaaiAllocator *allocator)
{
  GstAllocator *alloc = GST_ALLOCATOR (allocator);

  alloc->mem_type = GST_ALLOCATOR_SIMAAI;
  alloc->mem_map_full = (GstMemoryMapFullFunction) mem_map_full;
  alloc->mem_unmap_full = (GstMemoryUnmapFullFunction) mem_unmap_full;

  GST_OBJECT_FLAG_SET (allocator, GST_ALLOCATOR_FLAG_CUSTOM_ALLOC);
}

void
gst_simaai_memory_init_once (void)
{
  static gsize _init = 0;
  GstAllocator *alloc;

  if (g_once_init_enter (&_init)) {
    alloc = (GstAllocator *)g_object_new (GST_TYPE_SIMAAI_ALLOCATOR2, NULL);

    /* Clear floating flag */
    gst_object_ref_sink (alloc);

    gst_allocator_register (GST_ALLOCATOR_SIMAAI, (GstAllocator *)gst_object_ref(alloc));
    GST_DEBUG_OBJECT(alloc, "Initialized %s allocator", GST_ALLOCATOR_SIMAAI);
    g_once_init_leave (&_init, 1);
  }
}

GstAllocator *
gst_simaai_memory_get_allocator (void)
{
  return gst_allocator_find (GST_ALLOCATOR_SIMAAI);
}

guintptr
gst_simaai_memory_get_phys_addr (GstMemory * mem)
{
  if (mem != NULL && mem->allocator != NULL && GST_IS_SIMAAI_ALLOCATOR2(mem->allocator))
    return simaai_memory_get_phys(GST_SIMAAI_MEMORY_CAST(mem)->memory);
  else
    return 0;
}
