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

#include <mutex>
#include <map>

#include <gst/gst.h>
#include <simaai/simaai_memory.h>

#include "gstsimaaiallocator_old.h"

#define DEFAULT_DIM_W 640
#define DEFAULT_DIM_H 480
#define DEFAULT_FMT 1

#define GST_CAT_DEFUALT GST_CAT_SIMAAI_BUFFER_MEMORY
GST_DEBUG_CATEGORY_STATIC (GST_CAT_DEFUALT);

/**
 * A map to hold the key, value pair of buffer_id to virtual address map
 * and the mutex guarding access to the map.
 */ 
static std::map<uint64_t, std::pair<simaai_memory_t *, void*>> mappings;
static std::mutex mapping_mutex;

/**
 * @brief buffer_id_to_vaddr:
 * Attaches the buffer_id to simaai_memory_t handle, which lets us mmap the allocated buffer to process context. 
 * @param buffer_id 
 * @return Returns: virtual address of the mmap-ed region
 *
 */
static void* buffer_id_to_vaddr(uint64_t id)
{
    std::lock_guard<std::mutex> _lock(mapping_mutex);
    void *res;
    simaai_memory_t * m;

    auto search = mappings.find(id);

    if (search == mappings.end()) {
        m = simaai_memory_attach(id);
        res = simaai_memory_map(m);
        mappings[id] = std::make_pair(m, res);
        GST_DEBUG("Inserintg mapping of buffer id %#lx to %p\n", id, res);
    }

    return mappings[id].second;
}

static void remove_id_from_mappings (uint64_t id)
{
    std::lock_guard<std::mutex> _lock(mapping_mutex);

    auto search = mappings.find(id);

    if (search != mappings.end()) {
        GST_DEBUG("Erased memory mapping\n");
        simaai_memory_t * m = search->second.first;
        simaai_memory_unmap(m);
        simaai_memory_free(m);
        mappings.erase(search);
    }
    else
	 GST_ERROR("ID Not found");
    
    return;
}

/**
 * @brief Structure used by the allocator
 */
typedef struct sima_mem_s {
    GstMemory mem; ///< GstMemory object
    gpointer data; ///< Opaque pointer to the data

    // Meta Data,
    guint format; ///< format definition
    guint w; ///< width representation
    guint h; ///< Height 
    
    simaai_memory_t * memory; ///< SiMa memory handle used by simamemlib
    uint64_t id; // buffer_id
} simaai_mem_t;

/**
 * @brief GstAllocator object
 * Global memory allocator object
 */
static GstAllocator *simaai_allocator = NULL;

/**
 * @brief simaai_custom_mem_alloc
 * Callback function registered when gst_memory_alloc is invoked with the custom sima allocator handle
 * Take care of initialize the dims elsewhere if the metadata is necessary or consumed in the plugin
 *
 * @param allocator the custom allocator handle initialied
 * @param size: size of the memory block
 * @param params: GstAllocationParams to be used by the allocator
 * @return Returns: GstMemory handle if success or 'NULL' handle NULL in return
 * 
 */
static GstMemory *
simaai_mem_alloc (GstAllocator * allocator,
                         gsize size,
                         GstAllocationParams * params)
{
  simaai_mem_t *mem = g_slice_new (simaai_mem_t);
  gsize maxsize = size + params->prefix + params->padding;

  gst_memory_init (GST_MEMORY_CAST (mem), params->flags, allocator, NULL,
      maxsize, params->align, params->prefix, size);

  simaai_memory_t * memory = NULL;
  mem->memory = simaai_memory_alloc(size, SIMAAI_MEM_TARGET_GENERIC);
  if (!mem->memory) {
       GST_ERROR("ERROR: allocating contiguous memory");
       g_slice_free (simaai_mem_t, mem);
       return NULL;
  }

  mem->id = simaai_memory_get_phys(mem->memory);
  // Reinitialize these somewhere else
  mem->w = DEFAULT_DIM_W;
  mem->h = DEFAULT_DIM_H;
  mem->format = DEFAULT_FMT;
  return (GstMemory *) mem;
}

/**
 * @brief simaai_mem_free:
 * Callback function for gst_memory_free 
 * @param allocator: the custom allocator handle initialied
 * @param mem: GstMemory handle to be freed
 * 
 */
static void
simaai_mem_free (GstAllocator * allocator, GstMemory * mem)
{
  simaai_mem_t *mmem = (simaai_mem_t *) mem;

  remove_id_from_mappings (simaai_memory_get_phys(mmem->memory));
  simaai_memory_unmap(mmem->memory);
  simaai_memory_free (mmem->memory);
  g_slice_free (simaai_mem_t, mmem);
}

/**
 * @brief simaai_mem_map
 * Used to mmap the simaai-memlib allocated memory into the process/thread context.
 *
 * Returns a gpointer to the mapped memory, which is mapped using buffer_id_to_vaddr. 
 * the gpointer is a (void *)
 * 
 * @param mem: GstMemory handle to be freed
 * @param maxsize: Maximum size of the map
 * @param flags: GstMapFlags object defining mapping flags
 * 
 */
static gpointer
simaai_mem_map (simaai_mem_t * mem, gsize maxsize, GstMapFlags flags)
{
  return (gpointer) buffer_id_to_vaddr(mem->id);
}

/**
 * @simaai_mem_unmap:
 * NOTE this is unimplemented as we handle memory attach and detach differently
 * with sima memory lib. so just a stub
 * Unmap the mapped memory callback using simamemory lib.
 * @param mem: GstMemory handle to be freed
 * @return TRUE always. The memory will be unmapped when the sima buffer is freed. Once mapped the same
 * virtual address (stored in the mappings) can be reused in the given Linux process context.
 */
static gboolean
simaai_mem_unmap (simaai_mem_t * mem)
{
  return TRUE;
}

/**
 * @brief simaai_mem_share
 * NOT implemented, to share the existing memory block with existing memory block at an offset
 * 
 * @param mem: GstMemory handle to be freed
 * @param offset: Offset of sharable memory
 * @param size: Size of the sharable memory 
 * @return Returns: simaai_mem_t handle on success or NULL
 * 
 */
static simaai_mem_t *
simaai_mem_share (simaai_mem_t * mem, gssize offset, gsize size)
{
  return NULL;
}

GType simaai_allocator_get_type (void);
#define simaai_allocator_parent_class parent_class
G_DEFINE_TYPE (SimaAiAllocator, simaai_allocator,
    GST_TYPE_ALLOCATOR);

static void
simaai_allocator_class_init (SimaAiAllocatorClass * klass)
{
  GstAllocatorClass *allocator_class;

  allocator_class = (GstAllocatorClass *) klass;
  allocator_class->alloc = simaai_mem_alloc;
  allocator_class->free = simaai_mem_free;
}

static void
simaai_allocator_init (SimaAiAllocator * allocator)
{
  GstAllocator *alloc = GST_ALLOCATOR_CAST (allocator);

  alloc->mem_type = "simaai-allocator";
  alloc->mem_map = (GstMemoryMapFunction) simaai_mem_map;
  alloc->mem_unmap = (GstMemoryUnmapFunction) simaai_mem_unmap;
  alloc->mem_share = (GstMemoryShareFunction) simaai_mem_share;
}

/**
 * @brief gst_simaai_buffer_memory_init_once:
 *
 * Initialize the global allocator, the allocator init is thread safe.
 * 
 */
void
gst_simaai_buffer_memory_init_once (void)
{
  static gsize _init = 0;

  if (g_once_init_enter (&_init)) {
    GST_DEBUG("Initializing simaai-memory allocator");
    simaai_allocator = (GstAllocator *)g_object_new (simaai_allocator_get_type (), NULL);
    gst_allocator_register ("simaai-allocator", simaai_allocator);
    g_once_init_leave (&_init, 1);
  }
}

void
gst_simaai_buffer_memory_free_once (void)
{
  static gsize _deinit = 0;

  if (g_once_init_enter (&_deinit)) {
    if (simaai_allocator) {
      GST_DEBUG("Deinitializing simaai-memory allocator");
      g_clear_object(&simaai_allocator);
      g_once_init_leave (&_deinit, 1);
    } else {
      g_once_init_leave (&_deinit, 0);
    }
  }
}

GstAllocator * gst_simaai_get_allocator(void) {
     return simaai_allocator;
}

simaai_memory_t * simaai_get_memory_handle(GstMemory * mem) {
     simaai_mem_t * _mem = (simaai_mem_t *) mem;

     auto search = mappings.find(_mem->id);
     if (search == mappings.end())
       buffer_id_to_vaddr(_mem->id);

     return mappings[_mem->id].first;
}

/**
 * @brief simaai_target_mem_alloc:
 * This is the custom allocator to be used when we need to pass in CPU target, more details please check README.md
 *
 * @param target: The cpu target for the simamemlib to allocate accordingly
 * @param size: size of the memory block
 * @param buffer_id: Filled by the allocator
 * @return Returns: GstMemory is returned on success or NULL
 *
 */
GstMemory * simaai_target_mem_alloc (guint target, gsize size, gint64 * buffer_id)
{
  return simaai_target_mem_alloc_flags (target, size, buffer_id, SIMAAI_MEM_FLAG_DEFAULT);
}

/**
 * @brief simaai_target_mem_alloc_flags:
 * This is the custom allocator to be used when we need to pass in CPU target, more details please check README.md
 * For details on the flags please see this 
 * https://bitbucket.org/sima-ai/simaai-memory-lib/src/develop/simaai_memory.h
 * @param target: The cpu target for the simamemlib to allocate accordingly
 * @param size: size of the memory block
 * @param buffer_id: Filled by the allocator
 * @param flags: Allocation flags
 * @return Returns: GstMemory is returned on success or NULL
 *
 */
GstMemory * simaai_target_mem_alloc_flags (guint target, gsize size, gint64 * buffer_id, int flags)
{
  simaai_mem_t *mem = g_slice_new (simaai_mem_t);

  if (mem == NULL) {
    GST_ERROR("unable to create slice for simaai_mem_t %p", simaai_allocator);
    return NULL;
  }

  gsize maxsize = size;
  gsize align = gst_memory_alignment;

  GstAllocationParams params;
  gst_allocation_params_init (&params);

  if (simaai_allocator != NULL) {
    gst_memory_init (GST_MEMORY_CAST (mem),
                     params.flags,
                     simaai_allocator,
                     NULL,
                     maxsize,
                     align,
                     0,
                     size);

    // Validate target here
    mem->memory = simaai_memory_alloc_flags (size, target, flags);
    if (!mem->memory) {
      GST_ERROR("ERROR: allocating contiguous memory");
      g_slice_free (simaai_mem_t, mem);
      return NULL;
    }

    *buffer_id = (gint64)simaai_memory_get_phys(mem->memory);
    mem->id = *buffer_id;
    return (GstMemory *) mem;
  } else {
    GST_ERROR("Allocator not initialized");
    g_slice_free (simaai_mem_t, mem);
  }

  return NULL;
}

/**
 * @brief simaai_target_mem_free:
 * This is a GstMemory free function of the custom allocator
 * @param mem: GstMemory handle to be freed
 *
 */
void simaai_target_mem_free (GstMemory * mem)
{
  simaai_mem_t *mmem = (simaai_mem_t *) mem;

  remove_id_from_mappings (simaai_memory_get_phys(mmem->memory));
  simaai_memory_unmap(mmem->memory);
  simaai_memory_free (mmem->memory);
  g_slice_free (simaai_mem_t, mmem);
}
