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

#ifndef GSTSIMAAIALLOCATOR_OLD_H_
#define GSTSIMAAIALLOCATOR_OLD_H_

#include <simaai/simaai_memory.h>

#include <gst/gst.h>
#include <gst/gstallocator.h>
#include <gst/gstmemory.h>

#include "gstsimaaiallocator_common.h"

G_BEGIN_DECLS

#define GST_TYPE_SIMAAI_ALLOCATOR (gst_simaai_buffer_memory_allocator_get_type())
GType gst_simaai_allocator_get_type(void);

#define GST_IS_SIMAAI_ALLOCATOR(obj)              (G_TYPE_CHECK_INSTANCE_TYPE ((obj), GST_TYPE_SIMAAI_ALLOCATOR))
#define GST_IS_SIMAAI_ALLOCATOR_CLASS(klass)      (G_TYPE_CHECK_CLASS_TYPE ((klass), GST_TYPE_SIMAAI_ALLOCATOR))
#define GST_SIMAAI_ALLOCATOR_GET_CLASS(obj)       (G_TYPE_INSTANCE_GET_CLASS ((obj), GST_TYPE_SIMAAI_ALLOCATOR, GstSimaaiAllocatorClass))
#define GST_SIMAAI_ALLOCATOR(obj)                 (G_TYPE_CHECK_INSTANCE_CAST ((obj), GST_TYPE_SIMAAI_ALLOCATOR, GstSimaaiAllocator))
#define GST_SIMAAI_ALLOCATOR_CLASS(klass)         (G_TYPE_CHECK_CLASS_CAST ((klass), GST_TYPE_SIMAAI_ALLOCATOR, GstSimaaiAllocatorClass))
#define GST_SIMAAI_ALLOCATOR_CAST(obj)            ((GstSimaaiAllocator *)(obj))

typedef struct
{
  GstAllocator parent;
} SimaAiAllocator;

typedef struct
{
  GstAllocatorClass parent_class;
} SimaAiAllocatorClass;

void
gst_simaai_buffer_memory_init_once (void);

void
gst_simaai_buffer_memory_free_once (void);

GstAllocator *
gst_simaai_get_allocator(void);

simaai_memory_t *
simaai_get_memory_handle(GstMemory * mem);

GstMemory *
simaai_target_mem_alloc(guint target, gsize size, gint64 * buffer_id);

GstMemory *
simaai_target_mem_alloc_flags (guint target, gsize size, gint64 * buffer_id, int flags);

void
simaai_target_mem_free(GstMemory * mem);

/**
 * GST_ALLOCATOR_SIMAAI:
 *
 * The allocator name for the SimaAi memory allocator
 */
#define GST_ALLOCATOR_SIMAAI    "SimaaiMemory"

#define GST_TYPE_SIMAAI_ALLOCATOR2 (gst_simaai_allocator2_get_type())
G_DECLARE_FINAL_TYPE (GstSimaaiAllocator, gst_simaai_allocator2,
                      GST, SIMAAI_ALLOCATOR2, GstAllocator)

struct _GstSimaaiAllocator
{
  GstAllocator parent;
};

/*
* gst_simaai_allocation_query_str_to_sima_mem_flag
* Converts string representation of SiMa memory flags into enum
*/
GstSimaaiMemoryFlags gst_simaai_allocation_query_str_to_sima_mem_flag(const char * desc);

/*
* gst_simaai_allocation_query_str_to_sima_mem_type
* Converts string representation of SiMa memory type into enum
*/
GstSimaaiMemoryFlags gst_simaai_allocation_query_str_to_sima_mem_type(const char *desc);

/*
* gst_simaai_allocation_query_sima_mem_flag_to_str
* Converts SiMa memory flags into string representation
*/
const char * gst_simaai_allocation_query_sima_mem_flag_to_str(GstSimaaiMemoryFlags mem_flag);

/*
* gst_simaai_allocation_query_sima_mem_type_to_str
* Converts SiMa memory flags into string representation
*/
const char * gst_simaai_allocation_query_sima_mem_type_to_str(GstSimaaiMemoryFlags mem_type);

/*
 * gst_simaai_allocation_query_create_meta
 * Convinient API that simplifies creation GST_SIMAAI_ALLOCATION_META
 * Can be used in propose_allocation callbacks in GStreamer base classes
 * Params:
 * [in] mem_type - type of the SiMa memory
 * [in] mem_flag - the additional flags that shall be applied to SiMa memory
 * Returns: GstStructure with the GST_SIMAAI_ALLOCATION_META_API_TYPE
 *          that contains string represention of input parameters
*/
GstStructure * gst_simaai_allocation_query_create_meta(GstSimaaiMemoryFlags mem_type, GstSimaaiMemoryFlags mem_flag);


/*
* API to add allocation meta structure to allocation query
* Hides the exact Metadata API type from the user, so they shall not worry about it
*/
gboolean gst_simaai_allocation_query_add_meta(GstQuery *alloc_query, GstStructure *alloc_meta);

/*
 * gst_simaai_allocation_query_parse
 * Parse allocation query to retrieve the information about SiMa Memory type and flags
 * Those values are separeted so the User can implement more complex logic in the element to handle that
 * Params:
 * [in] query - ALLOCATION query to parse
 * [out] mem_type - SiMa memory type proposed by downstream element
 * [out] mem_flag - SiMa memory flag proposed by downstream element
 * Returns: TRUE if ALLOCATION query has been parsed successfully, FALSE otherwise
*/
gboolean gst_simaai_allocation_query_parse(GstQuery *query, GstSimaaiMemoryFlags *mem_type, GstSimaaiMemoryFlags *mem_flag);

/*
 * gst_allocation_query_send
 * Send ALLOCATION query to downstream element and retrieve information about required memory type from it.
 * Can be used in GstElement based classes
 * Params:
 * [in] pad - current element pad. ALLOCATION query will be sent to it's peer pad, so usually it is element's src pad.
 * [in] caps - GstCaps associated with the given 'pad'.
 * [out] mem_type - SiMa memory type that can be used by allocator
 * [out] mem_flag - SiMa memory flag that can be applied to SiMa memory
 * Returns: TRUE if ALLOCATION query has been parsed successfully, FALSE otherwise
*/
gboolean gst_simaai_allocation_query_send(GstPad *pad, GstCaps *caps, GstSimaaiMemoryFlags *mem_type, GstSimaaiMemoryFlags *mem_flag);

/**
 * gst_simaai_memory_init_once:
 *
 * Initialize the Simaai Memory allocator. The allocator init is thread safe.
 */
void gst_simaai_memory_init_once (void);

/*
 * gst_simaai_memory_get_allocator:
 *
 * Find a previously registered Simaai Memory allocator.
 *
 * Returns: a pointer to registered Simaai Memory allocator.
 */
GstAllocator * gst_simaai_memory_get_allocator (void);

/**
 * gst_simaai_memory_get_phys_addr:
 * @mem: a #GstMemory
 *
 * Returns: Simaai memory physical address that is backing @mem, or 0 if none.
 */
guintptr gst_simaai_memory_get_phys_addr (GstMemory * mem);

G_END_DECLS

#endif