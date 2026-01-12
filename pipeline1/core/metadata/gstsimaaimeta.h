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
#ifndef GST_SIMAAI_ALLOCATION_META_H
#define GST_SIMAAI_ALLOCATION_META_H

#include <gst/gst.h>

// ************************************************************************************
// This library provides a SiMa specific Metadata API types that can be used by GStreamer
// You can extend this library by adding a new API types for your needs
// ************************************************************************************


// Allocation Meta API
// Shall be used to pass information about memory type and its flags to upstream element
// Upstream element should try to fit those requirements if possible.
// This is done by passing a custom #GstStructure to gst_query_add_allocation_meta()
// when handling the ALLOCATION query.
// This structure should be named 'simaai-allocation-meta' and can have the following fields:
// - memory_type (gchar*) name of memory region to allocate memory
// - memory_flags (gchar*) additional flags that has to be applied to memory region

// Useful defines
#define GST_SIMAAI_ALLOCATION_META_MEMORY_TYPE_PROP_STR "memory_type"
#define GST_SIMAAI_ALLOCATION_META_MEMORY_FLAG_PROP_STR "memory_flag"
#define GST_SIMAAI_ALLOCATION_META_STRUCT_NAME_STR      "simaai-allocation-meta"


typedef struct _GstSimaaiAllocationMeta GstSimaaiAllocationMeta;

struct _GstSimaaiAllocationMeta {
  GstMeta      meta;

  gchar        *memory_type;
  gchar        *memory_flags;
};

GType gst_simaai_allocation_meta_api_get_type (void);
#define GST_SIMAAI_ALLOCATION_META_API_TYPE (gst_simaai_allocation_meta_api_get_type())

#define gst_buffer_get_simaai_allocation_meta(b) \
  ((GstSimaaiAllocationMeta*)gst_buffer_get_meta((b),GST_SIMAAI_ALLOCATION_META_API_TYPE))

const GstMetaInfo *gst_simaai_allocation_meta_get_info (void);
#define GST_SIMAAI_ALLOCATION_META_INFO (gst_simaai_allocation_meta_get_info())

GstSimaaiAllocationMeta * gst_buffer_add_simaai_allocation_meta (GstBuffer *buffer, const gchar *memory_type, const gchar *memory_flags);

#endif