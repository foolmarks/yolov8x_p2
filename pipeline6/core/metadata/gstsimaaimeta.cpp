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
#include "gstsimaaimeta.h"

GType
gst_simaai_allocation_meta_api_get_type (void)
{
  static GType type;
  static const gchar *tags[] = { "memregion", "memflags", NULL };

  if (g_once_init_enter (&type)) {
    GType _type = gst_meta_api_type_register ("GstSimaaiAllocationMetaAPI", tags);
    g_once_init_leave (&type, _type);
  }
  return type;
}

static gboolean
gst_simaai_allocation_meta_init (GstMeta * meta, gpointer params, GstBuffer * buffer)
{
  GstSimaaiAllocationMeta *emeta = (GstSimaaiAllocationMeta *) meta;

  emeta->memory_type = NULL;
  emeta->memory_flags = NULL;

  return TRUE;
}

static gboolean
gst_simaai_allocation_meta_transform (GstBuffer * transbuf, GstMeta * meta,
    GstBuffer * buffer, GQuark type, gpointer data)
{
  GstSimaaiAllocationMeta *emeta = (GstSimaaiAllocationMeta *) meta;

  // we always copy no matter what transform
  gst_buffer_add_simaai_allocation_meta (transbuf, emeta->memory_type, emeta->memory_flags);

  return TRUE;
}

static void
gst_simaai_allocation_meta_free (GstMeta * meta, GstBuffer * buffer)
{
  GstSimaaiAllocationMeta *emeta = (GstSimaaiAllocationMeta *) meta;

  g_free (emeta->memory_type);
  g_free (emeta->memory_flags);
  emeta->memory_type = NULL;
  emeta->memory_flags = NULL;
}

const GstMetaInfo *
gst_simaai_allocation_meta_get_info (void)
{
  static const GstMetaInfo *meta_info = NULL;

  if (g_once_init_enter (&meta_info)) {
    const GstMetaInfo *mi = gst_meta_register (GST_SIMAAI_ALLOCATION_META_API_TYPE,
        "GstSimaaiAllocationMeta",
        sizeof (GstSimaaiAllocationMeta),
        gst_simaai_allocation_meta_init,
        gst_simaai_allocation_meta_free,
        gst_simaai_allocation_meta_transform);
    g_once_init_leave (&meta_info, mi);
  }
  return meta_info;
}

GstSimaaiAllocationMeta *
gst_buffer_add_simaai_allocation_meta (GstBuffer   *buffer,
                                const gchar *memory_type,
                                const gchar *memory_flags)
{
  GstSimaaiAllocationMeta *meta;

  g_return_val_if_fail (GST_IS_BUFFER (buffer), NULL);

  meta = (GstSimaaiAllocationMeta *) gst_buffer_add_meta (buffer,
      GST_SIMAAI_ALLOCATION_META_INFO, NULL);

  meta->memory_type = g_strdup (memory_type);
  meta->memory_flags = g_strdup (memory_flags);

  return meta;
}