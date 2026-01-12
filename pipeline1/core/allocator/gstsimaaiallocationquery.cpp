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

#include "gstsimaaiallocator_old.h"

#include <gstsimaaimeta.h>


GstSimaaiMemoryFlags gst_simaai_allocation_query_str_to_sima_mem_flag(const char * desc)
{
    if (strcmp(desc, "GST_SIMAAI_MEMORY_FLAG_CACHED") == 0) {
        return GST_SIMAAI_MEMORY_FLAG_CACHED;
    }

    if (strcmp(desc, "GST_SIMAAI_MEMORY_FLAG_RDONLY") == 0) {
        return GST_SIMAAI_MEMORY_FLAG_RDONLY;
    }

    if (strcmp(desc, "GST_SIMAAI_MEMORY_FLAG_DEFAULT") == 0) {
        return GST_SIMAAI_MEMORY_FLAG_DEFAULT;
    }

    // default
    return GST_SIMAAI_MEMORY_FLAG_DEFAULT;
}

GstSimaaiMemoryFlags gst_simaai_allocation_query_str_to_sima_mem_type(const char *desc)
{
    if (strcmp(desc, "GST_SIMAAI_MEMORY_TARGET_GENERIC") == 0) {
        return GST_SIMAAI_MEMORY_TARGET_GENERIC;
    }

    if (strcmp(desc, "GST_SIMAAI_MEMORY_TARGET_OCM") == 0) {
        return GST_SIMAAI_MEMORY_TARGET_OCM;
    }

    if (strcmp(desc, "GST_SIMAAI_MEMORY_TARGET_DMS0") == 0) {
        return GST_SIMAAI_MEMORY_TARGET_DMS0;
    }

    if (strcmp(desc, "GST_SIMAAI_MEMORY_TARGET_DMS1") == 0) {
        return GST_SIMAAI_MEMORY_TARGET_DMS1;
    }

    if (strcmp(desc, "GST_SIMAAI_MEMORY_TARGET_DMS2") == 0) {
        return GST_SIMAAI_MEMORY_TARGET_DMS2;
    }

    if (strcmp(desc, "GST_SIMAAI_MEMORY_TARGET_DMS3") == 0) {
        return GST_SIMAAI_MEMORY_TARGET_DMS3;
    }

    if (strcmp(desc, "GST_SIMAAI_MEMORY_TARGET_EV74") == 0) {
        return GST_SIMAAI_MEMORY_TARGET_EV74;
    }

    return GST_SIMAAI_MEMORY_TARGET_GENERIC;
}

const char * gst_simaai_allocation_query_sima_mem_flag_to_str(GstSimaaiMemoryFlags mem_flag)
{
    if (mem_flag == GST_SIMAAI_MEMORY_FLAG_CACHED) {
        return "GST_SIMAAI_MEMORY_FLAG_CACHED";
    }

    if (mem_flag == GST_SIMAAI_MEMORY_FLAG_RDONLY) {
        return "GST_SIMAAI_MEMORY_FLAG_RDONLY";
    }

    if (mem_flag == GST_SIMAAI_MEMORY_FLAG_DEFAULT) {
        return "GST_SIMAAI_MEMORY_FLAG_DEFAULT";
    }

    // default
    return "GST_SIMAAI_MEMORY_FLAG_UNSUPPORTED";
}

const char * gst_simaai_allocation_query_sima_mem_type_to_str(GstSimaaiMemoryFlags mem_type)
{
    if (mem_type == GST_SIMAAI_MEMORY_TARGET_GENERIC) {
        return "GST_SIMAAI_MEMORY_TARGET_GENERIC";
    }

    if (mem_type == GST_SIMAAI_MEMORY_TARGET_OCM) {
        return "GST_SIMAAI_MEMORY_TARGET_OCM";
    }

    if (mem_type == GST_SIMAAI_MEMORY_TARGET_DMS0) {
        return "GST_SIMAAI_MEMORY_TARGET_DMS0";
    }

    if (mem_type == GST_SIMAAI_MEMORY_TARGET_DMS1) {
        return "GST_SIMAAI_MEMORY_TARGET_DMS1";
    }

    if (mem_type == GST_SIMAAI_MEMORY_TARGET_DMS2) {
        return "GST_SIMAAI_MEMORY_TARGET_DMS2";
    }

    if (mem_type == GST_SIMAAI_MEMORY_TARGET_DMS3) {
        return "GST_SIMAAI_MEMORY_TARGET_DMS3";
    }

    if (mem_type == GST_SIMAAI_MEMORY_TARGET_EV74) {
        return "GST_SIMAAI_MEMORY_TARGET_EV74";
    }

    return "GST_SIMAAI_MEMORY_TARGET_UNSUPPORTED";
}

GstStructure * gst_simaai_allocation_query_create_meta(GstSimaaiMemoryFlags mem_type, GstSimaaiMemoryFlags mem_flag)
{
    GstStructure * allocation_meta = gst_structure_new (GST_SIMAAI_ALLOCATION_META_STRUCT_NAME_STR,
    GST_SIMAAI_ALLOCATION_META_MEMORY_TYPE_PROP_STR, G_TYPE_STRING, gst_simaai_allocation_query_sima_mem_type_to_str(mem_type),
    GST_SIMAAI_ALLOCATION_META_MEMORY_FLAG_PROP_STR, G_TYPE_STRING, gst_simaai_allocation_query_sima_mem_flag_to_str(mem_flag), NULL);

    return allocation_meta;
}

gboolean gst_simaai_allocation_query_add_meta(GstQuery *alloc_query, GstStructure *alloc_meta)
{
    if (!alloc_query || !alloc_meta) {
        return FALSE;
    }

    gst_query_add_allocation_meta(alloc_query, GST_SIMAAI_ALLOCATION_META_API_TYPE, alloc_meta);

    return TRUE;
}

gboolean gst_simaai_allocation_query_parse(GstQuery *query, GstSimaaiMemoryFlags *mem_type, GstSimaaiMemoryFlags *mem_flag)
{
    guint idx;
    const GstStructure *params;

    if (gst_query_find_allocation_meta(query, GST_SIMAAI_ALLOCATION_META_API_TYPE, &idx)) {
        gst_query_parse_nth_allocation_meta (query, idx, &params);

        if (params) {
            const gchar * memory_type_str = gst_structure_get_string(params, GST_SIMAAI_ALLOCATION_META_MEMORY_TYPE_PROP_STR);
            const gchar * memory_flag_str = gst_structure_get_string(params, GST_SIMAAI_ALLOCATION_META_MEMORY_FLAG_PROP_STR);
            
            *mem_type = gst_simaai_allocation_query_str_to_sima_mem_type(memory_type_str);
            *mem_flag = gst_simaai_allocation_query_str_to_sima_mem_flag(memory_flag_str);

            return TRUE;

        } else {
            return FALSE;
        }
    }

    return FALSE;
}

gboolean gst_simaai_allocation_query_send(GstPad *pad, GstCaps *caps, GstSimaaiMemoryFlags *mem_type, GstSimaaiMemoryFlags *mem_flag)
{
    gboolean ret = FALSE;

    *mem_type = GST_SIMAAI_MEMORY_TARGET_GENERIC;
    *mem_flag = GST_SIMAAI_MEMORY_FLAG_DEFAULT;

    // We don't support GstBufferPool provided by downstream plugin
    GstQuery *allocation_query = gst_query_new_allocation(caps, FALSE);

    if (gst_pad_peer_query(pad, allocation_query)) {
        gst_simaai_allocation_query_parse(allocation_query, mem_type, mem_flag);
        ret = TRUE;
    }

    gst_query_unref(allocation_query);
    return ret;
}