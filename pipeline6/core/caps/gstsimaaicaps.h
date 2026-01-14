//**************************************************************************
//||                        SiMa.ai CONFIDENTIAL                          ||
//||   Unpublished Copyright (c) 2025 SiMa.ai, All Rights Reserved.       ||
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

#ifndef __GST_SIMAAI_CAPS_H__
#define __GST_SIMAAI_CAPS_H__

#include <gst/gst.h>
#include <json-glib/json-glib.h>

#define JSON_FIELD_NAME_CAPS        "caps"
#define JSON_FIELD_NAME_SINK_PADS   "sink_pads"
#define JSON_FIELD_NAME_SRC_PADS    "src_pads"
#define JSON_FIELD_NAME_MEDIA_TYPE  "media_type"
#define JSON_FIELD_NAME_PARAMS      "params"
#define JSON_FIELD_NAME_TYPE        "type"
#define JSON_FIELD_NAME_NAME        "name"
#define JSON_FIELD_NAME_VALUES      "values"
#define JSON_FIELD_NAME_JSON_FIELD  "json_field"

G_BEGIN_DECLS

typedef struct _GstSimaaiCaps GstSimaaiCaps;
struct _GstSimaaiCaps {
    gchar *config;

    JsonParser *parser;

    GstCaps *sink_caps;
    GstCaps *src_caps;
};

gboolean gst_simaai_caps_process_sink_caps(GstElement *element,
                                           GstSimaaiCaps *simaai_caps,
                                           GstEvent *event);

GstCaps *gst_simaai_caps_fixate_src_caps(GstElement *element,
                                         GstSimaaiCaps *simaai_caps,
                                         GstCaps *caps);

gboolean gst_simaai_caps_query(GstElement *element,
                               GstSimaaiCaps *simaai_caps,
                               GstQuery *query,
                               GstPadDirection pad_direction);

gboolean gst_simaai_caps_negotiate(GstElement *element,
                                   GstSimaaiCaps *simaai_caps);

gboolean gst_simaai_caps_parse_config(GstElement *element,
                                      GstSimaaiCaps *simaai_caps,
                                      const gchar *config);

GstSimaaiCaps *gst_simaai_caps_init(void);

void gst_simaai_caps_free(GstSimaaiCaps *simaai_caps);

G_END_DECLS

#endif /* __GST_SIMAAI_CAPS_H__ */
