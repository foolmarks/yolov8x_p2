/*
 * GStreamer
 * Copyright (C) 2023 SiMa.ai
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Library General Public
 * License as published by the Free Software Foundation; either
 * version 2 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Library General Public License for more details.
 *
 * You should have received a copy of the GNU Library General Public
 * License along with this library; if not, write to the
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * Boston, MA 02111-1307, USA.
 */

#ifndef GST_SIMAAIOVERLAY2_H_
#define GST_SIMAAIOVERLAY2_H_

#include <simaai/parser_types.h>
#include <simaai/parser.h>

#include <gst/gst.h>
#include "gstoverlaycommon.h"

/**
 * @brief Flag to print minimized log.
 */
#define DEFAULT_SILENT TRUE

/**
 * @brief Flag to enable KPI Transmission
 */
#define DEFAULT_TRANSMIT FALSE
#define PLUGIN_CPU_TYPE "APU"

/*
 * Meta information string for gstreamer
 * https://gstreamer.freedesktop.org/documentation/gstreamer/gstmeta.html?gi-language=c
*/
#define SIMAAI_META_STR "GstSimaMeta"

/* Max Size of buffer-name or "node name" */
#define MAX_NODE_NAME 32

/* Default resolution for output w and h */
#define TARGET_RES_DEFAULT 224

/* Maximum supported overlaying profiles, read render-info */
#define OVERLAY_MAX_SUPPORTED_PROFILES 4
#define OVERLAY_MAX_MEMINFO 4

G_BEGIN_DECLS

#define GST_TYPE_SIMAAI_OVERLAY2            (gst_simaai_overlay2_get_type ())
#define GST_SIMAAI_OVERLAY2(obj)            (G_TYPE_CHECK_INSTANCE_CAST ((obj), GST_TYPE_SIMAAI_OVERLAY2, GstSimaaiOverlay2))
#define GST_SIMAAI_OVERLAY2_CLASS(klass)    (G_TYPE_CHECK_CLASS_CAST ((klass), GST_TYPE_SIMAAI_OVERLAY2, GstSimaaiOverlay2Class))
#define GST_SIMAAI_OVERLAY2_GET_CLASS(obj)  (G_TYPE_INSTANCE_GET_CLASS ((obj), GST_TYPE_SIMAAI_OVERLAY2, GstSimaaiOverlay2Class))

typedef struct _GstSimaaiOverlay2 GstSimaaiOverlay2;
typedef struct _GstSimaaiOverlay2Class GstSimaaiOverlay2Class;

/**
 * @brief Custom subclass structure
 */
struct _GstSimaaiOverlay2
{
  GstAggregator parent; /**< Parent g_object class */
  GstSimaaiOverlay2Private *priv; /**< Private members of the overlay
                                   * plugin */
  gboolean silent; /**< true to print minimized log */
  gboolean transmit; /*< true to enable kpi transmission*/
};

struct _GstSimaaiOverlay2Class
{
  GstAggregatorClass parent_class;
};

GType gst_simaai_overlay2_get_type (void);

G_END_DECLS

#endif // GST_SIMAAIOVERLAY2_H_
