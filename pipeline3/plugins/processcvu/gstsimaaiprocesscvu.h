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

#ifndef GST_SIMAAIPROCESSCVU_H_
#define GST_SIMAAIPROCESSCVU_H_

#include <gst/gst.h>
#include <gst/base/gstaggregator.h>

#define CONFIG_PATH_LEN 128
#define DEFAULT_CONFIG_FILE "/mnt/host/evxx_pre_proc.json"
#define DEFAULT_SOURCE_NODE_NAME "allegrodec"
#define DEFAULT_IN_BUFFER_LIST "a65-topk"
#define DEFAULT_NUM_BUFFERS 5
#define MIN_POOL_SIZE 2

#define PAD_TEMPLATE_NAME_SINK  "sink_%u"
#define PAD_TEMPLATE_NAME_SRC   "src"

#define SIMAAI_META_STR "GstSimaMeta"

G_BEGIN_DECLS

#define GST_TYPE_SIMAAI_PROCESSCVU            (gst_simaai_processcvu_get_type ())
#define GST_SIMAAI_PROCESSCVU(obj)            (G_TYPE_CHECK_INSTANCE_CAST ((obj), GST_TYPE_SIMAAI_PROCESSCVU, GstSimaaiProcesscvu))
#define GST_SIMAAI_PROCESSCVU_CLASS(klass)    (G_TYPE_CHECK_CLASS_CAST ((klass), GST_TYPE_SIMAAI_PROCESSCVU, GstSimaaiProcesscvuClass))
#define GST_SIMAAI_PROCESSCVU_GET_CLASS(obj)  (G_TYPE_INSTANCE_GET_CLASS ((obj), GST_TYPE_SIMAAI_PROCESSCVU, GstSimaaiProcesscvuClass))

typedef struct _GstSimaaiProcesscvu GstSimaaiProcesscvu;
typedef struct _GstSimaaiProcesscvuClass GstSimaaiProcesscvuClass;
typedef struct _GstSimaaiProcesscvuPrivate GstSimaaiProcesscvuPrivate;
typedef std::chrono::time_point<std::chrono::steady_clock> TimePoint;
struct _GstSimaaiProcesscvu
{
  GstAggregator parent;
  gboolean silent; /**< true to print minimized log */
  GstSimaaiProcesscvuPrivate *priv;
  gboolean transmit;
};

struct _GstSimaaiProcesscvuClass
{
  GstAggregatorClass parent_class;
};

GType gst_simaai_processcvu_get_type (void);

G_END_DECLS

#endif // GST_SIMAAIPROCESSCVU_H_
