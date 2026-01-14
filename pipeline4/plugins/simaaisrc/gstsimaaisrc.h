/*
 * GStreamer
 * Copyright (C) 2022 SiMa.ai
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

#ifndef __GST_SIMAAI_SRC_H__
#define __GST_SIMAAI_SRC_H__

#include <gst/gst.h>
#include <gst/base/gstbasesrc.h>

#define DEFAULT_INDEX 0

G_BEGIN_DECLS
#define GST_TYPE_SIMAAISRC \
    (gst_simaaisrc_get_type())
#define GST_SIMAAISRC(obj) \
    (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_SIMAAISRC,GstSimaaiSrc))
#define GST_SIMAAISRC_CLASS(klass) \
    (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_SIMAAISRC,GstSimaaiSrcClass))
#define GST_IS_SIMAAISRC(obj) \
    (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_SIMAAISRC))
#define GST_IS_SIMAAISRC_CLASS(klass) \
    (G_TYPE_CHECK_CLASS_TYPE((klass), GST_TYPE_SIMAAISRC))
#define GST_SIMAAISRC_CAST(obj) ((GstSimaaiSrc *) (obj))
typedef struct _GstSimaaiSrc GstSimaaiSrc;
typedef struct _GstSimaaiSrcClass GstSimaaiSrcClass;
typedef struct _GstSimaaiSrcPrivate GstSimaaiSrcPrivate;

/**
 * @brief GstSimaaiSrc data structure.
 */
struct _GstSimaaiSrc
{
  GstBaseSrc element;
  GstSimaaiSrcPrivate * priv;
};

/**
 * @brief GstSimaaiSrcClass data structure.
 */
struct _GstSimaaiSrcClass
{
  GstBaseSrcClass parent_class;
};

GType gst_simaaisrc_get_type (void);

G_END_DECLS
#endif /* __GST_SIMAAI_SRC_H__ */
