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

#include <gst/gst.h>
#include <gst/base/gstaggregator.h>

#include "gstoverlaycommon.h"
#include "gstsimaaioverlay2.h"

#define STR_FORMAT_NV12 "NV12"
#define STR_FORMAT_I420 "I420"
#define STR_FORMAT_YUV420P "YUV420P"
#define STR_FORMAT_UNKNOWN "UNKNOWN"

/**
 * @brief list of properties supported by the overlay2 plugin
 */
enum {
  PROP_0,
  PROP_SILENT,
  PROP_TRANSMIT,
  PROP_OVERLAY_INFO, /**< This is a key::value pair like string separated by ',',
                        this represents where do we get input from for
                        the respective rendering stage **/
  PROP_LABELS_FILE, /**< Classification labels **/

  /* Visualization properties */
  PROP_OVERLAY_BBOX_LINE_COLOR, /**< Advanced support for setting up bounding box line color **/
  PROP_OVERLAY_TEXT_LINE_COLOR, /**< Advanced support for setting up text line color **/
  PROP_OVERLAY_TEXT_FONT_SIZE, /**< Advanced support for setting up text font size **/
  PROP_OVERLAY_TEXT_FONT_TYPE, /**< Advanced support for setting up text font type **/
  PROP_OVERLAY_LINE_THICKNESS, /**< Advanced support for setting up line font type **/
  PROP_OVERLAY_TEXT_FONT_SCALE, /**< Advanced support for setting up line font scale **/
  PROP_MEM_TARGET, /*< Memory type where buffer will be allocated*/
  PROP_DUMP_DATA, /**< Dump input buffer metadata */
  PROP_UNKNOWN
};

/**
 * @brief helper function to print information
 */
static void debug_overlay_profiles (const GstSimaaiOverlay2 * self)
{
  auto sz = self->priv->profile_map.size();
  g_message("Size of the map: %ld", sz);

  for (int i = 0; i < sz; i++) {
    auto pair = self->priv->profile_map[i];
    g_message ("Overlay type: %s", pair.first.c_str());
    g_message ("Overlay input name: %s", pair.second.c_str());
  }
}

/**
 * @brief, Helper API to parse the rendering information metadata
 * @todo, handle undefined cases
 * @todo, repeatative strings
 */
static gboolean gst_simaai_overlay2_parse_info (GstSimaaiOverlay2 * self, const gchar * infostr)
{
  guint rank = 0;
  guint64 val;
  gchar *info_string;
  gchar ** profile_str;
  gchar ** overlay_profile_str;

  if (infostr == NULL)
    return FALSE;

  info_string = g_strdup(infostr);
  g_strstrip(info_string);

  profile_str = g_strsplit(info_string, ",", OVERLAY_MAX_SUPPORTED_PROFILES);
  gint num_of_profiles = g_strv_length(profile_str);
  if (num_of_profiles > OVERLAY_MAX_SUPPORTED_PROFILES) {
    GST_ERROR("Unsupported number of profiles in the input string,"
              "max profiles supported in this version is %d", OVERLAY_MAX_SUPPORTED_PROFILES);
    g_strfreev(profile_str);
    g_free(info_string);
    return FALSE;
  }

  for (int i = 0; i < num_of_profiles; i++) {
    if (profile_str[i] != NULL) {
      overlay_profile_str = g_strsplit(profile_str[i], "::", 2);
      std::string s1(overlay_profile_str[0]);
      std::string s2(overlay_profile_str[1]);
      self->priv->profile_map.insert({i, std::make_pair(s1, s2)});
      g_strfreev(overlay_profile_str);
    }
  }

  // debug_overlay_profiles(self);
  g_strfreev(profile_str);
  g_free(info_string);
  return TRUE;
}

/**
 * @brief, Helper API to install plugin properties
 */
void gst_simaai_overlay2_install_properties(GObjectClass * gobj_class) {
  g_object_class_install_property (gobj_class, PROP_OVERLAY_INFO,
                                   g_param_spec_string ("render-info",
                                                        "Information",
                                                        "Overlay Input information",
                                                        DEFAULT_OVERLAY_INFO,
                                                        (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobj_class, PROP_LABELS_FILE,
                                   g_param_spec_string ("labels-file",
                                                        "LabelsFile",
                                                        "Labels File Data",
                                                        DEFAULT_LABELS_FILE,
                                                        (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  /**
   * GstSimaaiOverlay2::silent:
   *
   * The flag to enable/disable debugging messages.
   */
  g_object_class_install_property (gobj_class, PROP_SILENT,
                                   g_param_spec_boolean ("silent",
                                                         "Silent",
                                                         "Produce verbose output",
                                                         DEFAULT_SILENT,
                                                         (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobj_class, PROP_TRANSMIT,
                                   g_param_spec_boolean ("transmit",
                                                         "Transmit",
                                                         "Transmit KPI Messages",
                                                         DEFAULT_TRANSMIT,
                                                         (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobj_class, PROP_MEM_TARGET,
                                   g_param_spec_int ("mem-target",
                                                     "MemoryTarget",
                                                     "Memory type where output buffer will be allocated", 0, 10,
                                                     DEFAULT_MEM_TARGET, (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobj_class, PROP_DUMP_DATA,
                                   g_param_spec_boolean ("dump-data",
                                                         "DumpData",
                                                         "Dump buffer metadata to file",
                                                         FALSE,
                                                         (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
}

const char * enum_format_to_string(const simaai::overlay::OverlayImageFormat value)
{
  const char *ret = NULL;
  switch (value)
  {
  case simaai::overlay::OverlayImageFormat::SIMAAI_IMG_NV12:
    ret = STR_FORMAT_NV12;
    break;
  case simaai::overlay::OverlayImageFormat::SIMAAI_IMG_YUV420:
    ret = STR_FORMAT_I420;
    break;
  case simaai::overlay::OverlayImageFormat::SIMAAI_IMG_YUV444:
  case simaai::overlay::OverlayImageFormat::SIMAAI_IMG_RGB:
  default:
    ret = STR_FORMAT_UNKNOWN;
    break;
  }

  return ret;
}

simaai::overlay::OverlayImageFormat str_format_to_enum(const char * const str)
{
  simaai::overlay::OverlayImageFormat ret = simaai::overlay::OverlayImageFormat::SIMAAI_IMG_UNKNOWN;

  if (str == NULL)
    return simaai::overlay::OverlayImageFormat::SIMAAI_IMG_UNKNOWN;

  if (strcmp(str, STR_FORMAT_I420) == 0 || strcmp(str, STR_FORMAT_YUV420P) == 0) {
    ret = simaai::overlay::OverlayImageFormat::SIMAAI_IMG_YUV420;
  } else if (strcmp(str, STR_FORMAT_NV12) == 0) {
    ret = simaai::overlay::OverlayImageFormat::SIMAAI_IMG_NV12;
  }

  return ret;
}

/**
 * @brief, Helper/Setter API for the properties
 */
void gst_simaai_overlay2_set_property (GObject * object,
                                       guint prop_id,
                                       const GValue * value,
                                       GParamSpec * pspec)
{
  GstSimaaiOverlay2 * self = GST_SIMAAI_OVERLAY2 (object);

  switch(prop_id) {
    case PROP_SILENT:
      self->silent = g_value_get_boolean (value);
      GST_DEBUG_OBJECT (self, "Set silent = %d", self->silent);
      break;
    case PROP_TRANSMIT:
      self->transmit = g_value_get_boolean (value);
      GST_DEBUG_OBJECT (self, "Set transmit = %d", self->transmit);
      break;
    case PROP_OVERLAY_INFO:
      gst_simaai_overlay2_parse_info (self, g_value_get_string(value));
      break;
    case PROP_LABELS_FILE:
      self->priv->is_text_overlay = TRUE;
      g_string_assign(self->priv->labels_file, g_value_get_string(value));
      GST_DEBUG_OBJECT(self, "Config argument was changed to %s", self->priv->labels_file->str);
      break;
    case PROP_MEM_TARGET:
      self->priv->mem_target = (gint)g_value_get_int(value);
      GST_DEBUG_OBJECT(self, "Memory target set to %d", self->priv->mem_target);
      break;
    case PROP_DUMP_DATA:
      self->priv->dump_data = g_value_get_boolean(value);
      GST_DEBUG_OBJECT(self, "Set dump data = %d", self->priv->dump_data);
      break;
    default:
      GST_DEBUG_OBJECT(self, "Default case warning");
      G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
      break;
  }
}

/**
 * @brief, Helper/Getter API for the properties
 */
void gst_simaai_overlay2_get_property (GObject * object,
                                       guint prop_id,
                                       GValue * value,
                                       GParamSpec * pspec)
{
  GstSimaaiOverlay2 * self = GST_SIMAAI_OVERLAY2 (object);

  switch(prop_id) {
    case PROP_SILENT:
      g_value_set_boolean (value, self->silent);
      break;
    case PROP_TRANSMIT:
      g_value_set_boolean (value, self->transmit);
      break;
    case PROP_LABELS_FILE:
      g_value_set_string(value, self->priv->labels_file->str);
      break;
    case PROP_MEM_TARGET:
      g_value_set_int(value, self->priv->mem_target);
      break;
    case PROP_DUMP_DATA:
      g_value_set_boolean(value, self->priv->dump_data);
      break;
    case PROP_OVERLAY_INFO:
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
      break;
  }
}
