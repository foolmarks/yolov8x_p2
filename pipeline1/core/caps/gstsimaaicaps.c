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

#include <gst/base/gstaggregator.h>

#include "gstsimaaicaps.h"

static gboolean
fixate_caps(GString *caps, JsonObject *root, const gchar *name,
    const gchar *type, const gchar *json_field)
{
    JsonArray *values_arr;
    guint8 i;

    values_arr = json_object_get_array_member(root, json_field);
    if (!values_arr)
        if (g_strcmp0(type, "int") == 0)
            g_string_append_printf(caps, ", %s=(%s)%ld", name, type,
                json_object_get_int_member(root, json_field));
        else if (g_strcmp0(type, "float") == 0)
            g_string_append_printf(caps, ", %s=(%s)%f", name, type,
                json_object_get_double_member(root, json_field));
        else if (g_strcmp0(type, "string") == 0)
            g_string_append_printf(caps, ", %s=(%s)%s", name, type,
                json_object_get_string_member(root, json_field));
        else
            return FALSE;
    else
        for (i = 0; i < json_array_get_length(values_arr); ++i)
            if (g_strcmp0(type, "int") == 0)
                g_string_append_printf(caps, ", %s__%hhu=(%s)%ld", name, i,
                    type, json_array_get_int_element(values_arr, i));
            else if (g_strcmp0(type, "float") == 0)
                g_string_append_printf(caps, ", %s__%hhu=(%s)%f", name, i, type,
                    json_array_get_double_element(values_arr, i));
            else if (g_strcmp0(type, "string") == 0)
                g_string_append_printf(caps, ", %s__%hhu=(%s)%s", name, i, type,
                    json_array_get_string_element(values_arr, i));
            else
                return FALSE;

    return TRUE;
}

static gboolean
write_values(JsonObject *root, const gchar *json_field, const gchar *name,
    const gchar *type, JsonArray *values, GstStructure *s)
{
    const GValue *caps_value;
    guint8 i;
    const gchar *value_str;
    gchar *element_name_str;
    JsonNode *node;

    if (!values) {
        caps_value = gst_structure_get_value(s, name);
        if (!caps_value) {
            GST_ERROR("<%s>: Error retrieving a value for %s", G_STRFUNC, name);
            return FALSE;
        }

        if (g_strcmp0(type, "int") == 0) {
            json_object_set_int_member(root, json_field,
                g_value_get_int(caps_value));
        } else if (g_strcmp0(type, "float") == 0) {
            json_object_set_double_member(root, json_field,
                g_value_get_double(caps_value));
        } else if (g_strcmp0(type, "string") == 0) {
            value_str = g_value_get_string(caps_value);
            if (g_strcmp0(value_str, "I420") == 0)
                json_object_set_string_member(root, json_field, "IYUV");
            else
                json_object_set_string_member(root, json_field, value_str);
        } else {
            return FALSE;
        }
    } else {
        for (i = 0; i < json_array_get_length(values); ++i) {
            element_name_str = g_strdup_printf("%s__%hhu", name, i);

            caps_value = gst_structure_get_value(s, element_name_str);
            if (!caps_value) {
                GST_ERROR("<%s>: Error retrieving a value for %s", G_STRFUNC,
                    element_name_str);
                return FALSE;
            }

            node = json_array_get_element(values, i);

            if (g_strcmp0(type, "int") == 0) {
                json_node_set_int(node, g_value_get_int(caps_value));
            } else if (g_strcmp0(type, "float") == 0) {
                json_node_set_double(node, g_value_get_double(caps_value));
            } else if (g_strcmp0(type, "string") == 0) {
                value_str = g_value_get_string(caps_value);
                if (g_strcmp0(value_str, "I420") == 0)
                    json_node_set_string(node, "IYUV");
                else
                    json_node_set_string(node, value_str);
            } else {
                return FALSE;
            }

            g_free(element_name_str);
        }
    }

    return TRUE;
}

static void
append_values(GString *caps, const gchar *values, const gchar *name,
    const gchar *type)
{
    gchar **tokens, **toks, *tok_tmp;
    gboolean add_suffix = FALSE;
    guint8 i, j;
    GString *values_str = g_string_new(values);
    gchar *vals = values_str->str;

    g_string_replace(values_str, " ", "", 0);

    if (g_strrstr(vals, "(")) {
        g_string_erase(values_str, 0, 1);
        g_string_erase(values_str, values_str->len - 1, 1);

        add_suffix = TRUE;
    }

    tokens = g_strsplit(vals, "),(", -1);
    for (i = 0; tokens[i]; ++i) {
        if (g_strrstr(tokens[i], "-")) {
            toks = g_strsplit(tokens[i], "-", 2);

            if (add_suffix)
                g_string_append_printf(caps, ", %s__%hhu=(%s)[%s, %s]", name,
                    i, type, g_strstrip(toks[0]), g_strstrip(toks[1]));
            else
                g_string_append_printf(caps, ", %s=(%s)[%s, %s]", name,
                    type, g_strstrip(toks[0]), g_strstrip(toks[1]));
        } else {
            toks = g_strsplit(tokens[i], ",", -1);

            for (j = 0; toks[j]; ++j)
                g_strstrip(toks[j]);

            tok_tmp = g_strjoinv(", ", toks);

            if (add_suffix)
                g_string_append_printf(caps, ", %s__%hhu=(%s){%s}", name, i,
                    type, tok_tmp);
            else
                g_string_append_printf(caps, ", %s=(%s){%s}", name, type,
                    tok_tmp);

            g_free(tok_tmp);
        }

        g_strfreev(toks);
    }

    g_strfreev(tokens);
    g_string_free(values_str, TRUE);
}

static gboolean
parse_pads(GstElement *element, GstSimaaiCaps *simaai_caps, JsonObject *object,
    GstPadDirection pad_direction)
{
    const gchar *pads_name;
    JsonArray *pads_arr, *params_arr;
    guint8 i, j;
    JsonObject *pad_obj, *param_obj;
    GString *caps_str;
    const gchar *media_type_str, *type_str, *name_str, *values_str;
    GstCaps *caps = gst_caps_new_empty();

    if (pad_direction == GST_PAD_SINK)
        pads_name = JSON_FIELD_NAME_SINK_PADS;
    else
        pads_name = JSON_FIELD_NAME_SRC_PADS;

    pads_arr = json_object_get_array_member(object, pads_name);
    if (!pads_arr) {
        GST_WARNING_OBJECT(element, "<%s>: Failed to retrieve `%s` from JSON",
            G_STRFUNC, pads_name);
        return FALSE;
    }

    for (i = 0; i < json_array_get_length(pads_arr); ++i) {
        pad_obj = json_array_get_object_element(pads_arr, i);
        if (!pad_obj) {
            GST_WARNING_OBJECT(element, "<%s>: Failed to retrieve a pad from "
                "`%s`", G_STRFUNC, pads_name);
            return FALSE;
        }

        media_type_str = json_object_get_string_member(pad_obj,
            JSON_FIELD_NAME_MEDIA_TYPE);
        if (!media_type_str) {
            GST_WARNING_OBJECT(element, "<%s>: Failed to retrieve a pad media "
                "type from `%s`", G_STRFUNC, pads_name);
            return FALSE;
        }

        caps_str = g_string_new(media_type_str);

        params_arr = json_object_get_array_member(pad_obj,
            JSON_FIELD_NAME_PARAMS);
        if (!params_arr) {
            GST_WARNING_OBJECT(element, "<%s>: Failed to retrieve pad "
                "parameters from `%s`", G_STRFUNC, pads_name);
            goto parse_pads_caps_from_json_out;
        }

        for (j = 0; j < json_array_get_length(params_arr); ++j) {
            param_obj = json_array_get_object_element(params_arr, j);
            if (!param_obj) {
                GST_WARNING_OBJECT(element, "<%s>: Failed to retrieve a pad "
                    "parameter from `%s`", G_STRFUNC, pads_name);
                goto parse_pads_caps_from_json_out;
            }

            name_str = json_object_get_string_member(param_obj,
                JSON_FIELD_NAME_NAME);
            if (!name_str) {
                GST_WARNING_OBJECT(element, "<%s>: Failed to retrieve a "
                    "parameter name from `%s`", G_STRFUNC, pads_name);
                goto parse_pads_caps_from_json_out;
            }

            type_str = json_object_get_string_member(param_obj,
                JSON_FIELD_NAME_TYPE);
            if (!type_str) {
                GST_WARNING_OBJECT(element, "<%s>: Failed to retrieve a "
                    "parameter type from `%s`", G_STRFUNC, pads_name);
                goto parse_pads_caps_from_json_out;
            }

            values_str = json_object_get_string_member(param_obj,
                JSON_FIELD_NAME_VALUES);
            if (!values_str) {
                GST_WARNING_OBJECT(element, "<%s>: Failed to retrieve parameter"
                    " values from `%s`", G_STRFUNC, pads_name);
                goto parse_pads_caps_from_json_out;
            }

            append_values(caps_str, values_str, name_str, type_str);
        }

        gst_caps_append(caps, gst_caps_from_string(caps_str->str));

        g_string_free(caps_str, TRUE);
    }

    if (pad_direction == GST_PAD_SINK)
        gst_caps_replace(&simaai_caps->sink_caps, caps);
    else
        gst_caps_replace(&simaai_caps->src_caps, caps);

    GST_WARNING_OBJECT(element, "<%s>: Parsed from JSON caps for `%s`: %s",
        G_STRFUNC, pads_name, gst_caps_to_string(caps));

    return TRUE;

parse_pads_caps_from_json_out:
    g_string_free(caps_str, TRUE);
    return FALSE;
}

gboolean
gst_simaai_caps_process_sink_caps(GstElement *element,
    GstSimaaiCaps *simaai_caps, GstEvent *event)
{
    GstCaps *caps = NULL;
    GstStructure *s = NULL;
    JsonNode *root;
    JsonObject *root_obj, *caps_obj, *pad_obj, *param_obj;
    JsonArray *pads_arr, *params_arr, *values_arr;
    const gchar *media_type_peer_str;
    guint8 i, j;
    const gchar *media_type_str, *type_str, *name_str, *json_field_str;
    JsonGenerator *generator = json_generator_new();

    gst_event_parse_caps(event, &caps);
    if (!caps) {
        GST_ERROR_OBJECT(element, "<%s>: Error getting caps from the event",
            G_STRFUNC);
        return FALSE;
    }

    s = gst_caps_get_structure(caps, 0);
    if (!s) {
        GST_ERROR_OBJECT(element, "<%s>: Error finding a structure in the caps",
            G_STRFUNC);
        return FALSE;
    }

    root = json_parser_get_root(simaai_caps->parser);
    if (!JSON_NODE_HOLDS_OBJECT(root)) {
        GST_ERROR_OBJECT(element, "<%s>: Error retrieving a top level node",
            G_STRFUNC);
        return FALSE;
    }
    root_obj = json_node_get_object(root);

    caps_obj = json_object_get_object_member(root_obj, JSON_FIELD_NAME_CAPS);
    if (!caps_obj) {
        GST_ERROR_OBJECT(element, "<%s>: Error retrieving caps from JSON",
            G_STRFUNC);
        return FALSE;
    }

    pads_arr = json_object_get_array_member(caps_obj,
        JSON_FIELD_NAME_SINK_PADS);
    if (!pads_arr) {
        GST_ERROR_OBJECT(element, "<%s>: Error retrieving `%s` from JSON",
            G_STRFUNC, JSON_FIELD_NAME_SINK_PADS);
        return FALSE;
    }

    media_type_peer_str = gst_structure_get_name(s);

    for (i = 0; i < json_array_get_length(pads_arr); ++i) {
        pad_obj = json_array_get_object_element(pads_arr, i);
        if (!pad_obj) {
            GST_ERROR_OBJECT(element, "<%s>: Error retrieving a pad from `%s`",
                G_STRFUNC, JSON_FIELD_NAME_SINK_PADS);
            return FALSE;
        }

        media_type_str = json_object_get_string_member(pad_obj,
            JSON_FIELD_NAME_MEDIA_TYPE);
        if (!media_type_str) {
            GST_ERROR_OBJECT(element, "<%s>: Error retrieving a pad media type "
                "from `%s`", G_STRFUNC, JSON_FIELD_NAME_SINK_PADS);
            return FALSE;
        }

        if (g_strcmp0(media_type_str, media_type_peer_str) != 0)
            continue;

        params_arr = json_object_get_array_member(pad_obj,
            JSON_FIELD_NAME_PARAMS);
        if (!params_arr) {
            GST_ERROR_OBJECT(element, "<%s>: Error retrieving pad parameters",
                G_STRFUNC);
            return FALSE;
        }

        for (j = 0; j < json_array_get_length(params_arr); ++j) {
            param_obj = json_array_get_object_element(params_arr, j);
            if (!param_obj) {
                GST_ERROR_OBJECT(element, "<%s>: Error retrieving a pad "
                    "parameter", G_STRFUNC);
                return FALSE;
            }

            json_field_str = json_object_get_string_member(param_obj,
                JSON_FIELD_NAME_JSON_FIELD);
            if (!json_field_str) {
                GST_INFO_OBJECT(element, "<%s>: Unable to retrieve a JSON "
                    "field name", G_STRFUNC);
                continue;
            }

            name_str = json_object_get_string_member(param_obj,
                JSON_FIELD_NAME_NAME);
            if (!name_str) {
                GST_ERROR_OBJECT(element, "<%s>: Error retrieving a parameter "
                    "name", G_STRFUNC);
                return FALSE;
            }

            type_str = json_object_get_string_member(param_obj,
                JSON_FIELD_NAME_TYPE);
            if (!type_str) {
                GST_ERROR_OBJECT(element, "<%s>: Error retrieving a parameter "
                    "type", G_STRFUNC);
                return FALSE;
            }

            values_arr = json_object_get_array_member(root_obj, json_field_str);
            if (!write_values(root_obj, json_field_str, name_str, type_str,
                values_arr, s)) {
                GST_ERROR_OBJECT(element, "<%s>: Error processing the `%s` "
                    "parameter type", G_STRFUNC, type_str);
                return FALSE;
            }
        }
    }

    json_generator_set_root(generator, root);

    json_generator_set_pretty(generator, TRUE);
    json_generator_set_indent(generator, 4);

    if (!json_generator_to_file(generator, simaai_caps->config, NULL)) {
        GST_ERROR_OBJECT(element, "<%s>: Error writing to `%s`", G_STRFUNC,
            simaai_caps->config);
    }

    g_object_unref(generator);

    GST_WARNING_OBJECT(element, "<%s>: Processed caps: %s", G_STRFUNC,
        gst_caps_to_string(caps));

    return TRUE;
}

GstCaps *
gst_simaai_caps_fixate_src_caps(GstElement *element, GstSimaaiCaps *simaai_caps,
    GstCaps *caps)
{
    GstCaps *fixed_caps = NULL;
    JsonNode *root;
    JsonObject *root_obj, *caps_obj, *pad_obj, *param_obj;
    JsonArray *pads_arr, *params_arr;
    const gchar *media_type_str;
    const gchar *type_str, *name_str, *values_str, *json_field_str;
    GString *caps_str;
    guint8 i;
    gchar **tokens;

    root = json_parser_get_root(simaai_caps->parser);
    if (!JSON_NODE_HOLDS_OBJECT(root)) {
        GST_ERROR_OBJECT(element, "<%s>: Error retrieving a top level node",
            G_STRFUNC);
        return NULL;
    }
    root_obj = json_node_get_object(root);

    caps_obj = json_object_get_object_member(root_obj, JSON_FIELD_NAME_CAPS);
    if (!caps_obj) {
        GST_ERROR_OBJECT(element, "<%s>: Error retrieving caps from JSON",
            G_STRFUNC);
        return NULL;
    }

    pads_arr = json_object_get_array_member(caps_obj, JSON_FIELD_NAME_SRC_PADS);
    if (!pads_arr) {
        GST_ERROR_OBJECT(element, "<%s>: Error retrieving `%s` from JSON",
            G_STRFUNC, JSON_FIELD_NAME_SRC_PADS);
        return NULL;
    }

    pad_obj = json_array_get_object_element(pads_arr, 0);
    if (!pad_obj) {
        GST_ERROR_OBJECT(element, "<%s>: Error retrieving a pad", G_STRFUNC);
        return NULL;
    }

    media_type_str = json_object_get_string_member(pad_obj,
        JSON_FIELD_NAME_MEDIA_TYPE);
    if (!media_type_str) {
        GST_ERROR_OBJECT(element, "<%s>: Error retrieving a pad media type",
            G_STRFUNC);
        return NULL;
    }

    caps_str = g_string_new(media_type_str);

    params_arr = json_object_get_array_member(pad_obj, JSON_FIELD_NAME_PARAMS);
    if (!params_arr) {
        GST_ERROR_OBJECT(element, "<%s>: Error retrieving pad parameters",
            G_STRFUNC);
        goto fixate_src_caps_out;
    }

    for (i = 0; i < json_array_get_length(params_arr); ++i) {
        param_obj = json_array_get_object_element(params_arr, i);
        if (!param_obj) {
            GST_ERROR_OBJECT(element, "<%s>: Error retrieving a pad parameter",
                G_STRFUNC);
            goto fixate_src_caps_out;
        }

        name_str = json_object_get_string_member(param_obj,
            JSON_FIELD_NAME_NAME);
        if (!name_str) {
            GST_ERROR_OBJECT(element, "<%s>: Error retrieving a parameter name",
                G_STRFUNC);
            goto fixate_src_caps_out;
        }

        type_str = json_object_get_string_member(param_obj,
            JSON_FIELD_NAME_TYPE);
        if (!type_str) {
            GST_ERROR_OBJECT(element, "<%s>: Error retrieving a parameter type",
                G_STRFUNC);
            goto fixate_src_caps_out;
        }

        json_field_str = json_object_get_string_member(param_obj,
            JSON_FIELD_NAME_JSON_FIELD);
        if (!json_field_str) {
            GST_INFO_OBJECT(element, "<%s>: Unable to retrieve a JSON field "
                "name", G_STRFUNC);

            values_str = json_object_get_string_member(param_obj,
                JSON_FIELD_NAME_VALUES);
            if (!values_str) {
                GST_ERROR_OBJECT(element, "<%s>: Failed to retrieve "
                    "parameter values", G_STRFUNC);
                goto fixate_src_caps_out;
            }

            if (g_strrstr(values_str, "-"))
                tokens = g_strsplit(values_str, "-", 2);
            else
                tokens = g_strsplit(values_str, ",", -1);

            g_string_append_printf(caps_str, ", %s=(%s)%s", name_str, type_str,
                g_strstrip(tokens[0]));

            g_strfreev(tokens);

            continue;
        }

        if (!fixate_caps(caps_str, root_obj, name_str, type_str,
            json_field_str)) {
            GST_ERROR_OBJECT(element, "<%s>: Error processing the `%s` "
                    "parameter type", G_STRFUNC, type_str);
            goto fixate_src_caps_out;
        }
    }

    fixed_caps = gst_caps_from_string(caps_str->str);

    GST_WARNING_OBJECT(element, "<%s>: Fixated source caps: %s", G_STRFUNC,
        caps_str->str);

fixate_src_caps_out:
    g_string_free(caps_str, TRUE);
    return fixed_caps;
}

gboolean
gst_simaai_caps_query(GstElement *element, GstSimaaiCaps *simaai_caps,
    GstQuery *query, GstPadDirection pad_direction)
{
    GstCaps *caps, *filter = NULL, *result;

    if (pad_direction == GST_PAD_SINK)
        caps = simaai_caps->sink_caps;
    else
        caps = simaai_caps->src_caps;

    gst_query_parse_caps(query, &filter);
    if (filter)
        result = gst_caps_intersect(caps, filter);
    else
        result = gst_caps_ref(caps);

    gst_query_set_caps_result(query, result);

    GST_WARNING_OBJECT(element, "<%s>: Resulting %s caps: %s", G_STRFUNC,
        pad_direction == GST_PAD_SINK ? "sink" : "source",
        gst_caps_to_string(result));

    gst_caps_unref(result);

    return TRUE;
}

gboolean
gst_simaai_caps_negotiate(GstElement *element, GstSimaaiCaps *simaai_caps)
{
    GstPad *src_pad;
    GstCaps *src_caps = simaai_caps->src_caps;
    GstCaps *peer_caps, *fixated_caps;
    gboolean retval = TRUE;

    src_pad = gst_element_get_static_pad(element, "src");
    if (!src_pad) {
        GST_ERROR_OBJECT(element, "<%s>: Error retrieving source pad",
            G_STRFUNC);
        return FALSE;
    }

    peer_caps = gst_pad_peer_query_caps(src_pad, src_caps);

    if (gst_caps_is_empty(peer_caps)) {
        GST_ERROR_OBJECT(element, "<%s>: Error negotiating caps", G_STRFUNC);
        retval = FALSE;
        goto negotiate_out;
    }

    if (gst_caps_is_any(src_caps))
        goto negotiate_out;

    if (!gst_caps_is_fixed(peer_caps)) {
        fixated_caps = gst_simaai_caps_fixate_src_caps(element, simaai_caps,
            peer_caps);
        if (!fixated_caps) {
            GST_ERROR_OBJECT(element, "<%s>: Error fixating caps", G_STRFUNC);
            retval = FALSE;
            goto negotiate_out;
        }

        gst_pad_push_event(src_pad, gst_event_new_caps(fixated_caps));
        //gst_aggregator_set_src_caps(element, fixated_caps);

        GST_WARNING_OBJECT(element, "<%s>: Negotiated source caps: %s",
            G_STRFUNC, gst_caps_to_string(fixated_caps));

        gst_caps_unref(fixated_caps);
    } else {
        gst_pad_push_event(src_pad, gst_event_new_caps(peer_caps));
        //gst_aggregator_set_src_caps(element, fixated_caps);

        GST_WARNING_OBJECT(element, "<%s>: Negotiated source caps: %s",
            G_STRFUNC, gst_caps_to_string(peer_caps));
    }

negotiate_out:
    gst_caps_unref(peer_caps);
    gst_object_unref(src_pad);
    return retval;
}

gboolean
gst_simaai_caps_parse_config(GstElement *element, GstSimaaiCaps *simaai_caps,
    const gchar *config)
{
    JsonNode *root;
    JsonObject *root_obj, *caps_obj;

    if (!json_parser_load_from_file(simaai_caps->parser, config, NULL)) {
        GST_ERROR_OBJECT(element, "<%s>: Error loading a JSON stream",
            G_STRFUNC);
        return FALSE;
    }

    root = json_parser_get_root(simaai_caps->parser);
    if (!JSON_NODE_HOLDS_OBJECT(root)) {
        GST_ERROR_OBJECT(element, "<%s>: Error retrieving a top level node",
            G_STRFUNC);
        return FALSE;
    }
    root_obj = json_node_get_object(root);

    caps_obj = json_object_get_object_member(root_obj, JSON_FIELD_NAME_CAPS);
    if (!caps_obj)
        GST_WARNING_OBJECT(element, "<%s>: Failed to retrieve caps from JSON",
            G_STRFUNC);

    if (!parse_pads(element, simaai_caps, caps_obj, GST_PAD_SINK))
        GST_WARNING_OBJECT(element, "<%s>: Failed to parse sink caps from JSON",
            G_STRFUNC);

    if (!parse_pads(element, simaai_caps, caps_obj, GST_PAD_SRC))
        GST_WARNING_OBJECT(element, "<%s>: Failed to parse source caps from "
            "JSON", G_STRFUNC);

    simaai_caps->config = g_strdup(config);

    return TRUE;
}

GstSimaaiCaps *
gst_simaai_caps_init(void)
{
    GstSimaaiCaps *simaai_caps = g_new0(GstSimaaiCaps, 1);

    simaai_caps->sink_caps = gst_caps_new_any();
    simaai_caps->src_caps = gst_caps_new_any();

    simaai_caps->parser = json_parser_new();

    return simaai_caps;
}

void
gst_simaai_caps_free(GstSimaaiCaps *simaai_caps)
{
    g_free(simaai_caps->config);

    g_object_unref(simaai_caps->parser);

    gst_caps_unref(simaai_caps->sink_caps);
    gst_caps_unref(simaai_caps->src_caps);

    g_free(simaai_caps);
}
