#include <time.h>

#include "utils.h"

#include <simaai/msgs/sgp.h>
// TODO: Auto include this when compiling, this is usecase specific
#include <simaai/platform/simaevxxipc.h>
#include <simaai/platform/simahostops.h>

void dump_buf_id_list(const sgp_msg_t * msg)
{
    if (msg != NULL) {
        GST_DEBUG("SGP MSG INFO");
        GST_DEBUG("msg->magic = 0x%x", msg->magic);
        GST_DEBUG("msg->frame_seq = 0x%x", msg->frame_seq);

        int i = 0;
        for (i = 0; i < SGPMSG_MAX_BUF_IDS; i++) {
            GST_DEBUG("NODE_NAME:BUF_ID = [%s]:[%d] OFFSET = [%d]", msg->buf_id_list[i].node_name, msg->buf_id_list[i].buf_id, msg->buf_id_list[i].offset);
        }
    }
}

simamm_buffer_id_t get_bufid(simaai_params_t * params, const sgp_msg_t * msg, const char * name)
{
    if (!msg)
        return -1;

    char *in_node_name = ((char *)parser_get_string(params, name));
    if (!in_node_name) {
        GST_ERROR("Please define input buf name");
        return -1;
    }

    int i = 0;
    for (i = 0; i < SGPMSG_MAX_BUF_IDS; i++) {
        if (!strcmp(msg->buf_id_list[i].node_name, in_node_name)) {
            return msg->buf_id_list[i].buf_id;
        }
    }

    GST_ERROR("Node name not found");
    return -1;
}

simamm_buffer_id_t get_bufid_generic(simaai_params_t * params, const sgp_msg_t * msg, const char * name)
{
    if (!msg)
        return -1;

    int i = 0;
    for (i = 0; i < SGPMSG_MAX_BUF_IDS; i++) {
        if (!strcmp(msg->buf_id_list[i].node_name, name)) {
            return msg->buf_id_list[i].buf_id;
        }
    }

    GST_ERROR("Node name not found");
    return -1;
}

simamm_buffer_id_t get_img_source_bufid(simaai_params_t * params, const sgp_msg_t * msg)
{
    if (!msg)
        return -1;

    char *in_node_name = "in-img-source";
    int i = 0;
    for (i = 0; i < SGPMSG_MAX_BUF_IDS; i++) {
        GST_DEBUG("GET_IMG_SOUCE_BUFID: name = %s, bufid = %d", msg->buf_id_list[i].node_name, msg->buf_id_list[i].buf_id);
        if (!strcmp(msg->buf_id_list[i].node_name, in_node_name)) {
            return msg->buf_id_list[i].buf_id;
        }
    }

    return -1;
}

size_t get_offset(simaai_params_t * params, const sgp_msg_t * msg, const char * name)
{
    if (!msg)
        return -1;

    char *in_node_name = ((char *)parser_get_string(params, name));
    if (!in_node_name) {
        GST_DEBUG("Please define input buf name");
        return -1;
    }

    int i = 0;
    for (i = 0; i < SGPMSG_MAX_BUF_IDS; i++) {
        if (!strcmp(msg->buf_id_list[i].node_name, in_node_name)) {
            return msg->buf_id_list[i].offset;
        }
    }

    GST_ERROR("Node name not found");
    return -1;
}

int32_t get_num_of_outbufs(simaai_params_t * params)
{
    return *((uint32_t *)parser_get_int(params, "no_of_outbuf"));
}

size_t get_img_source_offset(simaai_params_t * params, const sgp_msg_t * msg)
{
    if (!msg)
        return -1;

    char *in_node_name = "in-img-source";
    int i = 0;
    for (i = 0; i < SGPMSG_MAX_BUF_IDS; i++) {
        if (!strcmp(msg->buf_id_list[i].node_name, in_node_name)) {
            return msg->buf_id_list[i].offset;
        }
    }

    return -1;
}

size_t get_img_sz(simaai_params_t * params)
{
    int32_t h = 0, w = 0;
    int fmt = -1;

    h = *((uint32_t *)parser_get_int(params, "img_height"));
    w = *((uint32_t *)parser_get_int(params, "img_width"));

    fmt = *((int *)parser_get_int(params, "format"));

    switch(fmt) {
    case SIMA_FORMAT_RGB:
        return (h * w * 3);
    case SIMA_FORMAT_YUV420:
        return (h * w * 1.5);
    default:
        return -1;
    }
    return -1;
}

// TODO: FIXME, more dynamic configuration loading. May be write
// app-specific layers and let use call these APIs from there
ssize_t get_output_sz(simaai_params_t * params)
{
    ssize_t out_sz = *((int64_t *)parser_get_int(params, "out_sz"));
    return out_sz;
}

uint32_t get_input_sz(simaai_params_t * params)
{
    return *((uint32_t *)parser_get_int(params, "in_sz"));
}

const char * get_model_path(simaai_params_t * params)
{
    return ((char *)parser_get_string(params, "model_path"));
}

const char * get_node_name(simaai_params_t * params)
{
    return ((char *)parser_get_string(params, "node_name"));
}

int get_next_cpu(simaai_params_t * params)
{
    return *((int *)parser_get_int(params, "next_cpu"));
}

int get_cpu(simaai_params_t * params)
{
    return *((int *)parser_get_int(params, "cpu"));
}

const char * get_input_path(simaai_params_t * params)
{
    return ((char *)parser_get_string(params, "inpath"));
}

GstSimaaiMemoryFlags get_mem_target (sima_cpu_e cpu) {
    switch(cpu) {
    case SIMA_CPU_EVXX:
        return GST_SIMAAI_MEMORY_TARGET_EV74;
    case SIMA_CPU_MLA:
        return GST_SIMAAI_MEMORY_TARGET_DMS0;
    case SIMA_CPU_MOSAIC:
        return GST_SIMAAI_MEMORY_TARGET_DMS0;
    case SIMA_CPU_A65:
    case SIMA_CPU_DEV:
        return GST_SIMAAI_MEMORY_TARGET_GENERIC;
    default:
        return GST_SIMAAI_MEMORY_TARGET_GENERIC;
    }
}

long long cur_time_ms() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return  (tv.tv_sec) * 1000 + (tv.tv_usec) / 1000;
}

GstBufferPool * allocate_simaai_memory_buffer_pool(GstObject * obj,
                                                   guint buf_size,
                                                   guint min_buffers,
                                                   guint max_buffers,
                                                   GstSimaaiMemoryFlags flags)
{
    GstBufferPool * pool = gst_buffer_pool_new ();
    if (pool == NULL) {
        GST_ERROR_OBJECT (obj, "gst_buffer_pool_new failed");
        return NULL;
    }

    GstStructure *config = gst_buffer_pool_get_config (pool);
    if (config == NULL) {
        GST_ERROR_OBJECT (obj, "gst_buffer_pool_get_config failed");
        gst_object_unref (pool);
        return NULL;
    }

    gst_buffer_pool_config_set_params (config, NULL, buf_size, min_buffers, max_buffers);

    GstAllocationParams params;
    gst_allocation_params_init (&params);
    params.flags = (GstMemoryFlags)flags;

    gst_buffer_pool_config_set_allocator (config, gst_simaai_memory_get_allocator(),
                                          &params);

    gboolean res = gst_buffer_pool_set_config (pool, config);
    if (res == FALSE) {
        GST_ERROR_OBJECT (obj, "gst_buffer_pool_set_config failed");
        gst_object_unref (pool);
        return NULL;
    }

    res = gst_buffer_pool_set_active (pool, TRUE);
    if (res == FALSE) {
        GST_ERROR_OBJECT (obj, "gst_buffer_pool_set_active failed");
        gst_object_unref (pool);
        return NULL;
    }

    return pool;
}

gboolean free_simaai_memory_buffer_pool(GstBufferPool * pool)
{
    g_return_val_if_fail (pool != NULL, FALSE);
    g_return_val_if_fail (gst_buffer_pool_set_active (pool, FALSE), FALSE);
    gst_object_unref (pool);

    return TRUE;
}
