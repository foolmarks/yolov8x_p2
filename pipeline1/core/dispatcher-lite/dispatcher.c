#include <stdio.h>
#include <sys/time.h> 

#include "dispatcher.h"
#include <simaai/sgp_transport.h>
#include <simaai/msgs/sgp.h>
#include <simaai/simaai_memory.h>
#include <simaai/sgp_types.h>
#include <simaai/platform/simaevxxipc.h>
#include <simaai/platform/simahostops.h>
#include <simaai/gstsimaaiallocator.h>

#include "dispatcher_common.h"
#include "simamm.h"
#include "utils.h"

enum
{
  PROP_CALLBACK = 1,
  N_PROPERTIES
};

static GParamSpec *obj_properties[N_PROPERTIES] = { NULL};

#define GST_CAT_DEFAULT buffer_data_exchanger_debug
GST_DEBUG_CATEGORY_STATIC (GST_CAT_DEFAULT);

G_DEFINE_TYPE_WITH_CODE(BufferDataExchanger,
                        buffer_data_exchanger,
                        GST_TYPE_ELEMENT,
                        GST_DEBUG_CATEGORY_INIT (GST_CAT_DEFAULT, "simaoffloader", 0,
                                                 "debug category for simaoffloader"))

static void
buffer_data_exchanger_set_property (GObject      *object,
                                    guint         property_id,
                                    const GValue *value,
                                    GParamSpec   *pspec)
{
    BufferDataExchanger *self = GST_BUFFERDATAEXCHANGER (object);
    switch (property_id)
    {
      case PROP_CALLBACK:
        self->callback = g_value_get_pointer (value);
        break;
      default:
        G_OBJECT_WARN_INVALID_PROPERTY_ID (object, property_id, pspec);
        break;
    }
}

static void
buffer_data_exchanger_finalize (GObject *object)
{
    BufferDataExchanger * self = GST_BUFFERDATAEXCHANGER (object);

    self->is_initialized = FALSE;

    if (self->pool) {
        free_simaai_memory_buffer_pool(self->pool);
        self->pool = NULL;
    }

    // Used for standalone testing
    if (self->input_buff) {
      simaai_memory_unmap(self->input_buff);
      simaai_memory_free(self->input_buff);
      self->input_buff = NULL;
    }

    g_mutex_clear(&self->dispatcher_mutex);

    if (self->type) {
        g_free(self->type->p_mla_handle);
    }
    g_free(self->type);
    return;
}

#define TARGET_CPU_DEFAULT 0

static void
buffer_data_exchanger_class_init (BufferDataExchangerClass *klass)
{
    GObjectClass *object_class = G_OBJECT_CLASS (klass);

    static const gchar *tags[] = { NULL };
    if (gst_meta_get_info("GstSimaMeta") == NULL)
        gst_meta_register_custom ("GstSimaMeta", tags, NULL, NULL, NULL);

    object_class->set_property = buffer_data_exchanger_set_property;
    obj_properties[PROP_CALLBACK] =
        g_param_spec_pointer ("callback",
                              "Callback",
                              "Received Callback",
                              G_PARAM_CONSTRUCT_ONLY | G_PARAM_WRITABLE);

    g_object_class_install_properties (object_class,
                                       N_PROPERTIES,
                                       obj_properties);

    object_class->finalize = buffer_data_exchanger_finalize;
}

static void
buffer_data_exchanger_init (BufferDataExchanger *self)
{
    gst_simaai_memory_init_once();

    self->callback = NULL;
    self->is_initialized = FALSE;
    /* self->out_offset_idx = 0; */
    self->pool = NULL;
    self->mem_flags = GST_SIMAAI_MEMORY_FLAG_CACHED;
    self->frame_id = 0;
    self->batch_size = 0;
    self->batch_size_model = 0;
    self->in_tensor_size = 0;
    self->out_tensor_size = 0;
    self->type = NULL;

    g_mutex_init(&self->dispatcher_mutex);
}

gboolean dispatcher_init(BufferDataExchanger * self, simaai_params_t * params)
{
    gboolean ret = FALSE;

    // Only get CPU to decide the path here, rest move it to dispatcher
    self->cpu = *((int *)parser_get_int(params, "cpu"));
    self->params = params;
    self->type = (sgp_cpu_variant_t *) g_malloc0(sizeof(sgp_cpu_variant_t));

    // Initialize communication with dispatcher
    switch(self->cpu) {
    case SIMA_CPU_A65: {
        // Build the data structures before calling init.
        ret = INIT(host, self);
        if (ret == FALSE) {
            GST_DEBUG_OBJECT (self, "[PROCESS] Failed to initialize platform type:0x%x", self->cpu);
            break;
        }

        ret = CONFIGURE(host, self);
        if (ret == FALSE) {
            GST_DEBUG_OBJECT (self, "[PROCESS] Failed to configure cpu type:0x%x, ret = %d", self->cpu, ret);
            break;
        }
        break;
    }
    case SIMA_CPU_EVXX: {
        // Build the data structures before calling init.
        ret = INIT(EVXX, self);
        if (ret == FALSE) {
            GST_DEBUG_OBJECT(self, "[PROCESS] Failed to initialize platform type:0x%x", self->cpu);
            break;
        }

        /* ret = CONFIGURE(EVXX, self); */
        /* if (ret == FALSE) { */
        /*     GST_DEBUG_OBJECT (self, "[PROCESS] Failed to configure cpu type:0x%x, ret = %d", self->cpu, ret); */
        /*     break; */
        /* } */
        break;
    }
    case SIMA_CPU_MLA:
    case SIMA_CPU_MOSAIC:
        self->type->p_mla_handle = (sgp_mla_t *) g_malloc(sizeof(sgp_mla_t));
        if (self->type->p_mla_handle == NULL)
            return FALSE;

        ret = INIT(MLA, self);
        break;
    case SIMA_CPU_DEV:
        self->batch_size = *((int *)parser_get_int(params, "batch_size"));
        self->batch_size_model = *((int *)parser_get_int(params, "batch_sz_model"));

        self->in_tensor_size = *((int *)parser_get_int(params, "in_tensor_sz"));
        self->out_tensor_size = *((int *)parser_get_int(params, "out_tensor_sz"));

        ret = mla_client_init(self);
        break;
        // Development only
    default:
        GST_DEBUG_OBJECT(self, "CPU type unknown %d", self->cpu);
        return ret;
    }

    return ret;
}

GstFlowReturn buffer_data_dispatcher_send(BufferDataExchanger *self,
                                          GstBuffer * buffer)
{
    // Request to h/w flow ends here
    // Call from the global dispatchers
    int32_t ret = 0;
    gint64 in_buf_id = 0;
    gint64 in_buf_offset = 0;
    gint64 in_pcie_buf_id = 0;
    gboolean is_pcie = FALSE;
    gchar* stream_id;
    guint64 timestamp;
    
    int dump_data = *((int *)parser_get_int(self->params, "dump_data"));
    GstMapInfo map;
    gst_buffer_map(buffer, &map, GST_MAP_READ);
    
    GstCustomMeta * meta = gst_buffer_get_custom_meta(buffer, "GstSimaMeta");
    if (meta != NULL) {
	 GstStructure * s = gst_custom_meta_get_structure(meta);
	 if (s == NULL) {
	      g_message("Returning from ");
	      return GST_FLOW_OK;
	 } else {
        
        // Check if PCIe related metadta exists
        if ((gst_structure_get_int64(s, "pcie-buffer-id", &in_pcie_buf_id) == TRUE)) {
            is_pcie = TRUE;
            GST_DEBUG("[process2] pcie-buffer-id = %ld", in_pcie_buf_id);
        }

	    if ((gst_structure_get_int64(s, "buffer-id", &in_buf_id) == TRUE) &&
		(gst_structure_get_int64(s, "frame-id", &self->frame_id) == TRUE) &&
		(gst_structure_get_int64(s, "buffer-offset", &in_buf_offset) == TRUE) &&
        (gst_structure_get_uint64(s, "timestamp", &timestamp) == TRUE))
        {
            stream_id = (gchar *)gst_structure_get_string(s, "stream-id");
            GST_DEBUG("Copied metadata, in_buf_offset %ld", in_buf_offset);
        }
	 }
    }

    GstBuffer * outbuf;
    GstFlowReturn res = gst_buffer_pool_acquire_buffer(self->pool, &outbuf, NULL);

    if (G_LIKELY (res == GST_FLOW_OK)) {
      GST_DEBUG_OBJECT (self, "Output buffer from pool: %p", outbuf);
    } else {
      GST_ERROR_OBJECT (self, "Failed to allocate buffer");
      gst_buffer_unmap(buffer, &map);
      return res;
    }

    guintptr out_buf_id = gst_simaai_memory_get_phys_addr(gst_buffer_peek_memory(outbuf, 0));

    // TODO: Cleanup
    switch (self->cpu) {
    case SIMA_CPU_EVXX: {
	 uint32_t no_of_bufs = get_num_of_outbufs(self->params);

         // build the evxx specific request message
	 sgp_ev_req_t * req = (sgp_ev_req_t *) g_malloc(sizeof(sgp_ev_req_t));
	 sgp_ev_resp_t resp;
	 memset(req, 0, sizeof(sgp_ev_req_t));

	 // Build the request specific message
	 req->magic = SGPMSG_REQMAGIC;
	 req->graph_id = *((uint8_t *)parser_get_int(self->params, "graph_id"));
	 // Update frame from message;
	 req->req_id = self->frame_id;
	 // Pre-allocated
	 req->osize = get_output_sz(self->params);

         req->oaddr = out_buf_id;
	 req->in_img_addr = buffer_id_to_paddr(self->in_buf_id);
	 req->in_img_size = map.size; // get_img_sz(self->params);
	 req->iaddr = buffer_id_to_paddr(self->in_buf_id); // get_iaddr(self->ev_ki, self->params, msg);
	 req->request_type = SGP_REQUEST_DATA;
	 
	 // TODO: Dynamically calculate this
	 // No need of this when you are not allocating
	 // req->isize = get_input_sz(self->params);
	 GST_DEBUG_OBJECT(self, "Posting work to evxx graph_id:0x%x, iaddr:0x%x, oaddr:0x%x, imgaddr:0x%x, req:0x%x",
			  req->graph_id, req->iaddr, req->in_img_addr, req->oaddr, req);
	 GST_DEBUG_OBJECT(self, "Request id:0x%x, msg_frame:0x%x, msg addr:0x%x",
			  req->req_id, self->frame_id, map);

	 // Post work to dispatcher
	 POST(EVXX, self, req);

	 // Wait for graph to be done
	 GST_DEBUG_OBJECT(self, "wait done for request id:0x%x", req->req_id);
	 resp = WAIT_DONE(EVXX, self, req->req_id);
	 if (resp.op_result != 0) {
	      g_free(req);
	      goto drop_and_continue;
	 }
	 
	 if (dump_data) {
	      ret = GET_OUTBUF(EVXX, self, self->frame_id, outbuf);
	      if (ret != 0)
		   goto drop_and_continue;
	 }
	 g_free(req);
	 break;
    }
    case SIMA_CPU_MOSAIC:
    case SIMA_CPU_MLA: {
        uint32_t iaddr = buffer_id_to_paddr(in_buf_id);
        uint32_t oaddr = out_buf_id;

        g_message("iaddr = %u oaddr = %u", iaddr, oaddr);

        ret = RUN_MODEL(MLA, self, (iaddr + in_buf_offset), oaddr);
        if (ret != 0) {
            GST_ERROR("Failed to run model ret:0x%x", ret);
            goto drop_and_continue;
        }

        if (dump_data) {
            ret = GET_OUTBUF(MLA, self, self->frame_id, outbuf);
            if (ret != 0)
                goto drop_and_continue;
        }
        break;
    }
    case SIMA_CPU_DEV: {
        uint64_t iaddr = in_buf_id;
        uint64_t oaddr = out_buf_id;
        size_t out_sz = get_output_sz(self->params);

        if (out_sz <= 0) {
            GST_ERROR("Out Size is defined zero");
            goto drop_and_continue;
        }

        ret = mla_client_run_model(self, iaddr, oaddr, map.size);
        if (ret != 0) {
            GST_ERROR("Failed to run model ret:0x%x", ret);
            goto drop_and_continue;
        }

        if (dump_data) {
	    ret = mla_client_write_output_buffer(self, self->frame_id, outbuf);
            if (ret != 0)
                goto drop_and_continue;
        }
	break;
    }
    default:
        break;
    }

    /* // Update metadata */
    GstCustomMeta * meta2 = gst_buffer_add_custom_meta(outbuf, "GstSimaMeta");
    if (meta2 == NULL) {
	 GST_ERROR("Unable to add metadata info to the buffer");
	 return GST_FLOW_ERROR;
    }

    GstStructure *s = gst_custom_meta_get_structure (meta2);
    if (s != NULL) {
        gst_structure_set (s,
            "buffer-id", G_TYPE_INT64, out_buf_id,
			"buffer-name", G_TYPE_STRING, self->params->node_name,
			"buffer-offset", G_TYPE_INT64, (gint64)0,
			"frame-id", G_TYPE_INT64, self->frame_id,
            "stream-id", G_TYPE_STRING, stream_id,
            "timestamp", G_TYPE_UINT64, timestamp, NULL);
        
        // Add PCIe metadata if exists
        if (is_pcie) {
            gst_structure_set(s, "pcie-buffer-id", G_TYPE_INT64, in_pcie_buf_id, NULL);
        }
    
	 GST_DEBUG_OBJECT(self, "Attaching meta information out_buf_id:[%lld], node_name:[%s], frame_id:[%lld], stream_id[%s], timestamp:[%ld]",
                          out_buf_id, self->params->node_name, self->frame_id, stream_id, timestamp);
    }

    gst_buffer_unmap(buffer, &map);
    return buffer_data_dispatcher_recv(self, outbuf);

drop_and_continue:
    /* GST_INFO("Something's wrong for frame_id: 0x%x, dropping & continuing", msg->frame_seq); */
    gst_buffer_unmap(buffer, &map);
    gst_buffer_unref(outbuf);
    return GST_FLOW_OK;
}

GstFlowReturn buffer_data_dispatcher_recv(BufferDataExchanger *self,
                                          GstBuffer * buffer)
{
    self->callback->buffer_received(buffer, self->callback->priv);
    return GST_FLOW_OK;
}

BufferDataExchanger *buffer_data_exchanger_new (BufferDataExchangerCallback *pcallback)
{
  BufferDataExchanger *pexchanger =
        g_object_new(GST_BUFFERDATAEXCHANGER_TYPE,
                     "callback", pcallback,
                     NULL);

  return pexchanger;
}
