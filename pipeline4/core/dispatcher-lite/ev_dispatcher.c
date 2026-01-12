#include <stdio.h>
#include <errno.h>

#include <simaai/parser.h>
#include <simaai/parser_types.h>

#include "dispatcher.h"
#include "dispatcher_common.h"

#include "ev_dispatcher.h"
#include "utils.h"
#include <simaai/ev_c_api.h>

#include <simaai/gstsimaaiallocator.h>

#define GST_CAT_DEFAULT ev_dispatcher_debug
GST_DEBUG_CATEGORY_STATIC (GST_CAT_DEFAULT);

gboolean EVXX_init(BufferDataExchanger * self)
{
	int32_t ret = 0;
	gboolean gst_ret = TRUE;

	GST_DEBUG_CATEGORY_GET (GST_CAT_DEFAULT, "simaoffloader");

        int debug = *((int *)parser_get_int(self->params, "debug"));

        if (debug) {
            ret = EVXX_prepare_in_buff(self);
            if (ret != 0) {
                GST_ERROR("Unable to allocate from sima-mem %s, ret:%d", strerror(errno), ret);
		goto err;
            }
        }

	ret = EVXX_prepare_out_buff(self);
	if (ret != 0) {
            GST_ERROR("Unable to allocate from sima-mem %s, ret:%d", strerror(errno), ret);
            goto err;
	}

	ret = ev_init();
	if (ret != 0) {
            GST_ERROR("Failed ev_init %s", strerror(errno));
            goto err;
        }

	return gst_ret;
err:
        EVXX_fini(self);
	return (gst_ret = FALSE);
}

void EVXX_fini(BufferDataExchanger * self)
{
    if (free_simaai_memory_buffer_pool(self->pool))
        self->pool = NULL;
}

int32_t EVXX_prepare_out_buff(BufferDataExchanger * self)
{
	int32_t res = 0;

        sima_cpu_e next_cpu = *((int *)parser_get_int(self->params, "next_cpu"));
        if (next_cpu < 0) {
            GST_ERROR("Unknown CPU request 0x%x", next_cpu);
            return -1;
        }

        GstSimaaiMemoryFlags mem_target = get_mem_target(next_cpu)
                                          | GST_SIMAAI_MEMORY_FLAG_DEFAULT;

        uint32_t no_of_bufs = *((int32_t *)parser_get_int(self->params, "no_of_outbuf"));
        ssize_t out_sz = *((int32_t *)parser_get_int(self->params, "out_sz"));

        if (out_sz <= 0) {
            GST_ERROR_OBJECT(self, "Unknown output size :%zd", out_sz);
            return -1;
        }

        // Default to 'MIN_POOL_SIZE'
        if (no_of_bufs < MIN_POOL_SIZE)
            no_of_bufs = MIN_POOL_SIZE;

        self->pool = allocate_simaai_memory_buffer_pool(GST_OBJECT (self), out_sz,
                                                        no_of_bufs, no_of_bufs,
                                                        mem_target);

        if (self->pool == NULL)
          return -1;

        GST_DEBUG_OBJECT (self, "%u buffers of size %zu, target 0x%08" PRIx32,
                          no_of_bufs, out_sz, mem_target);
        
        return res;
}

static void memory_info_wrapper(simaai_memory_t *buf)
{
    GST_DEBUG(
        "Buffer: size = %zu, phys address = 0x%0lx\n",
        simaai_memory_get_size(buf),
        simaai_memory_get_phys(buf));
}

// Used for standalone testing only
int32_t EVXX_prepare_in_buff(BufferDataExchanger * self)
{
	FILE *ifp;
	size_t isz, risz;
	void *ivaddr;
	char *dumpaddr;
	int32_t res = 0, i;

        char * inpath = (const char *)parser_get_string(self->params, "inpath");

	ifp = fopen(inpath, "r");
	if(ifp == NULL) {
		res = -1;
		goto err_ifp;
	}

	fseek(ifp, 0L, SEEK_END);
	isz = ftell(ifp);
	fseek(ifp, 0L, SEEK_SET);

	GST_DEBUG_OBJECT(self, "TEST: Size of test input file %s: %d\n", inpath, isz);

	self->input_buff = simaai_memory_alloc(isz, SIMAAI_MEM_TARGET_EV74);
	if(self->input_buff == NULL) {
		res = -2;
		goto err_ialloc;
	} else {
		memory_info_wrapper(self->input_buff);
	}

	ivaddr = simaai_memory_map(self->input_buff);
	if(ivaddr == NULL) {
		GST_ERROR("TEST: FAILED to map buffer");
		res = -4;
		goto err_ifp;
	}

	risz = fread(ivaddr, 1, isz, ifp);
	if (isz != risz) {
		GST_ERROR("TEST: FAILED to read data risz: %ld, isz: %ld, vaddr : 0x%x, error: %s", risz, isz, ivaddr, strerror(errno));
		res = -4;
		goto err_iread;
	}

	simaai_memory_unmap(self->input_buff);
	fclose(ifp);
	return res;

err_iread:
	simaai_memory_unmap(self->input_buff);
err_oalloc:
	simaai_memory_free(self->input_buff);
err_ialloc:
	fclose(ifp);
err_ifp:
	return res;
}

void EVXX_post (BufferDataExchanger * self, sgp_ev_req_t * req)
{
        GST_DEBUG_OBJECT(self, "Inisde EVXX_post graph_id:0x%x, req_id:0x%x", req->graph_id, req->req_id);
        ev_post(req);
	return;
}

sgp_ev_resp_t EVXX_wait_done (BufferDataExchanger * self, uint32_t id)
{
        GST_DEBUG_OBJECT(self,"Inisde EVXX_wait_for_done");
	return ev_wait_for_done(id);
}

gboolean EVXX_configure (BufferDataExchanger * self)
{
	int ret = ev_configure(self->params);
        GST_DEBUG_OBJECT(self, "[EV_DISPATCHER] EVXX_configure ret:%d", ret);
	return (ret == 0) ? TRUE : FALSE;
}

int32_t
EVXX_write_output_buffer(BufferDataExchanger * self, uint32_t id, GstBuffer * buffer)
{
	FILE *ofp;
	size_t osz = *((int32_t *)parser_get_int(self->params, "out_sz"));
        size_t wosz;
	void *ovaddr;
	int32_t res = 0, i;
	char *dumpaddr;
	char full_opath[256];

	snprintf(full_opath, sizeof(full_opath) - 1, "/tmp/%s-%d.out", self->params->node_name, id);

	ofp = fopen(full_opath, "w");
	if(ofp == NULL) {
		res = -1;
		goto err_ofp;
	}

	GstMapInfo map;
	gst_buffer_map(buffer, &map, GST_MAP_READ);

	wosz = fwrite((char *)(map.data), 1, osz, ofp);
	if(osz != wosz) {
		res = -3;
		gst_buffer_unmap(buffer, &map);
		goto err_owrite;
	}
	gst_buffer_unmap(buffer, &map);     // Reinitialize these somewhere else

err_owrite:
err_omap:
	fclose(ofp);
err_ofp:
	return res;
}

static int _id = 0;

int32_t
EVXX_write_input_buffer(BufferDataExchanger * self, void * vaddr, gsize sz)
{
     FILE *ofp;
     size_t wosz;
     int32_t res = 0, i;
     char *dumpaddr;
     char full_opath[256];

     snprintf(full_opath, sizeof(full_opath) - 1, "/tmp/%s-%d.in", self->params->node_name, _id++);

     ofp = fopen(full_opath, "w");
     if(ofp == NULL) {
	  res = -1;
	  goto err_ofp;
     }

     wosz = fwrite(vaddr, 1, sz, ofp);
     if(sz != wosz) {
	  res = -3;
	  goto err_owrite;
     }

err_owrite:
err_omap:
     fclose(ofp);
err_ofp:
     return res;
}
