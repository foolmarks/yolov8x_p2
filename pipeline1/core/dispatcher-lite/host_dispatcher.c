#include <stdio.h>
#include <errno.h>

#include "dispatcher.h"
#include "dispatcher_common.h"
#include "utils.h"

#include "host_dispatcher.h"
#include <simaai/ev_c_api.h>

#define GST_CAT_DEFAULT host_dispatcher_debug
GST_DEBUG_CATEGORY_STATIC (GST_CAT_DEFAULT);

gboolean host_init(BufferDataExchanger * self)
{
	int32_t ret = 0;

	GST_DEBUG_CATEGORY_GET (GST_CAT_DEFAULT, "simaoffloader");

        int debug = *((int *)parser_get_int(self->params, "debug"));

        if (debug) {
            ret = PREPARE_INBUF(host, self);
            if (ret != 0) {
		GST_ERROR("Unable to allocate from sima-mem %s", strerror(errno));
		goto err;
            }
        }

	ret = PREPARE_OUTBUF(host, self);
	if (ret != 0) {
		GST_ERROR("Unable to allocate from sima-mem %s", strerror(errno));
		goto err;
	}

	ret = a65_host_init();
	if (ret != 0) {
	        host_fini(self);
		goto err;
	}

	return (ret == 0 ? TRUE: FALSE);
err:
	return FALSE;
}

void host_fini(BufferDataExchanger * self)
{
        if (free_simaai_memory_buffer_pool(self->pool))
                self->pool = NULL;
}

int32_t host_prepare_out_buff(BufferDataExchanger * self)
{
	int32_t res = 0;

        sima_cpu_e cpu = *((int *)parser_get_int(self->params, "next_cpu"));

        GstSimaaiMemoryFlags mem_target = get_mem_target(cpu)
                                          | GST_SIMAAI_MEMORY_FLAG_DEFAULT;

        ssize_t out_sz = get_output_sz(self->params);
        uint32_t no_of_bufs = *((int32_t *)parser_get_int(self->params, "no_of_outbuf"));

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
    GST_DEBUG("Buffer: size = %zu, phys address = 0x%0lx\n",
              simaai_memory_get_size(buf),
              simaai_memory_get_phys(buf));
}

// Used for standalone testing only
int32_t host_prepare_in_buff(BufferDataExchanger * self)
{
	FILE *ifp;
	size_t isz, risz;
	void *ivaddr;
	char *dumpaddr;
	int32_t res = 0, i;

	/* strcpy(self->cfg->inpath, "/root/ev74_test.yuv420"); */
        char * inpath = get_input_path(self->params);
	ifp = fopen(inpath, "r");
	if(ifp == NULL) {
		res = -1;
		goto err_ifp;
	}

	fseek(ifp, 0L, SEEK_END);
	isz = ftell(ifp);
	fseek(ifp, 0L, SEEK_SET);

	GST_DEBUG_OBJECT(self,"TEST: Size of test input file %s: %d\n", inpath, isz);

	self->input_buff = simaai_memory_alloc(isz, SIMAAI_MEM_TARGET_GENERIC);
	if(self->input_buff == NULL) {
		res = -2;
		goto err_ialloc;
	}
        /* else { */
        /*     memory_info_wrapper(self->input_buff); */
	/* } */

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

void host_post (BufferDataExchanger * self, sgp_host_req_t * req)
{
        a65_host_post(req);
	return;
}

sgp_host_resp_t host_wait_done (BufferDataExchanger * self, uint32_t id)
{
	return a65_host_wait_for_done(id);
}

gboolean host_configure (BufferDataExchanger * self)
{
	int ret = a65_host_configure(self->params);
	return (ret == 0) ? TRUE : FALSE;
}

int32_t
host_write_output_buffer(BufferDataExchanger * self, uint32_t id, GstBuffer * buffer)
{
	FILE *ofp;
	size_t osz = get_output_sz(self->params), wosz;
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
		goto err_owrite;
	}

#if 0
	dumpaddr = (char *)ovaddr;
	GST_DEBUG("TEST: Successfully stored output data: ");
	for(i = 0; i < 16; i++)
		GST_DEBUG("%02x", dumpaddr[i]);
	GST_DEBUG("\n");
#endif
err_owrite:
	gst_buffer_unmap(buffer, &map);
err_omap:
	fclose(ofp);
err_ofp:
	return res;
}
