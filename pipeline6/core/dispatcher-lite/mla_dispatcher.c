#include <stdio.h>

#include "dispatcher.h"
#include "dispatcher_common.h"

#include "mla_dispatcher.h"
#include "utils.h"

#include <simaai/gstsimaaiallocator.h>

#define GST_CAT_DEFAULT mla_dispatcher_debug
GST_DEBUG_CATEGORY_STATIC (GST_CAT_DEFAULT);

void MLA_get_handle(BufferDataExchanger * self)
{

	uint64_t * mlac_handle = &self->type->p_mla_handle->handle;
	GST_DEBUG_OBJECT(self,"mla_handle = 0x%x", mlac_handle);

	void *idata[1] = { mlac_handle };
	void *odata[1] = { mlac_handle };
	int32_t res;
	sgp_param_t param[1];

	param[0].pdir = SGP_PDIR_OUT;
	param[0].ptype = SGP_PTYPE_PTR;
	param[0].size = SGP_PTYPE_PTR_SIZE;

	if (self->transport != NULL) {
		res = sgp_transport_rpc(self->transport,
					self->type->p_mla_handle->req_id++,
					SGP_MLA_GET_HANDLE,
					1,
					param,
					idata,
					odata);
	}
}

gboolean MLA_load_model(BufferDataExchanger * self)
{
	sgp_param_t param[2];
	uint64_t * mla_model = &self->type->p_mla_handle->model;
	/* strcpy(self->type->p_mla_handle->path, "/usr/lib/1dma71_ifm4.lm"); */
	/* strcpy(self->type->p_mla_handle->path, "/root/centernet_4dma.lm"); */
	char * path = get_model_path(self->params);
	void *idata[2] = { path, mla_model };
	void *odata[1] = { mla_model };
	int32_t res;

	param[0].pdir = SGP_PDIR_IN;
	param[0].ptype = SGP_PTYPE_BUFFER;
	param[0].size = strlen(path) + 1;
	param[1].pdir = SGP_PDIR_OUT;
	param[1].ptype = SGP_PTYPE_PTR;
	param[1].size = SGP_PTYPE_PTR_SIZE;

	GST_DEBUG_OBJECT(self, "TEST: Calling mla_model_p mla_load_model(%s)\n", path);

	sgp_transport_t ** t = self->transport;
	if (t != NULL) {
		res = sgp_transport_rpc(t,
					self->type->p_mla_handle->req_id++,
					SGP_MLA_LOAD_MODEL,
					2,
					param,
					idata,
					odata);

	} else {
		GST_ERROR("Transport handle not initialized\n");
	}

	GST_DEBUG_OBJECT(self,"TEST: Result: %d, Output parameter value: %#lx\n", res, mla_model);
	if (res == SGP_ERROR_NONE)
		return FALSE;
	else
		return TRUE;
}

int32_t MLA_prepare_out_buff(BufferDataExchanger * self)
{
	int32_t res = 0;
	simaai_memory_t * output_buff;

        sima_cpu_e next_cpu = get_next_cpu(self->params);

        GstSimaaiMemoryFlags mem_target = get_mem_target(next_cpu)
                                          | GST_SIMAAI_MEMORY_FLAG_DEFAULT;

        uint32_t no_of_bufs = *((int32_t *)parser_get_int(self->params, "no_of_outbuf"));
        ssize_t out_sz = get_output_sz(self->params);

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

// Used for standalone testing only
int32_t MLA_prepare_in_buff(BufferDataExchanger * self)
{
	FILE *ifp;
	size_t isz, risz, osz = 1024;
	void *ivaddr;
	char *dumpaddr;
	int32_t res = 0, i;
	simaai_memory_t *input_buff;

        char * inpath = get_input_path(self->params);
        GST_DEBUG_OBJECT(self, "Input path = %s", inpath);

        ifp = fopen(inpath, "r");
	if(ifp == NULL) {
		res = -1;
		goto err_ifp;
	}

	fseek(ifp, 0L, SEEK_END);
	isz = ftell(ifp);
	fseek(ifp, 0L, SEEK_SET);

	input_buff = simaai_memory_alloc(isz, SIMAAI_MEM_TARGET_DMS0);
	if(input_buff == NULL) {
		res = -2;
		goto err_ialloc;
	}

	ivaddr = simaai_memory_map(input_buff);
	if(ivaddr == NULL) {
		res = -4;
		goto err_ifp;
	}

	risz = fread(ivaddr, 1, isz, ifp);
	if(isz != risz) {
		res = -4;
		goto err_iread;
	}

	self->input_buff = input_buff;

#if 0
	dumpaddr = (char *)ivaddr;
	GST_DEBUG("TEST: Successfully allocated buffers, input data: ");
	for(i = 0; i < 16; i++)
		GST_DEBUG("%02x", dumpaddr[i]);
	GST_DEBUG("\n");
#endif
	simaai_memory_unmap(input_buff);
	fclose(ifp);
	return res;

err_iread:
	simaai_memory_unmap(input_buff);
err_oalloc:
	simaai_memory_free(input_buff);
err_ialloc:
	fclose(ifp);
err_ifp:
	return res;
}

int32_t MLA_attach_input(BufferDataExchanger * self)
{

}

int32_t MLA_run_model(BufferDataExchanger * self,
					  uint32_t iaddr, uint32_t oaddr)
{
	sgp_param_t param[4];
	uint32_t phys = 1;
	uint64_t mla_model = self->type->p_mla_handle->model;
	void *idata[4] = {&mla_model, &iaddr, &oaddr, &phys};
	int32_t res;

	param[0].pdir = SGP_PDIR_IN;
	param[0].ptype = SGP_PTYPE_PTR;
	param[0].size = SGP_PTYPE_PTR_SIZE;

	param[1].pdir = SGP_PDIR_IN;
	param[1].ptype = SGP_PTYPE_UINT32;
	param[1].size = sizeof(iaddr);

	param[2].pdir = SGP_PDIR_IN;
	param[2].ptype = SGP_PTYPE_UINT32;
	param[2].size = sizeof(oaddr);

	param[3].pdir = SGP_PDIR_IN;
	param[3].ptype = SGP_PTYPE_UINT32;
	param[3].size = sizeof(phys);

	if (self->transport != NULL) {
		sgp_req_seq_t id = self->type->p_mla_handle->req_id++;
		sgp_transport_t ** t = self->transport;

		GST_DEBUG_OBJECT(self, "TEST: Calling int mla_run_model(%#lx, %#lx, %#lx, %d)\n", mla_model, iaddr, oaddr, id);
		res = sgp_transport_rpc(t,
					id,
					SGP_MLA_RUN_MODEL,
					4,
					param,
					idata,
					NULL);

	}

	return res;
}

int32_t MLA_wait_done(BufferDataExchanger * self)
{
	int32_t res = 0;

	GST_DEBUG_OBJECT(self, "TEST: Calling int mla_wait_done(void)\n");
	sgp_transport_t ** t = self->transport;

	res = sgp_transport_rpc(t,
				self->type->p_mla_handle->req_id++,
				SGP_MLA_WAIT_DONE,
				0,
				NULL, NULL, NULL);
	return res;
}

int32_t MLA_get_status(BufferDataExchanger * self)
{
	int32_t res;

	res = sgp_transport_rpc(self->transport,
				self->type->p_mla_handle->req_id++,
				SGP_MLA_GET_STATUS,
				0, NULL, NULL, NULL);
	return res;
}

int32_t
MLA_write_output_buffer(BufferDataExchanger * self, uint32_t id, GstBuffer * buffer)
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
		g_message("Size of mla buffer = %zu", map.size);
		gst_buffer_unmap(buffer, &map);
		goto err_owrite;
	}

	gst_buffer_unmap(buffer, &map);

err_owrite:
err_omap:
	fclose(ofp);
err_ofp:
	return res;
}

gboolean MLA_init(BufferDataExchanger * self)
{
	int32_t ret = 0;

	GST_DEBUG_CATEGORY_GET (GST_CAT_DEFAULT, "simaoffloader");

	if (self == NULL)
		goto err;

	self->transport = sgp_transport_create_socket_client(MLA_PORT);
	GST_DEBUG_OBJECT(self, "Dispatcher transport for mla 0x%x\n", self->transport);
	if (self->transport == NULL)
		goto err;

        // Used when debugging standalone node only
        int debug = *((int *)parser_get_int(self->params, "debug"));

        if (debug) {
            ret = PREPARE_INBUF(MLA, self);
            if (ret != 0x0)
		goto err;
        }

	ret = MLA_prepare_out_buff(self);
	if (ret != 0x0)
		goto err;

	MLA_get_handle(self);

	ret = MLA_load_model(self);
	if (ret != 0x0) {
	        MLA_fini(self);
		goto err;
	}

	return TRUE;
err:
	return FALSE;
}

void MLA_fini(BufferDataExchanger * self)
{
        if (free_simaai_memory_buffer_pool(self->pool))
                self->pool = NULL;
}
