#include <fcntl.h>
#include <inttypes.h>
#include <stdio.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/uio.h>

#include <stdio.h>
#include <sys/time.h>

#include "dispatcher.h"

#include "mla_client.h"
#include "utils.h"

#include "simamm.h"

#include <simaai/gstsimaaiallocator.h>
// MLA-rt client API
#include <simaai/gst-api.h>

#define GST_CAT_DEFAULT mla_client_debug
GST_DEBUG_CATEGORY_STATIC (GST_CAT_DEFAULT);

int32_t mla_client_load_model(BufferDataExchanger * self)
{
  char * path = (char *)get_model_path(self->params);
  self->mla_hdl.mla_model = (uint64_t) mla_load_model((mla_handle_p) self->mla_hdl.mla_handle, path);
  g_message("Loaded model from location %s, model:hdl: %p", path, self->mla_hdl.mla_model);
  return 0;
}

int32_t mla_client_prepare_out_buff(BufferDataExchanger * self)
{
  int32_t res = 0;

  sima_cpu_e next_cpu = (sima_cpu_e)get_next_cpu(self->params);

  GstSimaaiMemoryFlags mem_target = get_mem_target(next_cpu) | self->mem_flags;

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

static void mla_client_build_addr_list (uint64_t addr_list[],
                                        uint64_t start_addr,
                                        size_t size_of_each_chunk,
                                        int batch_size) {
    size_t offset = 0;
    for (int i = 0; i < batch_size; i++) {
        offset = i * size_of_each_chunk;
        addr_list[i] = start_addr + offset;
        /* fprintf(stderr, "start_addr:0x%lx, offset: %ld, chunk_size: %ld, in_addr:0x%lx, batch_size:%d\n", */
        /*         start_addr, offset, size_of_each_chunk, addr_list[i], batch_size); */
    }
}

/**
 * @brief helper to run mla in batched mode
 * @return 0 on success. -1 on failure 
*/
int8_t mla_run_batch(mla_client_hdl_t *mla_hdl,
                      uint64_t in_addr,
                      uint64_t out_addr, 
                      size_t batch_size,
                      size_t in_tensor_size,
                      size_t out_tensor_size) 
{
  uint64_t * in_addr_list = malloc(sizeof(uint64_t) * batch_size);
  uint64_t * out_addr_list = malloc(sizeof(uint64_t) * batch_size);

  mla_client_build_addr_list(in_addr_list, 
                              in_addr, 
                              in_tensor_size, 
                              batch_size);
  mla_client_build_addr_list(out_addr_list, 
                              out_addr, 
                              out_tensor_size, 
                              batch_size);

  if (mla_run_batch_model_phys((mla_model_p) mla_hdl->mla_model,
                               batch_size,
                               in_addr_list , in_tensor_size,
                               out_addr_list, out_tensor_size) != 0) {
      g_message("Model run failed");
      free(in_addr_list);
      free(out_addr_list);
      return -1;
  }
  free(in_addr_list);
  free(out_addr_list);
  return 0;
}

int32_t mla_client_run_model(BufferDataExchanger * self,
			     uint64_t in_addr,
			     uint64_t out_addr,
			     size_t in_data_sz)
{
  GST_DEBUG("in_addr: %p, out_addr: %p, model: 0x%x", in_addr, out_addr, self->mla_hdl.mla_model);

  if (self->batch_size < 1)
  {
    g_message("Invalid value for parameter \"batch_size\": %d", self->batch_size);
    return -1;
  }

  if (self->batch_size_model < 1)
  {
    g_message("Invalid value for parameter \"batch_size_model\": %d", self->batch_size);
    return -1;
  }
  
  // Define the strategy if batch processing is requested
  if (self->batch_size == 1)
  {
    // No batching
    if (mla_run_model_phys((mla_model_p) self->mla_hdl.mla_model, in_addr , out_addr) != 0)
    {
      g_message("Model run failed");
    }
  }
  else
  {
    // Use native batching
    if (self->batch_size <= self->batch_size_model)
    {
      if (mla_run_batch(&self->mla_hdl,
                        in_addr, 
                        out_addr,
                        self->batch_size,
                        in_data_sz / self->batch_size,
                        get_output_sz(self->params) / self->batch_size))
        return -1;
    }
    // User requested more batches to process than current model supports
    else
    {
      int run_cnt = 0;
      int completed = 0;


      run_cnt = self->batch_size / self->batch_size_model;
      
      uint64_t in_addr_offseted = in_addr;
      uint64_t out_addr_offseted = out_addr;

      for (int i = 0; i < run_cnt; i++) {
        if (mla_run_batch(&self->mla_hdl,
                        in_addr_offseted, 
                        out_addr_offseted,
                        self->batch_size_model,
                        self->in_tensor_size,
                        self->out_tensor_size)) {
          return -1;
        }
        GST_DEBUG("Run:[%d], in_addr_offset: 0x%lx, out_addr_offset: 0x%lx\n", i, in_addr_offseted, out_addr_offseted);

        completed += self->batch_size_model;

        in_addr_offseted += self->in_tensor_size * self->batch_size_model;
        out_addr_offseted += self->out_tensor_size * self->batch_size_model;
      }

      GST_DEBUG("\n Called mla_run_model : %d times\n", completed);
    
      if (completed < self->batch_size) {
        int left_over_batch = self->batch_size - completed;
        
        if (mla_run_batch(&self->mla_hdl,
                        in_addr_offseted, 
                        out_addr_offseted,
                        left_over_batch,
                        self->in_tensor_size,
                        self->out_tensor_size)) {
          return -1;
        }
        GST_DEBUG("\n Called the left over mla_run_model : %d \n", left_over_batch);
      }
    }
  }
  
  return 0;
}

gboolean mla_client_init(BufferDataExchanger * self)
{
  int32_t ret = 0;

  GST_DEBUG_CATEGORY_GET (GST_CAT_DEFAULT, "simaoffloader");

  g_message("Initialize dispatcher");
  
  if (self == NULL)
    goto err;

  ret = mla_client_prepare_out_buff(self);
  if (ret != 0x0) {
    goto err;
  }

  self->mla_hdl.mla_handle = (uint64_t) mla_get_handle();
  g_message("handle: 0x%x, %p", self->mla_hdl.mla_handle, self->mla_hdl.mla_handle);
  
  ret = mla_client_load_model(self);
  if (ret != 0x0) {
    mla_client_fini(self);
    goto err;
  }

  return TRUE;
err:
  return FALSE;
}

void mla_client_fini(BufferDataExchanger * self)
{
  if (free_simaai_memory_buffer_pool(self->pool))
    self->pool = NULL;
}

int32_t
mla_client_write_output_buffer (BufferDataExchanger * self, uint32_t id, GstBuffer * buffer)
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

  /* simaai_memory_invalidate_cache(simaai_get_memory_handle(self->mem)); */
  
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
