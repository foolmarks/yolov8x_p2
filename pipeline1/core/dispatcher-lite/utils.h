#ifndef UTILS_H_
#define UTILS_H_

#include <inttypes.h>

#include "dispatcher.h"

#include <simaai/gstsimaaiallocator.h>
#include <simaai/parser/types.h>
#include <simaai/parser.h>

typedef enum {
    SIMA_FORMAT_RGB = 0,
    SIMA_FORMAT_YUV420,
    SIMA_FORMAT_UNKNOWN = -1
} sima_format_e;

#ifdef __cplusplus
extern "C" {
#endif

ssize_t get_output_sz(simaai_params_t * params);
uint32_t get_input_sz(simaai_params_t * params);
GstSimaaiMemoryFlags get_mem_target(sima_cpu_e cpu);
const char * get_model_path(simaai_params_t * params);
int get_next_cpu(simaai_params_t * params);
int get_cpu(simaai_params_t * params);
const char * get_input_path(simaai_params_t * params);
void dump_buf_id_list(const sgp_msg_t * msg);
const char * get_node_name(simaai_params_t * params);
simamm_buffer_id_t get_bufid(simaai_params_t * params, const sgp_msg_t * msg, const char * name);
long long cur_time_ms();
int32_t get_num_of_outbufs(simaai_params_t * params);
GstBufferPool * allocate_simaai_memory_buffer_pool(GstObject * obj,
                                                   guint buf_size,
                                                   guint min_buffers,
                                                   guint max_buffers,
                                                   GstSimaaiMemoryFlags flags);
gboolean free_simaai_memory_buffer_pool(GstBufferPool * pool);

#ifdef __cplusplus
}
#endif

#endif // UTILS_H_
