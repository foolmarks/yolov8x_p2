#ifndef DISPATCHER_MLA_CLIENT_H_
#define DISPATCHER_MLA_CLIENT_H_

#include <simaai/sgp_transport.h>
#include <simaai/simaai_memory.h>
#include <simaai/msgs/sgp.h>

#include <simaai/parser/types.h>
#include <simaai/parser.h>

#include <simaai/gst-api.h>

#include "dispatcher.h"

#define MLA_OUTPUT_SIZE 1024

gboolean mla_client_load_model(BufferDataExchanger * self);

int32_t mla_client_prepare_out_buff(BufferDataExchanger * self);

int32_t mla_client_run_model(BufferDataExchanger * self,
			     uint64_t in_addr,
			     uint64_t out_addr,
			     size_t in_data_sz);

gboolean mla_client_init(BufferDataExchanger * self);

void mla_client_fini(BufferDataExchanger * self);

int32_t
mla_client_write_output_buffer(BufferDataExchanger * self, uint32_t id, GstBuffer * buffer);

#endif // DISPATCHER_MLA_CLIENT_H_
