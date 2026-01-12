#ifndef DISPATCHER_HOST_H_
#define DISPATCHER_HOST_H_

#include <simaai/simaai_memory.h>
#include <simaai/msgs/sgp.h>

#include <simaai/parser/types.h>
#include <simaai/parser.h>

#include "dispatcher.h"
#include <simaai/ev_c_api.h>

int32_t host_init(BufferDataExchanger * self);

void host_fini(BufferDataExchanger * self);

sgp_host_resp_t host_wait_done(BufferDataExchanger * self, uint32_t id);

void host_post(BufferDataExchanger * self, sgp_host_req_t * req);

int32_t host_prepare_out_buf(BufferDataExchanger * self);

int32_t host_prepare_in_buff(BufferDataExchanger * self);

gboolean host_configure (BufferDataExchanger * self);

void host_get_status(BufferDataExchanger * self);

void host_prepare_in_buf(BufferDataExchanger * self);

int32_t host_write_output_buffer(BufferDataExchanger * self, uint32_t id, GstBuffer * buffer);

#endif // DISPATCHER_HOST_H_
