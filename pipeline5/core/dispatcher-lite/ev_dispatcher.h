#ifndef DISPATCHER_EV_H_
#define DISPATCHER_EV_H_

#include <simaai/simaai_memory.h>
#include <simaai/msgs/sgp.h>

#include <simaai/parser/types.h>
#include <simaai/parser.h>


#include "dispatcher.h"
#include <simaai/ev_c_api.h>

int32_t EVXX_init(BufferDataExchanger * self);

sgp_ev_resp_t EVXX_wait_done(BufferDataExchanger * self, uint32_t id);

void EVXX_post(BufferDataExchanger * self, sgp_ev_req_t * req);

int32_t EVXX_prepare_out_buf(BufferDataExchanger * self);

int32_t EVXX_prepare_in_buff(BufferDataExchanger * self);

gboolean EVXX_configure (BufferDataExchanger * self);

void EVXX_get_status(BufferDataExchanger * self);

void EVXX_prepare_in_buf(BufferDataExchanger * self);

int32_t EVXX_write_output_buffer(BufferDataExchanger * self, uint32_t id, GstBuffer * buffer);

void EVXX_fini(BufferDataExchanger * self);

int32_t
EVXX_write_input_buffer(BufferDataExchanger * self, void * vaddr, gsize sz);

#endif // DISPATCHER_EV_H_
