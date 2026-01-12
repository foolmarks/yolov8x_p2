#ifndef DISPATCHER_MLA_H_
#define DISPATCHER_MLA_H_

#include <simaai/sgp_transport.h>
#include <simaai/simaai_memory.h>
#include <simaai/msgs/sgp.h>

#include <simaai/parser/types.h>
#include <simaai/parser.h>

#include "dispatcher.h"

#define MLA_OUTPUT_SIZE 1024

void MLA_get_handle(BufferDataExchanger * self);

gboolean MLA_load_model(BufferDataExchanger * self);

int32_t MLA_prepare_out_buff(BufferDataExchanger * self);

int32_t MLA_prepare_in_buff(BufferDataExchanger * self);

int32_t MLA_run_model(BufferDataExchanger * self,
                      uint32_t iaddr, uint32_t oaddr);

int32_t MLA_write_output_buffer(BufferDataExchanger * self, uint32_t id, GstBuffer * buffer);

int32_t MLA_wait_done(BufferDataExchanger * self);

int32_t MLA_get_status(BufferDataExchanger * self);

gboolean MLA_init(BufferDataExchanger * self);

void MLA_fini(BufferDataExchanger * self);

#endif // DISPATCHER_MLA_H_
