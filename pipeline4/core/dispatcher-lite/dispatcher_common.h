#ifndef DISPATCHER_COMMON_H_
#define DISPATCHER_COMMON_H_

#include "mla_dispatcher.h"
#include "mla_client.h"
#include "ev_dispatcher.h"
#include "host_dispatcher.h"

/* #define mla "MLA" */
/* #define evxx "EVXX" */
/* #define a65 "A65" */

#define GET_HANDLE(cpu, ... ) cpu##_##get_handle(__VA_ARGS__)
// Models are MLA specific
#define LOAD_MODEL(cpu, ... ) cpu##_##load_model(__VA_ARGS__)
#define RUN_MODEL(cpu, ...) cpu##_##run_model(__VA_ARGS__)
#define PREPARE_INBUF(cpu, ... ) cpu##_##prepare_in_buff(__VA_ARGS__)
#define PREPARE_OUTBUF(cpu, ... ) cpu##_##prepare_out_buff(__VA_ARGS__)
#define WAIT_DONE(cpu, ...) cpu##_##wait_done(__VA_ARGS__)
#define GET_OUTBUF(cpu, ...) cpu##_##write_output_buffer(__VA_ARGS__)
#define INIT(cpu, ...) cpu##_##init(__VA_ARGS__)
#define POST(cpu, ...) cpu##_##post(__VA_ARGS__)
#define CONFIGURE(cpu, ...) cpu##_##configure(__VA_ARGS__)

#endif // DISPATCHER_COMMON_H_
