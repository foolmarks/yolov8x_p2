#ifndef HANDLERS_H_
#define HANDLERS_H_

#include <inttypes.h>
#include <simaai/dispatcher_types.h>

#define MAX_MODEL_FPATH_LEN 256

typedef uint64_t sgp_req_id_t;

typedef struct sgp_evxx_s {
  sgp_req_id_t req_id;
  uint32_t handle;
  
  // Image params
  int32_t in_w;
  int32_t in_h;
  int32_t out_w;
  int32_t out_h;
  int32_t step;
} sgp_evxx_t;

typedef struct sgp_mla_s {
  sgp_req_id_t req_id;
  uint64_t handle;
  uint64_t model;
  char path[MAX_MODEL_FPATH_LEN];
} sgp_mla_t;

typedef struct sgp_host_s {
  uint32_t handle;
  sgp_req_id_t req_id;
} sgp_host_t;

// Use this cpu variant for testing on x86 
typedef struct sgp_dev_s {
  uint32_t handle;
  sgp_req_id_t req_id;
} sgp_dev_t;

#endif // HANDLERS_H_
