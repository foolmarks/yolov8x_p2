#ifndef LIVE_TRACE_READER_API_H
#define LIVE_TRACE_READER_API_H

#include <unistd.h>
#include <inttypes.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
	LIVE_TRACE_READER_RUNNING_STATUS_STOP = 0,
	LIVE_TRACE_READER_RUNNING_STATUS_RUN  = 1
};

struct live_trace_reader_init_data_t {
    const char *session_url;
    const char *pipeline_id;
    pid_t pid;
    uint64_t plugins_count;
};

void live_trace_reader_set_running_status(const int new_status);
int  live_trace_reader_run(const struct live_trace_reader_init_data_t params);

#ifdef __cplusplus
}
#endif

#endif // LIVE_TRACE_READER_API_H