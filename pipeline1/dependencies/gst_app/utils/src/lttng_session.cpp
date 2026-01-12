#include "lttng_session.h"

#define SESSION_URL_STR "net://localhost/host/sb-board33/my-session"
#define EVENT_NAME_REMOTE_CORE__EV74 "remote_core:EV74"
#define EVENT_NAME_REMOTE_CORE__ANY "remote_core:*"
#define EVENT_NAME_PIPELINE__ANY "pipeline:*"
#define SESSION_NAME "my-session"

namespace utils {
LttngSession::LttngSession(const char *session_name, const pid_t pid) {
    std::string url = create_session_url(session_name);

    int code = 0;
    do {
        code = lttng_create_session_live(session_name, url.c_str(), 1000000);
        if (abs(code) == LTTNG_ERR_EXIST_SESS) {
            printf("Restart session\n");
            lttng_destroy_session(session_name);
        } else if (code < 0) {
            fprintf(stderr, "Error creating session: %d '%s'\n", code, lttng_strerror(code));
            return;
        }
    } while (code < 0);

    struct lttng_domain domain;
    domain.type = LTTNG_DOMAIN_UST;
    domain.buf_type = LTTNG_BUFFER_PER_UID;
    domain.attr.pid = pid;

    _handle = lttng_create_handle(session_name, &domain);
    if (_handle == NULL) {
        fprintf(stderr, "Error lttng_handle is NULL\n");
        return;
    }

    struct lttng_channel *chan = lttng_channel_create(&domain);
    if (chan == NULL) {
        fprintf(stderr, "Error lttng_channel_create is NULL\n");
        return;
    }

    chan->attr.overwrite = 1;
    chan->attr.num_subbuf = 4;
    chan->attr.subbuf_size = 262144; // 256 KB

    code = lttng_enable_channel(_handle, chan);
    if (code < 0) {
        fprintf(stderr, "Error enabling channel: %s\n", lttng_strerror(code));
        return;
    }

    // Enable remote_core events
    struct lttng_event ev;
    memset(&ev, 0, sizeof(ev));
    ev.type = LTTNG_EVENT_TRACEPOINT;

    strncpy(ev.name, EVENT_NAME_REMOTE_CORE__ANY, strlen(EVENT_NAME_REMOTE_CORE__ANY) + 1);

    if (lttng_enable_event(_handle, &ev, NULL) < 0) {
        fprintf(stderr, "Error enabling event1\n");
        return;
    }

    // Enable pipeline events
    memset(&ev, 0, sizeof(ev));
    ev.type = LTTNG_EVENT_TRACEPOINT;
    strncpy(ev.name, EVENT_NAME_PIPELINE__ANY, strlen(EVENT_NAME_PIPELINE__ANY) + 1);

    if (lttng_enable_event(_handle, &ev, NULL) < 0) {
        fprintf(stderr, "Error enabling event2\n");
        return;
    }

    // Start tracing
    if (lttng_start_tracing(session_name) < 0) {
        fprintf(stderr, "Error starting tracing\n");
        return;
    }

    printf("Tracing started.\n");
}

std::string LttngSession::create_session_url(const char *session_name) {
    std::string url("net4://localhost/host/");

    char hostname[HOST_NAME_MAX];
    if (gethostname(hostname, HOST_NAME_MAX)) {
        fprintf(stderr, "Error: cannot get hostname\n");
        return NULL;
    }

    url.append(hostname);
    url.append("/");
    url.append(session_name);

    return url;
}

LttngSession::~LttngSession() {
    if (_handle == nullptr) {
        return;
    }

    printf("Stopping and destroing a lttng session\n");

    int ret = lttng_stop_tracing(_handle->session_name);
    if (ret < 0) {
        fprintf(stderr, "Error during stopping tracing: '%s'\n", lttng_strerror(ret));
    }

    lttng_destroy_session(_handle->session_name);
    lttng_destroy_handle(_handle);
    printf("Destroyed a lttng session\n");
}
} // namespace utils
