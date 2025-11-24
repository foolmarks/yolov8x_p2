#include <babeltrace2/babeltrace.h>
#include <stdio.h>
#include <unistd.h>
#include <simaai/simaailog.h>

#include "live_trace_reader_api.h"

#define MAX_RETRY_COUNT (5)

static int live_trace_reader_running_status = LIVE_TRACE_READER_RUNNING_STATUS_RUN;

void live_trace_reader_set_running_status(const int new_status)
{
	live_trace_reader_running_status = new_status;
}

const bt_component_class_source *load_source_class() {
    const bt_plugin *ctf_plugin = NULL;
    bt_plugin_find_status found = bt_plugin_find("ctf", BT_TRUE, BT_FALSE, BT_TRUE, BT_TRUE, BT_FALSE, &ctf_plugin);
    if (found != BT_PLUGIN_FIND_STATUS_OK) {
        simaailog(SIMAAILOG_ERR, "ctf plugin did not found");
        return NULL;
    }

    const bt_component_class_source * src_class = bt_plugin_borrow_source_component_class_by_name_const(ctf_plugin, "lttng-live");
    if (!src_class) {
        simaailog(SIMAAILOG_ERR, "Failed to find lttng-live component class");
        return NULL;
    }

    return src_class;
}

bt_value * make_params_for_source(const char *session_url) {
    bt_value *inputs_array = bt_value_array_create();
    bt_value *url_value = bt_value_string_create_init(session_url);
    bt_value_array_append_element(inputs_array, url_value);

    bt_value *params = bt_value_map_create();
    bt_value_map_insert_entry(params, "inputs", inputs_array);

    return params;
}

const bt_component_class_sink *load_kpi_sender_class() {
    const bt_plugin *kpi_sender_class = NULL;
    bt_plugin_find_status found = bt_plugin_find("simaai_kpi", BT_TRUE, BT_TRUE, BT_TRUE, BT_TRUE, BT_TRUE, &kpi_sender_class);
    if (found != BT_PLUGIN_FIND_STATUS_OK) {
        simaailog(SIMAAILOG_ERR, "simaai_kpi plugin did not found");
        return NULL;
    }

    const bt_component_class_sink * sink_class = bt_plugin_borrow_sink_component_class_by_name_const(kpi_sender_class, "kpi_sink");
    if (!sink_class) {
        simaailog(SIMAAILOG_ERR, "Failed to find kpi_sink component class");
        return NULL;
    }

    return sink_class;
}

bt_value * make_params_for_kpi_sender(const char *pipeline_id, const pid_t pid, const uint64_t plugins_count)
{
    bt_value *params = bt_value_map_create();

    bt_value *pipeline_id_value = bt_value_string_create_init(pipeline_id);
    bt_value_map_insert_entry(params, "pipeline_id", pipeline_id_value);

    bt_value *pipeline_pid_value = bt_value_integer_signed_create_init(pid);
    bt_value_map_insert_entry(params, "pid", pipeline_pid_value);

    bt_value *plugins_count_value = bt_value_integer_unsigned_create_init(plugins_count);
    bt_value_map_insert_entry(params, "plugins_count", plugins_count_value);

    return params;
}

bt_graph *live_trace_reader_make_graph(const struct live_trace_reader_init_data_t params) {
    // Graph creation
    bt_graph *graph = bt_graph_create(0);
    if (!graph) {
        simaailog(SIMAAILOG_ERR, "Failed to create graph.");
        return NULL;
    }

    // ctf.lttng-live source component creation and connection
    const bt_component_class_source *src_class = load_source_class();
    if (src_class == NULL) {
        simaailog(SIMAAILOG_ERR, "Failed to load component class source");
        return NULL;
    }

    bt_value *src_params = make_params_for_source(params.session_url);
    if (src_params == NULL) {
        simaailog(SIMAAILOG_ERR, "Failed to make params for source component");
        return NULL;
    }

    const bt_component_source *src = NULL;
    bt_graph_add_component_status add_component_status = bt_graph_add_source_component(graph, src_class, "src", src_params, BT_LOGGING_LEVEL_WARNING, &src);
    if (add_component_status != BT_GRAPH_ADD_COMPONENT_STATUS_OK) {
        simaailog(SIMAAILOG_ERR, "Failed to add source component.");

        bt_value_put_ref(src_params);
        return NULL;
    }

    // text.pretty sink component creation and connection
    const bt_component_class_sink *sink_class = load_kpi_sender_class();
    if (sink_class == NULL) {
        simaailog(SIMAAILOG_ERR, "Failed to load component class sink");

        bt_value_put_ref(src_params);
        return NULL;
    }

    bt_value *sink_params = make_params_for_kpi_sender(params.pipeline_id, params.pid, params.plugins_count);
    if (sink_params == NULL) {
        simaailog(SIMAAILOG_ERR, "Failed to make params for sink component");

        bt_value_put_ref(src_params);
        return NULL;
    }

    const bt_component_sink *sink = NULL;
    add_component_status = bt_graph_add_sink_component(graph, sink_class, "sink", sink_params, BT_LOGGING_LEVEL_WARNING, &sink);
    if (add_component_status != BT_GRAPH_ADD_COMPONENT_STATUS_OK) {
        simaailog(SIMAAILOG_ERR, "Failed to add sink component.");

        bt_value_put_ref(src_params);
        bt_value_put_ref(sink_params);
        return NULL;
    }

    // Connect ports
    const bt_port_output *src_out_port = bt_component_source_borrow_output_port_by_index_const(src, 0);
    const bt_port_input *sink_in_port = bt_component_sink_borrow_input_port_by_index_const(sink, 0);
    if (!src_out_port || !sink_in_port) {
        simaailog(SIMAAILOG_ERR, "Failed to get ports.");

        bt_value_put_ref(src_params);
        bt_value_put_ref(sink_params);
        return NULL;
    }

    const bt_connection *connection_src_sink = NULL;
    bt_graph_connect_ports(graph, src_out_port, sink_in_port, &connection_src_sink);
    if (!connection_src_sink) {
        simaailog(SIMAAILOG_ERR, "Failed to connect ports.");

        bt_value_put_ref(src_params);
        bt_value_put_ref(sink_params);
        return NULL;
    }

    bt_value_put_ref(src_params);
    bt_value_put_ref(sink_params);
    return graph;
}

int live_trace_reader_run(const struct live_trace_reader_init_data_t params)
{
    bt_graph *graph = NULL;
    bt_graph_run_once_status run_status;
    int retry_count = 0;

    // Run graph
    do {
        if (!graph) {
            graph = live_trace_reader_make_graph(params);
        }

        run_status = bt_graph_run_once(graph);

        if (run_status == BT_GRAPH_RUN_ONCE_STATUS_ERROR) {
            retry_count++;
            fprintf(stderr, "Graph run failed. Try #%d\n", retry_count);

            if (retry_count >= MAX_RETRY_COUNT) {
                break;
            }
            sleep(1);

            bt_graph_put_ref(graph);
            graph = NULL;

            bt_current_thread_clear_error();
            run_status = BT_GRAPH_RUN_ONCE_STATUS_AGAIN;
        }
    } while (run_status >= 0 && live_trace_reader_running_status);

    if (run_status < 0) {
        fprintf(stderr, "Graph run failed. No more retires. LTR is not working.\n");
    } else {
        printf("Finished LTR\n");
    }

    bt_graph_put_ref(graph);

    return run_status;
}
