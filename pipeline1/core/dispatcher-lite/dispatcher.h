#ifndef DISPATCHER_H_
#define DISPATCHER_H_

#include <gst/gst.h>

#include <simaai/gstsimaaiallocator.h>
#include <simaai/sgp_transport.h>
#include <simaai/dispatcher_types.h>
#include <simaai/simaai_memory.h>
#include <simaai/sgp_types.h>
// TODO: Remove this
#include <simaai/parser/types.h>
#include <simaai/parser.h>
#include <simaai/msgs/sgp.h>
#include <simaai/gst-api.h>

#include "handlers.h"

G_BEGIN_DECLS

#define GST_BUFFERDATAEXCHANGER_TYPE (buffer_data_exchanger_get_type())

#define GST_BUFFERDATAEXCHANGER(obj)                                    \
    (G_TYPE_CHECK_INSTANCE_CAST((obj), GST_BUFFERDATAEXCHANGER_TYPE, BufferDataExchanger))

#define GST_BUFFERDATAEXCHANGER_CLASS(klass)                            \
    (G_TYPE_CHECK_CLASS_CAST((klass),GST_BUFFERDATAEXCHANGER_TYPE,BufferDataExchangerClass))
#define GST_IS_BUFFERDATAEXCHANGER(obj)                                 \
    (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_BUFFERDATAEXCHANGER_TYPE))
#define GST_IS_BUFFERDATAEXCHANGER_CLASS(klass)                         \
    (G_TYPE_CHECK_CLASS_TYPE((klass),GST_BUFFERDATAEXCHANGER_TYPE))

/* G_DECLARE_FINAL_TYPE(BufferDataExchanger, */
/*                      buffer_data_exchanger, */
/*                      DATAEXCHANGER, */
/*                      BUFFER, */
/*                      GObject); */

typedef struct _BufferDataExchanger BufferDataExchanger;
typedef struct _BufferDataExchangerClass BufferDataExchangerClass;

typedef struct _BufferDataExchangerCallback {
    //notification of buffer received. It is the callbacks responsibility
    // to unref the buffer
    void (*buffer_received)(gpointer buffer, void *priv);
    void *priv;
} BufferDataExchangerCallback;

struct _BufferDataExchangerClass {
    GstElementClass parent_class;
};

typedef struct sgp_cpu_variant_s {
    sgp_mla_t * p_mla_handle;
    sgp_evxx_t * p_evxx_handle;
    sgp_host_t * p_host_handle;
    sgp_dev_t * p_dev_handle;
} sgp_cpu_variant_t;

typedef struct mla_client_hdl_s {
     uint64_t mla_handle;
     uint64_t mla_model;
} mla_client_hdl_t;

#define MAX_IN_PATH 256
#define SGP_MAGIC_NUMBER 0xe67042
#define MAX_BUFS 5
#define MIN_POOL_SIZE 2

struct _BufferDataExchanger {
    GstElement element;
    /* Other members, including private data. */
    BufferDataExchangerCallback *callback;
    sima_cpu_e cpu;
    gboolean is_initialized;

    GMutex dispatcher_mutex;

    mla_client_hdl_t mla_hdl;
    sgp_transport_t * transport;
    sgp_cpu_variant_t * type;

    // Use this for testing only, the in_buff should come from a src_pad
    // on sgp_msg_t always
    simaai_memory_t* input_buff;
    // This might not be needed
    void *ivaddr;
    simaai_params_t * params;

    GstBufferPool *pool; // Buffer pool
    gint64 in_buf_id;
    gint64 in_pcie_buf_id;
    GstSimaaiMemoryFlags mem_flags;

    gint64 frame_id;
    
    gint batch_size;        // batch size requested by user
    gint batch_size_model;  // batch size the particular model supports
    size_t in_tensor_size;  // input tensor size. For example, tensor with shape "100:3:128:256" will have size: 100*3*128*256 = 9830400
    size_t out_tensor_size; // output tensor size. For example, tensor with shape "96:2048" = 96*2048 = 196608;
};

BufferDataExchanger *buffer_data_exchanger_new (BufferDataExchangerCallback *callback);

// Init
gboolean dispatcher_init(BufferDataExchanger * self, simaai_params_t * params);
/* gboolean dispatcher_pt_init(BufferDataExchanger * self); */
// Deinit
void deinit_dispatcher(sima_cpu_e cpu);

// Data
// Send
GstFlowReturn buffer_data_dispatcher_send(BufferDataExchanger * self, GstBuffer * buffer);

// Receive
GstFlowReturn buffer_data_dispatcher_recv(BufferDataExchanger * self, GstBuffer * buffer);

// Utils
GType buffer_data_exchanger_get_type ();

G_END_DECLS

#endif // DISPATCHER_H_
