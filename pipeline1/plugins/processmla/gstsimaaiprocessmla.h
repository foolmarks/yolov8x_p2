#ifndef GST_SIMAAIPROCESS_MLA_H_
#define GST_SIMAAIPROCESS_MLA_H_

#include <gst/gst.h>
#include <gst/base/gstbasetransform.h>

#include <simaai/sgp_types.h>

#define TARGET_CPU_DEFAULT 3
#define DEFAULT_CONFIG_FILE "/mnt/host/evxx_pre_proc.json"
#define DEFAULT_LM_FILE "/mnt/host/model.lm"
#define DEFAULT_NODE_NAME "sima-MLA"
#define DEFAULT_NUM_BUFFERS 5
#define DEFAULT_BATCH_SIZE 1
#define SIMAAI_META_STR "GstSimaMeta"
#define MIN_POOL_SIZE 2

#define PAD_TEMPLATE_NAME_SINK  "sink"
#define PAD_TEMPLATE_NAME_SRC   "src"

G_BEGIN_DECLS

#define GST_TYPE_SIMAAI_PROCESS_MLA            (gst_simaai_process_mla_get_type ())
#define GST_SIMAAI_PROCESS_MLA(obj)            (G_TYPE_CHECK_INSTANCE_CAST ((obj), GST_TYPE_SIMAAI_PROCESS_MLA, GstSimaaiProcessMLA))
#define GST_SIMAAI_PROCESS_MLA_CLASS(klass)    (G_TYPE_CHECK_CLASS_CAST ((klass), GST_TYPE_SIMAAI_PROCESS_MLA, GstSimaaiProcessMLAClass))
#define GST_SIMAAI_PROCESS_MLA_GET_CLASS(obj)  (G_TYPE_INSTANCE_GET_CLASS ((obj), GST_TYPE_SIMAAI_PROCESS_MLA, GstSimaaiProcessMLAClass))

typedef struct _GstSimaaiProcessMLA GstSimaaiProcessMLA;
typedef struct _GstSimaaiProcessMLAClass GstSimaaiProcessMLAClass;
typedef struct _GstSimaaiProcessMLA_Private GstSimaaiProcessMLA_Private;
typedef std::chrono::time_point<std::chrono::steady_clock> TimePoint;
struct _GstSimaaiProcessMLA
{
  GstBaseTransform parent;
  gboolean silent; /**< true to print minimized log */
  GstSimaaiProcessMLA_Private *priv;
  gboolean transmit;
};

struct _GstSimaaiProcessMLAClass
{
  GstBaseTransformClass parent_class;
};

GType gst_simaai_process_mla_get_type (void);

G_END_DECLS

#endif // GST_SIMAAIPROCESS_MLA_H_
