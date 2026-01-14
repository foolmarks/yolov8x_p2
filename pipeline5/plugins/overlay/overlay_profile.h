#ifndef OVERLAY_PROFILE_H_
#define OVERLAY_PROFILE_H_

#define MAX_OP_NAME 64
#define MAX_SUPPORTED_PROFILES 4
#define MAX_TYPE_NAME 64

enum class OverlayType {
  SIMAAI_SOURCE_IMAGE = 0,
  SIMAAI_OVERLAY_BBOX,
  SIMAAI_OVERLAY_TEXT,
  SIMAAI_OVERLAY_UNKNOWN = -1
};

typedef struct simaai_overlay_s {
  OverlayType type;
  char type_name[MAX_TYPE_NAME];
  char op_name[MAX_OP_NAME];
} simaai_overlay_t ;

simaai_overlay_t profiles[MAX_SUPPORTED_PROFILES] = {
  {OverlayType::SIMAAI_OVERLAY_BBOX, "bbox", "topk"},
  {OverlayType::SIMAAI_OVERLAY_BBOX, "bbox", "tracker"},
  {OverlayType::SIMAAI_OVERLAY_TEXT, "text", "transform"},
  {OverlayType::SIMAAI_OVERLAY_UNKNOWN, "unknown", "unknown"},
};

#endif // OVERLAY_PROFILE_H_
