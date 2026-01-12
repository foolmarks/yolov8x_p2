//**************************************************************************
//||                        SiMa.ai CONFIDENTIAL                          ||
//||   Unpublished Copyright (c) 2022-2023 SiMa.ai, All Rights Reserved.  ||
//**************************************************************************
// NOTICE:  All information contained herein is, and remains the property of
// SiMa.ai. The intellectual and technical concepts contained herein are 
// proprietary to SiMa and may be covered by U.S. and Foreign Patents, 
// patents in process, and are protected by trade secret or copyright law.
//
// Dissemination of this information or reproduction of this material is 
// strictly forbidden unless prior written permission is obtained from 
// SiMa.ai.  Access to the source code contained herein is hereby forbidden
// to anyone except current SiMa.ai employees, managers or contractors who 
// have executed Confidentiality and Non-disclosure agreements explicitly 
// covering such access.
//
// The copyright notice above does not evidence any actual or intended 
// publication or disclosure  of  this source code, which includes information
// that is confidential and/or proprietary, and is a trade secret, of SiMa.ai.
//
// ANY REPRODUCTION, MODIFICATION, DISTRIBUTION, PUBLIC PERFORMANCE, OR PUBLIC
// DISPLAY OF OR THROUGH USE OF THIS SOURCE CODE WITHOUT THE EXPRESS WRITTEN
// CONSENT OF SiMa.ai IS STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE 
// LAWS AND INTERNATIONAL TREATIES. THE RECEIPT OR POSSESSION OF THIS SOURCE
// CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS TO 
// REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE, USE, OR
// SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.                
//
//**************************************************************************

#ifndef _SIMAAI_OVERLAY_TYPES_H_
#define _SIMAAI_OVERLAY_TYPES_H_

namespace simaai {
namespace overlay {

/**
 * @brief OverlayType information
 */
enum class OverlayType {
  SIMAAI_OVERLAY_INPUT = 0,
  SIMAAI_OVERLAY_BBOX,
  SIMAAI_OVERLAY_TEXT,
  SIMAAI_OVERLAY_POINTS
};

/**
 * @brief Overlay Buffer Type information
 */
enum class OverlayBufType {
  SIMAAI_BUF_SOURCE = 0,
  SIMAAI_BUF_TOPK,
  SIMAAI_BUF_YOLO,
  SIMAAI_BUF_TRANSFORM,
  SIMAAI_BUF_TRACKER,
  SIMAAI_BUF_CMN,
  SIMAAI_BUF_UNKNOWN = -1
};

/**
 * @brief Overlay Image Format information
 */
enum class OverlayImageFormat {
  SIMAAI_IMG_YUV420 = 0,
  SIMAAI_IMG_YUV444,
  SIMAAI_IMG_RGB,
  SIMAAI_IMG_NV12,
  SIMAAI_IMG_UNKNOWN = -1
};

}  // namespace overlay
} // namespace simaai

#endif /* _SIMAAI_OVERLAY_TYPES_H_ */
