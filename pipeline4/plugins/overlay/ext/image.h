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

#ifndef IMAGE_H_
#define IMAGE_H_

#include <iostream>

#include <simaai/cv_helpers.h>

/**
 * @brief The below is the base class for image type support,
 * @todo THIS IS NOT USED YET, would be used with next iteration of development
 */
namespace simaai {
namespace overlay {
/**
 * @brief Enum of ImageFormats supported
 */

class Image {
 public:
  Image() = default;
  virtual ~Image(){};

  Image(simaai::cv_helpers::ImageFormat fmt, int w, int h, int s)
      :format(fmt),
       width(w),
       height(h),
       stride(s)
  {};

  virtual void draw_rectangle (const simaai::cv_helpers::Points & points);

  virtual void draw_text (const simaai::cv_helpers::Points & points, std::string & classification_text);

  virtual void draw_points (simaai::cv_helpers::Points & points);

 private:
  simaai::cv_helpers::ImageFormat format;
  int width, height, stride;
};
}
}

#endif // IMAGE_H_
