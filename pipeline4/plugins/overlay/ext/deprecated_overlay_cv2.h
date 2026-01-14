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

// Render-in-place
// Render-push
// Render

// CV2 overlay
#include <stdio.h>
#include <string>
#include <iostream>
#include <stdlib.h>
#include <fstream>
#include <opencv2/opencv.hpp>

using OverlayInfoMap = std::map<OverlayType , gpointer>;

enum class TransformType {
  OVERLAY_BBOX,
  OVERLAY_TEXT,
  OVERLAY_POINTS,
};

class Overlay {
 public:
  Overlay() = default;
  virtual ~Overlay() = default;

  void set_overlay_render_config(const RenderConfig & cfg)
  {};

  OverlayInfoMap get_map_type(void)
  {};

  bool render(void)
  {};

 private:
  std::map<OverlayType , void *> overlay_info_map;
}

// class to handle YUV images
class YUVImage {
public:
  enum YUVFormat {
    YUV420P,
    YUV444P
  };

  YUVImage(unsigned char * yuvPixels, unsigned width, unsigned height, YUVFormat format) {
    unsigned lumaSize = width * height;
    unsigned chromaSize = width/2 * height/2;
    if (format == YUV444P) {
      chromaSize = lumaSize;
    }

    y_ = cv::Mat(cv::Size(width, height),  CV_8UC1, yuvPixels);
    u_ = cv::Mat(cv::Size(width/2, height/2),  CV_8UC1, &yuvPixels[lumaSize]);
    v_ = cv::Mat(cv::Size(width/2, height/2),  CV_8UC1, &yuvPixels[lumaSize + chromaSize]);
    format_ = format;
    y_stride = y_.step;
    u_stride = u_.step;
    v_stride = v_.step;
  }

  YUVImage(std::string fname, unsigned width, unsigned height, YUVFormat format) {
    FILE * f = fopen(fname.c_str(),"rb");
    if ( !f ) {
      std::string out = "Unable to open input file " + fname;
      throw std::runtime_error(out);
    }

    unsigned lumaSize = width * height;
    unsigned chromaSize = width/2 * height/2;
    if (format == YUV444P) {
      chromaSize = lumaSize;
    }
    unsigned frameSize =  lumaSize + 2 * chromaSize;
    std::cout << "frameSize = " << frameSize << "\n";
    unsigned char* yuvPixels = new unsigned char[frameSize];

    if (fread(yuvPixels, frameSize, 1, f) != 1) {
      std::string out = "Unable to read " + std::to_string(frameSize) + " bytesfrom file " + fname;
      throw std::runtime_error(out);
    }
    fclose(f);
    y_ = cv::Mat(cv::Size(width, height),  CV_8UC1, yuvPixels);
    u_ = cv::Mat(cv::Size(width/2, height/2),  CV_8UC1, &yuvPixels[lumaSize]);
    v_ = cv::Mat(cv::Size(width/2, height/2),  CV_8UC1, &yuvPixels[lumaSize + chromaSize]);
    format_ = format;
  }

  void I420toNV12(const cv::Mat& input, cv::Mat& output) {
    int width = input.cols;
    int height = input.rows * 2 / 3;
    int stride = (int)input.step[0];

    input.copyTo(output);

    cv::Mat inU = cv::Mat(cv::Size(width / 2, height / 2), CV_8UC1, (unsigned char*)input.data + stride * height, stride / 2);
    cv::Mat inV = cv::Mat(cv::Size(width/2, height/2), CV_8UC1, (unsigned char*)input.data + stride*height + (stride/2)*(height/2), stride/2);

    for (int row = 0; row < height / 2; row++) {
      for (int col = 0; col < width / 2; col++) {
        output.at<uchar>(height + row, 2 * col) = inU.at<uchar>(row, col);
        output.at<uchar>(height + row, 2 * col + 1) = inV.at<uchar>(row, col);
      }
    }
  }

  std::string type2str(int type) {
    std::string r;

    uchar depth = type & CV_MAT_DEPTH_MASK;
    uchar chans = 1 + (type >> CV_CN_SHIFT);

    switch ( depth ) {
      case CV_8U:  r = "8U"; break;
      case CV_8S:  r = "8S"; break;
      case CV_16U: r = "16U"; break;
      case CV_16S: r = "16S"; break;
      case CV_32S: r = "32S"; break;
      case CV_32F: r = "32F"; break;
      case CV_64F: r = "64F"; break;
      default:     r = "User"; break;
    }

    r += "C";
    r += (chans+'0');

    return r;
  }

  void get_nv12_image (uint8_t * dst, int w, int h) {
    if (dst == NULL)
      return;

    int uBitDepth = 8,  h_scale = 2,  v_scale = 2;

    char * y_src = (char *)y_.data;
    char * u_src = (char *)u_.data;
    char * v_src = (char *)v_.data;

    char * y_dst = (char *)dst;
    char * uv_dst =  (char *)dst + w * h;

    int offset = w * (uBitDepth >= 10 ? 2 : 1);

    for (int i = 0; i < h; i++)
    {
      memcpy(y_dst, y_src, offset);

      y_src += y_stride;
      y_dst += y_stride;
    }

    // Luma
    int iWidthC = (w + h_scale - 1) / h_scale;
    int iHeightC = (h + v_scale - 1) / v_scale;

    for(int iH = 0; iH < iHeightC; ++iH)
    {
      for(int iW = 0; iW < iWidthC; ++iW)
      {
        uv_dst[iW * 2] = u_src[iW];
        uv_dst[iW * 2 + 1] = v_src[iW];
      }

      uv_dst += y_stride;
      u_src += u_stride;
      v_src += v_stride;
    }
  }

  cv::Mat convert2bgr() {
    if (format_ == YUV420P) {
      unsigned w = y_.cols;
      unsigned h = y_.rows;
      cv::Mat u_scaled;
      cv::resize(u_, u_scaled, cv::Size(w, h), 0, 0, cv::INTER_LINEAR);
      cv::Mat v_scaled;
      cv::resize(v_, v_scaled, cv::Size(w, h), 0, 0, cv::INTER_LINEAR);
      cv::Mat bgr;
      std::vector<cv::Mat> channels = {y_, u_scaled, v_scaled};
      cv::Mat merged;
      cv::merge(channels, merged);
      // cv::cvtColor(merged, bgr, cv::COLOR_YUV2BGR);
      return merged;
    } else if (format_ == YUV444P) {
      cv::Mat bgr;
      std::vector<cv::Mat> channels = {y_, u_, v_};
      cv::Mat merged;
      cv::merge(channels, merged);
      // cv::cvtColor(merged, bgr, cv::COLOR_YUV2BGR);
      return merged;
    }
    return cv::Mat();
  }

  void putText(const std::string& text, cv::Point text_pos, int fontFace, double fontScale,
               const cv::Scalar& color_rgb, int thickness = 1, int lineType = 8, bool bottomLeftOrigin = false) {

    cv::Mat color_rgb_m(cv::Size(1,1), CV_8UC3, {color_rgb[0], color_rgb[1], color_rgb[2]});
    cv::Mat color_yuv;
    cv::cvtColor(color_rgb_m, color_yuv, cv::COLOR_RGB2YUV);
    cv::Vec3b color_yuv_pix = color_yuv.at<cv::Vec3b>(0,0);


    cv::putText(y_, text, text_pos, fontFace, fontScale, color_yuv_pix[0], thickness, lineType, bottomLeftOrigin);

    text_pos = cv::Point(int(text_pos.x/2), int(text_pos.y/2));
    thickness = int(thickness/2);

    cv::putText(u_, text, text_pos, fontFace, fontScale/2, color_yuv_pix[1], thickness, lineType, bottomLeftOrigin);
    cv::putText(v_, text, text_pos, fontFace, fontScale/2, color_yuv_pix[2], thickness, lineType, bottomLeftOrigin);
  }

  void rectangle(cv::Point start_pt, cv::Point end_pt,
                 const cv::Scalar& color_rgb, int thickness = 1, int lineType = 8) {

    cv::Mat color_rgb_m(cv::Size(1,1), CV_8UC3, {color_rgb[0], color_rgb[1], color_rgb[2]});
    cv::Mat color_yuv;
    cv::cvtColor(color_rgb_m, color_yuv, cv::COLOR_RGB2YUV);
    cv::Vec3b color_yuv_pix = color_yuv.at<cv::Vec3b>(0,0);

    cv::rectangle(y_, start_pt, end_pt, color_yuv_pix[0], thickness, lineType);

    start_pt = cv::Point(int(start_pt.x/2), int(start_pt.y/2));
    end_pt = cv::Point(int(end_pt.x/2), int(end_pt.y/2));
    thickness = int(thickness/2);

    cv::rectangle(u_, start_pt, end_pt, color_yuv_pix[1], thickness, lineType);
    cv::rectangle(v_, start_pt, end_pt, color_yuv_pix[2], thickness, lineType);
  }

  void circle(cv::Point center, const cv::Scalar& color_rgb, int thickness = 5, int radius = 5) {
    cv::Mat color_rgb_m(cv::Size(1,1), CV_8UC3, {color_rgb[0], color_rgb[1], color_rgb[2]});
    cv::Mat color_yuv;
    cv::cvtColor(color_rgb_m, color_yuv, cv::COLOR_RGB2YUV);
    cv::Vec3b color_yuv_pix = color_yuv.at<cv::Vec3b>(0,0);

    // cv::circle(y_, center, thickness, color_yuv_pix[0], radius);

    thickness = int(thickness/2);

    cv::circle(u_, center, thickness, color_yuv_pix[1], radius);
    cv::circle(v_, center, thickness, color_yuv_pix[2], radius);
  }

private:
  cv::Mat y_, u_, v_;
  YUVFormat format_;
  int y_stride, u_stride, v_stride;
};

