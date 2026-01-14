//**************************************************************************
//||                        SiMa.ai CONFIDENTIAL                          ||
//||   Unpublished Copyright (c) 2025 SiMa.ai, All Rights Reserved.       ||
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

#include <aggregator/agg_template.h>
#include <simaai/boxdecode.h>

std::atomic<int> UserContext::global_instance_counter{0};
extern "C" int run(int instance, void * in_data, int in_data_size, void * out_data, int out_data_size);

UserContext::UserContext(const char *json_file_name) {
    instance_id = global_instance_counter++;
    if (::configure(instance_id, json_file_name) < 0)
        GST_ERROR("Error parsing JSON config for instance %d", instance_id);
}

void UserContext::run(std::vector<Input> &input, std::span<uint8_t> output) {
    if (::run(instance_id, static_cast<void *>(input[0].getData().data()),
        static_cast<int>(input[0].getDataSize()),
        static_cast<void *>(output.data()),
        static_cast<int>(output.size())) < 0)
        GST_ERROR("Error processing data");
}

UserContext::~UserContext() {
}
