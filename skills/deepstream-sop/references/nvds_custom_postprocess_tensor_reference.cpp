/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <string.h>

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <sstream>
#include <string>
#include <unordered_map>

#include "infer_custom_process.h"
#include "nvbufsurface.h"
#include "nvdsmeta.h"

typedef struct _GstBuffer GstBuffer;

/** This is a example how DeepStream Triton plugin(gst-nvinferserver) do
 * custom extra input preprocess and custom postprocess on triton based models.
 */

// enable debug log
// user can set DS_LOG_DEBUG=1 to enable debug log
static const bool ENABLE_DEBUG() {
    // get env variable DS_LOG_DEBUG
    const char* ds_log_debug = getenv("LOG_DEBUG");
    if (ds_log_debug) {
        return atoi(ds_log_debug) != 0;
    }
    return false;
}

// single batch sliding windows size = 2 * FRAMES_PER_SIDE + 1
static const int FRAMES_PER_SIDE =
    atoi(getenv("FRAMES_PER_SIDE") ? getenv("FRAMES_PER_SIDE") : "5");

// sliding windows size = 2 * FRAMES_PER_SIDE + SEQUENCE_BATCH
static const int SEQUENCE_BATCH =
    atoi(getenv("SEQUENCE_BATCH") ? getenv("SEQUENCE_BATCH") : "8");
static const int SLIDING_WINDOWS_SIZE = 2 * FRAMES_PER_SIDE + SEQUENCE_BATCH;
static const int OFFSET = FRAMES_PER_SIDE + SEQUENCE_BATCH - 1;

namespace dsis = nvdsinferserver;

#define LOG_DEBUG(fmt, ...)                                                   \
    if (ENABLE_DEBUG()) {                                                     \
        fprintf(stdout, "%s:%d" fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__); \
    }

#define LOG_ERROR(fmt, ...) \
    fprintf(stderr, "%s:%d" fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__)

#ifndef INFER_ASSERT
#define INFER_ASSERT(expr)                                                     \
    do {                                                                       \
        if (!(expr)) {                                                         \
            fprintf(stderr, "%s:%d ASSERT(%s) \n", __FILE__, __LINE__, #expr); \
            std::abort();                                                      \
        }                                                                      \
    } while (0)
#endif

#define CSTR(str) (str).empty() ? "" : (str).c_str()

static const std::vector<std::string> kClassLabels = {"boundary"};

/** Define a function for custom processor for DeepStream Triton
 * plugin(nvinferserver) do custom extra input preprocess and custom postprocess
 * on triton based models. The sysmbol is loaded through infer_config {
 *     custom_lib {  path: "path/to/custom_impl_process.so" }
 *     extra {
 *       custom_process_funcion: "CreateInferServerCustomProcess"
 *     }}
 */
extern "C" dsis::IInferCustomProcessor* CreateInferServerCustomProcess(
    const char* config, uint32_t configLen);

namespace {
using namespace dsis;

std::string memType2Str(InferMemType t) {
    switch (t) {
        case InferMemType::kGpuCuda:
            return "kGpuCuda";
        case InferMemType::kCpu:
            return "kCpu";
        case InferMemType::kCpuCuda:
            return "kCpuPinned";
        default:
            return "Unknown";
    }
}

std::string dataType2Str(dsis::InferDataType t) {
    switch (t) {
        case InferDataType::kFp32:
            return "kFp32";
        case InferDataType::kFp16:
            return "kFp16";
        case InferDataType::kInt8:
            return "kInt8";
        case InferDataType::kInt32:
            return "kInt32";
        case InferDataType::kInt16:
            return "kInt16";
        case InferDataType::kUint8:
            return "kUint8";
        case InferDataType::kUint16:
            return "kUint16";
        case InferDataType::kUint32:
            return "kUint32";
        case InferDataType::kFp64:
            return "kFp64";
        case InferDataType::kInt64:
            return "kInt64";
        case InferDataType::kUint64:
            return "kUint64";
        case InferDataType::kString:
            return "kString";
        case InferDataType::kBool:
            return "kBool";
        default:
            return "Unknown";
    }
}

// return buffer description string
std::string strOfBufDesc(const dsis::InferBufferDescription& desc) {
    std::stringstream ss;
    ss << "*" << desc.name << "*, shape: ";
    for (uint32_t i = 0; i < desc.dims.numDims; ++i) {
        if (i != 0) {
            ss << "x";
        } else {
            ss << "[";
        }
        ss << desc.dims.d[i];
        if (i == desc.dims.numDims - 1) {
            ss << "]";
        }
    }
    ss << ", dataType:" << dataType2Str(desc.dataType);
    ss << ", memType:" << memType2Str(desc.memType);
    return ss.str();
}

}  // namespace

/** Example of a Custom process instance for
 * deepstream-triton(gst-nvinferserver) plugin It is derived from
 * nvdsinferserver::IInferCustomProcessor If should be loaded through
 * config_triton_inferserver_primary_fasterRCNN.txt:
 *   infer_config {
 *     custom_lib {  path: "path/to/custom_impl_process.so" }
 *     extra {
 *       custom_process_funcion: "CreateInferServerCustomProcess"
 *     }
 *   }
 */
class NvInferServerCustomProcess : public dsis::IInferCustomProcessor {
   private:
    std::map<uint64_t, std::vector<float>> _streamFeedback;
    std::mutex _streamMutex;

   public:
    ~NvInferServerCustomProcess() override = default;

    /**
     * override function
     * Do custom processing on extra inputs.
     * @primaryInput is already preprocessed. DO NOT update it again.
     * @extraInputs, do custom processing and fill all data according the tensor
     * shape
     * @options, it has most of the common Deepstream metadata along with
     * primary data. e.g. NvDsBatchMeta, NvDsObjectMeta, NvDsFrameMeta, stream
     * ids... see infer_ioptions.h to see all the potential key name and
     * structures in the key-value table.
     */
    NvDsInferStatus extraInputProcess(
        const std::vector<dsis::IBatchBuffer*>&
            primaryInputs,  // primary tensor(image) has been processed
        std::vector<dsis::IBatchBuffer*>& extraInputs,
        const dsis::IOptions* options) override {
        return NVDSINFER_SUCCESS;
    }

    /** override function
     * Custom processing for inferenced output tensors.
     * output memory types is controlled by gst-nvinferserver config file
     *     config_triton_inferserver_primary_fasterRCNN.txt:
     *       infer_config {
     *         backend {  output_mem_type: MEMORY_TYPE_CPU }
     *     }
     * User can even attach parsed metadata into GstBuffer from this function
     */
    NvDsInferStatus inferenceDone(const dsis::IBatchArray* outputs,
                                  const dsis::IOptions* inOptions) override {
        INFER_ASSERT(inOptions);
        std::vector<uint64_t> streamIds;
        INFER_ASSERT(inOptions->getValueArray(OPTION_NVDS_SREAM_IDS,
                                              streamIds) == NVDSINFER_SUCCESS);
        INFER_ASSERT(!streamIds.empty());
        int batchSize = (int)streamIds.size();
        LOG_DEBUG("custom_postprocess_tensor: streamIds size: %d\n", batchSize);

        GstBuffer* gstBuf = nullptr;
        NvDsBatchMeta* batchMeta = nullptr;
        std::vector<NvDsFrameMeta*> frameMetaList;
        NvBufSurface* bufSurf = nullptr;
        std::vector<NvBufSurfaceParams*> surfParamsList;
        int64_t unique_id = 0;

        // get GstBuffer
        if (inOptions->hasValue(OPTION_NVDS_GST_BUFFER)) {
            INFER_ASSERT(inOptions->getObj(OPTION_NVDS_GST_BUFFER, gstBuf) ==
                         NVDSINFER_SUCCESS);
        }
        INFER_ASSERT(gstBuf);

        // get NvBufSurface
        if (inOptions->hasValue(OPTION_NVDS_BUF_SURFACE)) {
            INFER_ASSERT(inOptions->getObj(OPTION_NVDS_BUF_SURFACE, bufSurf) ==
                         NVDSINFER_SUCCESS);
        }
        INFER_ASSERT(bufSurf);

        // get NvDsBatchMeta
        if (inOptions->hasValue(OPTION_NVDS_BATCH_META)) {
            INFER_ASSERT(inOptions->getObj(OPTION_NVDS_BATCH_META, batchMeta) ==
                         NVDSINFER_SUCCESS);
        }
        INFER_ASSERT(batchMeta);

        // get all frame meta list into vector<NvDsFrameMeta*>
        if (inOptions->hasValue(OPTION_NVDS_FRAME_META_LIST)) {
            INFER_ASSERT(inOptions->getValueArray(OPTION_NVDS_FRAME_META_LIST,
                                                  frameMetaList) ==
                         NVDSINFER_SUCCESS);
        }
        LOG_DEBUG("custom_postprocess_tensor: frameMetaList size: %d\n",
                  (int)frameMetaList.size());

        // get unique_id
        if (inOptions->hasValue(OPTION_NVDS_UNIQUE_ID)) {
            INFER_ASSERT(inOptions->getInt(OPTION_NVDS_UNIQUE_ID, unique_id) ==
                         NVDSINFER_SUCCESS);
        }

        // get all surface params list into vector<NvBufSurfaceParams*>
        if (inOptions->hasValue(OPTION_NVDS_BUF_SURFACE_PARAMS_LIST)) {
            INFER_ASSERT(
                inOptions->getValueArray(OPTION_NVDS_BUF_SURFACE_PARAMS_LIST,
                                         surfParamsList) == NVDSINFER_SUCCESS);
        }

        std::unordered_map<std::string, SharedIBatchBuffer> tensors;
        for (uint32_t iTensor = 0; iTensor < outputs->getSize(); ++iTensor) {
            const auto& outTensor = outputs->getSafeBuf(iTensor);
            INFER_ASSERT(outTensor);
            auto desc = outTensor->getBufDesc();
            LOG_DEBUG("out tensor: %s, desc: %s \n", CSTR(desc.name),
                      strOfBufDesc(desc).c_str());
            tensors.emplace(desc.name, outTensor);
        }

        if (tensors.find("output_0") == tensors.end()) {
            LOG_ERROR("custom_postprocess_tensor: no output_0 not found\n");
            return NVDSINFER_SUCCESS;
        }

        auto ouptut0 = tensors["output_0"];
        INFER_ASSERT(ouptut0);
        auto desc = ouptut0->getBufDesc();
        batchSize = desc.dims.d[0];
        LOG_DEBUG("custom_postprocess_tensor: batchSize: %d\n", batchSize);
        float* input0Ptr = (float*)ouptut0->getBufPtr(0);
        int frameRange = desc.dims.d[1];

        for (int idx = 0; idx < batchSize; ++idx) {
            NvDsFrameMeta* frameMeta = frameMetaList[idx];
            INFER_ASSERT(frameMeta);
            float* scorePtr = input0Ptr + idx * frameRange;
            for (int i = 0; i < frameRange; ++i) {
                float score = scorePtr[i];
                NvDsObjectMeta* objMeta =
                    nvds_acquire_obj_meta_from_pool(batchMeta);
                objMeta->unique_component_id = unique_id;
                objMeta->confidence = score;

                /* This is not for tracking obj but for frame number since
                 * offset is observed. */
                objMeta->object_id = frameMeta->frame_num - OFFSET + i;
                objMeta->class_id = 0;
                LOG_DEBUG(
                    "custom_postprocess_tensor: i: %d, frame_num: %d, score: "
                    "%f\n",
                    i, (int)objMeta->object_id, score);

                NvOSD_RectParams& rect_params = objMeta->rect_params;
                NvOSD_TextParams& text_params = objMeta->text_params;

                rect_params.left = 0;
                rect_params.top = 0;
                rect_params.width = frameMeta->pipeline_width;
                rect_params.height = frameMeta->pipeline_height;

                /* Border of width 3. */
                rect_params.border_width = 3;
                rect_params.has_bg_color = 0;
                rect_params.border_color = (NvOSD_ColorParams){1, 0, 0, 1};
                text_params.display_text =
                    g_strdup(kClassLabels[objMeta->class_id].c_str());
                strncpy(objMeta->obj_label,
                        kClassLabels[objMeta->class_id].c_str(),
                        MAX_LABEL_SIZE - 1);
                objMeta->obj_label[MAX_LABEL_SIZE - 1] = 0;

                /* Display text above the left top corner of the object. */
                text_params.x_offset = rect_params.left;
                text_params.y_offset = rect_params.top - 10;
                /* Set black background for the text. */
                text_params.set_bg_clr = 1;
                text_params.text_bg_clr = (NvOSD_ColorParams){0, 0, 0, 1};
                /* Font face, size and color. */
                text_params.font_params.font_name = (gchar*)"Serif";
                text_params.font_params.font_size = 11;
                text_params.font_params.font_color =
                    (NvOSD_ColorParams){1, 1, 1, 1};

                nvds_acquire_meta_lock(batchMeta);
                nvds_add_obj_meta_to_frame(frameMeta, objMeta, NULL);
                frameMeta->bInferDone = TRUE;
                nvds_release_meta_lock(batchMeta);
            }
        }

        return NVDSINFER_SUCCESS;
    }

    /** override function
     * Receiving errors if anything wrong inside lowlevel lib
     */
    void notifyError(NvDsInferStatus s) override {
        LOG_ERROR("nvds_custom_postprocess_tensor: notifyError %d\n", (int)s);
    }
};

/** Implementation to Create a custom processor for DeepStream Triton
 * plugin(nvinferserver) to do custom extra input preprocess and custom
 * postprocess on triton based models.
 */
extern "C" {
dsis::IInferCustomProcessor* CreateInferServerCustomProcess(
    const char* config, uint32_t configLen) {
    LOG_DEBUG("custom_postprocess_tensor: CreateInferServerCustomProcess\n");
    LOG_DEBUG("custom_postprocess_tensor: FRAMES_PER_SIDE: %d\n",
              FRAMES_PER_SIDE);
    LOG_DEBUG("custom_postprocess_tensor: SEQUENCE_BATCH: %d\n",
              SEQUENCE_BATCH);
    LOG_DEBUG("custom_postprocess_tensor: SLIDING_WINDOWS_SIZE: %d\n",
              SLIDING_WINDOWS_SIZE);
    LOG_DEBUG("custom_postprocess_tensor: OFFSET: %d\n", OFFSET);
    // Ownership of the returned IInferCustomProcessor* is transferred across
    // the extern "C" ABI to the nvinferserver low-level lib, which deletes it.
    // make_unique(...).release() keeps RAII during construction while handing a
    // raw owning pointer through the fixed C contract (no manual `new`).
    return std::make_unique<NvInferServerCustomProcess>().release();
}
}
