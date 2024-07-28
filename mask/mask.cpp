#include <glog/logging.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vitis/ai/env_config.hpp>
#include <xir/graph/graph.hpp>

#include "vart/dpu/vitis_dpu_runner_factory.hpp"
#include "vart/mm/host_flat_tensor_buffer.hpp"
#include "vart/runner_ext.hpp"
#include "vart/tensor_buffer.hpp"

static std::vector<float> convert_fixpoint_to_float(vart::TensorBuffer *tensor,
                                                    float scale);
static std::vector<float> softmax(const std::vector<float> &input);

static std::unique_ptr<vart::TensorBuffer> create_cpu_flat_tensor_buffer(
    const xir::Tensor *tensor)
{
    return std::make_unique<vart::mm::HostFlatTensorBuffer>(tensor);
}

static std::vector<float> convert_fixpoint_to_float(
    vart::TensorBuffer *tensor_buffer, float scale)
{
    uint64_t data = 0u;
    size_t size = 0u;
    std::tie(data, size) = tensor_buffer->data(std::vector<int>{0, 0});
    signed char *data_c = (signed char *)data;
    auto ret = std::vector<float>(size);
    transform(data_c, data_c + size, ret.begin(),
              [scale](signed char v)
              { return ((float)v) * scale; });
    return ret;
}

static std::vector<float> softmax(const std::vector<float> &input)
{
    auto output = std::vector<float>(input.size());
    std::transform(input.begin(), input.end(), output.begin(), expf);
    auto sum = accumulate(output.begin(), output.end(), 0.0f, std::plus<float>());
    std::transform(output.begin(), output.end(), output.begin(),
                   [sum](float v)
                   { return v / sum; });
    return output;
}

static std::vector<std::pair<int, float>> topk(const std::vector<float> &score,
                                               int K)
{
    auto indices = std::vector<int>(score.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::partial_sort(indices.begin(), indices.begin() + K, indices.end(),
                      [&score](int a, int b)
                      { return score[a] > score[b]; });
    auto ret = std::vector<std::pair<int, float>>(K);
    std::transform(
        indices.begin(), indices.begin() + K, ret.begin(),
        [&score](int index)
        { return std::make_pair(index, score[index]); });
    return ret;
}

template <typename T>
std::ostream &operator<<(std::ostream &os, std::vector<T> v)
{
    os << '{';
    std::copy(v.begin(), v.end(), std::ostream_iterator<T>(os, " "));
    os << '}';
    return os;
}

struct __attribute__((packed)) Candidate
{
    uint8_t x;
    uint8_t y;
    uint8_t width;
    uint8_t height;
    uint8_t object_confidence;
    uint8_t class_confidences[2];
};

enum Tag
{
    Fine,
    Violated,
};

struct Result
{
    cv::Rect2f box;
    Tag tag;
    float confidence;
};

class Mask
{
private:
    inline static const std::string KERNEL = "face_mask_0";

    std::vector<Result> nms(const std::vector<Result> &inputs, float threshold)
    {
        std::vector<Result> ret;
        std::vector<bool> eliminated(inputs.size());
        for (size_t i = 0; i < inputs.size(); i++)
        {
            if (!eliminated[i])
            {
                const auto &a = inputs[i];
                for (size_t j = i + 1; j < inputs.size(); j++)
                {
                    const auto &b = inputs[j];
                    auto I = (a.box & b.box).area();
                    auto U = a.box.area() + b.box.area() - I;
                    float iou = I / U;
                    if (iou > threshold)
                    {
                        eliminated[j] = true;
                    }
                }
                ret.push_back(a);
            }
        }
        return ret;
    }

    cv::Mat fill_input(const cv::Mat &image, vart::TensorBuffer &buffer)
    {
        cv::Mat resized;
        cv::resize(image, resized, cv::Size(512, 512));
        uint64_t offset = 0;
        size_t size = 0;
        std::tie(offset, size) = buffer.data(std::vector<int>{0, 0, 0, 0});
        uint8_t *start = static_cast<uint8_t *>(reinterpret_cast<void *>(offset));
        for (auto i = 0; i < resized.rows; i++)
        {
            for (auto j = 0; j < resized.cols; j++)
            {
                auto rgb = resized.at<cv::Vec3b>(i, j);
                *(start++) = static_cast<uint8_t>(static_cast<float>(rgb[0]) / 255.0f * 64.0f);
                *(start++) = static_cast<uint8_t>(static_cast<float>(rgb[1]) / 255.0f * 64.0f);
                *(start++) = static_cast<uint8_t>(static_cast<float>(rgb[2]) / 255.0f * 64.0f);
            }
        }
        return resized;
    }

    std::vector<Result> select(vart::TensorBuffer &buffer)
    {
        uint64_t offset = 0;
        size_t size = 0;
        auto tensor = buffer.get_tensor();
        std::tie(offset, size) = buffer.data(std::vector<int>{0, 0, 0, 0});
        Candidate *cursor = static_cast<Candidate *>(reinterpret_cast<void *>(offset));
        std::vector<Candidate> candidates;
        size_t length = size / 7;
        while (length--)
        {
            uint16_t value;
            value = static_cast<uint16_t>(cursor->class_confidences[0]) *
                    static_cast<uint16_t>(cursor->object_confidence);
            value /= 255u;
            cursor->class_confidences[0] = static_cast<uint8_t>(value);
            value = static_cast<uint16_t>(cursor->class_confidences[1]) *
                    static_cast<uint16_t>(cursor->object_confidence);
            value /= 255u;
            cursor->class_confidences[1] = static_cast<uint8_t>(value);
            if (cursor->width >= 2 && cursor->height >= 2)
            {
                candidates.push_back(*cursor);
            }
            cursor++;
        }
        std::sort(candidates.begin(), candidates.end(), [](const Candidate &a, const Candidate &b)
                  { return std::max(a.class_confidences[0], a.class_confidences[1]) >
                           std::max(b.class_confidences[0], b.class_confidences[1]); });
        auto top = candidates[0];
        std::vector<Candidate> top_candidates(candidates.begin(), candidates.begin() + 32);
        std::vector<Result> ret;
        for (const auto candidate : top_candidates)
        {
            float x = static_cast<float>(candidate.x) / 255.0f * 512.0f;
            float y = static_cast<float>(candidate.y) / 255.0f * 512.0f;
            float w = static_cast<float>(candidate.width) / 255.0f * 512.0f;
            float h = static_cast<float>(candidate.height) / 255.0f * 512.0f;
            cv::Rect2f rect(
                std::clamp(x - w / 2.0f, 0.0f, 512.0f),
                std::clamp(y - h / 2.0f, 0.0f, 512.0f),
                w,
                h);
            Result r;
            r.box = rect;
            if (candidate.class_confidences[0] > candidate.class_confidences[1])
            {
                r.confidence = candidate.class_confidences[0];
                r.tag = Violated;
            }
            else
            {
                r.confidence = candidate.class_confidences[1];
                r.tag = Fine;
            }
            ret.push_back(r);
        }
        return ret;
    }

public:
    std::unique_ptr<vart::Runner> runner;
    std::vector<std::unique_ptr<vart::Runner>> runners;

    Mask(const std::string &filename) : runner(
                                            vart::dpu::DpuRunnerFactory::create_dpu_runner(filename, Mask::KERNEL))
    {
    }

    bool run(const std::string &path)
    {
        auto image = cv::imread(path);
        auto inputs = this->runner->get_input_tensors();
        auto outputs = this->runner->get_output_tensors();
        auto input_buffer = create_cpu_flat_tensor_buffer(inputs[0]);
        auto output_buffers = std::array{
            create_cpu_flat_tensor_buffer(outputs[0]),
            create_cpu_flat_tensor_buffer(outputs[1]),
        };
        auto resized = this->fill_input(image, *input_buffer);

        std::cerr << "Input: ";
        for (auto const &input : inputs)
        {
            std::cerr << input->get_shape();
        }
        std::cerr << ' ' << 'x' << vart::get_input_scale(inputs[0]);
        std::cerr << std::endl;
        std::cerr << "Output: ";
        for (auto const &output : outputs)
        {
            std::cerr << output->get_shape();
        }
        auto scales = vart::get_output_scale(outputs);
        std::cerr << ' ' << 'x' << scales[0] << ' ' << 'x' << scales[1];
        std::cerr << std::endl;

        uint32_t job;
        int status;
        std::tie(job, status) = this->runner->execute_async(
            {input_buffer.get()},
            {
                output_buffers[0].get(),
                output_buffers[1].get(),
            });
        this->runner->wait(job, -1);

        auto results = this->select(*output_buffers[0]);
        auto filtered = this->nms(results, 0.6);
        std::sort(filtered.begin(), filtered.end(), [](const Result &a, const Result &b)
                  { return a.box.area() > b.box.area(); });
        auto result = filtered[0];
        if (result.tag == Fine)
        {
            std::cerr << "Fine" << result.box << std::endl;
            cv::rectangle(resized, result.box, cv::Scalar(76, 177, 34));
        }
        else
        {
            std::cerr << "Violated" << result.box << std::endl;
            cv::rectangle(resized, result.box, cv::Scalar(36, 28, 237));
        }
        cv::imwrite("output.jpg", resized, {});

        return result.tag == Fine;
    }
};

int main(int argc, char *argv[])
{
    Mask m("face_mask_detection_pt.xmodel");
    if (argc >= 2)
    {
        if (m.run(argv[1]))
        {
            return EXIT_SUCCESS;
        }
        else
        {
            return EXIT_FAILURE;
        }
    }
    return EXIT_FAILURE;
}

