#include "aidget/aidget_infer.h"
// #include "stdio.h"
#include <iostream>
#include <vector>
#include <string>

struct MetaData {
    std::vector<std::string> input_name;
    std::vector<std::string> output_name;
    aidget::Interpreter*     net;
    aidget::Session*         session;
};

MetaData* Init(const char* model) {
    aidget::Interpreter* net = aidget::Interpreter::LoadModelFromFile(model);
    aidget::AidgetConfig config;
    config.type = kCPUInference;
    aidget::Session* session = net->CreateSession(config);
    MetaData* meta = new MetaData();
    meta->input_name.resize(5);
    meta->output_name.resize(5);
    meta->input_name[0] = "input_audio_1"; // 1, 320
    meta->input_name[1] = "streaming_1/speech_features/data_frame_2input_state"; // 1, 640
    meta->input_name[2] = "streaming_1/stream/stream/ExternalState";     // 1, 3, 20, 1
    meta->input_name[3] = "streaming_1/stream_1/stream_1/ExternalState"; // 1, 5, 18, 16
    meta->input_name[4] = "streaming_1/gru_4input_state"; // 1, 256
    meta->output_name[0] = "streaming_1/net/BiasAdd";
    meta->output_name[1] = "streaming_1/speech_features/data_frame_2/concat"; // 1, 640
    meta->output_name[2] = "streaming_1/stream/concat";   // 1, 3, 20, 1
    meta->output_name[3] = "streaming_1/stream_1/concat"; // 1, 5, 18, 16
    meta->output_name[4] = "streaming_1/gru_4/cell/add_3"; // 1, 256
    meta->net = net;
    meta->session = session;
    // for debug
    net->PrintAllInputTensorInfo(session);
    net->PrintAllOutputTensorInfo(session);
    return meta;
}
aidget::Tensor* Infer(MetaData* meta) {
    aidget::Interpreter* net = meta->net;
    aidget::Session* session = meta->session;
    for (auto& input_name: meta->input_name) {
        aidget::Tensor* input_tensor = net->GetSessionInputByName(session, input_name.c_str());
        for (int i = 0; i < input_tensor->GetElemNum(); ++i) {
            input_tensor->host<float>()[i] = 1.0f;
        }
    }
    net->RunSession(session);
    aidget::Tensor* output_tensor = net->GetSessionOutputByName(session, meta->output_name[0].c_str());
    return output_tensor;
}

int main(int argc, char** argv) {
    const char* model_name = argv[1];
    MetaData* meta = Init(model_name);
    aidget::Tensor* result = Infer(meta);
    result->DebugTensor();
    std::cout << "Inference Done!" << std::endl;
    return 0;
}