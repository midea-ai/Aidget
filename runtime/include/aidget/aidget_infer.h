/*************************************************
 *
 *  Created by Aidget on 2022/11/30.           
 *  Copyright © 2022,  developed by Midea AIIC 
 *
 *************************************************/

#ifndef __AIDGET_INFER_H__
#define __AIDGET_INFER_H__

#include <functional>
#include <map>
#include <string>
#include <vector>
#include <memory>
#include "aidget/aidget_common.h"

namespace aidget {

#define MAX_TENSOR_DIM 6

class Session;
class Tensor;
class Backend;
struct Content;
struct AidgetContext;

class AIDGET_API Tensor {
public:
    struct TensorDesc;
    enum DimensionType {
        TENSORFLOW, // NHWC
        CAFFE,      // NCHW
        CAFFE_C4    // NC4HW4
    };

public:
    Tensor(int dim = 4, DimensionType type = CAFFE);
    Tensor(const Tensor* tensor, DimensionType type = CAFFE, bool alloc = true);
    virtual ~Tensor();

    static Tensor* Create(const std::vector<int>& shape,
                          PrecisionType type,
                          DimensionType dim_type = TENSORFLOW) {
        return Allocate(shape, type, nullptr, dim_type, false);
    }
    template <typename T>
    static Tensor* Create(const std::vector<int>& shape,
                          DimensionType dim_type = TENSORFLOW) {
        return Allocate(shape, PrecisionTypeOf<T>(), nullptr, dim_type, false);
    }
    static Tensor* Allocate(const std::vector<int>& shape,
                            PrecisionType type,
                            void* data = nullptr,
                            DimensionType dim_type = TENSORFLOW,
                            bool alloc = true);
    template <typename T>
    static Tensor* Allocate(const std::vector<int>& shape,
                            void* data,
                            DimensionType dim_type) {
        return Allocate(shape, PrecisionTypeOf<T>(), data, dim_type);
    }
    template <typename T>
    static Tensor* Allocate(const std::vector<int>& shape,
                            DimensionType dim_type) {
        return Allocate(shape, PrecisionTypeOf<T>(), nullptr, dim_type);
    }
    template <typename T>
    static Tensor* Allocate(const std::vector<int>& shape,
                            void* data) {
        return Allocate(shape, PrecisionTypeOf<T>(), data);
    }
    template <typename T>
    static Tensor* Allocate(const std::vector<int>& shape) {
        return Allocate(shape, PrecisionTypeOf<T>());
    }

    static Tensor* Copy(const Tensor* tensor, bool copy = true);

    bool CopyFrom(const Tensor* tensor);
    bool CopyTo(Tensor* tensor) const;

    const BufferDesc& buffer() const {
        return buffer_;
    }
    BufferDesc& buffer() {
        return buffer_;
    }

    DimensionType GetDimensionType() const;
    void SetPrecisionType(int type);
    inline PrecisionType GetPrecisionType() const {
        return buffer_.type;
    }
    template <typename T>
    T* host() const {
        return (T*)buffer_.host;
    }
    int GetDimension() const {
        return buffer_.dimensions;
    }
    std::vector<int> GetShape() const;
    int GetBytesNum() const;
    inline int GetElemNum() const {
        return GetBytesNum() / buffer_.type.bytes();
    }

    inline int width() const {
        if (GetDimensionType() == TENSORFLOW) {
            return buffer_.dim[2].extent;
        }
        return buffer_.dim[3].extent;
    }
    inline int height() const {
        if (GetDimensionType() == TENSORFLOW) {
            return buffer_.dim[1].extent;
        }
        return buffer_.dim[2].extent;
    }
    inline int channel() const {
        if (GetDimensionType() == TENSORFLOW) {
            if (GetDimension() == 2) {
                return buffer_.dim[1].extent;
            } else {
                return buffer_.dim[3].extent;
            }
        }
        return buffer_.dim[1].extent;
    }
    inline int batch() const {
        return buffer_.dim[0].extent;
    }
    inline int stride(int index) const {
        return buffer_.dim[index].stride;
    }
    inline int length(int index) const {
        return buffer_.dim[index].extent;
    }
    inline void SetStride(int index, int stride) {
        buffer_.dim[index].stride = stride;
    }
    inline void SetLength(int index, int length) {
        buffer_.dim[index].extent = length;
    }
    inline void SetName(const std::string& name) { name_ = name; }
    inline const std::string& name() const { return name_; }
    void DebugTensor() const;
    void DebugShape() const;
private:
    BufferDesc buffer_;
    struct TensorDesc* desc_;
    std::string name_;
    friend class TensorUtils;
    Tensor(const Tensor& tensor)  = delete;
    Tensor(const Tensor&& tensor) = delete;
    Tensor& operator=(const Tensor&) = delete;
    Tensor& operator=(const Tensor&&) = delete;
};

class AIDGET_API OperatorInfo {
    struct Info;
public:
    const std::string& name() const;
    const std::string& type() const;
    float flops() const;

protected:
    OperatorInfo();
    ~OperatorInfo();
    Info* mContent;
};

typedef std::function<bool(const std::vector<Tensor*>&, const std::string&)> TensorCallBack;
typedef std::function<bool(const std::vector<Tensor*>&, const OperatorInfo*)> TensorCallBackWithInfo;

class AIDGET_API Interpreter {
public:
    static Interpreter* LoadModelFromFile(const char* file);
    static Interpreter* LoadModelFromBuffer(const void* buffer, size_t size);
    virtual ~Interpreter();

public:
    Session* CreateSession(const AidgetConfig& config);
    Session* CreateMultiPathSession(const std::vector<AidgetConfig>& configs);
    bool ReleaseSession(Session* session);
    void ResizeSession(Session* session);
    void ReleaseModel();
    AidgetStatus RunSession(Session* session) const;
    AidgetStatus RunSessionWithCallBack(const Session* session, const TensorCallBack& before, const TensorCallBack& end,
                                        bool sync = false) const;
    AidgetStatus RunSessionWithCallBackInfo(const Session* session, const TensorCallBackWithInfo& before,
                                            const TensorCallBackWithInfo& end, bool sync = false) const;
    Tensor* GetSessionInputByName(const Session* session, const char* name);
    Tensor* GetSessionInputByIndex(const Session* session, int index);
    Tensor* GetSessionOutputByName(const Session* session, const char* name);
    Tensor* GetSessionOutputByIndex(const Session* session, int index);

    const std::map<std::string, Tensor*>& GetSessionAllOutput(const Session* session) const;
    const std::map<std::string, Tensor*>& GetSessionAllInput(const Session* session) const;
    void PrintAllInputTensorInfo(const Session* session);
    void PrintAllOutputTensorInfo(const Session* session);
    void ReleaseMemory();
    void ResizeTensor(Tensor* tensor, const std::vector<int>& dims);
    void ResizeTensor(Tensor* tensor, int batch, int channel, int height, int width);
private:
    static Interpreter* LoadModelFromBufferInternal(Content* net);
    Content* mNet;
    AidgetContext* context_;
    Interpreter(Content* net);
    Interpreter(const Interpreter&)  = delete;
    Interpreter(const Interpreter&&) = delete;
    Interpreter& operator=(const Interpreter&) = delete;
    Interpreter& operator=(const Interpreter&&) = delete;
};
} // namespace aidget
#endif
