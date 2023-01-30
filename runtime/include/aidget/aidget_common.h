/*************************************************
 *
 *  Created by Aidget on 2022/11/30.           
 *  Copyright © 2022,  developed by Midea AIIC 
 *
 *************************************************/

#ifndef __AIDGET_COMMON_H__
#define __AIDGET_COMMON_H__

#include <stdint.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <assert.h>

#ifndef BUILD_SHARED_LIB
#define AIDGET_API
#else
#define AIDGET_API __attribute__((visibility("default")))
#endif

#if defined(__APPLE__)
#include <TargetConditionals.h>
#if TARGET_OS_IPHONE
#define BUILD_FOR_IOS
#endif
#endif

#ifdef USE_LOGCAT
#include <android/log.h>
#define AIDGET_ERROR(format, ...) __android_log_print(ANDROID_LOG_ERROR, "AidgetJNI", format, ##__VA_ARGS__)
#define AIDGET_PRINT(format, ...) __android_log_print(ANDROID_LOG_INFO, "AidgetJNI", format, ##__VA_ARGS__)
#else
#define AIDGET_PRINT(format, ...) printf(format, ##__VA_ARGS__)
#define AIDGET_ERROR(format, ...) printf(format, ##__VA_ARGS__)
#endif

#ifdef DEBUG
#define AIDGET_ASSERT(x)                                            \
    {                                                            \
        int res = (x);                                           \
        if (!res) {                                              \
            AIDGET_ERROR("Error for %s, %d\n", __FILE__, __LINE__); \
            assert(res);                                         \
        }                                                        \
    }
#else
#define AIDGET_ASSERT(x)                                            \
    {                                                            \
        int res = (x);                                           \
        if (!res) {                                              \
            AIDGET_ERROR("Error for %s, %d\n", __FILE__, __LINE__); \
        }                                                        \
    }
#endif

#define FUNC_PRINT(x) AIDGET_PRINT(#x "=%d in %s, %d \n", x, __func__, __LINE__);
#define FUNC_PRINT_ALL(x, type) AIDGET_PRINT(#x "=" #type " %" #type " in %s, %d \n", x, __func__, __LINE__);

#define AIDGET_CHECK(success, log) \
    if(!(success)){ \
        AIDGET_ERROR("Check failed: %s ==> %s\n", #success, #log); \
    }

typedef enum {
    kSuccess           = 0,
    kFailure           = 1,
    kMemoryOverflow    = 2,
    kNotSupport        = 3,
    kInferShapeError   = 4,
    kKernelNotFound    = 5,
    kInvalidValue      = 6,
    kInputDataError    = 7,
    kCallBackError     = 8,
} AidgetStatus;

typedef enum {
    kCPUInference   = 0,
    kHIFI5Inference = 1,
    kRISCVInference = 2,
} AidgetInferType;

enum PowerMode {
    Power_Normal = 0,
    Power_High   = 1,
    Power_Low    = 2,
};

#ifdef __cplusplus
#include <vector>
#include <string>
namespace aidget {
struct AidgetConfig {
    AidgetInferType type = kCPUInference;
    bool   dynamic_model = false;
    int       thread_num = 1;
    PowerMode      power = Power_Normal;
};
}; // namespace aidget

#endif  // __cplusplus

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum PrecisionCode {
    kIntType = 0,
    kUintType = 1,
    kFloatType = 2,
    kHandleType = 3,
} PrecisionCode;

struct PrecisionType {
#if __cplusplus >= 201103L
    __attribute__((aligned(1))) PrecisionCode code;
#else
    __attribute__((aligned(1))) uint8_t code;
#endif
    __attribute__((aligned(1))) uint8_t bits;
    __attribute__((aligned(2))) uint16_t lanes;

#ifdef __cplusplus
    inline PrecisionType(PrecisionCode code, uint8_t bits, uint16_t lanes = 1)
        : code(code), bits(bits), lanes(lanes) {
    }
    inline PrecisionType() : code((PrecisionCode)0), bits(0), lanes(0) {}

    inline bool operator==(const PrecisionType &other) const {
        return (code == other.code &&
                bits == other.bits &&
                lanes == other.lanes);
    }

    inline bool operator!=(const PrecisionType &other) const {
        return !(*this == other);
    }

    inline int bytes() const { return (bits + 7) / 8; }
#endif
};

typedef struct DimInfo {
    int32_t extent, stride;

#ifdef __cplusplus
    inline DimInfo() : extent(0), stride(0) {}
    inline DimInfo(int32_t e, int32_t s) :
        extent(e), stride(s) {}

    inline bool operator==(const DimInfo &other) const {
        return (extent == other.extent) &&
            (stride == other.stride);
    }

    inline bool operator!=(const DimInfo &other) const {
        return !(*this == other);
    }
#endif
} DimInfo;

#ifdef __cplusplus
} // extern "C"
#endif

typedef struct BufferDesc {
    uint64_t device;
    uint8_t* host;
    int32_t dimensions;
    DimInfo *dim;
    struct PrecisionType type;
} BufferDesc;


#ifdef __cplusplus

template<typename T>
inline PrecisionType PrecisionTypeOf() {
    return PrecisionType(kHandleType, 64);
}

#ifdef ENABLE_ARM82
#include <arm_fp16.h>
template<>
inline PrecisionType PrecisionTypeOf<float16_t>() {
    return PrecisionType(kFloatType, 16);
}
#endif

template<>
inline PrecisionType PrecisionTypeOf<float>() {
    return PrecisionType(kFloatType, 32);
}

template<>
inline PrecisionType PrecisionTypeOf<double>() {
    return PrecisionType(kFloatType, 64);
}

template<>
inline PrecisionType PrecisionTypeOf<bool>() {
    return PrecisionType(kUintType, 1);
}

template<>
inline PrecisionType PrecisionTypeOf<uint8_t>() {
    return PrecisionType(kUintType, 8);
}

template<>
inline PrecisionType PrecisionTypeOf<uint16_t>() {
    return PrecisionType(kUintType, 16);
}

template<>
inline PrecisionType PrecisionTypeOf<uint32_t>() {
    return PrecisionType(kUintType, 32);
}

template<>
inline PrecisionType PrecisionTypeOf<uint64_t>() {
    return PrecisionType(kUintType, 64);
}

template<>
inline PrecisionType PrecisionTypeOf<int8_t>() {
    return PrecisionType(kIntType, 8);
}

template<>
inline PrecisionType PrecisionTypeOf<int16_t>() {
    return PrecisionType(kIntType, 16);
}

template<>
inline PrecisionType PrecisionTypeOf<int32_t>() {
    return PrecisionType(kIntType, 32);
}

template<>
inline PrecisionType PrecisionTypeOf<int64_t>() {
    return PrecisionType(kIntType, 64);
}

#endif
#endif // __AIDGET_COMMON_H__
