# **aidget convert introduction**

## **Environmental Requirements**

- protobuf (version >= 3.0 is required)

## **Usage**
``` bash
Usage:
  ./aidget_convert [OPTION...]

  -h, --help              convert onnx/tflite model format to aidget model
                          
      --model_type arg    source model type, [ONNX,TFLITE]
      --model_file arg    onnx model file or tflite model file
      --aidget_model arg  converted aidget model, *.aidget
      --target arg        target device, [ARM,HIFI5], default ARM
      --quantized_model   indicate source model is a tflite quantized 
                          model, default: false
```
