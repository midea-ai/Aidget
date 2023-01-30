## **aidget user guide**
### **Directory Structure**
```
├── converter
│   ├── aidget_converter
│── runtime
│   ├── libs
│   |   ├── android
│   |   |   ├── armv8
│   |   |   |   ├── libaidget.so
│   |   |   ├── armv7
│   |   |   |   ├── libaidget.so
│   |   ├── raspberry-pi
│   |   |   ├── libaidget.so(armv8)
│   |   ├── rk3568
│   |   |   ├── libaidget.so(armv8)
│   |   ├── rq3800-alios
│   |   |   ├── libaidget.a
│   ├── example
│   |   ├── model_test.cpp
│   ├── models
│   |   ├── kws_model.aidget
```
### **Build aidget**
``` bash
cd runtime && mkdir build && cd build
```
- for android armv8
``` bash
cmake .. -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake -DCMAKE_BUILD_TYPE=Release -DANDROID_ABI="arm64-v8a" -DANDROID_STL=c++_static
```
- for android armv7
``` bash
cmake .. -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake  -DCMAKE_BUILD_TYPE=Release -DANDROID_ABI="armeabi-v7a" -DANDROID_STL=c++_static
```
- for raspberry-pi armv8
``` bash
export TOOLCHAIN_PATH=/path/to/raspberry_toolchain
cmake .. -DCMAKE_CXX_COMPILER=$TOOLCHAIN_PATH/aarch64-linux-gnu-g++ -DCMAKE_C_COMPILER=$TOOLCHAIN_PATH/aarch64-linux-gnu-gcc -DCMAKE_SYSTEM_PROCESSOR=aarch64 -DCMAKE_SYSTEM_NAME=Linux -DBUILD_PLATFORM=raspberry-pi
```
- for rk356x(rk3566/rk3568) armv8
``` bash
export TOOLCHAIN_PATH=/path/to/rk356x_toolchain
cmake .. -DCMAKE_CXX_COMPILER=$TOOLCHAIN_PATH/aarch64-linux-g++ -DCMAKE_C_COMPILER=$TOOLCHAIN_PATH/aarch64-linux-gcc -DCMAKE_SYSTEM_PROCESSOR=aarch64 -DCMAKE_SYSTEM_NAME=Linux -DBUILD_PLATFORM=rk3568
```
``` bash
make -j12
```
### **Run aidget**
``` bash
adb push libs/$BUILD_PLATFORM/libaidget.so ${path_to_run}
adb push build/aidget_demo ${path_to_run}
adb push models/kws_model.aidget ${path_to_run}
adb shell "cd ${path_to_run} && ./aidget_demo kws_model.aidget"
```