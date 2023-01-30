cmake_minimum_required(VERSION 3.0)
project(aidget_demo)

set(BUILD_PLATFORM "android" CACHE STRING "select build platform type")
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(${CMAKE_SOURCE_DIR}/include)

if(BUILD_PLATFORM STREQUAL android)
    if (CMAKE_SYSTEM_PROCESSOR MATCHES "^aarch64" OR ARCHS STREQUAL "arm64")
      set(LIB_PATH ${CMAKE_SOURCE_DIR}/libs/android/armv8)
    else()
      set(LIB_PATH ${CMAKE_SOURCE_DIR}/libs/android/armv7)
    endif()
elseif (BUILD_PLATFORM STREQUAL raspberry-pi)
    set(LIB_PATH ${CMAKE_SOURCE_DIR}/libs/raspberry-pi)
elseif (BUILD_PLATFORM STREQUAL rk3568)
    set(LIB_PATH ${CMAKE_SOURCE_DIR}/libs/rk3568)
else()
    message(FATAL_ERROR "Unsupport Platform!")
endif()

add_executable(aidget_demo example/model_test.cpp)

if(CMAKE_SYSTEM_NAME MATCHES "^Linux")
  target_link_libraries(aidget_demo -pthread)
elseif(CMAKE_SYSTEM_NAME MATCHES "^Android")
  target_link_libraries(aidget_demo log android m)
endif()

target_link_libraries(aidget_demo ${LIB_PATH}/libaidget.so)
