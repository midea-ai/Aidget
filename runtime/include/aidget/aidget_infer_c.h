/*************************************************
 *
 *  Created by Aidget on 2022/11/30.           
 *  Copyright © 2022,  developed by Midea AIIC 
 *
 *************************************************/

#ifndef __AIDGET_INFER_C_H__
#define __AIDGET_INFER_C_H__

#include "aidget/aidget_common.h"
#ifdef __cplusplus
extern "C" {
#endif
void* inferEngineInit(unsigned char* model, unsigned int model_size, AidgetInferType target_device, int num_thread);
int inferEngineResize(void* sess);
int inferEngineRun(void* sess);

int resizeInputTensorByIndex(void* sess, int index, int* dims, int dim_size);
int resizeInputTensorByName(void* sess, char* tensor_name, int* dims, int dim_size);

void* getInputTensorBufferByName(void* sess, char* tensor_name);
void* getOutputTensorBufferByName(void* sess, char* tensor_name);

void* getInputTensorBufferByIndex(void* sess, int index);
void* getOutputTensorBufferByIndex(void* sess, int index);

int getModelInputSize(void* sess);
int getModelOutputSize(void* sess);

int getInputTensorElemSizeByName(void* sess, char* tensor_name);
int getInputTensorElemSizeByIndex(void* sess, int index);

int getInputTensorByteSizeByName(void* sess, char* tensor_name);
int getInputTensorByteSizeByIndex(void* sess, int index);

int getOutputTensorElemSizeByName(void* sess, char* name);
int getOutputTensorElemSizeByIndex(void* sess, int index);

int getOutputTensorByteSizeByName(void* sess, char* tensor_name);
int getOutputTensorByteSizeByIndex(void* sess, int index);

void printAllInputTensorInfo(void* sess);
void printAllOutputTensorInfo(void* sess);

void profileOpInferSummary(void* sess, int run_time);

#ifdef __cplusplus
}
#endif
#endif
