#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdlib>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iostream>
#include <string>
#include <sys/stat.h>
#include <unordered_map>
#include <vector>

#include "NvInfer.h"

#include "CUDANode.h"
#include "CUDANodeIO.h"
#include "CUDAPipeline.h"
#include "DIGITSDetector.h"
#include "TensorRTEngine.h"

#define CACHE_FILE "detection.tensorcache"
#define MODEL_FILE "detectnet.prototxt"
#define WEIGHTS_FILE "ped-100.caffemodel"

#define NUM_SAMPLES 10
#define BATCH_SIZE 1

#define MODEL_IMAGE_WIDTH 1024
#define MODEL_IMAGE_HEIGHT 512
#define MODEL_IMAGE_DEPTH 3
#define MODEL_IMAGE_ELESIZE 4

#define INPUT_IMAGE_WIDTH 640
#define INPUT_IMAGE_HEIGHT 480
#define INPUT_IMAGE_DEPTH 4
#define INPUT_IMAGE_ELESIZE 4

#define NB_CLASSES 1
#define CLASS_ELESIZE 4

using namespace nvinfer1;
using namespace std;
using namespace jetson_tensorrt;

int main(int argc, char **argv) {

  DIGITSDetector engine =
      DIGITSDetector(MODEL_FILE, WEIGHTS_FILE, CACHE_FILE, MODEL_IMAGE_DEPTH,
                     MODEL_IMAGE_WIDTH, MODEL_IMAGE_HEIGHT, NB_CLASSES);

  std::cout << engine.engineSummary() << std::endl;

  // Loads the image into device memory and preprocesses it
  CUDAPipeline *preprocessPipeline = CUDAPipeline::createRGBAfImageNetPipeline(
      INPUT_IMAGE_WIDTH, INPUT_IMAGE_HEIGHT, MODEL_IMAGE_WIDTH,
      MODEL_IMAGE_HEIGHT, make_float3(0.0, 0.0, 0.0));

  /* Create input structure for predictions */
  LocatedExecutionMemory input = engine.allocInputs(MemoryLocation::NONE);
  input.location = MemoryLocation::DEVICE;

  /* Create and allocate output structure */
  LocatedExecutionMemory output = engine.allocOutputs(MemoryLocation::HOST);

  // Allocate the image memory
  CUDANodeIO preprocessInput = CUDANodeIO(
      MemoryLocation::HOST,
      new float[INPUT_IMAGE_DEPTH * INPUT_IMAGE_WIDTH * INPUT_IMAGE_HEIGHT],
      INPUT_IMAGE_DEPTH *INPUT_IMAGE_WIDTH *INPUT_IMAGE_HEIGHT);

  for (;;) {
    int totalMs = 0;

    for (int i = 0; i < NUM_SAMPLES; i++) {
      auto t_start = std::chrono::high_resolution_clock::now();

      // Preprocess the image (and load to GPU if needed)
      CUDANodeIO preprocessOutput = preprocessPipeline->pipe(preprocessInput);

      // Load the image into the network inputs (1st batch, 1st input)
      input[0][0] = preprocessOutput.data;

      std::vector<ClassRectangle> detections =
          engine.detect(input, output, 0.5);

      std::cout << detections.size() << std::endl;

      auto t_end = std::chrono::high_resolution_clock::now();
      auto ms =
          std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start)
              .count();

      totalMs += ms;
    }

    totalMs /= NUM_SAMPLES;
    std::cout << "Average over " << NUM_SAMPLES << " runs is " << totalMs
              << " ms." << std::endl;
  }
}
