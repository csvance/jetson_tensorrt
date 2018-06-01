#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iostream>
#include <string>
#include <sys/stat.h>
#include <unordered_map>
#include <cassert>
#include <vector>

#include "NvInfer.h"

#include "TensorRTEngine.h"
#include "DIGITSDetector.h"
#include "RTExceptions.h"

#define CACHE_FILE "detection.tensorcache"
#define MODEL_FILE "detectnet.prototxt"
#define WEIGHTS_FILE "ped-100.caffemodel"

#define NUM_SAMPLES 10
#define BATCH_SIZE 1

#define IMAGE_WIDTH 224
#define IMAGE_HEIGHT 224
#define IMAGE_DEPTH 3
#define IMAGE_ELESIZE 4

#define NB_CLASSES 1
#define CLASS_ELESIZE 4

using namespace nvinfer1;
using namespace std;
using namespace jetson_tensorrt;

int main(int argc, char** argv) {

	DIGITSDetector engine = DIGITSDetector(MODEL_FILE, WEIGHTS_FILE, CACHE_FILE, IMAGE_DEPTH, IMAGE_WIDTH, IMAGE_HEIGHT, NB_CLASSES, BATCH_SIZE);

	std::cout << engine.engineSummary() << std::endl;

	/* Allocate memory for detections */
	LocatedExecutionMemory input = LocatedExecutionMemory(LocatedExecutionMemory::Location::HOST, vector<vector<void*>>(BATCH_SIZE));

	for (int b=0; b < BATCH_SIZE; b++) {
		//Inputs
		input[b].push_back(new unsigned char[IMAGE_DEPTH * IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_ELESIZE]);
	}

	for (;;) {
		int totalMs = 0;

		for (int i = 0; i < NUM_SAMPLES; i++) {
			auto t_start = std::chrono::high_resolution_clock::now();

			LocatedExecutionMemory result = engine.predict(input);
			for (int b = 0; b < result.size(); b++)
				delete ((unsigned char*)result[b][0]);

			auto t_end = std::chrono::high_resolution_clock::now();
			auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();

			totalMs += ms;

		}

		totalMs /= NUM_SAMPLES;
		std::cout << "Average over " << NUM_SAMPLES << " runs is " << totalMs
				<< " ms." << std::endl;

	}
}


