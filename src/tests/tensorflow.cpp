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
#include "TensorflowRTEngine.h"
#include "RTExceptions.h"

#define CACHE_FILE "./tensorflow_cache.tensorcache"
#define MODEL_FILE "./inception_v3.uff"
#define NUM_SAMPLES 10
#define BATCH_SIZE 1

#define IMAGE_WIDTH	299
#define IMAGE_HEIGHT 299
#define IMAGE_DEPTH 3
#define IMAGE_ELESIZE 4

#define CLASS_ELESIZE 4
#define NB_CLASSES 1001

using namespace nvinfer1;
using namespace std;

int main(int argc, char** argv) {

	TensorflowRTEngine engine = TensorflowRTEngine();
	engine.addInput("input", DimsCHW(IMAGE_DEPTH, IMAGE_HEIGHT, IMAGE_WIDTH), IMAGE_ELESIZE);

	Dims outputDims; outputDims.nbDims = 1; outputDims.d[0] = NB_CLASSES;
	engine.addOutput("InceptionV3/Predictions/Reshape_1", outputDims, CLASS_ELESIZE);

	try{
		engine.loadCache(CACHE_FILE, BATCH_SIZE);
	} catch (ModelDeserializeException& e){
		engine.loadModel(MODEL_FILE, (size_t) BATCH_SIZE);
		engine.saveCache(CACHE_FILE);
	}

	std::cout << engine.engineSummary() << std::endl;

	/* Allocate memory for predictions */
	vector<vector<void*>> batch(BATCH_SIZE);
	for (int b=0; b < BATCH_SIZE; b++) {

		//Inputs
		batch[b].push_back(new unsigned char[IMAGE_DEPTH * IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_ELESIZE]);
	}

	for (;;) {
		int totalMs = 0;

		for (int i = 0; i < NUM_SAMPLES; i++) {
			auto t_start = std::chrono::high_resolution_clock::now();

			vector<vector<void*>> batchOutputs = engine.predict(batch);
			for (int b = 0; b < batchOutputs.size(); b++)
				delete ((unsigned char*)batchOutputs[b][0]);

			auto t_end = std::chrono::high_resolution_clock::now();
			auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();

			totalMs += ms;

		}

		totalMs /= NUM_SAMPLES;
		std::cout << "Average over " << NUM_SAMPLES << " runs is " << totalMs
				<< " ms." << std::endl;

	}
}


