#include <cuda_runtime.h>
#include <cuda.h>

cudaError_t cudaPreImageNet( float4* input, size_t inputWidth, size_t inputHeight,
				         float* output, size_t outputWidth, size_t outputHeight );
cudaError_t cudaPreImageNetMean( float4* input, size_t inputWidth, size_t inputHeight,
				             float* output, size_t outputWidth, size_t outputHeight, const float3& mean_value );
