cuda_add_library(
  jetson_tensorrt_cuda STATIC
  cudaImageNetPreprocess.cu
  cudaNormalize.cu
  cudaOverlay.cu
  cudaResize.cu
  cudaRGB.cu
  cudaYUV-NV12.cu
  cudaYUV-YUYV.cu
  cudaYUV-YV12.cu
)
