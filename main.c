// C standard includes
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

// OpenCL includes
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.h>
#define SPNG_STATIC
#include "spng/spng.h"

#ifdef __unix__
#include <assert.h>
#include <errno.h>

  typedef int errno_t;
  errno_t fopen_s(FILE **f, const char *name, const char *mode) {
    errno_t ret = 0;
    assert(f);
    *f = fopen(name, mode);
    /* Can't be sure about 1-to-1 mapping of errno and MS' errno_t */
    if (!*f)
      ret = errno;
    return ret;
  }
#endif

uint8_t getByte(uint32_t i, uint32_t off) {
  return (uint8_t)((i >> off * 8) & 0xff);
}
uint8_t getByte16(uint16_t i, uint32_t off) {
  return (uint8_t)((i >> off * 8) & 0xff);
}
#define MAX_SOURCE_SIZE (0x1000000)
#define errorf(format, err, ...)                                               \
  do {                                                                         \
    if (err) {                                                                 \
      fprintf(stderr, format, err, __VA_ARGS__);                               \
    }                                                                          \
  } while (0)

struct Dims {
  double xmin;
  double xmax;
  double ymin;
  double ymax;
  int width;
  int height;
  double zoom;
  int zoom_level;
};
int main() {
  cl_platform_id platform_id = NULL;
  cl_device_id device_id = NULL;
  cl_device_id *devices_id = NULL;
  cl_context context = NULL;
  cl_command_queue command_queue = NULL;
  cl_mem Cmobj = NULL;
  cl_mem Args = NULL;
  cl_program program = NULL;
  cl_kernel kernel = NULL;
  cl_uint ret_num_devices;
  cl_uint ret_num_platforms;
  cl_int ret;

  int width = 7680 ;
  int height = 4320 ;
  char *C;
  int size = 3 * width * height;
  C = (char *)calloc(size, sizeof(char));

  FILE *fp;
  const char fileName[] = "./kernel.cl";
  size_t source_size;
  char *source_str;

  /* Load kernel source file */
  int err = fopen_s(&fp, fileName, "r");
  if (!fp) {
    fprintf(stderr, "Failed to load kernel.");
    exit(-1);
  }
  source_str = (char *)malloc(MAX_SOURCE_SIZE);
  source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
  fclose(fp);

  /* Get Platform/Device Information */
  ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
  errorf("%d at line %d \r\n", ret, __LINE__);
  ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id,
                       &ret_num_devices);
  errorf("%d at line %d \r\n", ret, __LINE__);

  /* Create OpenCL Context */
  context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
  errorf("%d at line %d \r\n", ret, __LINE__);

  /*Create command queue */
  command_queue = clCreateCommandQueue(context, device_id, 0, &ret);
  errorf("%d at line %d \r\n", ret, __LINE__);

  /*Create Buffer Object */
  double cordx = -0.16498674718560924;
  double cordy = -1.0394110666891514;
  double zoom_level = 15;
  double zoom = powf(0.75,zoom_level) ;
  Cmobj = clCreateBuffer(context, CL_MEM_READ_WRITE, size, NULL, &ret);
  errorf("%d at line %d \r\n", ret, __LINE__);
  Args = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(struct Dims), NULL,
                        &ret);
  struct Dims args;
  double fwidth = (double)width;
  double fheight = (double)height;
  double ratio = fmax(fwidth, fheight) / fmin(fwidth, fheight);
  args.xmin = (-ratio)* zoom  + cordx;
  args.xmax = (ratio)* zoom    + cordx ;
  args.ymin = (-1.0 ) *zoom + cordy ;  
  args.ymax = (1.0 )*zoom + cordy ;
  args.width = width;
  args.height = height;
  args.zoom = zoom;
  args.zoom_level = zoom_level;
  printf("%f %f",args.xmin, args.xmax);

  /* Copy input data to the memory buffer */
  ret = clEnqueueWriteBuffer(command_queue, Args, CL_TRUE, 0,
                             sizeof(struct Dims), &args, 0, NULL, NULL);
  errorf("%d at line %d \r\n", ret, __LINE__);
  /* Create kernel program from source file*/
  program = clCreateProgramWithSource(context, 1, (const char **)&source_str,
                                      (const size_t *)&source_size, &ret);
  errorf("%d at line %d \r\n", ret, __LINE__);

  ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
  if (1 || ret != 0) {
    size_t log_size;
    clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL,
                          &log_size);

    char *log = (char *)calloc(log_size, sizeof(char));

    clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size,
                          log, NULL);
    printf("%s\n", log);
    free(log);
  }

  errorf("%d at line %d \r\n", ret, __LINE__);
  /* Create data parallel OpenCL kernel */
  kernel = clCreateKernel(program, "mandlebrot", &ret);
  errorf("%d at line %d \r\n", ret, __LINE__);

  /* Set OpenCL kernel arguments */
  ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&Cmobj);
  errorf("%d at line %d \r\n", ret, __LINE__);
  ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&Args);
  errorf("%d at line %d \r\n", ret, __LINE__);
  // ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*) (&Args +
  // sizeof(int))); ret = clSetKernelArg(kernel, 3, sizeof(cl_mem),
  // (void*)(&Args + sizeof(int) * 2));

  size_t global_item_size = width * height;
  size_t local_item_size = 1;

  /* Execute OpenCL kernel as data parallel */
  ret =
      clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_item_size,
                             &local_item_size, 0, NULL, NULL);
  errorf("%d at line %d \r\n", ret, __LINE__);

  /* Transfer result to host */
  ret = clEnqueueReadBuffer(command_queue, Cmobj, CL_TRUE, 0,
                            size * sizeof(char), C, 0, NULL, NULL);
  errorf("%d at line %d \r\n", ret, __LINE__);

  char *header = (char *)calloc(0x20, sizeof(char));

  struct spng_ihdr ihdr = {0};
  spng_ctx *ctx = NULL;
  ctx = spng_ctx_new(SPNG_CTX_ENCODER);
  spng_set_option(ctx,SPNG_ENCODE_TO_BUFFER,1);
  ihdr.width = width;
  ihdr.height = height;
  ihdr.color_type  = SPNG_COLOR_TYPE_TRUECOLOR;
  ihdr.bit_depth = 8;
  spng_set_ihdr(ctx,&ihdr);
  int fmt = SPNG_FMT_PNG;
  int png_ret = spng_encode_image(ctx, C, size, fmt, SPNG_ENCODE_FINALIZE);
  if(png_ret){
    printf("spng_encode_image() error: %s\n",spng_strerror(png_ret));
  }
  size_t png_size;
  void *png_buf = NULL;

  /* Get the internal buffer of the finished PNG */
  png_buf = spng_get_png_buffer(ctx, &png_size, &png_ret);

  if(png_buf == NULL)
  {
      printf("spng_get_png_buffer() error: %s\n", spng_strerror(ret));
  } 
  
  /* Display Results */
  FILE *f;
  err = fopen_s(&f, "mandlebrot.png", "wb");
  fwrite(png_buf, sizeof(char), png_size, f);
  fclose(f);

  /* Finalization */
  ret = clFlush(command_queue);
  ret = clFinish(command_queue);
  ret = clReleaseKernel(kernel);
  ret = clReleaseProgram(program);
  ret = clReleaseMemObject(Cmobj);
  ret = clReleaseMemObject(Args);
  ret = clReleaseCommandQueue(command_queue);
  ret = clReleaseContext(context);

  free(source_str);
  free(png_buf);

  free(header);
  free(C);

  return ret;
}
