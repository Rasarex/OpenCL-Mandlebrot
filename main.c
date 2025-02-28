// C standard includes
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

// OpenCL includes
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.h>

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

uint32_t BMP_HEADER(char *buff, uint32_t size, uint16_t height,
                    uint16_t width) {

  buff[0x00] = 'B';
  buff[0x01] = 'M';
  buff[0x02] = getByte(size, 0);
  buff[0x03] = getByte(size, 1);
  buff[0x04] = getByte(size, 2);
  buff[0x05] = getByte(size, 3);

  buff[0x0A] = 0x20;
  buff[0x0B] = 0;
  buff[0x0C] = 0;
  buff[0x0D] = 0;

  buff[0x0E] = 12;
  buff[0x0F] = 0;
  buff[0x10] = 0;
  buff[0x11] = 0;

  buff[0x12] = getByte16(width, 0);
  buff[0x13] = getByte16(width, 1);
  buff[0x14] = getByte16(height, 0);
  buff[0x15] = getByte16(height, 1);

  buff[0x16] = 1;
  buff[0x17] = 0;
  buff[0x18] = 24;
  buff[0x19] = 0;
  return 0x20;
}
struct Dims {
  float xmin;
  float xmax;
  float ymin;
  float ymax;
  int width;
  int height;
  float zoom;
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

  int width = 7680;
  int height = 4320;
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
  float cordx = -0.16498674718560924;
  float cordy = -1.0394110666891514;
  float zoom =1   ;
  Cmobj = clCreateBuffer(context, CL_MEM_READ_WRITE, size, NULL, &ret);
  errorf("%d at line %d \r\n", ret, __LINE__);
  Args = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(struct Dims), NULL,
                        &ret);
  struct Dims args;
  float fwidth = (float)width;
  float fheight = (float)height;
  float ratio = fmaxf(fwidth, fheight) / fminf(fwidth, fheight);
  args.xmin = (-ratio) / zoom + cordx;
  args.xmax = (ratio) / zoom + cordx; //-0.80;
  args.ymin = (-1.0) / zoom - cordy;  // ( ((float)height)/zoom/2.0) + cordy;
  args.ymax =
      (1.0) / zoom - cordy; //(((float)height)/zoom/2.0) - cordy;//-0.80;
  args.width = width;
  args.height = height;
  args.zoom = zoom;

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

  /* Display Results */
  FILE *f;
  err = fopen_s(&f, "img.bmp", "wb");
  BMP_HEADER(header, size, height, width);
  fwrite(header, sizeof(char), 0x20, f);
  fwrite(C, sizeof(char), size, f);
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

  free(header);
  free(C);

  return ret;
}
