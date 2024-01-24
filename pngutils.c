#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "lua.h"
#include "lauxlib.h"
#include "png.h"

void hwc_to_chw(const uint8_t* input, size_t h, size_t w, float** output, size_t* output_count) {
  size_t stride = h * w;
  *output_count = stride * 3;
  float* output_data = (float*)malloc(*output_count * sizeof(float));
  assert(output_data != NULL);
  for (size_t i = 0; i != stride; ++i) {
    for (size_t c = 0; c != 3; ++c) {
      output_data[c * stride + i] = input[i * 3 + c];
    }
  }
  *output = output_data;
}

int read_image_file(const char* input_file, size_t* height, size_t* width, float** out, size_t* output_count) {
  png_image image; /* The control structure used by libpng */
  /* Initialize the 'png_image' structure. */
  memset(&image, 0, (sizeof image));
  image.version = PNG_IMAGE_VERSION;
  if (png_image_begin_read_from_file(&image, input_file) == 0) {
    return -1;
  }
  uint8_t* buffer;
  image.format = PNG_FORMAT_BGR;
  size_t input_data_length = PNG_IMAGE_SIZE(image);
  if (input_data_length != 720 * 720 * 3) {
    printf("input_data_length:%zd\n", input_data_length);
    return -1;
  }
  buffer = (uint8_t*)malloc(input_data_length);
  memset(buffer, 0, input_data_length);
  if (png_image_finish_read(&image, NULL /*background*/, buffer, 0 /*row_stride*/, NULL /*colormap*/) == 0) {
    return -1;
  }
  hwc_to_chw(buffer, image.height, image.width, out, output_count);
  free(buffer);
  *width = image.width;
  *height = image.height;
  return 0;
}

int write_image_file(uint8_t* model_output_bytes, unsigned int height,
                     unsigned int width, const char* output_file){
  png_image image;
  memset(&image, 0, (sizeof image));
  image.version = PNG_IMAGE_VERSION;
  image.format = PNG_FORMAT_BGR;
  image.height = height;
  image.width = width;
  int ret = 0;
  if (png_image_write_to_file(&image, output_file, 0 /*convert_to_8bit*/, model_output_bytes, 0 /*row_stride*/,
			      NULL /*colormap*/) == 0) {
    printf("write to '%s' failed:%s\n", output_file, image.message);
    ret = -1;
  }
  return ret;
}

static const struct luaL_Reg luaort [] = {
    {"read", lpu_createenv},
    {"write", lpu_createsessionoptions},
    {NULL, NULL}
};

int luaopen_pngutils(lua_State *L) {
    return 0;
}