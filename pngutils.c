#include <stdlib.h>
#include "lua.h"
#include "lauxlib.h"
#include "png.h"

static int l_hwc_to_chw (lua_State *L) {
  size_t len;
  const char* input = luaL_checklstring(L, 1, &len);
  size_t stride = len / 3;

  float* output_data = (float*)malloc(len * sizeof(float));
  luaL_argcheck(L, output_data != NULL, 1, "Out of memory");
  for (size_t i = 0; i != stride; ++i) {
    for (size_t c = 0; c != 3; ++c) {
      output_data[c * stride + i] = input[i * 3 + c];
    }
  }
  
  lua_pushlstring(L, (const char *)output_data, len * sizeof(float));

  return 1;
}

static int l_chw_to_hwc (lua_State *L) {
  size_t len;
  const float* input = (const float*)luaL_checklstring(L, 1, &len);
  len = len / sizeof(float);
  size_t stride = len / 3;

  char* output_data = (char*)malloc(len);
  luaL_argcheck(L, output_data != NULL, 1, "out of memry");
  for (size_t c = 0; c != 3; ++c) {
    size_t t = c * stride;
    for (size_t i = 0; i != stride; ++i) {
      float f = input[t + i];
      if (f < 0.f || f > 255.0f) f = 0;
      output_data[i * 3 + c] = (char)f;
    }
  }
 
  lua_pushlstring(L, output_data, len);

  return 1;
}

static int l_read_image_file (lua_State *L) {
  const char* input_file = luaL_checkstring(L, 1);

  png_image image; /* The control structure used by libpng */
  /* Initialize the 'png_image' structure. */
  memset(&image, 0, (sizeof image));
  image.version = PNG_IMAGE_VERSION;
  if (png_image_begin_read_from_file(&image, input_file) == 0) {
    luaL_error(L, "Reading image file failed");
  }
  char* buffer;
  image.format = PNG_FORMAT_BGR;
  size_t input_data_length = PNG_IMAGE_SIZE(image);

  buffer = (char*)malloc(input_data_length);
  memset(buffer, 0, input_data_length);
  if (png_image_finish_read(&image, NULL /*background*/, buffer, 0 /*row_stride*/, NULL /*colormap*/) == 0) {
    luaL_error(L, "Finish reading image file failed");
  }
  
  lua_pushinteger(L, image.width);
  lua_pushinteger(L, image.height);
  lua_pushlstring(L, buffer, input_data_length);
  return 3;
}


static int l_write_image_file (lua_State *L) {
  const char* image_data = luaL_checkstring(L, 1);
  unsigned int height = (unsigned int)luaL_checkinteger(L, 2);
  unsigned int width = (unsigned int)luaL_checkinteger(L, 3);
  const char* output_file = luaL_checkstring(L, 4);
  
  png_image image;
  memset(&image, 0, (sizeof image));
  image.version = PNG_IMAGE_VERSION;
  image.format = PNG_FORMAT_BGR;
  image.height = height;
  image.width = width;
  int ret = 0;
  if (png_image_write_to_file(&image, output_file, 0 /*convert_to_8bit*/, image_data, 0 /*row_stride*/,
			      NULL /*colormap*/) == 0) {
    luaL_error(L, "write to '%s' failed:%s\n", output_file, image.message);
  }

  return 0;
}

static const struct luaL_Reg luapngutils [] = {
    {"read", l_read_image_file},
    {"write", l_write_image_file},
    {"hwc2chw", l_hwc_to_chw},
    {"chw2hwc", l_chw_to_hwc},
    {NULL, NULL}
};

int luaopen_pngutils(lua_State *L) {
    luaL_newlib(L, luapngutils);

    return 1;
}