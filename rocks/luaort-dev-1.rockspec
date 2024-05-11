package = "luaort"
version = "dev-1"
source = {
   url = "git+https://git@github.com/MrSyabro/luaort.git",
   branch = "master",
}
description = {
   homepage = "https://github.com/MrSyabro/luaort",
   license = "MIT/X11",
   maintainer = "MrSyabro",
}
dependencies = {
   "lua >= 5.2"
}

local ort_incdir = "$(ONNXRUNTIME_INCDIR)"
local ort_libdir = "$(ONNXRUNTIME_LIBDIR)"

build = {
   type = "builtin",
   modules = {
      luaort = {
        sources = { "ort.c", },
        libraries = {"onnxruntime"}
      }
   },
}

if ort_incdir then
   build.modules.luaort.incdirs = {ort_incdir}
   build.modules.luaort.libdirs = {ort_libdir}
end