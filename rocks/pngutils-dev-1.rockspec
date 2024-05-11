package = "pngutils"
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

local png_incdir = "$(PNG_INCDIR)"
local png_libdir = "$(PNG_LIBDIR)"

build = {
   type = "builtin",
   modules = {
      pngutils = {
        sources = { "pngutils.c", },
        libraries = {"png"},
      }
   },
}

if png_incdir then
   build.modules.pngutils.incdirs = {png_incdir}
   build.modules.pngutils.libdirs = {png_libdir}
end