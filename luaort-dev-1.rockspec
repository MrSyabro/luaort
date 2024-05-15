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

build = {
   type = "builtin",
   modules = {
      luaort = {
        sources = { "src/ort.c", },
        libraries = {"onnxruntime"}
      }
   },
}