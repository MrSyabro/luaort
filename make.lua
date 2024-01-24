local lm = require "luamake"

local vcpkg = "D:\\vcpkg\\packages"
local libpngpath = vcpkg .. "\\libpng_x64-windows"
local zlibpath = vcpkg .. "\\zlib_x64-windows"
--local ortlibpath = "D:\\onnxruntime-win-x64-1.10.0"
local ortlibpath = "D:\\onnxruntime-training-win-x64-1.16.3"

lm:lua_dll "pngutils" {
    includes = {
        libpngpath .. "\\include",
        zlibpath .. "\\include"
    },
    linkdirs = {
        libpngpath .. "\\lib",
        zlibpath .. "\\lib"
    },
    links = {"libpng16", "zlib"},
    sources = "pngutils.c",
    mode = "debug",
    msvc = {
        flags = "/utf-8",
    },
}

lm:lua_dll "luaort" {
    includes = ortlibpath .. "\\include",
    linkdirs = ortlibpath .. "\\lib",
    links = "onnxruntime",
    sources = "ort.c",
    mode = "debug",
    msvc = {
        flags = "/utf-8",
    },
}