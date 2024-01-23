--local ortlibpath = "D:\\onnxruntime-win-x64-1.10.0"
local ortlibpath = "D:\\onnxruntime-training-win-x64-1.16.3"

local lm = require "luamake"
lm:lua_dll "luaort" {
    includes = ortlibpath .. "\\include",
    linkdirs = ortlibpath .. "\\lib",
    links = "onnxruntime",
    sources = "ort.c",
    msvc = {
        flags = "/utf-8",
    },
}