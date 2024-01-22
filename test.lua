local Ort = require "luaort"

local Env = Ort.CreateEnv()
local SessionOptions = Ort.CreateSessionOptions()

--local Session = Env:CreateSession("test.onnx", SessionOptions)

print(Env)
print(SessionOptions)

Env = nil
SessionOptions = nil

collectgarbage()

while true do end