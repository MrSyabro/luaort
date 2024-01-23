local Ort = require "luaort"

local Env = Ort.CreateEnv()
print(Env)

local SessionOptions = Ort.CreateSessionOptions()
print(SessionOptions)

local Session = Env:CreateSession("candy.onnx", SessionOptions)
print(Session)

local mi = Ort.CreateMemoryInfo()
print(mi)

local inputvalue = mi:CreateTensor({1, 2, 3}, {1,3,720,720})
print(inputvalue)

print("is tensor:", inputvalue:isTensor())

inputvalue = nil
collectgarbage()

while true do end