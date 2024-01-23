local Ort = require "luaort" --[[@as Ort]]

local Env = Ort.CreateEnv()
print(Env)

local SessionOptions = Ort.CreateSessionOptions()
print(SessionOptions)

local Session = Env:CreateSession("candy.onnx", SessionOptions)
print(Session)

local mi = Ort.CreateCPUMemoryInfo("Arena", "Default")
print(mi)

local inputvalue = mi:CreateTensor({1, 2, 3}, {1,3,720,720}, "FLOAT")
print(inputvalue)

print("is tensor:", inputvalue:isTensor())