local Ort = require "luaort" --[[@as Ort]]

local Env = Ort.CreateEnv()

local SessionOptions = Ort.CreateSessionOptions()

local Session = Env:CreateSession("candy.onnx", SessionOptions)
print("Session InputCount", Session:GetInputCount())
print("Session OutputCount", Session:GetOutputCount())

local mi = Ort.CreateCPUMemoryInfo("Arena", "Default")

local inputvalue = mi:CreateTensor("testdata", {1,3,720,720}, "FLOAT")
print("in is tensor:", inputvalue:isTensor())

local outputvalue = Session:Run({"inputImage"}, inputvalue, {"outputImage"})
print("out isTensor:", outputvalue:isTensor())

