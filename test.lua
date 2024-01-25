local Ort = require "luaort" --[[@as Ort]]
local png = require "pngutils" --[[@as PNGUtils]]

local w, h, d = png.read("in.png")
print("Png size", w, h)
print("string size", #d)
d = png.hwc2chw(d)

local Env = Ort.CreateEnv()

local SessionOptions = Ort.CreateSessionOptions()

local Session = Env:CreateSession("la_muse.onnx", SessionOptions)
print("Session InputCount", Session:GetInputCount())
print("Session OutputCount", Session:GetOutputCount())

local mi = Ort.CreateCPUMemoryInfo("Arena", "Default")

local inputvalue = mi:CreateTensor(d, {1,3,720,720}, "FLOAT")
print("in is tensor:", inputvalue:isTensor())

local outputvalue = Session:Run({"inputImage"}, inputvalue, {"outputImage"})
print("out isTensor:", outputvalue:isTensor())

local d = outputvalue:GetData()
print("size of outdata", #d)

d = png.chw2hwc(d)
print("data2 size", #d)

png.write(d, w, h, "out.png")

os.execute('start out.png')