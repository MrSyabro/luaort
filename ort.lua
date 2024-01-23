---@meta

---@alias OrtAllocatorType
---| "Invalid"
---|>"Device"
---| "Arena"

---Memory types for allocated memory, execution provider specific types should be extended in each provider.
---@alias OrtMemType
---| "CPUInput"              --- Any CPU memory used by non-CPU execution provider
---| "CPUOutput"             --- CPU accessible memory outputted by non-CPU execution provider, i.e. CUDA_PINNED
---|>"Default"               --- The default allocator for execution provider

---@alias TENSOR_DATA_ELEMENT_TYPE
---| "UNDEFINED"
---|>"FLOAT"
---| "UINT8"
---| "INT8"
---| "UINT16"
---| "INT16"
---| "INT32"
---| "INT64"
---| "STRING"
---| "BOOL"
---| "FLOAT16"
---| "DOUBLE"
---| "UINT32"
---| "UINT64"
---| "COMPLEX64"
---| "COMPLEX128"
---| "BFLOAT16"
---|
---| "FLOAT8E4M3FN"
---| "FLOAT8E4M3FNUZ"
---| "FLOAT8E5M2"
---| "FLOAT8E5M2FNUZ"

---@class OrtEnv
Env = {}

---Create an OrtSession from a model file
---@param modelpath string
---@param sessionoptions OrtSessionOptions
---@return OrtSession
function Env:CreateSession(modelpath, sessionoptions)
end

---@class OrtSession
Session = {}

---@class OrtSessionOptions
SessionOptions = {}

function SessionOptions:AppendExecutionProvider_DML()
end

---Append CUDA provider to session options
function SessionOptions:AppendExecutionProvider_CUDA()
end


---@class OrtMemoryInfo
MemoryInfo = {}

---@param inputdata table
---@param inputshape table
---@param datatype TENSOR_DATA_ELEMENT_TYPE?
---@return OrtValue
function MemoryInfo:CreateTensor(inputdata, inputshape, datatype)
end

---@class OrtValue
Value = {}

---@return boolean
function Value:isTensor()
end

---@class Ort
Ort = {}

---Создает OrtEnv
---@return OrtEnv
function Ort.CreateEnv()
end

---Create an ::OrtSessionOptions object
---@return OrtSessionOptions
function Ort.CreateSessionOptions()
end

---Create an ::OrtMemoryInfo for CPU memory with OrtArenaAllocator and OrtMemTypeDefault
---@param allocator OrtAllocatorType?
---@param memorytype OrtMemType?
---@return OrtMemoryInfo
function Ort.CreateCPUMemoryInfo(allocator, memorytype)
end

return Ort