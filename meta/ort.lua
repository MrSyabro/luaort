---@meta

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

---@return string[]
function Session:GetInputs()
end

---@return string[]
function Session:GetOutputs()
end

---@param index integer
---@return string
---@return integer[] dimensions
function Session:GetInputType(index)
end

---@param index integer
---@return string
---@return integer[] dimensions
function Session:GetOutputType(index)
end

---Run the model in an ::OrtSession
---@param input_names string[]
---@param input_tensors OrtValue[]
---@param output_names string[]
---@return OrtValue[]
function Session:Run(input_names, input_tensors, output_names)
end

---@class OrtSessionOptions
SessionOptions = {}

function SessionOptions:AppendExecutionProvider_DML()
end

---@alias CudnnConvAlgoSearch
---|>"Exhaustive" expensive exhaustive benchmarking using cudnnFindConvolutionForwardAlgorithmEx
---| "Heuristic" lightweight heuristic based search using cudnnGetConvolutionForwardAlgorithm_v7
---| "Default" default algorithm using CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM

---@class CUDAProviderOptions
---@field device_id number
---@field cudnn_conv_algo_search CudnnConvAlgoSearch
---@field gpu_mem_limit number CUDA memory limit (To use all possible memory pass in maximum size_t)
---@field arena_extend_strategy number
---@field do_copy_in_default_stream number
---@field has_user_compute_stream number
---@field tunable_op_enable number
---@field tunable_op_tuning_enable number
---@field tunable_op_max_tuning_duration_ms number

---Append CUDA provider to session options
function SessionOptions:AppendExecutionProvider_CUDA()
end

---@alias OpenVINODeviceType
---| "CPU_FP32"
---| "CPU_FP16"
---| "GPU_FP32"
---| "GPU_FP16"

---@class OpenVINOProviderOptions
---@field device_type OpenVINODeviceType

---Append OpenVINO provider to session options
---@param config OpenVINOProviderOptions?
function SessionOptions:AppendExecutionProvider_OpenVINO(config)
end

---@param provider_name string
---@param provider_options table<string, string|number>
function SessionOptions:AppendExecutionProvider(provider_name, provider_options)
end

---@class OrtValue
Value = {}

---@return boolean
function Value:isTensor()
end

---@return string rawdata
function Value:GetData()
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

---@param inputshape table
---@param datatype TENSOR_DATA_ELEMENT_TYPE?
---@param inputdata string?
---@return OrtValue
function Ort.CreateValue(inputshape, datatype, inputdata)
end

return Ort