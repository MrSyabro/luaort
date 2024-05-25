#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "lua.h"
#include "lauxlib.h"
#include "onnxruntime_c_api.h"

const OrtApi* g_ort = NULL;

#define ORT_LUA_ERROR(lua, expr)                            \
OrtStatus* onnx_status = (expr);                            \
if (onnx_status != NULL) {                                  \
    const char* msg = g_ort->GetErrorMessage(onnx_status);  \
    g_ort->ReleaseStatus(onnx_status);                      \
    luaL_error((L), "[ORT] %s\n", msg);                     \
    return 0;                                               \
}

const static char* lort_tensort_elemennt_data_type [] = {
    "UNDEFINED",
    "FLOAT",
    "UINT8",
    "INT8",
    "UINT16",
    "INT16",
    "INT32",
    "INT64",
    "STRING",
    "BOOL",
    "FLOAT16",
    "DOUBLE",
    "UINT32",
    "UINT64",
    NULL
};

static const char* lort_AllocatorType [] = {
    "Invalid",
    "Device",
    "Arena",
    NULL
};

static const char* lort_MemType [] = {
    "CPUInput",
    "CPUOutput",
    "Default",
    NULL
};

static double *lort_todarray(lua_State *L, int index, size_t *count) {
    *count = lua_rawlen(L, index);
    double* array = calloc(*count, sizeof(double));
    for (int i = 1; i <= *count; i++) {
        lua_rawgeti(L, 2, i);
        array[i-1] = (double)lua_tonumber(L, -1);
        lua_pop(L, 1);
    }

    return array;
}

static int64_t * lort_toiarray(lua_State *L, int index, size_t* count) {
    *count = lua_rawlen(L, index);
    int64_t *array = calloc(*count, sizeof(int64_t));
    for (int i = 1; i <= *count; i++) {
        lua_rawgeti(L, index, i);
        array[i-1] = (int64_t)lua_tointeger(L, -1);
        lua_pop(L, 1);
    }

    return array;
}

static int lort_createenv (lua_State *L) {
    OrtEnv* env;
    ORT_LUA_ERROR(L, g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "OrtLua", &env));
    if (env == NULL) { luaL_error(L, "Failed env creating."); }

    OrtEnv** luaptr = (OrtEnv**)lua_newuserdata(L, sizeof(OrtEnv*));
    *luaptr = env;

    luaL_getmetatable(L, "Ort.Env");    
    lua_setmetatable(L, -2);

    return 1;
}

static int lort_createsessionoptions (lua_State *L) {
    OrtSessionOptions* session_options;
    ORT_LUA_ERROR(L, g_ort->CreateSessionOptions(&session_options));
    if (session_options == NULL) { luaL_error(L, "Failed options creating."); }

    OrtSessionOptions** luaptr = (OrtSessionOptions**)lua_newuserdata(L, sizeof(OrtSessionOptions*));
    *luaptr = session_options;

    luaL_getmetatable(L, "Ort.SessionOptions");    
    lua_setmetatable(L, -2);

    return 1;
}

static int lort_createvalue (lua_State *L) {
    luaL_checktype(L, 1, LUA_TTABLE);
    int element_data_type = luaL_checkoption(L, 2, "FLOAT", lort_tensort_elemennt_data_type);

    if (!lua_isnoneornil(L, 3))
        if (element_data_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING) {
            luaL_checktype(L, 3, LUA_TSTRING);
        } else {
            luaL_checktype(L, 3, LUA_TSTRING);
        }

    size_t input_shape_len;
    int64_t *input_shape = lort_toiarray(L, 1, &input_shape_len);

    OrtAllocator* default_allocator = NULL;
    g_ort->GetAllocatorWithDefaultOptions(&default_allocator);

    OrtValue* input_tensor = NULL;
    ORT_LUA_ERROR(L, g_ort->CreateTensorAsOrtValue(default_allocator, input_shape, input_shape_len, element_data_type, &input_tensor));
    luaL_argcheck(L, input_tensor != NULL, 1, "Failed creting tensor");

    if (lua_istable(L, 3) && element_data_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING) {
        int strings_count = (int)luaL_len(L, 3);

        const char** s = calloc(sizeof(char* const*), strings_count);

        for (int i = 1; i <= strings_count; i++) {
            lua_geti(L, 3, i);
            s[i] = lua_tostring(L, -1);
            lua_pop(L, 1);
        }

        ORT_LUA_ERROR(L, g_ort->FillStringTensor(input_tensor, s, strings_count));

        free(s);
    } else if (lua_isstring(L, 3)) {
        size_t modelort_input_ele_count;
        const char *modelort_input = luaL_checklstring(L, 3, &modelort_input_ele_count);

        void *modelort_inputc = NULL;
        ORT_LUA_ERROR(L, g_ort->GetTensorMutableData(input_tensor, (void**)&modelort_inputc));

        memcpy(modelort_inputc, modelort_input, modelort_input_ele_count);
    }

    OrtValue** luaptr = (OrtValue**)lua_newuserdata(L, sizeof(OrtValue*));
    *luaptr = input_tensor;
    luaL_getmetatable(L, "Ort.Value");    
    lua_setmetatable(L, -2);

    free(input_shape);

    return 1;
}

static const struct luaL_Reg luaort [] = {
    {"CreateEnv", lort_createenv},
    {"CreateSessionOptions", lort_createsessionoptions},
    {"CreateValue", lort_createvalue},
    {NULL, NULL}
};


// Env

static int lort_env_createsession (lua_State *L) {
    luaL_checktype(L, 1, LUA_TUSERDATA);
    luaL_checktype(L, 2, LUA_TSTRING);
    luaL_checktype(L, 3, LUA_TUSERDATA);

    OrtEnv* env = *(OrtEnv**)luaL_checkudata(L, 1, "Ort.Env");
    size_t pathlen;
    const char* modelpath = lua_tolstring(L, 2, &pathlen);
    wchar_t* wmodelpath = (wchar_t*)calloc(pathlen+1, sizeof(wchar_t));
    mbstowcs(wmodelpath, modelpath, pathlen);
    OrtSessionOptions* session_options = *(OrtSessionOptions**)luaL_checkudata(L, 3, "Ort.SessionOptions");

    OrtSession* session;
    ORT_LUA_ERROR(L, g_ort->CreateSession(env, wmodelpath, session_options, &session));
    if (session == NULL) { luaL_error(L, "Failed env creating."); }

    OrtSession** luaptr = (OrtSession**)lua_newuserdata(L, sizeof(OrtSession*));
    *luaptr = session;
    luaL_getmetatable(L, "Ort.Session");    
    lua_setmetatable(L, -2);

    free(wmodelpath);

    return 1;
}

static int lort_env_release (lua_State *L) {
    luaL_checktype(L, 1, LUA_TUSERDATA);
    OrtEnv* env = *(OrtEnv**)luaL_checkudata(L, 1, "Ort.Env");

    g_ort->ReleaseEnv(env);

    return 0;
}

static const struct luaL_Reg env_m [] = {
    {"CreateSession", lort_env_createsession},
    {"__gc", lort_env_release},
    {NULL, NULL}
};


// SessionOptions

static int lort_sessionoptions_AppendExecutionProvider_DML (lua_State *L) {
    #ifdef USE_DML
        luaL_checktype(L, 1, LUA_TUSERDATA);
        OrtSessionOptions* session_options = *(OrtSessionOptions**)luaL_checkudata(L, 1, "Ort.SessionOptions");
        ORT_LUA_ERROR(L, g_ort->OrtSessionOptionsAppendExecutionProvider_DML(session_options, 0));
    #else
        luaL_error(L, "DirectML is not enabled in this build.");
    #endif
    
    return 0;
}

static int lort_sessionoptions_AppendExecutionProvider_OpenVINO (lua_State *L) {
    luaL_checktype(L, 1, LUA_TUSERDATA);
    OrtSessionOptions* session_options = *(OrtSessionOptions**)luaL_checkudata(L, 1, "Ort.SessionOptions");

    OrtOpenVINOProviderOptions provider_options;
    memset(&provider_options, 0, sizeof(provider_options));

    if (lua_istable(L, 2)) {
        lua_getfield(L, 2, "device_type");
        provider_options.device_type = lua_tostring(L, -1);
    }

    ORT_LUA_ERROR(L, g_ort->SessionOptionsAppendExecutionProvider_OpenVINO(session_options, &provider_options));

    return 0;
}

static int lort_sessionoptions_AppendExecutionProvider_CUDA (lua_State *L) {
    luaL_checktype(L, 1, LUA_TUSERDATA);
    OrtSessionOptions* session_options = *(OrtSessionOptions**)luaL_checkudata(L, 1, "Ort.SessionOptions");

    OrtCUDAProviderOptions o;
    // Here we use memset to initialize every field of the above data struct to zero.
    memset(&o, 0, sizeof(o));

    o.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
    o.gpu_mem_limit = SIZE_MAX;
    ORT_LUA_ERROR(L, g_ort->SessionOptionsAppendExecutionProvider_CUDA(session_options, &o));
    
    return 0;
}

static int lort_sessionoptions_AppendExecutionProvider (lua_State *L) {
    luaL_checktype(L, 1, LUA_TUSERDATA);
    OrtSessionOptions* session_options = *(OrtSessionOptions**)luaL_checkudata(L, 1, "Ort.SessionOptions");
    const char* provider_name = luaL_checkstring(L, 2);

    //ORT_LUA_ERROR(L, g_ort->SessionOptionsAppendExecutionProvider(session_options));

    return 0;
}

static int lort_sessionoptions_release (lua_State *L) {
    luaL_checktype(L, 1, LUA_TUSERDATA);
    OrtSessionOptions* session_options = *(OrtSessionOptions**)luaL_checkudata(L, 1, "Ort.SessionOptions");

    g_ort->ReleaseSessionOptions(session_options);

    return 0;
}

static const struct luaL_Reg sessionoptions_m [] = {
    {"AppendExecutionProvider_DML", lort_sessionoptions_AppendExecutionProvider_DML},
    {"AppendExecutionProvider_CUDA", lort_sessionoptions_AppendExecutionProvider_CUDA},
    {"AppendExecutionProvider_OpenVINO", lort_sessionoptions_AppendExecutionProvider_OpenVINO},
    {"__gc", lort_sessionoptions_release},
    {NULL, NULL}
};


// Session

static int lort_session_GetInputCount(lua_State *L) {
    luaL_checktype(L, 1, LUA_TUSERDATA);
    OrtSession* session = *(OrtSession**)luaL_checkudata(L, 1, "Ort.Session");
    
    size_t count;
    g_ort->SessionGetInputCount(session, &count);

    lua_pushnumber(L, (int)count);
    return 1;
}

static int lort_session_GetOutputCount(lua_State *L) {
    luaL_checktype(L, 1, LUA_TUSERDATA);
    OrtSession* session = *(OrtSession**)luaL_checkudata(L, 1, "Ort.Session");
    
    size_t count;
    g_ort->SessionGetOutputCount(session, &count);

    lua_pushnumber(L, (int)count);
    return 1;
}

static int lort_session_release(lua_State *L) {
    luaL_checktype(L, 1, LUA_TUSERDATA);
    OrtSession* session = *(OrtSession**)luaL_checkudata(L, 1, "Ort.Session");

    g_ort->ReleaseSession(session);

    return 0;
}

static char** sarray (lua_State *L, int index, size_t *count) {
    *count = luaL_len(L, index);
    char** array = calloc(*count + 1, sizeof(char*));

    for (int i = 1; i <= *count; i++) {
        lua_rawgeti(L, index, i);
        array[i - 1] = (char *)lua_tostring(L, -1);
        lua_pop(L, 1);
    }
    array[*count + 1] = NULL;

    return array;
}

static int lort_session_run (lua_State *L) {
    luaL_checktype(L, 1, LUA_TUSERDATA);
    luaL_checktype(L, 2, LUA_TTABLE);
    luaL_checktype(L, 3, LUA_TUSERDATA);
    luaL_checktype(L, 4, LUA_TTABLE);

    OrtSession* session = *(OrtSession**)luaL_checkudata(L, 1, "Ort.Session");
    size_t input_count;
    char** input_names = sarray(L, 2, &input_count);
    OrtValue* input_tensor = *(OrtValue**)luaL_checkudata(L, 3, "Ort.Value");
    size_t output_count;
    char** output_names = sarray(L, 4, &output_count);

    OrtValue* output_tensor = NULL;
    ORT_LUA_ERROR(L, g_ort->Run(session, NULL, (const char* const*)input_names, 
                                    (const OrtValue* const*)&input_tensor,
                                    input_count, (const char* const*)output_names,
                                    output_count, &output_tensor));
    luaL_argcheck(L, output_tensor != NULL, 1, "Failed runing");

    OrtValue** luaptr = (OrtValue**)lua_newuserdata(L, sizeof(OrtValue*));
    *luaptr = output_tensor;
    luaL_getmetatable(L, "Ort.Value");    
    lua_setmetatable(L, -2);

    return 1;
}

static const struct luaL_Reg session_m [] = {
    {"GetInputCount", lort_session_GetInputCount},
    {"GetOutputCount", lort_session_GetOutputCount},
    {"Run", lort_session_run},
    {"__gc", lort_session_release},
    {NULL, NULL}
};

// OrtValue

static int lort_value_istensor (lua_State *L) {
    luaL_checktype(L, 1, LUA_TUSERDATA);
    OrtValue* value = *(OrtValue**)luaL_checkudata(L, 1, "Ort.Value");

    int is_tensor;
    g_ort->IsTensor(value, &is_tensor);

    lua_pushboolean(L, is_tensor);

    return 1;
}

static size_t getsize(ONNXTensorElementDataType datatype) {
    switch (datatype)
    {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: return (sizeof(float)); break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8: return (sizeof(uint8_t)); break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8: return (sizeof(int8_t)); break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16: return (sizeof(uint16_t)); break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16: return (sizeof(int16_t)); break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: return (sizeof(int32_t)); break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: return (sizeof(int64_t)); break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: return (sizeof(int16_t)); break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE: return (sizeof(double)); break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32: return (sizeof(uint32_t)); break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64: return (sizeof(uint64_t)); break;
    default:
        break;
    }
    return 0;
}

static int lort_value_getdata (lua_State *L) { // TODO сделать лучше размер данных
    luaL_checktype(L, 1, LUA_TUSERDATA);
    OrtValue* value = *(OrtValue**)luaL_checkudata(L, 1, "Ort.Value");

    char* output_tensor_data = NULL;
    ORT_LUA_ERROR(L, g_ort->GetTensorMutableData(value, (void**)&output_tensor_data));

    OrtTensorTypeAndShapeInfo* typeandshape = NULL;
    g_ort->GetTensorTypeAndShape(value, &typeandshape);

    size_t count;
    g_ort->GetTensorShapeElementCount(typeandshape, &count);
    ONNXTensorElementDataType datatype;
    g_ort->GetTensorElementType(typeandshape, &datatype);

    size_t sizeofel = getsize(datatype);

    g_ort->ReleaseTensorTypeAndShapeInfo(typeandshape);

    lua_pushlstring(L, output_tensor_data, count * sizeofel);

    return 1;
}

static int lort_value_release (lua_State *L) {
    luaL_checktype(L, 1, LUA_TUSERDATA);
    OrtValue* value = *(OrtValue**)luaL_checkudata(L, 1, "Ort.Value");

    g_ort->ReleaseValue(value);

    return 0;
}

static const struct luaL_Reg value_m [] = {
    {"isTensor", lort_value_istensor},
    {"GetData", lort_value_getdata},
    {"__gc", lort_value_release},
    {NULL, NULL}
};

/* Оставшиеся классы
ORT_RUNTIME_CLASS(IoBinding);
ORT_RUNTIME_CLASS(RunOptions);
ORT_RUNTIME_CLASS(TypeInfo);
ORT_RUNTIME_CLASS(TensorTypeAndShapeInfo);
ORT_RUNTIME_CLASS(CustomOpDomain);
ORT_RUNTIME_CLASS(MapTypeInfo);
ORT_RUNTIME_CLASS(SequenceTypeInfo);
ORT_RUNTIME_CLASS(ModelMetadata);
ORT_RUNTIME_CLASS(ThreadPoolParams);
ORT_RUNTIME_CLASS(ThreadingOptions);
ORT_RUNTIME_CLASS(ArenaCfg);
ORT_RUNTIME_CLASS(PrepackedWeightsContainer);
ORT_RUNTIME_CLASS(TensorRTProviderOptionsV2);
*/

#if defined(_WIN32) || defined(_WIN64)
__declspec(dllexport)
#endif
int luaopen_luaort(lua_State *L) {
    g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (!g_ort) {
        luaL_error(L, "Failed to init ONNX Runtime engine.\n");
        return 0;
    }

    luaL_newmetatable(L, "Ort.Env");
    lua_pushvalue(L, -1);
    lua_setfield(L, -2, "__index");
    luaL_setfuncs(L, env_m, 0);

    luaL_newmetatable(L, "Ort.Session");
    lua_pushvalue(L, -1);
    lua_setfield(L, -2, "__index");
    luaL_setfuncs(L, session_m, 0);

    luaL_newmetatable(L, "Ort.SessionOptions");
    lua_pushvalue(L, -1);
    lua_setfield(L, -2, "__index");
    luaL_setfuncs(L, sessionoptions_m, 0);

    luaL_newmetatable(L, "Ort.Value");
    lua_pushvalue(L, -1);
    lua_setfield(L, -2, "__index");
    luaL_setfuncs(L, value_m, 0);

    luaL_newlib(L, luaort);

    return 1;
}