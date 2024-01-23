#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "lua.h"
#include "lauxlib.h"
#include "onnxruntime_c_api.h"

const OrtApi* g_ort = NULL;

#define ORT_LUA_ERROR(lua, expr)                            \
OrtStatus* onnx_status = (expr);                            \
if (onnx_status != NULL) {                                  \
    const char* msg = g_ort->GetErrorMessage(onnx_status);  \
    g_ort->ReleaseStatus(onnx_status);                      \
    luaL_error((L), "ORT: %s\n", msg);                      \
    return 0;                                               \
}

static int l_createenv (lua_State *L) {
    OrtEnv* env;
    ORT_LUA_ERROR(L, g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "OrtLua", &env));
    if (env == NULL) { luaL_error(L, "Failed env creating."); }

    OrtEnv** luaptr = (OrtEnv**)lua_newuserdata(L, sizeof(OrtEnv*));
    *luaptr = env;

    luaL_getmetatable(L, "Ort.Env");    
    lua_setmetatable(L, -2);

    return 1;
}

static int l_createsessionoptions (lua_State *L) {
    OrtSessionOptions* session_options;
    ORT_LUA_ERROR(L, g_ort->CreateSessionOptions(&session_options));
    if (session_options == NULL) { luaL_error(L, "Failed options creating."); }

    OrtSessionOptions** luaptr = (OrtSessionOptions**)lua_newuserdata(L, sizeof(OrtSessionOptions*));
    *luaptr = session_options;

    luaL_getmetatable(L, "Ort.SessionOptions");    
    lua_setmetatable(L, -2);

    return 1;
}

static int l_creatememoryinfo (lua_State *L) {
    OrtMemoryInfo* memory_info;
    ORT_LUA_ERROR(L, g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info));

    OrtMemoryInfo** luaptr = (OrtMemoryInfo**)lua_newuserdata(L, sizeof(OrtMemoryInfo*));
    *luaptr = memory_info;

    luaL_getmetatable(L, "Ort.MemoryInfo");    
    lua_setmetatable(L, -2);

    return 1;
}

static const struct luaL_Reg luaort [] = {
    {"CreateEnv", l_createenv},
    {"CreateSessionOptions", l_createsessionoptions},
    {"CreateMemoryInfo", l_creatememoryinfo},
    {NULL, NULL}
};


// Env

static int l_env_createsession (lua_State *L) {
    luaL_checktype(L, 1, LUA_TUSERDATA);
    luaL_checktype(L, 2, LUA_TSTRING);
    luaL_checktype(L, 3, LUA_TUSERDATA);

    OrtEnv* env = *(OrtEnv**)luaL_checkudata(L, 1, "Ort.Env");
    size_t pathlen;
    const char* modelpath = lua_tolstring(L, 2, &pathlen);
    const wchar_t* wmodelpath = calloc(pathlen+1, sizeof(wchar_t));
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

static int l_env_release (lua_State *L) {
    luaL_checktype(L, 1, LUA_TUSERDATA);
    OrtEnv* env = *(OrtEnv**)luaL_checkudata(L, 1, "Ort.Env");

    g_ort->ReleaseEnv(env);

    return 0;
}

static const struct luaL_Reg env_m [] = {
    {"CreateSession", l_env_createsession},
    {"__gc", l_env_release},
    {NULL, NULL}
};


// SessionOptions

static int l_sessionoptions_AppendExecutionProvider_DML (lua_State *L) {
    #ifdef USE_DML
        luaL_checktype(L, 1, LUA_TUSERDATA);
        OrtSessionOptions* session_options = *(OrtSessionOptions**)luaL_checkudata(L, 1, "Ort.SessionOptions");
        ORT_LUA_ERROR(L, g_ort->OrtSessionOptionsAppendExecutionProvider_DML(session_options, 0));
    #else
        luaL_error(L, "DirectML is not enabled in this build.");
    #endif
    
    return 0;
}

static int l_sessionoptions_AppendExecutionProvider_CUDA (lua_State *L) {
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

static int l_sessionoptions_release (lua_State *L) {
    luaL_checktype(L, 1, LUA_TUSERDATA);
    OrtSessionOptions* session_options = *(OrtSessionOptions**)luaL_checkudata(L, 1, "Ort.SessionOptions");

    g_ort->ReleaseSessionOptions(session_options);

    return 0;
}

static const struct luaL_Reg sessionoptions_m [] = {
    {"AppendExecutionProvider_DML", l_sessionoptions_AppendExecutionProvider_DML},
    {"AppendExecutionProvider_CUDA", l_sessionoptions_AppendExecutionProvider_CUDA},
    {"__gc", l_sessionoptions_release},
    {NULL, NULL}
};


// Session

static int l_session_GetInputCount(lua_State *L) {
    luaL_checktype(L, 1, LUA_TUSERDATA);
    OrtSession* session = *(OrtSession**)luaL_checkudata(L, 1, "Ort.Session");
    
    size_t count;
    g_ort->SessionGetInputCount(session, &count);

    lua_pushnumber(L, (int)count);
    return 1;
}

static int l_session_GetOutputCount(lua_State *L) {
    luaL_checktype(L, 1, LUA_TUSERDATA);
    OrtSession* session = *(OrtSession**)luaL_checkudata(L, 1, "Ort.Session");
    
    size_t count;
    g_ort->SessionGetOutputCount(session, &count);

    lua_pushnumber(L, (int)count);
    return 1;
}

static int l_session_release(lua_State *L) {
    luaL_checktype(L, 1, LUA_TUSERDATA);
    OrtSession* session = *(OrtSession**)luaL_checkudata(L, 1, "Ort.Session");

    g_ort->ReleaseSession(session);

    return 0;
}

static const struct luaL_Reg session_m [] = {
    {"GetInputCount", l_session_GetInputCount},
    {"GetOutputCount", l_session_GetOutputCount},
    {"__gc", l_session_release},
    {NULL, NULL}
};

static double *darray(lua_State *L, int index, size_t *count) {
    *count = lua_rawlen(L, index);
    double* array = calloc(*count, sizeof(double));
    for (int i = 1; i <= *count; i++) {
        lua_rawgeti(L, 2, i);
        array[i-1] = (double)lua_tonumber(L, -1);
        lua_pop(L, 1);
    }

    return array;
}

static int64_t * iarray(lua_State *L, int index, size_t* count) {
    *count = lua_rawlen(L, index);
    int64_t *array = calloc(*count, sizeof(int64_t));
    for (int i = 1; i <= *count; i++) {
        lua_rawgeti(L, 2, i);
        array[i-1] = (int64_t)lua_tointeger(L, -1);
        lua_pop(L, 1);
    }

    return array;
}

// MemoryInfo

static int l_memoryinfo_CreateTensor (lua_State *L) {
    luaL_checktype(L, 1, LUA_TUSERDATA);
    luaL_checktype(L, 2, LUA_TTABLE);
    luaL_checktype(L, 3, LUA_TTABLE);

    OrtMemoryInfo* memory_info = *(OrtMemoryInfo**)luaL_checkudata(L, 1, "Ort.MemoryInfo");

    size_t model_input_ele_count;
    double *model_input = darray(L, 2, &model_input_ele_count);
    const size_t model_input_len = model_input_ele_count * sizeof(double);

    size_t input_shape_len;
    int64_t *input_shape = iarray(L, 3, &input_shape_len);   //const int64_t input_shape[] = {1, 3, 720, 720};

    OrtValue* input_tensor = NULL;
    ORT_LUA_ERROR(L, g_ort->CreateTensorWithDataAsOrtValue(memory_info, model_input, model_input_len, input_shape,
                                                            input_shape_len, ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,
                                                            &input_tensor));
    luaL_argcheck(L, input_tensor != NULL, 1, "Failed creting tensor");

    OrtValue** luaptr = (OrtValue**)lua_newuserdata(L, sizeof(OrtValue*));
    *luaptr = input_tensor;
    luaL_getmetatable(L, "Ort.Value");    
    lua_setmetatable(L, -2);

    free(input_shape);

    return 1;
}

static int l_memoryinfo_release (lua_State *L) {
    luaL_checktype(L, 1, LUA_TUSERDATA);
    OrtMemoryInfo* memory_info = *(OrtMemoryInfo**)luaL_checkudata(L, 1, "Ort.MemoryInfo");

    g_ort->ReleaseMemoryInfo(memory_info);

    return 0;
}

static const struct luaL_Reg memoryinfo_m [] = {
    {"CreateTensor", l_memoryinfo_CreateTensor},
    {"__gc", l_memoryinfo_release},
    {NULL, NULL}
};


// OrtValue

static int l_istensor (lua_State *L) {
    luaL_checktype(L, 1, LUA_TUSERDATA);
    OrtValue* value = *(OrtValue**)luaL_checkudata(L, 1, "Ort.Value");

    int is_tensor;
    g_ort->IsTensor(value, &is_tensor);

    lua_pushboolean(L, is_tensor);

    return 1;
}

static int l_value_release (lua_State *L) {
    luaL_checktype(L, 1, LUA_TUSERDATA);
    OrtValue* value = *(OrtValue**)luaL_checkudata(L, 1, "Ort.Value");

    void *data;
    g_ort->GetTensorMutableData(value, &data); //TODO а так можно?
    free(data);

    g_ort->ReleaseValue(value);

    return 0;
}

static const struct luaL_Reg value_m [] = {
    {"isTensor", l_istensor},
    {"__gc", l_value_release},
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

    luaL_newmetatable(L, "Ort.MemoryInfo");
    lua_pushvalue(L, -1);
    lua_setfield(L, -2, "__index");
    luaL_setfuncs(L, memoryinfo_m, 0);

    luaL_newmetatable(L, "Ort.Value");
    lua_pushvalue(L, -1);
    lua_setfield(L, -2, "__index");
    luaL_setfuncs(L, value_m, 0);

    // Регистрируем функцию в глобальной таблице Lua
    luaL_newlib(L, luaort);

    // Возвращаем 0 для успешной загрузки библиотеки
    return 1;
}