#include <math.h>
#include <stdio.h>

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

static const struct luaL_Reg luaort [] = {
    {"CreateEnv", l_createenv},
    {"CreateSessionOptions", l_createsessionoptions},
    {NULL, NULL}
};


// Env

static int l_env_createsession (lua_State *L) {
    luaL_checktype(L, 1, LUA_TUSERDATA);
    luaL_checktype(L, 2, LUA_TSTRING);
    luaL_checktype(L, 3, LUA_TUSERDATA);

    OrtEnv* env = *(OrtEnv**)luaL_checkudata(L, 1, "Ort.Env");
    const char* modelpath = lua_tostring(L, 2);
    OrtSessionOptions* session_options = *(OrtSessionOptions**)luaL_checkudata(L, 3, "Ort.SessionOptions");

    OrtSession* session;
    ORT_LUA_ERROR(L, g_ort->CreateSession(env, modelpath, session_options, &session));
    if (session == NULL) { luaL_error(L, "Failed env creating."); }

    OrtSession** luaptr = (OrtSession**)lua_newuserdata(L, sizeof(OrtSession*));
    *luaptr = session;
    luaL_getmetatable(L, "Ort.Session");    
    lua_setmetatable(L, -2);

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

    luaL_newmetatable(L, "Ort.SessionOptions");
    lua_pushvalue(L, -1);
    lua_setfield(L, -2, "__index");
    luaL_setfuncs(L, sessionoptions_m, 0);

    // Регистрируем функцию в глобальной таблице Lua
    luaL_newlib(L, luaort);

    // Возвращаем 0 для успешной загрузки библиотеки
    return 1;
}