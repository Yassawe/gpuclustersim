#pragma once

#ifdef _WIN32
#define ENABLE_EXPORT __declspec(dllexport)
#else
#define ENABLE_EXPORT __attribute__((visibility("default")))
#endif