#ifndef OPENMM_WINDOWSEXPORTGLUED_H_
#define OPENMM_WINDOWSEXPORTGLUED_H_

#ifdef _MSC_VER
    #pragma warning(disable:4996)
    #pragma warning(disable:4251)
    #if defined(GLUED_BUILDING_SHARED_LIBRARY)
        #define OPENMM_EXPORT_GLUED __declspec(dllexport)
    #elif defined(GLUED_BUILDING_STATIC_LIBRARY) || defined(GLUED_USE_STATIC_LIBRARIES)
        #define OPENMM_EXPORT_GLUED
    #else
        #define OPENMM_EXPORT_GLUED __declspec(dllimport)
    #endif
#else
    #define OPENMM_EXPORT_GLUED
#endif

#endif // OPENMM_WINDOWSEXPORTGLUED_H_
