#include "GluedForce.h"
#include "GluedForceProxy.h"
#include "internal/windowsExportGlued.h"
#include "openmm/serialization/SerializationProxy.h"

// Serialization proxies use DllMain / constructor-attribute auto-registration
// so they are available as soon as the serialization library is loaded.
// This is separate from kernel factory registration.

#if defined(WIN32)
    #include <windows.h>
    extern "C" OPENMM_EXPORT_GLUED void registerGluedSerializationProxies();
    BOOL WINAPI DllMain(HANDLE hModule, DWORD ul_reason_for_call, LPVOID lpReserved) {
        if (ul_reason_for_call == DLL_PROCESS_ATTACH)
            registerGluedSerializationProxies();
        return TRUE;
    }
#else
    extern "C" void __attribute__((constructor)) registerGluedSerializationProxies();
#endif

using namespace GluedPlugin;
using namespace OpenMM;

extern "C" OPENMM_EXPORT_GLUED void registerGluedSerializationProxies() {
    SerializationProxy::registerProxy(typeid(GluedForce),
                                      new GluedForceProxy());
}
