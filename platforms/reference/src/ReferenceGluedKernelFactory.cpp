#include "ReferenceGluedKernelFactory.h"
#include "ReferenceGluedKernels.h"
#include "GluedKernels.h"
#include "internal/windowsExportGlued.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/OpenMMException.h"
#include "openmm/Platform.h"
#include "openmm/reference/ReferencePlatform.h"

using namespace GluedPlugin;
using namespace OpenMM;

// Called by OpenMM's plugin loader (Platform::loadPluginsFromDirectory /
// Platform::loadPluginLibrary) — NOT via DllMain or constructor attribute.
// See: openmm/olla/src/Platform.cpp, initializePlugins().
extern "C" OPENMM_EXPORT_GLUED void registerKernelFactories() {
    for (int i = 0; i < Platform::getNumPlatforms(); i++) {
        Platform& platform = Platform::getPlatform(i);
        if (dynamic_cast<ReferencePlatform*>(&platform) != NULL) {
            platform.registerKernelFactory(
                CalcGluedForceKernel::Name(),
                new ReferenceGluedKernelFactory());
        }
    }
}

KernelImpl* ReferenceGluedKernelFactory::createKernelImpl(
    std::string name, const Platform& platform, ContextImpl& context) const {
    if (name == CalcGluedForceKernel::Name())
        return new ReferenceCalcGluedForceKernel(name, platform);
    throw OpenMMException(
        "ReferenceGluedKernelFactory: unknown kernel name: " + name);
}
