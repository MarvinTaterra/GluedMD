#include "OpenCLGluedKernelFactory.h"
#include "CommonGluedKernels.h"
#include "GluedKernels.h"
#include "internal/windowsExportGlued.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/OpenMMException.h"
#include "openmm/Platform.h"
#include "openmm/opencl/OpenCLPlatform.h"
#include "openmm/opencl/OpenCLContext.h"

using namespace GluedPlugin;
using namespace OpenMM;

// Thin OpenCL subclass — type-distinguished only; all logic is in Common layer.
class OpenCLCalcGluedForceKernel : public CommonCalcGluedForceKernel {
public:
    OpenCLCalcGluedForceKernel(std::string name,
                                    const Platform& platform,
                                    ComputeContext& cc)
        : CommonCalcGluedForceKernel(name, platform, cc) {}
};

extern "C" OPENMM_EXPORT_GLUED void registerKernelFactories() {
    for (int i = 0; i < Platform::getNumPlatforms(); i++) {
        Platform& platform = Platform::getPlatform(i);
        if (dynamic_cast<OpenCLPlatform*>(&platform) != NULL) {
            platform.registerKernelFactory(
                CalcGluedForceKernel::Name(),
                new OpenCLGluedKernelFactory());
        }
    }
}

KernelImpl* OpenCLGluedKernelFactory::createKernelImpl(
    std::string name, const Platform& platform, ContextImpl& context) const {
    if (name == CalcGluedForceKernel::Name()) {
        OpenCLPlatform::PlatformData& data =
            *static_cast<OpenCLPlatform::PlatformData*>(context.getPlatformData());
        OpenCLContext& cl = *data.contexts[0];
        return new OpenCLCalcGluedForceKernel(name, platform, cl);
    }
    throw OpenMMException(
        "OpenCLGluedKernelFactory: unknown kernel name: " + name);
}
