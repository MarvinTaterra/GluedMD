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

// Thin OpenCL subclass — adds the CV_ENERGY device path's inner-context accessor;
// all other logic is in the Common layer.
class OpenCLCalcGluedForceKernel : public CommonCalcGluedForceKernel {
public:
    OpenCLCalcGluedForceKernel(std::string name,
                                    const Platform& platform,
                                    ComputeContext& cc)
        : CommonCalcGluedForceKernel(name, platform, cc) {}

protected:
    // CV_ENERGY device path: the linked inner Context's OpenCLContext, read from its
    // PlatformData. Mirrors OpenCLCalcCustomCVForceKernel::getInnerComputeContext.
    ComputeContext& getInnerComputeContext(ContextImpl& innerContext) override {
        return *reinterpret_cast<OpenCLPlatform::PlatformData*>(
            innerContext.getPlatformData())->contexts[0];
    }
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
