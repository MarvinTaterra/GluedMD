#include "CudaGluedKernelFactory.h"
#include "CommonGluedKernels.h"
#include "GluedKernels.h"
#include "internal/windowsExportGlued.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/OpenMMException.h"
#include "openmm/Platform.h"
#include "openmm/cuda/CudaPlatform.h"
#include "openmm/cuda/CudaContext.h"
#include "openmm/cuda/CudaArray.h"

using namespace GluedPlugin;
using namespace OpenMM;

// CUDA subclass: exposes the native CUDA stream and device pointers so that
// the common layer's GPU-native PyTorch path can share OpenMM's CUDA stream
// with torch (via c10::cuda::CUDAStreamGuard) and access raw device pointers
// for zero-copy CUDA tensor creation and D2D gradient copies.
class CudaCalcGluedForceKernel : public CommonCalcGluedForceKernel {
public:
    CudaCalcGluedForceKernel(std::string name,
                                   const Platform& platform,
                                   ComputeContext& cc)
        : CommonCalcGluedForceKernel(name, platform, cc) {}

protected:
    // Returns OpenMM's CUstream cast to void* so the common layer can pass it
    // to c10::cuda::getStreamFromExternal without a CUDA type in the common header.
    void* getNativeCudaStream() const override {
        CUstream s = static_cast<CudaContext&>(cc_).getCurrentStream();
        return reinterpret_cast<void*>(s);
    }

    // Returns the raw device pointer (CUdeviceptr → void*) for a ComputeArray
    // so the common layer can call torch::from_blob / cudaMemcpyAsync without
    // platform-specific types in the common header.
    void* getComputeArrayDevPtr(OpenMM::ComputeArray& arr) const override {
        CudaContext& cu = static_cast<CudaContext&>(cc_);
        CudaArray& cudaArr = cu.unwrap(arr.getArray());
        CUdeviceptr ptr = cudaArr.getDevicePointer();
        return reinterpret_cast<void*>(static_cast<uintptr_t>(ptr));
    }

    // CV_ENERGY device path: the linked inner Context's CudaContext, read from its
    // PlatformData. Mirrors CudaCalcCustomCVForceKernel::getInnerComputeContext.
    ComputeContext& getInnerComputeContext(ContextImpl& innerContext) override {
        return *reinterpret_cast<CudaPlatform::PlatformData*>(
            innerContext.getPlatformData())->contexts[0];
    }
};

// Called by OpenMM's plugin loader. Must be named exactly "registerKernelFactories".
// Platform load order: OpenMMCUDA.dll is loaded before this plugin DLL, so
// dynamic_cast<CudaPlatform*> will succeed at the time this runs.
extern "C" OPENMM_EXPORT_GLUED void registerKernelFactories() {
    for (int i = 0; i < Platform::getNumPlatforms(); i++) {
        Platform& platform = Platform::getPlatform(i);
        if (dynamic_cast<CudaPlatform*>(&platform) != NULL) {
            platform.registerKernelFactory(
                CalcGluedForceKernel::Name(),
                new CudaGluedKernelFactory());
        }
    }
}

KernelImpl* CudaGluedKernelFactory::createKernelImpl(
    std::string name, const Platform& platform, ContextImpl& context) const {
    if (name == CalcGluedForceKernel::Name()) {
        CudaPlatform::PlatformData& data =
            *static_cast<CudaPlatform::PlatformData*>(context.getPlatformData());
        CudaContext& cu = *data.contexts[0];
        return new CudaCalcGluedForceKernel(name, platform, cu);
    }
    throw OpenMMException(
        "CudaGluedKernelFactory: unknown kernel name: " + name);
}
