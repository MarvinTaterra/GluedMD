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
// Also provides getSharedBiasPtrs() for multiwalker B2 support.
class CudaCalcGluedForceKernel : public CommonCalcGluedForceKernel {
public:
    CudaCalcGluedForceKernel(std::string name,
                                   const Platform& platform,
                                   ComputeContext& cc)
        : CommonCalcGluedForceKernel(name, platform, cc) {}

    // Returns raw device pointers for the specified bias's shared arrays.
    // biasType=BIAS_METAD: returns [grid_ptr]
    // biasType=BIAS_OPES:  returns [centers, sigmas, logweights, numKernels, numAllocated]
    std::vector<long long> getMultiWalkerPtrs(int biasType, int localIdx) override {
        std::vector<long long> ptrs;
        CudaContext& cu = static_cast<CudaContext&>(cc_);
        if (biasType == GluedForce::BIAS_METAD && localIdx < (int)metaDGridBiases_.size()) {
            auto& m = metaDGridBiases_[localIdx];
            CudaArray& arr = cu.unwrap(m.grid.getArray());
            ptrs.push_back((long long)arr.getDevicePointer());
        } else if (biasType == GluedForce::BIAS_OPES && localIdx < (int)opesBiases_.size()) {
            auto& o = opesBiases_[localIdx];
            ptrs.push_back((long long)cu.unwrap(o.kernelCenters.getArray()).getDevicePointer());
            ptrs.push_back((long long)cu.unwrap(o.kernelSigmas.getArray()).getDevicePointer());
            ptrs.push_back((long long)cu.unwrap(o.kernelLogWeights.getArray()).getDevicePointer());
            ptrs.push_back((long long)cu.unwrap(o.numKernelsGPU.getArray()).getDevicePointer());
            ptrs.push_back((long long)cu.unwrap(o.numAllocatedGPU.getArray()).getDevicePointer());
        }
        return ptrs;
    }

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
