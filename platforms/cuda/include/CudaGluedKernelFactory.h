#ifndef CUDA_GLUED_KERNEL_FACTORY_H_
#define CUDA_GLUED_KERNEL_FACTORY_H_

#include "openmm/KernelFactory.h"

namespace GluedPlugin {

class CudaGluedKernelFactory : public OpenMM::KernelFactory {
public:
    OpenMM::KernelImpl* createKernelImpl(std::string name,
                                          const OpenMM::Platform& platform,
                                          OpenMM::ContextImpl& context) const override;
};

} // namespace GluedPlugin

#endif // CUDA_GLUED_KERNEL_FACTORY_H_
