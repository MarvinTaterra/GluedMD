#ifndef OPENCL_GLUED_KERNEL_FACTORY_H_
#define OPENCL_GLUED_KERNEL_FACTORY_H_

#include "openmm/KernelFactory.h"

namespace GluedPlugin {

class OpenCLGluedKernelFactory : public OpenMM::KernelFactory {
public:
    OpenMM::KernelImpl* createKernelImpl(std::string name,
                                          const OpenMM::Platform& platform,
                                          OpenMM::ContextImpl& context) const override;
};

} // namespace GluedPlugin

#endif // OPENCL_GLUED_KERNEL_FACTORY_H_
