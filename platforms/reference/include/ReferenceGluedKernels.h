#ifndef REFERENCE_GLUED_KERNELS_H_
#define REFERENCE_GLUED_KERNELS_H_

#include "GluedKernels.h"
#include "openmm/Platform.h"
#include "openmm/System.h"

namespace GluedPlugin {

class ReferenceCalcGluedForceKernel : public CalcGluedForceKernel {
public:
    ReferenceCalcGluedForceKernel(std::string name,
                                       const OpenMM::Platform& platform)
        : CalcGluedForceKernel(name, platform) {}

    void initialize(const OpenMM::System& system,
                    const GluedForce& force) override;

    double execute(OpenMM::ContextImpl& context,
                   bool includeForces, bool includeEnergy) override;

    void updateState(OpenMM::ContextImpl& context, int step) override;

    void getCurrentCVs(OpenMM::ContextImpl& context,
                       std::vector<double>& values) override;

    std::vector<double> downloadCVValues() override;

    std::vector<char> getBiasStateBytes() override;
    void setBiasStateBytes(const std::vector<char>& bytes) override;

    std::vector<double> getOPESMetrics(int biasIndex) override;

private:
    int numParticles_ = 0;
};

} // namespace GluedPlugin

#endif // REFERENCE_GLUED_KERNELS_H_
