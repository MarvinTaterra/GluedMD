#ifndef OPENMM_GLUEDFORCEIMPL_H_
#define OPENMM_GLUEDFORCEIMPL_H_

#include "GluedForce.h"
#include "GluedKernels.h"
#include "openmm/internal/ForceImpl.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/Kernel.h"
#include <map>
#include <string>
#include <vector>
#include "internal/windowsExportGlued.h"

namespace GluedPlugin {

/**
 * Internal implementation of GluedForce.  Bridges the public Force API to
 * the platform kernel (CalcGluedForceKernel).
 *
 * Key design constraints:
 * - Inherits ForceImpl directly (NOT CustomCPPForceImpl) so that all computation
 *   stays on the GPU without a CPU↔GPU round-trip.
 * - The lastStepIndex_ guard in updateContextState() ensures bias deposition
 *   fires exactly once per step even when execute() is called multiple times.
 */
class OPENMM_EXPORT_GLUED GluedForceImpl : public OpenMM::ForceImpl {
public:
    GluedForceImpl(const GluedForce& owner);
    ~GluedForceImpl() override;

    void initialize(OpenMM::ContextImpl& context) override;

    const OpenMM::Force& getOwner() const override { return owner_; }

    void updateContextState(OpenMM::ContextImpl& context,
                            bool& forcesInvalid) override;

    double calcForcesAndEnergy(OpenMM::ContextImpl& context,
                               bool includeForces, bool includeEnergy,
                               int groups) override;

    std::map<std::string, double> getDefaultParameters() override;

    std::vector<std::string> getKernelNames() override;

    // Forwarded to kernel for checkpoint/restore
    std::vector<char> getBiasStateBytes() const;
    void setBiasStateBytes(const std::vector<char>& bytes);

    // Forwarded to kernel for live CV query
    void getCurrentCVs(std::vector<double>& values);

    // Download CV values from the last force evaluation
    std::vector<double> downloadCVValues();

    // Returns [zed, rct, nker, neff] for the biasIndex-th OPES bias.
    std::vector<double> getOPESMetrics(int biasIndex);

    // Multiwalker B2: get shared GPU array pointers from primary, or redirect to primary's arrays.
    std::vector<long long> getMultiWalkerPtrs(int biasType, int localIdx);
    void redirectToPrimaryBias(int biasType, int localIdx, const std::vector<long long>& ptrs);

private:
    const GluedForce& owner_;
    OpenMM::Kernel kernel_;
    int lastStepIndex_ = -1;
};

} // namespace GluedPlugin

#endif // OPENMM_GLUEDFORCEIMPL_H_
