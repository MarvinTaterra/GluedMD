#ifndef GLUED_KERNELS_H_
#define GLUED_KERNELS_H_

#include "GluedForce.h"
#include "openmm/KernelImpl.h"
#include "openmm/Platform.h"
#include "openmm/System.h"
#include "openmm/internal/ContextImpl.h"
#include <string>
#include <vector>

namespace GluedPlugin {

/**
 * Abstract kernel invoked by GluedForceImpl to evaluate CVs, apply biases,
 * and scatter forces on each step.
 *
 * The separation between execute() and updateState() is critical: execute() may
 * be called multiple times per step (e.g. during minimization or constraint
 * iteration), while updateState() is called exactly once per step and is the
 * correct hook for bias deposition.
 */
class CalcGluedForceKernel : public OpenMM::KernelImpl {
public:
    static std::string Name() { return "CalcGluedForce"; }

    CalcGluedForceKernel(std::string name, const OpenMM::Platform& platform)
        : OpenMM::KernelImpl(name, platform) {}

    virtual void initialize(const OpenMM::System& system,
                            const GluedForce& force) = 0;

    /**
     * Evaluate all CV kernels, all bias energy/gradient kernels, and apply
     * chain-rule force scatter.  Returns the total bias energy.
     */
    virtual double execute(OpenMM::ContextImpl& context,
                           bool includeForces, bool includeEnergy) = 0;

    /**
     * Called once per time step.  Handles bias deposition (METAD, OPES, etc.)
     * which must not fire more than once per step.
     */
    virtual void updateState(OpenMM::ContextImpl& context, int step) = 0;

    virtual void getCurrentCVs(OpenMM::ContextImpl& context,
                                std::vector<double>& values) = 0;

    virtual std::vector<char> getBiasStateBytes() = 0;
    virtual void setBiasStateBytes(const std::vector<char>& bytes) = 0;

    virtual std::vector<double> downloadCVValues() = 0;

    /**
     * Returns diagnostic metrics for the biasIndex-th OPES bias:
     *   [0] zed  = exp(logZ)          — normalization estimate
     *   [1] rct  = kT * logZ          — convergence indicator c(t)
     *   [2] nker = numKernels         — compressed kernel count
     *   [3] neff = eff. sample size   — exp(2*logSumW - logSumW2)
     */
    virtual std::vector<double> getOPESMetrics(int biasIndex) = 0;

    /**
     * Multiwalker B2: get raw device pointers for the specified bias's shared arrays.
     * biasType=BIAS_METAD: returns [grid_ptr]
     * biasType=BIAS_OPES:  returns [centers, sigmas, logweights, numKernels, numAllocated]
     * Default implementation returns empty vector (non-CUDA platforms).
     * localIdx: 0-based index within that bias type's list (not the global bias index).
     */
    virtual std::vector<long long> getMultiWalkerPtrs(int biasType, int localIdx) { return {}; }

    /**
     * Multiwalker B2: redirect this walker's bias kernels to use external shared GPU arrays.
     * Called after context creation on secondary walkers.
     * ptrs: raw CUDA device pointers (as long long) from primary's getMultiWalkerPtrs().
     * localIdx: 0-based index within that bias type's list.
     */
    virtual void redirectToPrimaryBias(int biasType, int localIdx, const std::vector<long long>& ptrs) {}
};

} // namespace GluedPlugin

#endif // GLUED_KERNELS_H_
