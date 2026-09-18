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
     * Hand the kernel a linked inner Context (clone of the System minus this
     * GluedForce) used to evaluate the unbiased total potential energy U and
     * forces F for a CV_ENERGY (OPES multithermal). Called by GluedForceImpl
     * after createLinkedContext. The matching ContextImpl* is passed alongside
     * because the kernel cannot reach Context::getImpl() (private); the device
     * path drives the inner evaluation through it (calcForcesAndEnergy,
     * getPlatformData). No-op default for platforms without an energy CV.
     */
    virtual void setInnerContext(OpenMM::Context* inner,
                                 OpenMM::ContextImpl* innerImpl) {}

    /**
     * Evaluate all CV kernels, all bias energy/gradient kernels, and apply
     * chain-rule force scatter.  Returns the total bias energy.
     */
    virtual double execute(OpenMM::ContextImpl& context,
                           bool includeForces, bool includeEnergy) = 0;

    /**
     * Called once per time step, before the step's force evaluation. Commits
     * bias history (deposits, running statistics, auxiliary coordinates) against
     * the current coordinates. Returns true if the bias potential changed, so the
     * integrator can discard cached forces.
     */
    virtual bool updateState(OpenMM::ContextImpl& context, long long step) = 0;

    virtual void getCurrentCVs(OpenMM::ContextImpl& context,
                                std::vector<double>& values) = 0;

    virtual std::vector<char> getBiasStateBytes() = 0;
    virtual void setBiasStateBytes(const std::vector<char>& bytes) = 0;

    virtual std::vector<double> downloadCVValues() = 0;

    // Total applied bias energy (kJ/mol) from the last force evaluation —
    // cached GPU read-back (no re-evaluation), so safe to call from a reporter
    // even with a PyTorch CV. Default 0 for platforms not implementing it.
    virtual double downloadLastBias() { return 0.0; }

    /**
     * Returns diagnostic metrics for the biasIndex-th OPES bias:
     *   [0] zed  = exp(logZ)          — normalization estimate
     *   [1] rct  = kT * logZ          — convergence indicator c(t)
     *   [2] nker = numKernels         — compressed kernel count
     *   [3] neff = eff. sample size   — exp(2*logSumW - logSumW2)
     */
    virtual std::vector<double> getOPESMetrics(int biasIndex) = 0;

    /**
     * Download the per-CV σ values of all deposited OPES kernels for a bias.
     * Returns a flat float vector of length numKernels * numCVsBias in row-
     * major order (kernel index outer, CV index inner). Returns an empty
     * vector if biasIndex is invalid, the bias is not OPES, or no kernels
     * have been deposited yet.
     */
    virtual std::vector<float> getKernelSigmas(int biasIndex) = 0;
};

} // namespace GluedPlugin

#endif // GLUED_KERNELS_H_
