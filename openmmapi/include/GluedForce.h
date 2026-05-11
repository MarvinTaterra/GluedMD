#ifndef OPENMM_GLUEDFORCE_H_
#define OPENMM_GLUEDFORCE_H_

#include "openmm/Context.h"
#include "openmm/Force.h"
#include <vector>
#include <string>
#include "internal/windowsExportGlued.h"

namespace GluedPlugin {

class GluedForceImpl;  // forward-declared so Force can hold impl_ pointer

/**
 * GluedForce is an OpenMM Force that provides GPU-resident enhanced sampling.
 * All collective variable (CV) evaluation, bias evaluation, and chain-rule force
 * scatter run natively inside OpenMM's GPU kernel infrastructure — no CPU↔GPU
 * round-trip is required.
 *
 * CVs are identified by integer type codes (GLUED_CV_DISTANCE = 1, etc.) so
 * that new CV kinds can be added incrementally without reshuffling the API.
 * Biases follow the same convention.
 */
class OPENMM_EXPORT_GLUED GluedForce : public OpenMM::Force {
public:
    // CV type codes — populated in Stage 3
    enum CVType {
        CV_DISTANCE           = 1,
        CV_ANGLE              = 2,
        CV_DIHEDRAL           = 3,
        CV_COM_DISTANCE       = 4,
        CV_GYRATION           = 5,
        CV_COORDINATION       = 6,
        CV_RMSD               = 7,
        CV_EXPRESSION         = 8,
        CV_PYTORCH            = 9,
        // CV_PATH produces 2 output values per call: [s (progress), z (distance)]
        // addCollectiveVariable returns the index of s; z is at index+1.
        // params = [lambda, N_frames, frame0_x0, frame0_y0, frame0_z0, ...]
        CV_PATH               = 10,
        // Stage 3.10 — Cartesian position: params[0] = component (0=x,1=y,2=z)
        CV_POSITION           = 11,
        // Stage 3.11 — Distance RMSD: atoms = flat pair list, params = ref distances
        CV_DRMSD              = 12,
        // Stage 3.12 — Contact map: atoms = pair list, params = per-pair [r0,n,m,w,ref]
        CV_CONTACTMAP         = 13,
        // Stage 3.13 — Plane distance: atoms[0..2] define plane, atoms[3] is query atom
        CV_PLANE              = 14,
        // Stage 3.13 — Projection on axis: atoms[0,1] = vector, atoms[2,3] = axis
        CV_PROJECTION         = 15,
        // Stage 3.14 — Ring puckering (Cremer-Pople): params[0]=ring size, params[1]=component
        CV_PUCKERING          = 16,
        // Stage 3.15 — Dipole: params[0] = component (0=magnitude,1=x,2=y,3=z)
        CV_DIPOLE             = 17,
        // Stage 3.16 — Box volume (nm^3); no atoms, no params
        CV_VOLUME             = 18,
        // Stage 3.16 — Cell length/angle: params[0] = component (0=a,1=b,2=c,3=alpha,4=beta,5=gamma)
        CV_CELL               = 19,
        // Stage 3.17 — Secondary structure: params[0]=subtype (0=alpha,1=antibeta,2=parabeta)
        CV_SECONDARY_STRUCTURE = 20,
        // Stage 3.18 — PCA projection: atoms = selection, params = flattened PC vectors + mean
        CV_PCA                = 21,
        // Stage 3.19 — eRMSD for RNA (deferred); specialized nucleotide reference frames
        CV_ERMSD              = 22
    };

    // Bias type codes — populated in Stage 5
    enum BiasType {
        BIAS_HARMONIC         = 1,
        BIAS_MOVING_RESTRAINT = 2,
        BIAS_METAD            = 3,
        BIAS_PBMETAD          = 4,
        BIAS_OPES             = 5,
        BIAS_EXTERNAL         = 6,
        BIAS_ABMD             = 7,
        // Stage 5.8  — One-sided polynomial wall: params = [at, kappa, exp_factor, n]
        BIAS_UPPER_WALL       = 8,
        BIAS_LOWER_WALL       = 9,
        // Stage 5.9  — Linear coupling: V = k*cv; params = [k]
        BIAS_LINEAR           = 10,
        // Stage 5.10 — OPES expanded ensemble (multi-thermal/-baric); deferred past 5.5
        BIAS_OPES_EXPANDED    = 11,
        // Stage 5.11 — Extended Lagrangian / AFED; deferred (requires integrator changes)
        BIAS_EXT_LAGRANGIAN   = 12,
        // Stage 5.12 — Maximum Entropy / EDS experimental restraints; deferred past Sec 7
        BIAS_MAXENT           = 13,
        BIAS_EDS              = 14
    };

    GluedForce();
    ~GluedForce() override;

    // --- CV management ---

    // Returns the first CV value index for this variable (s for PATH, the single
    // value for all other types). CV_PATH occupies 2 consecutive value slots.
    int addCollectiveVariable(int type, const std::vector<int>& atoms,
                              const std::vector<double>& parameters);
    // Total number of CV output values (PATH counts as 2, all others as 1).
    int getNumCollectiveVariables() const;
    // Number of addCollectiveVariable calls (always <= getNumCollectiveVariables()).
    int getNumCollectiveVariableSpecs() const;
    void getCollectiveVariableInfo(int idx, int& type, std::vector<int>& atoms,
                                   std::vector<double>& parameters) const;

    // --- Bias management ---

    int addBias(int type, const std::vector<int>& cvIndices,
                const std::vector<double>& parameters,
                const std::vector<int>& integerParameters);
    int getNumBiases() const;
    void getBiasInfo(int idx, int& type, std::vector<int>& cvIndices,
                     std::vector<double>& parameters,
                     std::vector<int>& integerParameters) const;

    // --- Global config ---

    void setTemperature(double kelvin);
    double getTemperature() const;

    // Convenience: convert a free-energy BARRIER (kJ/mol) to the OPES biasfactor
    // gamma that addBias() expects.  Formula: gamma = BARRIER/kT (Invernizzi 2020).
    // kT_kJ = 8.314462618e-3 * T_kelvin  (R in kJ/mol/K).
    static double barrierToGamma(double barrier_kJ, double kT_kJ) {
        return barrier_kJ / kT_kJ;
    }
    static double kTFromTemperature(double temperatureKelvin) {
        return 8.314462618e-3 * temperatureKelvin;  // R * T in kJ/mol
    }

    bool usesPeriodicBoundaryConditions() const override { return usesPBC_; }
    void setUsesPeriodicBoundaryConditions(bool yes) { usesPBC_ = yes; }

    // --- Bias state checkpoint/restore (Stage 6) ---

    std::vector<char> getBiasStateBytes() const;
    void setBiasStateBytes(const std::vector<char>& bytes);

    // --- Live CV query (Stage 6) ---

    void getCurrentCollectiveVariables(OpenMM::Context& context,
                                       std::vector<double>& values) const;

    /**
     * Returns convergence diagnostics for the biasIndex-th OPES bias:
     *   [0] zed  = exp(logZ)        — normalization estimate
     *   [1] rct  = kT * logZ        — c(t) convergence indicator (flattens at convergence)
     *   [2] nker = numKernels       — compressed kernel count
     *   [3] neff = effective sample size
     */
    std::vector<double> getOPESMetrics(OpenMM::Context& context, int biasIndex) const;

    /**
     * Get the deposited per-kernel σ values for an OPES bias. Returns a flat
     * float vector of length numKernels * numCVsBias in row-major order
     * (kernel index outer, CV index inner). Empty if no kernels have been
     * deposited or if biasIndex does not refer to an OPES bias.
     */
    std::vector<float> getKernelSigmas(OpenMM::Context& context, int biasIndex) const;

    // --- Stage 2 test-only API ---
    void setTestForce(int mode, double scale);
    int getTestForceMode() const;
    double getTestForceScale() const;

    // --- Stage 3/4 test-only API ---
    // Set bias gradients (dU/dCV_i) used by the scatter kernel for testing.
    // Must be called before Context creation; length must equal numCVs.
    void setTestBiasGradients(const std::vector<double>& gradients);
    std::vector<double> getTestBiasGradients() const;

    // --- Stage 3.7: Expression CVs ---
    // expression: algebraic string; cv0..cvN-1 map to inputCVIndices[0..N-1].
    // Returns the CV value index for this expression CV.
    int addExpressionCV(const std::string& expression, const std::vector<int>& inputCVIndices);
    void getExpressionCVInfo(int specIdx, std::string& expression, std::vector<int>& inputCVIndices) const;

    // --- Stage 3.8: PyTorch (TorchScript) CVs ---
    // torchScriptPath: path to a TorchScript model saved with model.save().
    // atomIndices: user-space atom indices whose positions are fed to the model.
    // parameters: optional scalar hyperparameters (passed as extra tensor to model if non-empty).
    // Model contract: input tensor[N,3] float32 nm positions → scalar output.
    // Returns the CV value index for this PyTorch CV.
    // NOTE: Only supported on the CUDA platform compiled with libtorch.
    int addPyTorchCV(const std::string& torchScriptPath,
                     const std::vector<int>& atomIndices,
                     const std::vector<double>& parameters);
    std::string getPyTorchCVModelPath(int specIdx) const;

    // Download CV values computed during the last force evaluation.
    std::vector<double> getLastCVValues(OpenMM::Context& context) const;

    // --- Multiwalker B2: shared GPU bias arrays ---
    // These methods enable multiple OpenMM Contexts (walkers) on the same GPU to
    // share a single bias grid (MetaD) or kernel list (OPES), with all deposits
    // going to the shared GPU arrays atomically — no CPU roundtrip, no periodic merge.
    // Requires CUDA platform; biasIdx is 0-based index of the bias (order of addBias calls).

    /**
     * Get raw device pointers for the specified bias's shared GPU arrays (primary walker).
     * biasType=BIAS_METAD: returns [grid_ptr]
     * biasType=BIAS_OPES:  returns [centers, sigmas, logweights, numKernels, numAllocated]
     * Returns empty vector on non-CUDA platforms or unsupported bias types.
     */
    std::vector<long long> getMultiWalkerPtrs(OpenMM::Context& context, int biasIdx) const;

    /**
     * Set up this walker to share bias arrays with a primary walker (secondary walker).
     * Must be called AFTER context creation. ptrs comes from primary's getMultiWalkerPtrs().
     * biasIdx: 0-based index of the bias in this walker's force (same bias type and order).
     */
    void setMultiWalkerPtrs(OpenMM::Context& context, int biasIdx, const std::vector<long long>& ptrs);

protected:
    OpenMM::ForceImpl* createImpl() const override;

private:
    friend class GluedForceImpl;   // allows impl to register itself
    struct CV {
        int type;
        std::vector<int> atoms;
        std::vector<double> params;
        std::string exprString;  // non-empty only for CV_EXPRESSION
    };
    struct Bias {
        int type;
        std::vector<int> cvIndices;
        std::vector<double> params;
        std::vector<int> intParams;
    };

    std::vector<CV> cvs_;
    int numCVValues_ = 0;   // total CV output values (PATH counts as 2)
    std::vector<Bias> biases_;
    double temperature_ = -1.0;
    bool usesPBC_ = false;
    int testForceMode_ = 0;
    double testForceScale_ = 0.0;
    std::vector<double> testBiasGradients_;
    // Set during Context creation; points to the most-recently-created impl.
    mutable GluedForceImpl* impl_ = nullptr;
};

} // namespace GluedPlugin

#endif // OPENMM_GLUEDFORCE_H_
