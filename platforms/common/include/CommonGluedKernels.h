#ifndef COMMON_GLUED_KERNELS_H_
#define COMMON_GLUED_KERNELS_H_

#include "GluedKernels.h"
#include "openmm/common/ComputeContext.h"
#include "openmm/common/ComputeArray.h"
#include "openmm/common/ComputeKernel.h"
#include "openmm/Platform.h"
#include "openmm/System.h"
#include <string>
#include <vector>

namespace GluedPlugin {

class CommonCalcGluedForceKernel : public CalcGluedForceKernel {
public:
    CommonCalcGluedForceKernel(std::string name,
                                    const OpenMM::Platform& platform,
                                    OpenMM::ComputeContext& cc)
        : CalcGluedForceKernel(name, platform), cc_(cc) {}

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

    // Returns [zed, rct, nker, neff] for the biasIndex-th OPES bias.
    std::vector<double> getOPESMetrics(int biasIndex) override;

    // Multiwalker B2: redirect this walker's bias kernels to use primary's shared GPU arrays.
    // ptrs are raw device pointers cast to long long, bias-type-specific.
    void redirectToPrimaryBias(int biasType, int localIdx, const std::vector<long long>& ptrs) override;

protected:
    OpenMM::ComputeContext& cc_;
    int forceGroupFlag_ = 1;

    // CUDA-specific virtual accessors — overridden by CudaCalcGluedForceKernel.
    // Default implementations return nullptr (non-CUDA platforms).
    virtual void* getNativeCudaStream() const { return nullptr; }
    virtual void* getComputeArrayDevPtr(OpenMM::ComputeArray& arr) const { return nullptr; }

private:
    // test kernel
    OpenMM::ComputeKernel testForceKernel_;
    int testForceMode_ = 0;
    double testForceScale_ = 0.0;

    // on-GPU data layout for CV evaluation and force scatter
    struct GpuPlan {
        int numCVs = 0;
        int numJacEntries = 0;

        OpenMM::ComputeArray cvValues;          // double[numCVs]
        OpenMM::ComputeArray cvBiasGradients;   // double[numCVs]

        // Shared bias energy buffer — one slot per bias eval kernel instance.
        // All eval kernels write to biasEnergies[biasEnergyIdx]; execute()
        // downloads once at the end; updateState() reads from the CPU mirror.
        OpenMM::ComputeArray biasEnergies;        // double[numBiasEnergySlots]
        std::vector<double>  biasEnergiesCPU;     // CPU mirror, downloaded once per execute()

        // Jacobian table: one row per (atom, CV) pair
        OpenMM::ComputeArray jacobianAtomIdx;   // int[numJacEntries]  GPU atom idx
        OpenMM::ComputeArray jacobianGradsX;    // float[numJacEntries]
        OpenMM::ComputeArray jacobianGradsY;    // float[numJacEntries]
        OpenMM::ComputeArray jacobianGradsZ;    // float[numJacEntries]
        OpenMM::ComputeArray jacobianCvIdx;     // int[numJacEntries]  which CV

        // Distance CV sub-plan (3.1)
        int numDistanceCVs = 0;
        int distanceFirstCVIndex = 0;
        int distanceFirstJacEntry = 0;
        OpenMM::ComputeArray distanceAtoms;     // int[2*numDistanceCVs] GPU atom pairs

        // Angle CV sub-plan (3.2)
        int numAngleCVs = 0;
        int angleFirstCVIndex = 0;
        int angleFirstJacEntry = 0;
        OpenMM::ComputeArray angleAtoms;        // int[3*numAngleCVs] GPU atom triplets

        // Dihedral CV sub-plan (3.3)
        int numDihedralCVs = 0;
        int dihedralFirstCVIndex = 0;
        int dihedralFirstJacEntry = 0;
        OpenMM::ComputeArray dihedralAtoms;     // int[4*numDihedralCVs] GPU atom quads

        // COM-distance CV sub-plan (3.4)
        // API: atoms[0]=n_group1, atoms[1..n_group1]=g1, atoms[n_group1+1..]=g2
        //      params = masses in same order as the atoms (after n_group1)
        // atomOffsets: int[2*N+1], interleaved group boundaries (g1Start,g2Start,g2End per CV)
        int numCOMDistanceCVs = 0;
        int comDistanceFirstCVIndex = 0;
        int comDistanceFirstJacEntry = 0;
        OpenMM::ComputeArray comDistanceAtomOffsets; // int[2*N+1]
        OpenMM::ComputeArray comDistanceAtoms;       // int[total_com_atoms] GPU idx
        OpenMM::ComputeArray comDistanceMasses;      // float[total_com_atoms]
        OpenMM::ComputeArray comDistanceTotalMasses; // float[2*N]

        // Gyration (Rg) CV sub-plan (3.5)
        // API: atoms = atom list, params = masses in same order
        int numGyrationCVs = 0;
        int gyrationFirstCVIndex = 0;
        int gyrationFirstJacEntry = 0;
        OpenMM::ComputeArray gyrationAtomOffsets;    // int[N+1]
        OpenMM::ComputeArray gyrationAtoms;          // int[total_rg_atoms] GPU idx
        OpenMM::ComputeArray gyrationMasses;         // float[total_rg_atoms]
        OpenMM::ComputeArray gyrationTotalMasses;    // float[N]

        // Coordination number CV sub-plan (3.6)
        // API: atoms[0]=nA, atoms[1..nA]=group A, rest=group B; params=[r0,n,m]
        // atomOffsets: int[2*N+1] interleaved (g1Start,g2Start,g2End per CV)
        int numCoordCVs = 0;
        int coordFirstCVIndex = 0;
        int coordFirstJacEntry = 0;
        OpenMM::ComputeArray coordAtomOffsets;       // int[2*N+1]
        OpenMM::ComputeArray coordAtoms;             // int[total_coord_atoms] GPU idx
        OpenMM::ComputeArray coordParams;            // float[3*N]: [r0,n,m] per CV

        // RMSD CV sub-plan (3.7) — no-fit (TYPE=SIMPLE) positional RMSD
        // API: atoms = atom list; params = [x0,y0,z0, x1,y1,z1, ...] reference nm
        int numRMSDCVs = 0;
        int rmsdFirstCVIndex = 0;
        int rmsdFirstJacEntry = 0;
        OpenMM::ComputeArray rmsdAtomOffsets;        // int[N+1]
        OpenMM::ComputeArray rmsdAtoms;              // int[total_rmsd_atoms] GPU idx
        OpenMM::ComputeArray rmsdRefPos;             // float[3*total_rmsd_atoms]

        // Path CV sub-plan (3.8) — produces 2 CV values: s (progress), z (dist)
        // API: atoms = atom list (M atoms, shared by all frames);
        //      params = [lambda, N_frames, frame0_x0, frame0_y0, frame0_z0, ...]
        // firstCVIndex = s; z is at firstCVIndex+1 per path CV.
        // Jacobian: 2*M entries per path CV (M for s, M for z).
        int numPathCVs = 0;
        int pathFirstCVIndex = 0;
        int pathFirstJacEntry = 0;
        OpenMM::ComputeArray pathAtomOffsets;        // int[N_pathCVs+1]
        OpenMM::ComputeArray pathAtoms;              // int[sum(M_i)] GPU idx
        OpenMM::ComputeArray pathRefOffsets;         // int[N_pathCVs+1] prefix sum of N_i*M_i
        OpenMM::ComputeArray pathRefPos;             // float[3*sum(N_i*M_i)]
        OpenMM::ComputeArray pathParams;             // float[2*N_pathCVs]: [lambda, N_frames]

        // PyTorch CV Jacobian layout
        // Jacobian entries immediately follow path entries: pytorchFirstJacEntry atoms total.
        int pytorchFirstJacEntry = 0;

        // Position CV (3.10): single atom, single Cartesian component.
        // atoms[0]=atom idx, params[0]=component (0=x,1=y,2=z).
        int numPositionCVs = 0;
        int positionFirstCVIndex = 0;
        int positionFirstJacEntry = 0;
        OpenMM::ComputeArray positionAtoms;       // int[N] GPU atom idx
        OpenMM::ComputeArray positionComponents;  // int[N] component 0/1/2

        // DRMSD (3.11): distance RMSD from reference pair distances.
        // atoms = flat pair list [a0,b0,a1,b1,...]; params = ref distances.
        int numDRMSDCVs = 0;
        int drmsdFirstCVIndex = 0;
        int drmsdFirstJacEntry = 0;
        OpenMM::ComputeArray drmsdPairOffsets;  // int[N+1] prefix sum of pair counts
        OpenMM::ComputeArray drmsdAtomPairs;    // int[2*total] GPU atom indices
        OpenMM::ComputeArray drmsdRefDists;     // float[total] reference distances

        // Contact map (3.12): weighted sum of rational switching functions.
        // atoms = flat pair list; params per pair = [r0, nn, mm, w] (4 floats).
        int numContactMapCVs = 0;
        int contactMapFirstCVIndex = 0;
        int contactMapFirstJacEntry = 0;
        OpenMM::ComputeArray contactMapPairOffsets; // int[N+1] prefix sum
        OpenMM::ComputeArray contactMapAtomPairs;   // int[2*total] GPU atom indices
        OpenMM::ComputeArray contactMapParams;      // float[4*total]: [r0,nn,mm,w] per pair

        // Plane CV (3.13): component of normalized normal to a 3-atom plane.
        // atoms=[a,b,c]; params=[component 0/1/2].
        // n = (b-a) × (c-a), CV = n_hat[component]; 3 Jacobian entries per CV.
        int numPlaneCVs = 0;
        int planeFirstCVIndex = 0;
        int planeFirstJacEntry = 0;
        OpenMM::ComputeArray planeAtoms;      // int[3*N] GPU atom indices
        OpenMM::ComputeArray planeComponents; // int[N] component 0/1/2

        // Projection CV (3.14): dot product of displacement with a fixed direction.
        // atoms=[a,b]; params=[nx,ny,nz] (unit-length direction stored pre-normalized).
        // CV = dot(r_b - r_a, d_hat); 2 Jacobian entries per CV.
        int numProjectionCVs = 0;
        int projectionFirstCVIndex = 0;
        int projectionFirstJacEntry = 0;
        OpenMM::ComputeArray projectionAtoms; // int[2*N] GPU atom indices
        OpenMM::ComputeArray projectionDirs;  // float[3*N] normalized direction vectors

        // Volume CV (3.15): simulation box volume. No atoms. Periodic only.
        // CV = boxVecX.x * boxVecY.y * boxVecZ.z. 0 Jacobian entries.
        int numVolumeCVs = 0;
        int volumeFirstCVIndex = 0;

        // Cell CV (3.16): box cell length. No atoms. Periodic only.
        // params=[component]: 0=|a|, 1=|b|, 2=|c|. 0 Jacobian entries.
        int numCellCVs = 0;
        int cellFirstCVIndex = 0;
        OpenMM::ComputeArray cellComponents; // int[max(N,1)] component per CV

        // Dipole CV (3.17): component or magnitude of Σ q_i r_i.
        // atoms=[a0..aN-1]; params=[q0..qN-1, component] (0=x,1=y,2=z,3=|μ|).
        // N Jacobian entries per CV.
        int numDipoleCVs = 0;
        int dipoleFirstCVIndex = 0;
        int dipoleFirstJacEntry = 0;
        OpenMM::ComputeArray dipoleAtomOffsets;  // int[N+1] prefix sum
        OpenMM::ComputeArray dipoleAtoms;        // int[total] GPU atom indices
        OpenMM::ComputeArray dipoleCharges;      // float[total] per-atom charges
        OpenMM::ComputeArray dipoleComponents;   // int[N] component 0/1/2/3

        // PCA CV (3.18): projection onto a principal component vector.
        // atoms=[a0..aN-1]; params=[mean_x0,mean_y0,mean_z0,...,ev_x0,ev_y0,ev_z0,...].
        // CV = dot(r_flat - mean_flat, eigvec); N Jacobian entries per CV.
        int numPCACVs = 0;
        int pcaFirstCVIndex = 0;
        int pcaFirstJacEntry = 0;
        OpenMM::ComputeArray pcaAtomOffsets;     // int[N+1] prefix sum
        OpenMM::ComputeArray pcaAtoms;           // int[total] GPU atom indices
        OpenMM::ComputeArray pcaRefPos;          // float[3*total] reference mean positions
        OpenMM::ComputeArray pcaEigvec;          // float[3*total] eigenvector components

        // Secondary structure CV (3.20) (ALPHARMSD, ANTIBETARMSD, PARABETARMSD)
        // atoms[]: Cα atoms; params[0]=subtype (0=alpha,1=antibeta,2=parabeta), params[1]=r0(nm)
        int numSecStrCVs = 0;
        int secStrFirstCVIndex = 0;
        int secStrFirstJacEntry = 0;
        int secStrTotalWindows = 0;
        OpenMM::ComputeArray secStrAtomOffsets;   // int[numSsCVs+1]
        OpenMM::ComputeArray secStrWindowOffsets; // int[numSsCVs+1] prefix sum of numWindows
        OpenMM::ComputeArray secStrAtoms;         // int[totalAtoms]
        OpenMM::ComputeArray secStrParams;        // float[2*numSsCVs]

        // Cremer-Pople ring puckering CV (3.16).
        // atoms: ring atoms in sequential order (exactly 5 or 6)
        // params: [ring_size (5.0 or 6.0), component (0.0/1.0/2.0)]
        // N Jacobian entries per CV (one per ring atom).
        int numPuckeringCVs    = 0;
        int puckerFirstCVIndex = 0;
        int puckerFirstJacEntry = 0;
        OpenMM::ComputeArray puckerAtomOffsets;  // int[numPuckeringCVs+1]
        OpenMM::ComputeArray puckerAtoms;        // int[totalRingAtoms]
        OpenMM::ComputeArray puckerParams;       // float[2*numPuckeringCVs]

        // eRMSD for RNA structure comparison (3.19) (Bottaro 2014).
        // atoms: [P1_0,P2_0,P3_0, P1_1,P2_1,P3_1, ..., P1_{N-1},P2_{N-1},P3_{N-1}]
        //        (3*N_residues atoms, grouped by residue)
        // params: [N_residues, ax, ay, az, G_ref_0_x, G_ref_0_y, G_ref_0_z, ...]
        //         length = 4 + 3*N*(N-1)/2
        // Supports up to N=64 residues per CV (local array limit in kernel).
        // 6*N*(N-1)/2 Jacobian entries per CV (3 for residue i + 3 for residue j per pair).
        int numErmsdCVs        = 0;
        int ermsdFirstCVIndex  = 0;
        int ermsdFirstJacEntry = 0;
        OpenMM::ComputeArray ermsdAtomOffsets;  // int[numErmsdCVs+1] prefix-sum of 3*N
        OpenMM::ComputeArray ermsdJacOffsets;   // int[numErmsdCVs+1] prefix-sum of 6*N*(N-1)
        OpenMM::ComputeArray ermsdRefGOffsets;  // int[numErmsdCVs+1] prefix-sum of 4*N*(N-1)
        OpenMM::ComputeArray ermsdNRes;         // int[numErmsdCVs]
        OpenMM::ComputeArray ermsdAtoms;        // int[total3N] GPU atom indices
        OpenMM::ComputeArray ermsdRefG;         // float[totalRefG] 4D Bottaro G-vectors
        OpenMM::ComputeArray ermsdCutoffs;      // float[numErmsdCVs]

        bool periodic = false;
    } plan_;

    OpenMM::ComputeKernel distanceKernel_;
    OpenMM::ComputeKernel angleKernel_;
    OpenMM::ComputeKernel dihedralKernel_;
    OpenMM::ComputeKernel comDistanceKernel_;
    OpenMM::ComputeKernel gyrationKernel_;
    OpenMM::ComputeKernel coordKernel_;
    OpenMM::ComputeKernel rmsdKernel_;
    OpenMM::ComputeKernel pathKernel_;
    OpenMM::ComputeKernel positionKernel_;
    OpenMM::ComputeKernel drmsdKernel_;
    OpenMM::ComputeKernel contactMapKernel_;
    OpenMM::ComputeKernel planeKernel_;
    OpenMM::ComputeKernel projectionKernel_;
    OpenMM::ComputeKernel volumeCellKernel_;  // handles both volume and cell CVs
    OpenMM::ComputeKernel dipoleKernel_;
    OpenMM::ComputeKernel pcaKernel_;
    OpenMM::ComputeKernel secStrKernel_;
    OpenMM::ComputeKernel puckerKernel_;
    OpenMM::ComputeKernel ermsdKernel_;
    OpenMM::ComputeKernel scatterKernel_;

    // PyTorch (TorchScript) CV plan.
    // GPU-native path (primary): posq → extractKernel → posBufGPU → torch (CUDA tensor,
    //   stream-shared) → output.item() → cvValues, grad D2D → gradBufGPU →
    //   deinterleavKernel → jacobianGradsXYZ.
    // CPU-intermediary fallback: posq download → torch CPU forward/backward → upload.
    // model holds a torch::jit::script::Module* via type-erased shared_ptr when HAS_TORCH.
    struct PyTorchCVPlan {
        int outputCVIndex = 0;
        int jacOffset = 0;
        int numAtoms = 0;
        std::vector<int> userAtoms;
        std::vector<int> gpuAtomIdx;         // CPU-side GPU-space indices; uploaded to atomIdxGPU
        std::shared_ptr<void> model;          // torch::jit::script::Module*

        // GPU-native inference buffers (initialized only when CUDA stream is available)
        OpenMM::ComputeArray atomIdxGPU;      // int[numAtoms]   GPU-space atom indices
        OpenMM::ComputeArray posBufGPU;       // float[numAtoms*3] interleaved positions
        OpenMM::ComputeArray gradBufGPU;      // float[numAtoms*3] interleaved grad from torch
        OpenMM::ComputeKernel extractKernel;  // posq → posBufGPU
        OpenMM::ComputeKernel deinterleavKernel; // gradBufGPU → jacGradsXYZ
    };

    // Linear coupling bias: V = sum_d k_d * cv_d.
    // params = [k_0, k_1, ...] (1 float per CV dim)
    struct LinearBias {
        int numCVsBias = 0;
        std::vector<int> cvIdxGlobal;
        OpenMM::ComputeArray cvIdxGPU;
        OpenMM::ComputeArray params;      // float[D]
        int biasEnergyIdx = -1;
        OpenMM::ComputeKernel evalKernel;
    };

    // One-sided polynomial wall bias.
    // wallType=0: UPPER_WALL (activates when s > at)
    // wallType=1: LOWER_WALL (activates when s < at)
    // V = kappa * max(0, delta)^n * exp(eps * delta)
    //   where delta = (s-at) for upper, (at-s) for lower
    // params per dim: [at, kappa, eps, n]  (4 floats)
    struct WallBias {
        int numCVsBias = 0;
        int wallType = 0;
        std::vector<int> cvIdxGlobal;
        OpenMM::ComputeArray cvIdxGPU;
        OpenMM::ComputeArray params;      // float[4*D]
        int biasEnergyIdx = -1;
        OpenMM::ComputeKernel evalKernel;
    };

    // bias state — one entry per addBias call

    struct HarmonicBias {
        int numCVsBias = 0;
        std::vector<int> cvIdxGlobal;
        OpenMM::ComputeArray cvIdxGPU;    // int[numCVsBias]
        OpenMM::ComputeArray params;      // float[2*numCVsBias]: [k, s0] per CV
        int biasEnergyIdx = -1;
        OpenMM::ComputeKernel evalKernel;
    };

    // OPES variants:
    //   variant=0 OPES_METAD     : adaptive sigma, finite gamma (biasfactor)
    //   variant=1 OPES_EXPLORE   : fixed sigma0, invGammaFactor=1 (gamma→∞)
    struct OpesBias {
        int numCVsBias = 0;
        std::vector<int> cvIdxGlobal;
        double kT = 2.479;     // kJ/mol at 298 K
        double gamma = 10.0;
        double sigmaMin = 1e-4;
        std::vector<double> sigma0;
        int pace = 500;
        int variant = 0;
        int maxKernels = 100000;
        int numKernels = 0;
        // 0 = mixed-adaptive (Welford at deposition, sigma0 as fallback).
        // >0 = fully adaptive: opesAccumulateWelford runs every step; deposit
        //      skips until nSamples reaches this threshold
        //      (default = 10*PACE).
        int adaptiveSigmaStride = 0;
        // CPU mirror of GPU nSamples; only used when adaptiveSigmaStride > 0.
        // Incremented in execute() each time welfordKernel fires.  Used in
        // updateState() to decide whether the GPU deposit kernel will actually
        // write a kernel (and so whether numKernels should be incremented).
        int nSamplesCPU = 0;

        // CPU mirrors for serialization — populated on demand via GPU downloads.
        // Not kept in sync during simulation; download happens only in getBiasStateBytes().
        double logZCPU = 0.0;
        std::vector<double> runningMean, runningM2;  // staging buffers
        int nSamples = 0;                            // staging counter
        std::vector<float> kernelCentersCPU;
        std::vector<float> kernelSigmasCPU;
        std::vector<float> kernelLogWeightsCPU;

        OpenMM::ComputeArray cvIdxGPU;
        OpenMM::ComputeArray kernelCenters;      // float[maxKernels*numCVsBias]
        OpenMM::ComputeArray kernelSigmas;       // float[maxKernels*numCVsBias]
        OpenMM::ComputeArray kernelLogWeights;   // float[maxKernels]
        // Welford + logZ — maintained by opesGatherDeposit on GPU
        OpenMM::ComputeArray runningMeanGPU;     // double[D]
        OpenMM::ComputeArray runningM2GPU;       // double[D]
        OpenMM::ComputeArray nSamplesGPU;        // int[1]
        OpenMM::ComputeArray sigma0GPU;          // double[D]
        OpenMM::ComputeArray logZGPU;            // double[1]
        OpenMM::ComputeArray numKernelsGPU;      // int[1] — committed kernel count (shared by evalKernel and gatherDepositKernel)
        OpenMM::ComputeArray numAllocatedGPU;    // int[1] — slot-claim counter for B2 multiwalker atomic deposits
        OpenMM::ComputeArray sumWeightsGPU;      // double[1] — KDNorm = Σ h_k (uncorrected weight sum)

        // Multiwalker B2 shared pointers (0 = use own arrays, non-zero = borrowed from primary)
        unsigned long long sharedCentersPtr    = 0;
        unsigned long long sharedSigmasPtr     = 0;
        unsigned long long sharedLogWeightsPtr = 0;
        unsigned long long sharedNumKernelsPtr = 0;
        unsigned long long sharedNumAllocPtr   = 0;
        // neff accumulation — maintained by opesUpdateNeff on GPU
        OpenMM::ComputeArray logSumWGPU;         // double[2]: {logSumW, logSumW2}
        OpenMM::ComputeArray stepCountGPU;       // int[1]
        int biasEnergyIdx = -1;
        OpenMM::ComputeKernel evalKernel;
        OpenMM::ComputeKernel gatherDepositKernel;
        OpenMM::ComputeKernel neffKernel;
        OpenMM::ComputeKernel welfordKernel;  // non-null only when adaptiveSigmaStride > 0
    };

    // Expression CV eval and prop plans
    struct ExpressionCVPlan {
        int numInputs = 0;
        int outputCVIndex = 0;
        OpenMM::ComputeArray inputCVIdx;   // int[max(numInputs,1)]
        OpenMM::ComputeArray partials;     // double[max(numInputs,1)]
        OpenMM::ComputeKernel evalKernel;
        OpenMM::ComputeKernel propKernel;  // invalid if numInputs==0
    };

    // Moving restraint bias (time-dependent harmonic).
    // schedule: double[M*(1+2*D)]: [step_0, k_0_cv0, at_0_cv0, ..., step_1, ...]
    // integerParameters[0] = M (number of schedule entries)
    struct MovingRestraintBias {
        int numCVsBias = 0;
        int numScheduleSteps = 0;
        std::vector<int> cvIdxGlobal;
        OpenMM::ComputeArray cvIdxGPU;   // int[D]
        OpenMM::ComputeArray schedule;   // double[M*(1+2*D)]
        int biasEnergyIdx = -1;
        double lastStep = -1.0;          // cache: avoid redundant setArg when step unchanged
        OpenMM::ComputeKernel evalKernel;
    };

    // ABMD ratchet bias: V = 0.5*k*max(0, maxCv-cv)^2.
    // parameters: [kappa_0, initial_max_0, kappa_1, initial_max_1, ...]
    // maxCv is tracked on the GPU every step by updateMaxKernel.
    // maxCvCPU is a serialization-only staging buffer; filled on demand.
    struct AbmdBias {
        int numCVsBias = 0;
        std::vector<int> cvIdxGlobal;
        std::vector<double> rhoMinCPU;      // staging buffer for serialization (rhoMin per dim)
        OpenMM::ComputeArray cvIdxGPU;      // int[D]
        OpenMM::ComputeArray rhoMin;        // double[D] — running min of (cv-TO)^2
        OpenMM::ComputeArray toGPU;         // double[D] — target values
        OpenMM::ComputeArray kappa;         // float[D]
        int biasEnergyIdx = -1;
        OpenMM::ComputeKernel evalKernel;
    };

    // Well-tempered metadynamics (grid-based).
    // parameters: [height0, sigma_0, ..., sigma_{D-1}, gamma, kT,
    //              origin_0, ..., origin_{D-1}, max_0, ..., max_{D-1}]
    // integerParameters: [pace, numBins_0, ..., numBins_{D-1},
    //                     periodic_0, ..., periodic_{D-1}]
    struct MetaDGridBias {
        int numCVsBias = 0;
        int D = 0;
        std::vector<int>    cvIdxGlobal;
        std::vector<int>    numBins;       // user-specified bins per dim
        std::vector<int>    actualPoints;  // numBins + (periodic?0:1)
        std::vector<double> origin;
        std::vector<double> spacing;
        std::vector<double> invSpacing;
        std::vector<double> maxVal;
        std::vector<int>    strides;
        std::vector<int>    isPeriodic;
        int    totalGridPoints = 0;
        double height0 = 0.0;
        std::vector<double> sigma;
        double gamma = 10.0;
        double kT    = 2.479;
        int    pace  = 500;
        int    numDeposited = 0;
        std::vector<double> gridCPU;       // staging buffer for serialization only
        OpenMM::ComputeArray cvIdxGPU;
        OpenMM::ComputeArray grid;          // double[totalGridPoints]
        OpenMM::ComputeArray originGPU;     // double[D]
        OpenMM::ComputeArray invSpacingGPU; // double[D]
        OpenMM::ComputeArray actualPointsGPU; // int[D]
        OpenMM::ComputeArray stridesGPU;    // int[D]
        OpenMM::ComputeArray periodicGPU;   // int[D]
        OpenMM::ComputeArray sigmaGPU;      // double[D]
        OpenMM::ComputeArray centerGPU;     // double[D] scratch for deposition
        double heightInvFactor = 0.0;       // 0 → flat height; else 1/((γ-1)*kT)
        int biasEnergyIdx = -1;
        OpenMM::ComputeKernel evalKernel;
        OpenMM::ComputeKernel depositKernel;
        OpenMM::ComputeKernel gatherCVsKernel; // gathers cvValues → centerGPU before deposit

        // Multiwalker B2: if non-zero, grid is borrowed from primary walker (do not free)
        unsigned long long sharedGridPtr = 0;  // raw CUDA device ptr of primary's grid (0 = use own)
    };

    // Parallel-bias MetaD: N independent 1-D MetaD grids, one per CV.
    // parameters: [height0, gamma, kT, sigma_0, origin_0, max_0,
    //              sigma_1, origin_1, max_1, ...] (stride 3 per CV after the 3 shared)
    // integerParameters: [pace, numBins_0, periodic_0, numBins_1, periodic_1, ...]
    struct PBMetaDGridBias {
        int pace = 500;
        double kT = 2.479;
        std::vector<MetaDGridBias> subGrids;  // one per CV, each D=1
        OpenMM::ComputeArray localEnergyGPU;     // double[N]: V_i per sub-grid
        OpenMM::ComputeArray localGradGPU;       // double[N]: dV_i/ds per sub-grid
        OpenMM::ComputeArray softmaxWeightsGPU;  // double[N]: w_i (updated by combineKernel)
        OpenMM::ComputeArray cvIdxListGPU;       // int[N]: global CV index per sub-grid
        int combinedEnergySlot = -1;             // one slot in plan_.biasEnergies for V_total
        OpenMM::ComputeKernel combineKernel;
    };

    // OPES Expanded ensemble bias.
    // Each CV index is an ECV (energy collective variable).
    // V(x) = -kT * (log Σ_λ w_λ exp(-ecv_λ/kT) − logZ)
    // params: [kT, w_0, ..., w_{D-1}]  (weights normalized internally)
    // intParams: [pace]
    struct OpesExpandedBias {
        int numStates = 0;
        double kT = 2.479;
        double invKT;
        int pace = 500;
        // logZ and numUpdates live on GPU; CPU copies are serialization-only staging.
        double logZCPU = 0.0;
        std::vector<int>    cvIdxGlobal;
        std::vector<double> logWeightsCPU;  // log(w_λ), pre-normalized
        OpenMM::ComputeArray cvIdxGPU;      // int[D]
        OpenMM::ComputeArray logWeightsGPU; // double[D]
        OpenMM::ComputeArray logZGPU;       // double[1]
        OpenMM::ComputeArray numUpdatesGPU; // int[1]
        int biasEnergyIdx = -1;
        OpenMM::ComputeKernel evalKernel;
        OpenMM::ComputeKernel updateLogZKernel;
    };

    // Extended Lagrangian / AFED.
    // Auxiliary coordinates s and momenta p are now integrated fully on the GPU
    // via extLagVerlet (velocity Verlet) called from updateState().
    // s/p CPU vectors are serialization-only staging buffers; filled on demand.
    // params = [kappa_0, mass_0, kappa_1, mass_1, ...]  (2*D values)
    struct ExtLagBias {
        std::vector<int>    cvIndices;
        std::vector<double> kappa;
        std::vector<double> massS;
        std::vector<double> s;            // staging buffer for serialization
        std::vector<double> p;            // staging buffer for serialization
        bool                initialized = false;
        double              lastDt = -1.0; // cache: avoid redundant setArg when dt unchanged
        OpenMM::ComputeArray cvIdxGPU;    // int[D]
        OpenMM::ComputeArray kappaGPU;    // double[D]
        OpenMM::ComputeArray massSGPU;    // double[D]
        OpenMM::ComputeArray sGPUArr;     // double[D] — maintained by verletKernel
        OpenMM::ComputeArray pGPUArr;     // double[D] — maintained by verletKernel
        int biasEnergyIdx = -1;
        OpenMM::ComputeKernel evalKernel;
        OpenMM::ComputeKernel initSKernel;
        OpenMM::ComputeKernel verletKernel;
    };

    // Error Diffusion Sampling (EDS).
    // Running average and lambda are maintained on the GPU by updateStateKernel.
    // lambda/runAvg CPU vectors are serialization-only staging buffers.
    // params = [target_0, sigma_0, target_1, sigma_1, ...]  (2*D values)
    // int_params = [pace, tau]
    struct EdsBias {
        std::vector<int>    cvIndices;
        std::vector<double> target;
        std::vector<double> max_range;  // max coupling range in kJ/mol (= RANGE * kbt)
        std::vector<double> lambda;     // staging buffer for serialization
        double kbt = 1.0;               // kB*T in kJ/mol
        int pace = 1;
        bool initialized = false;
        int lastDoUpdate = -1;  // cache: avoid redundant setArg on non-transition steps
        OpenMM::ComputeArray cvIdxGPU;     // int[D]
        OpenMM::ComputeArray lambdaGPU;    // double[D]
        OpenMM::ComputeArray meanGPU;      // double[D] — Welford running mean
        OpenMM::ComputeArray ssdGPU;       // double[D] — Welford sum of squared deviations
        OpenMM::ComputeArray accumGPU;     // double[D] — AdaGrad coupling accumulator
        OpenMM::ComputeArray countGPU;     // int[D] — step count within period
        OpenMM::ComputeArray targetGPU;    // double[D]
        OpenMM::ComputeArray maxRangeGPU;  // double[D]
        int biasEnergyIdx = -1;
        OpenMM::ComputeKernel evalKernel;
        OpenMM::ComputeKernel updateStateKernel;
    };

protected:
    // Bias state vectors — protected so CudaCalcGluedForceKernel can read
    // device pointers for B2 multiwalker sharing via getSharedBiasPtrs().
    std::vector<PyTorchCVPlan>        pytorchCVPlans_;
    std::vector<HarmonicBias>         harmonicBiases_;
    std::vector<OpesBias>             opesBiases_;
    std::vector<OpesExpandedBias>     opesExpandedBiases_;
    std::vector<MovingRestraintBias>  movingRestraintBiases_;
    std::vector<AbmdBias>             abmdBiases_;
    std::vector<MetaDGridBias>        metaDGridBiases_;
    std::vector<PBMetaDGridBias>      pbmetaDGridBiases_;
    std::vector<LinearBias>           linearBiases_;
    std::vector<WallBias>             wallBiases_;
    // External bias: fixed precomputed grid (no deposition).
    // Reuses MetaDGridBias for grid geometry/GPU arrays; deposit fields unused.
    // params = [origin_0, ..., origin_{D-1}, max_0, ..., max_{D-1}, grid_val_0, ...]
    // integerParameters = [numBins_0, ..., numBins_{D-1}, isPeriodic_0, ..., isPeriodic_{D-1}]
    std::vector<MetaDGridBias>        externalGridBiases_;
    std::vector<ExtLagBias>           extLagBiases_;
    std::vector<EdsBias>              edsBiases_;

private:

    // MaxEnt (Cesari et al. JCTC 2016): linear bias with adaptive Lagrange multiplier.
    // params    = [kT, sigma, alpha, at_0, kappa_0, tau_0, at_1, kappa_1, tau_1, ...]
    //             3 global scalars + 3 values per CV
    // intParams = [pace, type, errorType]
    //             type:      0=EQUAL, 1=INEQUAL_GT (CV>=at), 2=INEQUAL_LT (CV<=at)
    //             errorType: 0=GAUSSIAN (or no noise if sigma=0), 1=LAPLACE
    struct MaxentBias {
        std::vector<int>    cvIndices;
        std::vector<double> at;      // target values per CV
        std::vector<double> kappa;   // initial learning rate per CV
        std::vector<double> tau;     // decay time (in steps) per CV
        std::vector<double> lambda;  // staging buffer for serialization
        double kbt   = 2.479;        // kB*T in kJ/mol
        double sigma = 0.0;          // noise level (Gaussian or Laplace sigma)
        double alpha = 1.0;          // Laplace shape parameter
        int pace      = 100;
        int type      = 0;           // 0=EQUAL, 1=INEQUAL_GT, 2=INEQUAL_LT
        int errorType = 0;           // 0=GAUSSIAN, 1=LAPLACE
        int biasEnergyIdx = -1;
        OpenMM::ComputeArray cvIdxGPU;    // int[D]
        OpenMM::ComputeArray lambdaGPU;   // double[D]
        OpenMM::ComputeArray atGPU;       // double[D]
        OpenMM::ComputeArray kappaGPU;    // double[D]
        OpenMM::ComputeArray tauGPU;      // double[D]
        OpenMM::ComputeKernel evalKernel;
        OpenMM::ComputeKernel updateKernel;
    };
    std::vector<MaxentBias>           maxentBiases_;
    std::vector<ExpressionCVPlan>     expressionCVPlans_;
    int lastUpdateStep_ = -1;  // guards duplicate deposition in updateState()
    int lastKnownStep_ = 0;  // tracked every updateState(); read by moving restraint in execute()
    bool cvValuesReady_ = false;  // set by execute(); guards updateState() on step 0
    // Box-change cache: setPeriodicBoxArgs only fires when box differs from last step.
    OpenMM::Vec3 lastBoxA_, lastBoxB_, lastBoxC_;
    bool boxArgsNeedUpdate_ = true;

    std::vector<int>    distanceUserAtoms_;
    std::vector<int>    angleUserAtoms_;
    std::vector<int>    dihedralUserAtoms_;
    // COM-distance: flat [g1_cv0..., g2_cv0..., g1_cv1..., g2_cv1..., ...]
    std::vector<int>    comDistanceUserAtoms_;
    std::vector<int>    comDistanceNGroup1_;       // n_group1 for each CV
    std::vector<int>    comDistanceCVAtomCount_;   // ng1+ng2 for each CV
    std::vector<float>  comDistanceMassData_;      // masses in same order as userAtoms
    // Gyration: flat atom list per CV, appended in CV order
    std::vector<int>    gyrationUserAtoms_;
    std::vector<int>    gyrationNAtoms_;        // atom count for each CV
    std::vector<float>  gyrationMassData_;
    // Coordination number: flat atom list per CV, appended in CV order
    std::vector<int>    coordUserAtoms_;
    std::vector<int>    coordNGroup1_;          // nA for each CV
    std::vector<int>    coordCVAtomCount_;      // nA+nB for each CV
    std::vector<float>  coordParamData_;        // [r0,n,m] per CV (3 floats each)
    // RMSD: flat atom list per CV, appended in CV order
    std::vector<int>    rmsdUserAtoms_;
    std::vector<int>    rmsdNAtoms_;            // atom count per CV
    std::vector<float>  rmsdRefPosData_;        // [x,y,z] per atom (3 floats each)
    // Path CV: flat atom and reference data, appended in CV order
    std::vector<int>    pathUserAtoms_;
    std::vector<int>    pathNAtoms_;            // M atoms per CV
    std::vector<int>    pathNFrames_;           // N frames per CV
    std::vector<float>  pathLambdaData_;        // lambda per CV
    std::vector<float>  pathRefPosData_;        // [x,y,z] per (frame,atom), N*M*3 per CV
    // PyTorch CV: per-CV atom counts (used to compute Jacobian offsets in buildPlan)
    std::vector<int>    pytorchNAtoms_;         // numAtoms for each PyTorch CV
    // Position CV data
    std::vector<int>    positionUserAtoms_;
    std::vector<int>    positionComponentData_; // 0/1/2 per CV
    // DRMSD CV data
    std::vector<int>    drmsdUserPairAtoms_;    // flat [a0,b0,a1,b1,...] user idx
    std::vector<int>    drmsdNPairs_;           // pair count per CV
    std::vector<float>  drmsdRefDistData_;      // reference distances
    // ContactMap CV data
    std::vector<int>    contactMapUserPairAtoms_; // flat [a0,b0,...] user idx
    std::vector<int>    contactMapNPairs_;        // pair count per CV
    std::vector<float>  contactMapParamData_;     // [r0,nn,mm,w] per pair
    // Plane CV data
    std::vector<int>    planeUserAtoms_;          // flat [a0,b0,c0,...] user idx
    std::vector<int>    planeComponentData_;      // component 0/1/2 per CV
    // Projection CV data
    std::vector<int>    projectionUserAtoms_;     // flat [a0,b0,...] user idx
    std::vector<float>  projectionDirData_;       // [nx,ny,nz] normalized per CV
    // Cell CV data
    std::vector<int>    cellComponentData_;       // component 0/1/2 per CV
    // Dipole CV data
    std::vector<int>    dipoleUserAtoms_;         // flat atom list (user idx)
    std::vector<int>    dipoleNAtoms_;            // atom count per CV
    std::vector<float>  dipoleChargeData_;        // charge per atom (same order)
    std::vector<int>    dipoleComponentData_;     // 0=x,1=y,2=z,3=|μ| per CV
    // PCA CV data
    std::vector<int>    pcaUserAtoms_;            // flat atom list (user idx)
    std::vector<int>    pcaNAtoms_;               // atom count per CV
    std::vector<float>  pcaRefPosData_;           // [x0,y0,z0,...] reference mean
    std::vector<float>  pcaEigvecData_;           // [x0,y0,z0,...] eigenvector
    // Secondary structure CV (3.20) data
    std::vector<int>    secStrUserAtoms_;
    std::vector<int>    secStrNAtoms_;            // atom count per CV
    std::vector<float>  secStrParamData_;         // [subtype, r0] per CV
    // Puckering CV data
    std::vector<int>    puckerUserAtoms_;         // flat ring atom list (user idx)
    std::vector<int>    puckerNAtoms_;            // ring size (5 or 6) per CV
    std::vector<float>  puckerParamData_;         // [ring_size, component] per CV
    // ERMSD CV data
    std::vector<int>   ermsdUserAtoms_;           // flat [P1,P2,P3,...] user idx
    std::vector<int>   ermsdNRes_;                // N residues per CV
    std::vector<float> ermsdRefGData_;            // reference G-vectors (4D Bottaro, all ordered pairs)
    std::vector<float> ermsdCutoffData_;          // cutoff per CV
    std::vector<double> testBiasGradients_;

    void buildPlan(const OpenMM::System& system, const GluedForce& force);
    void compileCVKernels(const GluedForce& force);
    void setupBiases(const GluedForce& force);
    void rebuildGpuAtomIndices();
};

} // namespace GluedPlugin

#endif // COMMON_GLUED_KERNELS_H_
