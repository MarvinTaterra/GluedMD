#include "CommonGluedKernels.h"
#include "GluedForce.h"
#include "openmm/common/ContextSelector.h"
#include <cmath>
#include <string>
#include <vector>

using namespace GluedPlugin;
using namespace OpenMM;
using namespace std;

// Kernel source strings defined in CommonGluedKernels.cpp
extern const string kScatterKernelSrc;
extern const string kHarmonicKernelSrc;
extern const string kLinearKernelSrc;
extern const string kWallKernelSrc;
extern const string kOPESKernelSrc;
extern const string kOPESExpandedKernelSrc;
extern const string kMovingRestraintKernelSrc;
extern const string kABMDKernelSrc;
extern const string kExtLagKernelSrc;
extern const string kEdsKernelSrc;
extern const string kMaxEntKernelSrc;
extern const string kMetaDKernelSrc;
extern const string kPBMetaDKernelSrc;


// Allocate GPU state and compile kernels for every registered bias.
// Also compiles the chain-rule scatter kernel (depends on Jacobian arrays).
// Called from initialize() after compileCVKernels().
void CommonCalcGluedForceKernel::setupBiases(const GluedForce& force) {
 // Bias kernels — set up one entry per addBias call.
 // Bias kernels run after CV kernels but before scatter so that
 // cvBiasGradients is fully populated before chain-rule scatter.
 int numBiases = force.getNumBiases();

 // Reserve all bias vectors before the loop to prevent reallocation.
 // Reallocation would invalidate pointers stored in kernel addArg() calls.
 // Also count totalEnergySlots: PBMetaD has one slot per subgrid (per CV dim).
 {
 int nH=0, nMv=0, nOp=0, nAb=0, nMd=0, nPb=0, nEx=0, nLin=0, nWall=0, nOpEx=0, nEl=0, nEds=0, nMxe=0;
 int totalEnergySlots = 0;
 for (int b = 0; b < numBiases; b++) {
 int t; vector<int> ci; vector<double> p; vector<int> ip;
 force.getBiasInfo(b, t, ci, p, ip);
 if (t == GluedForce::BIAS_HARMONIC) { nH++; totalEnergySlots++; }
 else if (t == GluedForce::BIAS_MOVING_RESTRAINT) { nMv++; totalEnergySlots++; }
 else if (t == GluedForce::BIAS_OPES) { nOp++; totalEnergySlots++; }
 else if (t == GluedForce::BIAS_ABMD) { nAb++; totalEnergySlots++; }
 else if (t == GluedForce::BIAS_METAD) { nMd++; totalEnergySlots++; }
 else if (t == GluedForce::BIAS_PBMETAD) { nPb++; totalEnergySlots += 1; }
 else if (t == GluedForce::BIAS_EXTERNAL) { nEx++; totalEnergySlots++; }
 else if (t == GluedForce::BIAS_LINEAR) { nLin++; totalEnergySlots++; }
 else if (t == GluedForce::BIAS_UPPER_WALL ||
 t == GluedForce::BIAS_LOWER_WALL) { nWall++; totalEnergySlots++; }
 else if (t == GluedForce::BIAS_OPES_EXPANDED) { nOpEx++; totalEnergySlots++; }
 else if (t == GluedForce::BIAS_EXT_LAGRANGIAN) { nEl++; totalEnergySlots++; }
 else if (t == GluedForce::BIAS_EDS) { nEds++; totalEnergySlots++; }
 else if (t == GluedForce::BIAS_MAXENT) { nMxe++; totalEnergySlots++; }
 }
 // Allocate the shared bias energy buffer and CPU mirror.
 int slots = max(1, totalEnergySlots);
 plan_.biasEnergies.initialize<double>(cc_, slots, "biasEnergies");
 plan_.biasEnergiesCPU.assign(totalEnergySlots, 0.0);
 harmonicBiases_.reserve(nH);
 movingRestraintBiases_.reserve(nMv);
 opesBiases_.reserve(nOp);
 abmdBiases_.reserve(nAb);
 metaDGridBiases_.reserve(nMd);
 pbmetaDGridBiases_.reserve(nPb);
 externalGridBiases_.reserve(nEx);
 linearBiases_.reserve(nLin);
 wallBiases_.reserve(nWall);
 opesExpandedBiases_.reserve(nOpEx);
 extLagBiases_.reserve(nEl);
 edsBiases_.reserve(nEds);
 maxentBiases_.reserve(nMxe);
 }

 int biasEnergyCounter = 0; // monotonically assigned slot in plan_.biasEnergies
 for (int bIdx = 0; bIdx < numBiases; bIdx++) {
 int bType;
 vector<int> cvIndices;
 vector<double> bParams;
 vector<int> bIntParams;
 force.getBiasInfo(bIdx, bType, cvIndices, bParams, bIntParams);

 if (bType == GluedForce::BIAS_HARMONIC) {
 harmonicBiases_.emplace_back();
 HarmonicBias& h = harmonicBiases_.back();
 h.numCVsBias = (int)cvIndices.size();
 h.cvIdxGlobal = cvIndices;
 h.cvIdxGPU.initialize<int>(cc_, h.numCVsBias,
 "harmBias_cvIdx_" + to_string(bIdx));
 h.cvIdxGPU.upload(vector<int>(cvIndices.begin(), cvIndices.end()));

 // params layout: [k_0, s0_0, k_1, s0_1, ...]
 vector<float> pf;
 for (int d = 0; d < h.numCVsBias; d++) {
 pf.push_back((float)bParams[2*d]);
 pf.push_back((float)bParams[2*d + 1]);
 }
 h.params.initialize<float>(cc_, 2*h.numCVsBias,
 "harmBias_params_" + to_string(bIdx));
 h.params.upload(pf);
 h.biasEnergyIdx = biasEnergyCounter++;

 ComputeProgram hProg = cc_.compileProgram(kHarmonicKernelSrc, {});
 h.evalKernel = hProg->createKernel("harmonicEvalBias");
 h.evalKernel->addArg(plan_.cvValues); // 0
 h.evalKernel->addArg(h.cvIdxGPU); // 1
 h.evalKernel->addArg(h.params); // 2
 h.evalKernel->addArg(h.numCVsBias); // 3
 h.evalKernel->addArg(plan_.cvBiasGradients); // 4
 h.evalKernel->addArg(plan_.biasEnergies); // 5
 h.evalKernel->addArg(h.biasEnergyIdx); // 6
 h.evalKernel->addArg(); // 7: spare (8.5.1 fix)

 } else if (bType == GluedForce::BIAS_MOVING_RESTRAINT) {
 movingRestraintBiases_.emplace_back();
 MovingRestraintBias& mv = movingRestraintBiases_.back();
 mv.numCVsBias = (int)cvIndices.size();
 mv.cvIdxGlobal = cvIndices;
 int M = bIntParams.size() > 0 ? bIntParams[0] : 1;
 mv.numScheduleSteps = M;
 int D = mv.numCVsBias;
 int stride = 1 + 2*D;
 mv.cvIdxGPU.initialize<int>(cc_, D,
 "movRestraint_cvIdx_" + to_string(bIdx));
 mv.cvIdxGPU.upload(vector<int>(cvIndices.begin(), cvIndices.end()));
 mv.schedule.initialize<double>(cc_, M * stride,
 "movRestraint_sched_" + to_string(bIdx));
 mv.schedule.upload(vector<double>(bParams.begin(), bParams.end()));
 mv.biasEnergyIdx = biasEnergyCounter++;
 ComputeProgram mvProg = cc_.compileProgram(kMovingRestraintKernelSrc, {});
 mv.evalKernel = mvProg->createKernel("movingRestraintEvalBias");
 mv.evalKernel->addArg(plan_.cvValues); // 0
 mv.evalKernel->addArg(mv.cvIdxGPU); // 1
 mv.evalKernel->addArg(mv.schedule); // 2
 mv.evalKernel->addArg(M); // 3
 mv.evalKernel->addArg(D); // 4
 mv.evalKernel->addArg(0.0); // 5: currentStep (setArg in execute)
 mv.evalKernel->addArg(plan_.cvBiasGradients); // 6
 mv.evalKernel->addArg(plan_.biasEnergies); // 7
 mv.evalKernel->addArg(mv.biasEnergyIdx); // 8
 mv.evalKernel->addArg(); // 9: spare (8.5.1 fix)

 } else if (bType == GluedForce::BIAS_ABMD) {
 abmdBiases_.emplace_back();
 AbmdBias& a = abmdBiases_.back();
 a.numCVsBias = (int)cvIndices.size();
 a.cvIdxGlobal = cvIndices;
 int D = a.numCVsBias;
 a.rhoMinCPU.resize(D, -1.0); // -1 = sentinel: first call always updates
 vector<float> kappaVec(D);
 vector<double> toVec(D);
 for (int d = 0; d < D; d++) {
 kappaVec[d] = (float)bParams[2*d];
 toVec[d] = bParams[2*d + 1]; // target value TO
 }
 a.cvIdxGPU.initialize<int>(cc_, D, "abmd_cvIdx_" + to_string(bIdx));
 a.cvIdxGPU.upload(vector<int>(cvIndices.begin(), cvIndices.end()));
 a.rhoMin.initialize<double>(cc_, D, "abmd_rhoMin_" + to_string(bIdx));
 a.rhoMin.upload(a.rhoMinCPU);
 a.toGPU.initialize<double>(cc_, D, "abmd_to_" + to_string(bIdx));
 a.toGPU.upload(toVec);
 a.kappa.initialize<float>(cc_, D, "abmd_kappa_" + to_string(bIdx));
 a.kappa.upload(kappaVec);
 a.biasEnergyIdx = biasEnergyCounter++;
 ComputeProgram abmdProg = cc_.compileProgram(kABMDKernelSrc, {});
 a.evalKernel = abmdProg->createKernel("abmdEvalBias");
 a.evalKernel->addArg(plan_.cvValues); // 0
 a.evalKernel->addArg(a.cvIdxGPU); // 1
 a.evalKernel->addArg(a.toGPU); // 2
 a.evalKernel->addArg(a.rhoMin); // 3
 a.evalKernel->addArg(a.kappa); // 4
 a.evalKernel->addArg(D); // 5
 a.evalKernel->addArg(plan_.cvBiasGradients); // 6
 a.evalKernel->addArg(plan_.biasEnergies); // 7
 a.evalKernel->addArg(a.biasEnergyIdx); // 8
 a.evalKernel->addArg(); // 9: spare

 } else if (bType == GluedForce::BIAS_METAD) {
 metaDGridBiases_.emplace_back();
 MetaDGridBias& m = metaDGridBiases_.back();
 m.numCVsBias = (int)cvIndices.size();
 m.D = m.numCVsBias;
 m.cvIdxGlobal = cvIndices;

 m.height0 = bParams[0];
 for (int d = 0; d < m.D; d++) m.sigma.push_back(bParams[1 + d]);
 m.gamma = bParams[1 + m.D];
 m.kT = bParams[2 + m.D];
 for (int d = 0; d < m.D; d++) m.origin.push_back(bParams[3 + m.D + d]);
 for (int d = 0; d < m.D; d++) m.maxVal.push_back(bParams[3 + 2*m.D + d]);

 m.pace = bIntParams[0];
 for (int d = 0; d < m.D; d++) m.numBins.push_back(bIntParams[1 + d]);
 for (int d = 0; d < m.D; d++) m.isPeriodic.push_back(bIntParams[1 + m.D + d]);

 m.spacing.resize(m.D); m.invSpacing.resize(m.D);
 m.actualPoints.resize(m.D); m.strides.resize(m.D);
 for (int d = 0; d < m.D; d++) {
 m.spacing[d] = (m.maxVal[d] - m.origin[d]) / m.numBins[d];
 m.invSpacing[d] = m.numBins[d] / (m.maxVal[d] - m.origin[d]);
 m.actualPoints[d] = m.numBins[d] + (m.isPeriodic[d] ? 0 : 1);
 }
 m.strides[0] = 1;
 for (int d = 1; d < m.D; d++)
 m.strides[d] = m.strides[d-1] * m.actualPoints[d-1];
 m.totalGridPoints = m.strides[m.D-1] * m.actualPoints[m.D-1];

 m.gridCPU.assign(m.totalGridPoints, 0.0);

 m.cvIdxGPU.initialize<int>(cc_, m.D, "metaD_cvIdx_" + to_string(bIdx));
 m.cvIdxGPU.upload(vector<int>(cvIndices.begin(), cvIndices.end()));
 m.grid.initialize<double>(cc_, m.totalGridPoints, "metaD_grid_" + to_string(bIdx));
 m.grid.upload(m.gridCPU);
 m.originGPU.initialize<double>(cc_, m.D, "metaD_origin_" + to_string(bIdx));
 m.originGPU.upload(m.origin);
 m.invSpacingGPU.initialize<double>(cc_, m.D, "metaD_invSp_" + to_string(bIdx));
 m.invSpacingGPU.upload(m.invSpacing);
 m.actualPointsGPU.initialize<int>(cc_, m.D, "metaD_npts_" + to_string(bIdx));
 m.actualPointsGPU.upload(vector<int>(m.actualPoints.begin(), m.actualPoints.end()));
 m.stridesGPU.initialize<int>(cc_, m.D, "metaD_strides_" + to_string(bIdx));
 m.stridesGPU.upload(vector<int>(m.strides.begin(), m.strides.end()));
 m.periodicGPU.initialize<int>(cc_, m.D, "metaD_periodic_" + to_string(bIdx));
 m.periodicGPU.upload(vector<int>(m.isPeriodic.begin(), m.isPeriodic.end()));
 m.sigmaGPU.initialize<double>(cc_, m.D, "metaD_sigma_" + to_string(bIdx));
 m.sigmaGPU.upload(m.sigma);
 m.centerGPU.initialize<double>(cc_, m.D, "metaD_center_" + to_string(bIdx));
 m.biasEnergyIdx = biasEnergyCounter++;
 m.heightInvFactor = (m.gamma <= 1.0) ? 0.0 : 1.0 / ((m.gamma - 1.0) * m.kT);

 ComputeProgram mdProg = cc_.compileProgram(kMetaDKernelSrc, {});

 m.evalKernel = mdProg->createKernel("metaDEvalBias");
 m.evalKernel->addArg(plan_.cvValues); // 0
 m.evalKernel->addArg(m.cvIdxGPU); // 1
 m.evalKernel->addArg(m.grid); // 2
 m.evalKernel->addArg(m.originGPU); // 3
 m.evalKernel->addArg(m.invSpacingGPU); // 4
 m.evalKernel->addArg(m.actualPointsGPU); // 5
 m.evalKernel->addArg(m.stridesGPU); // 6
 m.evalKernel->addArg(m.periodicGPU); // 7
 m.evalKernel->addArg(m.D); // 8
 m.evalKernel->addArg(plan_.cvBiasGradients); // 9
 m.evalKernel->addArg(plan_.biasEnergies); // 10
 m.evalKernel->addArg(m.biasEnergyIdx); // 11
 m.evalKernel->addArg(); // 12: spare (8.5.1 fix)

 m.depositKernel = mdProg->createKernel("metaDDeposit");
 m.depositKernel->addArg(m.grid); // 0
 m.depositKernel->addArg(m.originGPU); // 1
 m.depositKernel->addArg(m.invSpacingGPU); // 2
 m.depositKernel->addArg(m.actualPointsGPU); // 3
 m.depositKernel->addArg(m.stridesGPU); // 4
 m.depositKernel->addArg(m.periodicGPU); // 5
 m.depositKernel->addArg(m.D); // 6
 m.depositKernel->addArg(m.totalGridPoints); // 7
 m.depositKernel->addArg(m.centerGPU); // 8
 m.depositKernel->addArg(m.sigmaGPU); // 9
 m.depositKernel->addArg(plan_.biasEnergies); // 10
 m.depositKernel->addArg(m.biasEnergyIdx); // 11
 m.depositKernel->addArg(m.height0); // 12
 m.depositKernel->addArg(m.heightInvFactor); // 13
 m.depositKernel->addArg(); // 14: spare

 m.gatherCVsKernel = mdProg->createKernel("metaDGatherCVs");
 m.gatherCVsKernel->addArg(plan_.cvValues); // 0
 m.gatherCVsKernel->addArg(m.cvIdxGPU); // 1
 m.gatherCVsKernel->addArg(m.centerGPU); // 2
 m.gatherCVsKernel->addArg(m.D); // 3

 } else if (bType == GluedForce::BIAS_PBMETAD) {
 // N independent 1-D MetaD grids combined via log-sum-exp (PBMetaD, Pfaendtner & Bonomi 2015).
 // params: [height0, gamma, kT, sigma_0, origin_0, max_0, sigma_1, ...]
 // intParams: [pace, numBins_0, periodic_0, numBins_1, periodic_1, ...]
 pbmetaDGridBiases_.emplace_back();
 PBMetaDGridBias& pb = pbmetaDGridBiases_.back();
 int N = (int)cvIndices.size();
 double height0 = bParams[0];
 double gamma = bParams[1];
 double kT = bParams[2];
 pb.pace = bIntParams[0];
 pb.kT = kT;
 pb.subGrids.resize(N);
 pb.combinedEnergySlot = biasEnergyCounter++;  // ONE slot for entire PBMETAD bias

 // Shared buffers for combining: localEnergy[N], localGrad[N], softmaxWeights[N], cvIdxList[N]
 string pbtag = "pb_" + to_string(bIdx);
 pb.localEnergyGPU.initialize<double>(cc_, N, "pbMetaD_localE_" + pbtag);
 pb.localGradGPU.initialize<double>(cc_, N, "pbMetaD_localG_" + pbtag);
 pb.softmaxWeightsGPU.initialize<double>(cc_, N, "pbMetaD_softmax_" + pbtag);
 pb.cvIdxListGPU.initialize<int>(cc_, N, "pbMetaD_cvList_" + pbtag);
 {
  vector<double> zeros(N, 0.0);
  vector<double> uniform(N, 1.0 / N);
  pb.localEnergyGPU.upload(zeros);
  pb.localGradGPU.upload(zeros);
  pb.softmaxWeightsGPU.upload(uniform);
  vector<int> cvList(cvIndices.begin(), cvIndices.end());
  pb.cvIdxListGPU.upload(cvList);
 }

 ComputeProgram mdProg  = cc_.compileProgram(kMetaDKernelSrc,   {});
 ComputeProgram pbProg  = cc_.compileProgram(kPBMetaDKernelSrc, {});

 for (int d = 0; d < N; d++) {
  MetaDGridBias& m = pb.subGrids[d];
  m.numCVsBias = 1;
  m.D = 1;
  m.cvIdxGlobal = { cvIndices[d] };

  m.height0 = height0;
  m.sigma.push_back(bParams[3 + 3*d]);
  m.gamma = gamma;
  m.kT = kT;
  m.origin.push_back(bParams[3 + 3*d + 1]);
  m.maxVal.push_back(bParams[3 + 3*d + 2]);
  m.pace = pb.pace;

  m.numBins.push_back(bIntParams[1 + 2*d]);
  m.isPeriodic.push_back(bIntParams[1 + 2*d + 1]);

  m.spacing.resize(1); m.invSpacing.resize(1);
  m.actualPoints.resize(1); m.strides.resize(1);
  m.spacing[0] = (m.maxVal[0] - m.origin[0]) / m.numBins[0];
  m.invSpacing[0] = m.numBins[0] / (m.maxVal[0] - m.origin[0]);
  m.actualPoints[0] = m.numBins[0] + (m.isPeriodic[0] ? 0 : 1);
  m.strides[0] = 1;
  m.totalGridPoints = m.actualPoints[0];
  m.gridCPU.assign(m.totalGridPoints, 0.0);
  m.biasEnergyIdx = -1;  // not used; combined slot is pb.combinedEnergySlot
  m.heightInvFactor = (gamma <= 1.0) ? 0.0 : 1.0 / ((gamma - 1.0) * kT);

  string tag = "pb_" + to_string(bIdx) + "_d" + to_string(d);
  m.cvIdxGPU.initialize<int>(cc_, 1, "pbMetaD_cvIdx_" + tag);
  m.cvIdxGPU.upload(vector<int>{ cvIndices[d] });
  m.grid.initialize<double>(cc_, m.totalGridPoints, "pbMetaD_grid_" + tag);
  m.grid.upload(m.gridCPU);
  m.originGPU.initialize<double>(cc_, 1, "pbMetaD_origin_" + tag);
  m.originGPU.upload(m.origin);
  m.invSpacingGPU.initialize<double>(cc_, 1, "pbMetaD_invSp_" + tag);
  m.invSpacingGPU.upload(m.invSpacing);
  m.actualPointsGPU.initialize<int>(cc_, 1, "pbMetaD_npts_" + tag);
  m.actualPointsGPU.upload(vector<int>(m.actualPoints.begin(), m.actualPoints.end()));
  m.periodicGPU.initialize<int>(cc_, 1, "pbMetaD_periodic_" + tag);
  m.periodicGPU.upload(vector<int>(m.isPeriodic.begin(), m.isPeriodic.end()));
  m.sigmaGPU.initialize<double>(cc_, 1, "pbMetaD_sigma_" + tag);
  m.sigmaGPU.upload(m.sigma);
  m.centerGPU.initialize<double>(cc_, 1, "pbMetaD_center_" + tag);

  // Sub-grid eval: writes V_i and dV_i/ds to localEnergy/localGrad[d].
  m.evalKernel = pbProg->createKernel("pbmetaDSubgridEval");
  m.evalKernel->addArg(plan_.cvValues);      // 0
  m.evalKernel->addArg(m.cvIdxGPU);          // 1
  m.evalKernel->addArg(m.grid);              // 2
  m.evalKernel->addArg(m.originGPU);         // 3
  m.evalKernel->addArg(m.invSpacingGPU);     // 4
  m.evalKernel->addArg(m.actualPointsGPU);   // 5
  m.evalKernel->addArg(m.periodicGPU);       // 6
  m.evalKernel->addArg(pb.localEnergyGPU);   // 7
  m.evalKernel->addArg(pb.localGradGPU);     // 8
  m.evalKernel->addArg(d);                   // 9: subgridIdx

  // Sub-grid deposit: uses localEnergy[d] and softmaxWeights[d].
  m.depositKernel = pbProg->createKernel("pbmetaDDeposit");
  m.depositKernel->addArg(m.grid);               // 0
  m.depositKernel->addArg(m.originGPU);           // 1
  m.depositKernel->addArg(m.invSpacingGPU);       // 2
  m.depositKernel->addArg(m.actualPointsGPU);     // 3
  m.depositKernel->addArg(m.periodicGPU);         // 4
  m.depositKernel->addArg(m.totalGridPoints);     // 5
  m.depositKernel->addArg(m.centerGPU);           // 6
  m.depositKernel->addArg(m.sigmaGPU);            // 7
  m.depositKernel->addArg(pb.localEnergyGPU);     // 8
  m.depositKernel->addArg(d);                     // 9: subgridIdx
  m.depositKernel->addArg(m.height0);             // 10
  m.depositKernel->addArg(m.heightInvFactor);     // 11
  m.depositKernel->addArg(pb.softmaxWeightsGPU);  // 12

  m.gatherCVsKernel = mdProg->createKernel("metaDGatherCVs");
  m.gatherCVsKernel->addArg(plan_.cvValues); // 0
  m.gatherCVsKernel->addArg(m.cvIdxGPU);    // 1
  m.gatherCVsKernel->addArg(m.centerGPU);   // 2
  m.gatherCVsKernel->addArg(m.D);           // 3
 }

 // Combine kernel: log-sum-exp over localEnergy[0..N-1] → V_total in biasEnergies.
 pb.combineKernel = pbProg->createKernel("pbmetaDCombine");
 pb.combineKernel->addArg(pb.localEnergyGPU);          // 0
 pb.combineKernel->addArg(pb.localGradGPU);            // 1
 pb.combineKernel->addArg(pb.cvIdxListGPU);            // 2
 pb.combineKernel->addArg(N);                          // 3
 pb.combineKernel->addArg(kT);                         // 4
 pb.combineKernel->addArg(plan_.biasEnergies);         // 5
 pb.combineKernel->addArg(pb.combinedEnergySlot);      // 6
 pb.combineKernel->addArg(plan_.cvBiasGradients);      // 7
 pb.combineKernel->addArg(pb.softmaxWeightsGPU);       // 8

 } else if (bType == GluedForce::BIAS_EXTERNAL) {
 // External bias: user-supplied grid, no deposition.
 // params = [origin_0, ..., origin_{D-1}, max_0, ..., max_{D-1}, grid_vals...]
 // intParams = [numBins_0, ..., numBins_{D-1}, isPeriodic_0, ..., isPeriodic_{D-1}]
 externalGridBiases_.emplace_back();
 MetaDGridBias& m = externalGridBiases_.back();
 m.numCVsBias = (int)cvIndices.size();
 m.D = m.numCVsBias;
 m.cvIdxGlobal = cvIndices;

 for (int d = 0; d < m.D; d++) m.origin.push_back(bParams[d]);
 for (int d = 0; d < m.D; d++) m.maxVal.push_back(bParams[m.D + d]);

 for (int d = 0; d < m.D; d++) m.numBins.push_back(bIntParams[d]);
 for (int d = 0; d < m.D; d++) m.isPeriodic.push_back(bIntParams[m.D + d]);

 m.spacing.resize(m.D); m.invSpacing.resize(m.D);
 m.actualPoints.resize(m.D); m.strides.resize(m.D);
 for (int d = 0; d < m.D; d++) {
 m.spacing[d] = (m.maxVal[d] - m.origin[d]) / m.numBins[d];
 m.invSpacing[d] = m.numBins[d] / (m.maxVal[d] - m.origin[d]);
 m.actualPoints[d] = m.numBins[d] + (m.isPeriodic[d] ? 0 : 1);
 }
 m.strides[0] = 1;
 for (int d = 1; d < m.D; d++)
 m.strides[d] = m.strides[d-1] * m.actualPoints[d-1];
 m.totalGridPoints = m.strides[m.D-1] * m.actualPoints[m.D-1];

 int gridOffset = 2 * m.D;
 m.gridCPU.assign(bParams.begin() + gridOffset,
 bParams.begin() + gridOffset + m.totalGridPoints);

 m.cvIdxGPU.initialize<int>(cc_, m.D, "extBias_cvIdx_" + to_string(bIdx));
 m.cvIdxGPU.upload(vector<int>(cvIndices.begin(), cvIndices.end()));
 m.grid.initialize<double>(cc_, m.totalGridPoints, "extBias_grid_" + to_string(bIdx));
 m.grid.upload(m.gridCPU);
 m.originGPU.initialize<double>(cc_, m.D, "extBias_origin_" + to_string(bIdx));
 m.originGPU.upload(m.origin);
 m.invSpacingGPU.initialize<double>(cc_, m.D, "extBias_invSp_" + to_string(bIdx));
 m.invSpacingGPU.upload(m.invSpacing);
 m.actualPointsGPU.initialize<int>(cc_, m.D, "extBias_npts_" + to_string(bIdx));
 m.actualPointsGPU.upload(vector<int>(m.actualPoints.begin(), m.actualPoints.end()));
 m.stridesGPU.initialize<int>(cc_, m.D, "extBias_strides_" + to_string(bIdx));
 m.stridesGPU.upload(vector<int>(m.strides.begin(), m.strides.end()));
 m.periodicGPU.initialize<int>(cc_, m.D, "extBias_periodic_" + to_string(bIdx));
 m.periodicGPU.upload(vector<int>(m.isPeriodic.begin(), m.isPeriodic.end()));
 m.biasEnergyIdx = biasEnergyCounter++;

 ComputeProgram mdProg = cc_.compileProgram(kMetaDKernelSrc, {});
 m.evalKernel = mdProg->createKernel("metaDEvalBias");
 m.evalKernel->addArg(plan_.cvValues); // 0
 m.evalKernel->addArg(m.cvIdxGPU); // 1
 m.evalKernel->addArg(m.grid); // 2
 m.evalKernel->addArg(m.originGPU); // 3
 m.evalKernel->addArg(m.invSpacingGPU); // 4
 m.evalKernel->addArg(m.actualPointsGPU); // 5
 m.evalKernel->addArg(m.stridesGPU); // 6
 m.evalKernel->addArg(m.periodicGPU); // 7
 m.evalKernel->addArg(m.D); // 8
 m.evalKernel->addArg(plan_.cvBiasGradients); // 9
 m.evalKernel->addArg(plan_.biasEnergies); // 10
 m.evalKernel->addArg(m.biasEnergyIdx); // 11
 m.evalKernel->addArg(); // 12: spare

 } else if (bType == GluedForce::BIAS_OPES) {
 opesBiases_.emplace_back();
 OpesBias& o = opesBiases_.back();
 o.numCVsBias = (int)cvIndices.size();
 o.cvIdxGlobal = cvIndices;
 o.kT = bParams[0];
 o.gamma = bParams[1];
 for (int d = 0; d < o.numCVsBias; d++)
 o.sigma0.push_back(bParams[2 + d]);
 o.sigmaMin = bParams[2 + o.numCVsBias];
 o.variant = bIntParams.size() > 0 ? bIntParams[0] : 0;
 o.pace = bIntParams.size() > 1 ? bIntParams[1] : 500;
 o.maxKernels = bIntParams.size() > 2 ? bIntParams[2] : 100000;
 // sigma0 <= 0 signals fully-adaptive mode (no fixed sigma provided).
 // adaptiveSigmaStride may be overridden via bIntParams[3]; defaults to 10*pace.
 bool adaptiveMode = (o.sigma0[0] <= 0.0);
 o.adaptiveSigmaStride = adaptiveMode
 ? (bIntParams.size() > 3 ? bIntParams[3] : 10 * o.pace)
 : 0;

 // Staging buffers (for serialization only, not kept in sync during simulation).
 o.runningMean.assign(o.numCVsBias, 0.0);
 o.runningM2.assign(o.numCVsBias, 0.0);

 // Pre-allocate full CPU arrays for serialization downloads.
 o.kernelCentersCPU.assign(o.maxKernels * o.numCVsBias, 0.0f);
 o.kernelSigmasCPU.assign(o.maxKernels * o.numCVsBias, 1.0f);
 o.kernelLogWeightsCPU.assign(o.maxKernels, 0.0f);

 o.cvIdxGPU.initialize<int>(cc_, o.numCVsBias,
 "opesBias_cvIdx_" + to_string(bIdx));
 o.cvIdxGPU.upload(vector<int>(cvIndices.begin(), cvIndices.end()));

 int D = o.numCVsBias;
 o.kernelCenters.initialize<float>(cc_, o.maxKernels * D,
 "opesBias_centers_" + to_string(bIdx));
 o.kernelSigmas.initialize<float>(cc_, o.maxKernels * D,
 "opesBias_sigmas_" + to_string(bIdx));
 o.kernelLogWeights.initialize<float>(cc_, o.maxKernels,
 "opesBias_logW_" + to_string(bIdx));

 // GPU-resident Welford state and logZ
 o.runningMeanGPU.initialize<double>(cc_, D, "opesBias_runMean_" + to_string(bIdx));
 o.runningMeanGPU.upload(o.runningMean);
 o.runningM2GPU.initialize<double>(cc_, D, "opesBias_runM2_" + to_string(bIdx));
 o.runningM2GPU.upload(o.runningM2);
 o.nSamplesGPU.initialize<int>(cc_, 1, "opesBias_nSamp_" + to_string(bIdx));
 o.nSamplesGPU.upload(vector<int>{0});
 o.sigma0GPU.initialize<double>(cc_, D, "opesBias_sigma0_" + to_string(bIdx));
 o.sigma0GPU.upload(o.sigma0);
 o.logZGPU.initialize<double>(cc_, 1, "opesBias_logZ_" + to_string(bIdx));
 o.logZGPU.upload(vector<double>{0.0}); // sum_uprob = 0 until first deposit
 // Linear weight accumulators {sum_w, sum_w2}.
 // Sentinel = exp(-gamma): ensures neff≈1 for early deposits.
 // sumW[0] = sum_w (Σ exp(V/kT)), sumW[1] = sum_w2 (Σ exp(2V/kT)).
 o.logSumWGPU.initialize<double>(cc_, 2, "opesBias_sumW_" + to_string(bIdx));
 o.logSumWGPU.upload(vector<double>{exp(-o.gamma), exp(-2.0*o.gamma)});
 o.stepCountGPU.initialize<int>(cc_, 1, "opesBias_stepCnt_" + to_string(bIdx));
 o.stepCountGPU.upload(vector<int>{0});

 o.biasEnergyIdx = biasEnergyCounter++;

 double invGF = (o.variant == 1) ? 1.0
 : (o.gamma - 1.0) / o.gamma;

 ComputeProgram oProg = cc_.compileProgram(kOPESKernelSrc, {});
 o.evalKernel = oProg->createKernel("opesEvalBias");
 o.numKernelsGPU.initialize<int>(cc_, 1, "numKernelsGPU_" + to_string(bIdx));
 o.numKernelsGPU.upload(vector<int>(1, 0));
 // B2 multiwalker slot-claim counter (must be initialized alongside numKernelsGPU).
 o.numAllocatedGPU.initialize<int>(cc_, 1, "numAllocatedGPU_" + to_string(bIdx));
 o.numAllocatedGPU.upload(vector<int>{0});
 // KDNorm initial value = exp(-gamma) sentinel (ensures no zero-division before first deposit).
 o.sumWeightsGPU.initialize<double>(cc_, 1, "opesBias_sumW_" + to_string(bIdx));
 o.sumWeightsGPU.upload(vector<double>{exp(-o.gamma)});
 // cutoff2 = 2*gamma/invGF = 2*gamma^2/(gamma-1)
 double cutoff2 = (invGF > 0.0) ? 2.0 * o.gamma / invGF : 2.0 * o.gamma;

 o.evalKernel->addArg(plan_.cvValues); // 0
 o.evalKernel->addArg(o.cvIdxGPU); // 1
 o.evalKernel->addArg(o.kernelCenters); // 2
 o.evalKernel->addArg(o.kernelSigmas); // 3
 o.evalKernel->addArg(o.kernelLogWeights); // 4
 o.evalKernel->addArg(o.numKernelsGPU); // 5
 o.evalKernel->addArg(D); // 6: numCVsBias
 o.evalKernel->addArg(invGF); // 7: invGammaFactor
 o.evalKernel->addArg(o.kT); // 8: kT
 o.evalKernel->addArg(o.logZGPU); // 9: logSumPairwiseGPU
 o.evalKernel->addArg(o.sumWeightsGPU); // 10: KDNorm (Σ h_k)
 o.evalKernel->addArg(cutoff2); // 11: 2*gamma/(gamma-1)
 o.evalKernel->addArg(plan_.cvBiasGradients); // 12
 o.evalKernel->addArg(plan_.biasEnergies); // 13
 o.evalKernel->addArg(o.biasEnergyIdx); // 14

 o.gatherDepositKernel = oProg->createKernel("opesGatherDeposit");
 o.gatherDepositKernel->addArg(plan_.cvValues); // 0
 o.gatherDepositKernel->addArg(o.cvIdxGPU); // 1
 o.gatherDepositKernel->addArg(o.runningMeanGPU); // 2
 o.gatherDepositKernel->addArg(o.runningM2GPU); // 3
 o.gatherDepositKernel->addArg(o.nSamplesGPU); // 4
 o.gatherDepositKernel->addArg(o.sigma0GPU); // 5
 o.gatherDepositKernel->addArg(o.sigmaMin); // 6
 o.gatherDepositKernel->addArg(o.kernelCenters); // 7
 o.gatherDepositKernel->addArg(o.kernelSigmas); // 8
 o.gatherDepositKernel->addArg(o.kernelLogWeights); // 9
 o.gatherDepositKernel->addArg(o.logZGPU); // 10: logSumPairwiseGPU
 o.gatherDepositKernel->addArg(o.numKernelsGPU); // 11
 o.gatherDepositKernel->addArg(D); // 12
 o.gatherDepositKernel->addArg(o.variant); // 13
 o.gatherDepositKernel->addArg(o.adaptiveSigmaStride); // 14
 o.gatherDepositKernel->addArg(plan_.biasEnergies); // 15
 o.gatherDepositKernel->addArg(o.biasEnergyIdx); // 16
 o.gatherDepositKernel->addArg(1.0 / o.kT); // 17: 1/kT — weight = exp(V/kT)
 o.gatherDepositKernel->addArg(o.sumWeightsGPU); // 18: KDNorm
 o.gatherDepositKernel->addArg(cutoff2); // 19: 2*gamma/(gamma-1)
 o.gatherDepositKernel->addArg(o.logSumWGPU); // 20: {sum_w, sum_w2} linear accumulators
 o.gatherDepositKernel->addArg(o.stepCountGPU); // 21: step count for neff
 o.gatherDepositKernel->addArg(o.numAllocatedGPU); // 22: B2 multiwalker slot-claim counter
 o.gatherDepositKernel->addArg(o.maxKernels); // 23: buffer capacity limit

 // In fully-adaptive mode, compile the every-step Welford accumulator kernel.
 // It runs in execute() (after CV kernels) and manages nSamples/runningMean/runningM2;
 // gatherDepositKernel then reads the pre-built variance at deposition time.
 if (adaptiveMode) {
 o.welfordKernel = oProg->createKernel("opesAccumulateWelford");
 o.welfordKernel->addArg(plan_.cvValues); // 0
 o.welfordKernel->addArg(o.cvIdxGPU); // 1
 o.welfordKernel->addArg(o.runningMeanGPU); // 2
 o.welfordKernel->addArg(o.runningM2GPU); // 3
 o.welfordKernel->addArg(o.nSamplesGPU); // 4
 o.welfordKernel->addArg(D); // 5
 }

 o.neffKernel = oProg->createKernel("opesUpdateNeff");
 o.neffKernel->addArg(plan_.biasEnergies); // 0
 o.neffKernel->addArg(o.biasEnergyIdx); // 1
 o.neffKernel->addArg(1.0 / o.kT); // 2: invKT
 o.neffKernel->addArg(o.logSumWGPU); // 3
 o.neffKernel->addArg(o.stepCountGPU); // 4

 } else if (bType == GluedForce::BIAS_LINEAR) {
 linearBiases_.emplace_back();
 LinearBias& lin = linearBiases_.back();
 lin.numCVsBias = (int)cvIndices.size();
 lin.cvIdxGlobal = cvIndices;
 int D = lin.numCVsBias;
 lin.cvIdxGPU.initialize<int>(cc_, D, "linBias_cvIdx_" + to_string(bIdx));
 lin.cvIdxGPU.upload(vector<int>(cvIndices.begin(), cvIndices.end()));
 vector<float> kVec;
 for (int d = 0; d < D; d++) kVec.push_back((float)bParams[d]);
 lin.params.initialize<float>(cc_, D, "linBias_params_" + to_string(bIdx));
 lin.params.upload(kVec);
 lin.biasEnergyIdx = biasEnergyCounter++;
 ComputeProgram linProg = cc_.compileProgram(kLinearKernelSrc, {});
 lin.evalKernel = linProg->createKernel("linearEvalBias");
 lin.evalKernel->addArg(plan_.cvValues); // 0
 lin.evalKernel->addArg(lin.cvIdxGPU); // 1
 lin.evalKernel->addArg(lin.params); // 2
 lin.evalKernel->addArg(D); // 3
 lin.evalKernel->addArg(plan_.cvBiasGradients); // 4
 lin.evalKernel->addArg(plan_.biasEnergies); // 5
 lin.evalKernel->addArg(lin.biasEnergyIdx); // 6
 lin.evalKernel->addArg(); // 7: spare (8.5.1 fix)

 } else if (bType == GluedForce::BIAS_UPPER_WALL ||
 bType == GluedForce::BIAS_LOWER_WALL) {
 wallBiases_.emplace_back();
 WallBias& wall = wallBiases_.back();
 wall.numCVsBias = (int)cvIndices.size();
 wall.wallType = (bType == GluedForce::BIAS_UPPER_WALL) ? 0 : 1;
 wall.cvIdxGlobal = cvIndices;
 int D = wall.numCVsBias;
 wall.cvIdxGPU.initialize<int>(cc_, D, "wallBias_cvIdx_" + to_string(bIdx));
 wall.cvIdxGPU.upload(vector<int>(cvIndices.begin(), cvIndices.end()));
 // params = [at_0, kappa_0, eps_0, n_0, at_1, ...] (4 per dim)
 vector<float> pVec;
 for (int d = 0; d < D; d++) {
 pVec.push_back((float)bParams[4*d]);
 pVec.push_back((float)bParams[4*d + 1]);
 pVec.push_back((float)bParams[4*d + 2]);
 pVec.push_back((float)bParams[4*d + 3]);
 }
 wall.params.initialize<float>(cc_, 4*D, "wallBias_params_" + to_string(bIdx));
 wall.params.upload(pVec);
 wall.biasEnergyIdx = biasEnergyCounter++;
 ComputeProgram wallProg = cc_.compileProgram(kWallKernelSrc, {});
 wall.evalKernel = wallProg->createKernel("wallEvalBias");
 wall.evalKernel->addArg(plan_.cvValues); // 0
 wall.evalKernel->addArg(wall.cvIdxGPU); // 1
 wall.evalKernel->addArg(wall.params); // 2
 wall.evalKernel->addArg(D); // 3
 wall.evalKernel->addArg(wall.wallType); // 4
 wall.evalKernel->addArg(plan_.cvBiasGradients); // 5
 wall.evalKernel->addArg(plan_.biasEnergies); // 6
 wall.evalKernel->addArg(wall.biasEnergyIdx); // 7
 wall.evalKernel->addArg(); // 8: spare (8.5.1 fix)

 } else if (bType == GluedForce::BIAS_OPES_EXPANDED) {
 opesExpandedBiases_.emplace_back();
 OpesExpandedBias& oe = opesExpandedBiases_.back();
 int D = (int)cvIndices.size();
 oe.numStates = D;
 oe.cvIdxGlobal = cvIndices;
 oe.kT = bParams[0];
 oe.invKT = 1.0 / oe.kT;
 oe.pace = (bIntParams.size() > 0) ? bIntParams[0] : 500;

 // Normalize weights and convert to log space
 double wSum = 0.0;
 for (int l = 0; l < D; l++) wSum += bParams[1 + l];
 if (wSum <= 0.0) wSum = D; // fallback: uniform
 oe.logWeightsCPU.resize(D);
 for (int l = 0; l < D; l++)
 oe.logWeightsCPU[l] = std::log(bParams[1 + l] / wSum);

 oe.cvIdxGPU.initialize<int>(cc_, D,
 "opesExpBias_cvIdx_" + to_string(bIdx));
 oe.cvIdxGPU.upload(vector<int>(cvIndices.begin(), cvIndices.end()));
 oe.logWeightsGPU.initialize<double>(cc_, D,
 "opesExpBias_logW_" + to_string(bIdx));
 oe.logWeightsGPU.upload(oe.logWeightsCPU);
 oe.logZGPU.initialize<double>(cc_, 1,
 "opesExpBias_logZ_" + to_string(bIdx));
 oe.logZGPU.upload(vector<double>{0.0});
 oe.numUpdatesGPU.initialize<int>(cc_, 1,
 "opesExpBias_nUpd_" + to_string(bIdx));
 oe.numUpdatesGPU.upload(vector<int>{0});
 oe.biasEnergyIdx = biasEnergyCounter++;

 ComputeProgram oeProg = cc_.compileProgram(kOPESExpandedKernelSrc, {});
 oe.evalKernel = oeProg->createKernel("opesExpandedEvalBias");
 oe.evalKernel->addArg(plan_.cvValues); // 0
 oe.evalKernel->addArg(oe.cvIdxGPU); // 1
 oe.evalKernel->addArg(oe.logWeightsGPU); // 2
 oe.evalKernel->addArg(D); // 3: numStates
 oe.evalKernel->addArg(oe.invKT); // 4: invKT
 oe.evalKernel->addArg(oe.logZGPU); // 5: logZGPU (updated by updateLogZKernel)
 oe.evalKernel->addArg(plan_.cvBiasGradients); // 6
 oe.evalKernel->addArg(plan_.biasEnergies); // 7
 oe.evalKernel->addArg(oe.biasEnergyIdx); // 8
 oe.evalKernel->addArg(); // 9: spare (8.5.1 fix)

 oe.updateLogZKernel = oeProg->createKernel("opesExpandedUpdateLogZ");
 oe.updateLogZKernel->addArg(plan_.cvValues); // 0
 oe.updateLogZKernel->addArg(oe.cvIdxGPU); // 1
 oe.updateLogZKernel->addArg(oe.logWeightsGPU); // 2
 oe.updateLogZKernel->addArg(oe.logZGPU); // 3
 oe.updateLogZKernel->addArg(oe.numUpdatesGPU); // 4
 oe.updateLogZKernel->addArg(D); // 5: numStates
 oe.updateLogZKernel->addArg(oe.invKT); // 6

 } else if (bType == GluedForce::BIAS_EXT_LAGRANGIAN) {
 // Extended Lagrangian / AFED
 extLagBiases_.emplace_back();
 ExtLagBias& el = extLagBiases_.back();
 int D = (int)cvIndices.size();
 el.cvIndices.assign(cvIndices.begin(), cvIndices.end());
 for (int i = 0; i < D; i++) {
 el.kappa.push_back(bParams[2*i]);
 el.massS.push_back(bParams[2*i + 1]);
 }
 el.s.assign(D, 0.0);
 el.p.assign(D, 0.0);
 el.initialized = false;

 el.cvIdxGPU.initialize<int>(cc_, D,
 "elBias_cvIdx_" + to_string(bIdx));
 el.cvIdxGPU.upload(vector<int>(cvIndices.begin(), cvIndices.end()));
 el.kappaGPU.initialize<double>(cc_, D,
 "elBias_kappa_" + to_string(bIdx));
 el.kappaGPU.upload(el.kappa);
 el.massSGPU.initialize<double>(cc_, D,
 "elBias_massS_" + to_string(bIdx));
 el.massSGPU.upload(el.massS);
 el.sGPUArr.initialize<double>(cc_, D,
 "elBias_s_" + to_string(bIdx));
 el.sGPUArr.upload(el.s); // zeros; initSKernel sets real values at step 0
 el.pGPUArr.initialize<double>(cc_, D,
 "elBias_p_" + to_string(bIdx));
 el.pGPUArr.upload(el.p); // zeros
 el.biasEnergyIdx = biasEnergyCounter++;

 ComputeProgram elProg = cc_.compileProgram(kExtLagKernelSrc, {});

 el.initSKernel = elProg->createKernel("extLagInitS");
 el.initSKernel->addArg(plan_.cvValues); // 0
 el.initSKernel->addArg(el.cvIdxGPU); // 1
 el.initSKernel->addArg(el.sGPUArr); // 2
 el.initSKernel->addArg(D); // 3
 el.initSKernel->addArg(); // 4: spare

 el.verletKernel = elProg->createKernel("extLagVerlet");
 el.verletKernel->addArg(plan_.cvValues); // 0
 el.verletKernel->addArg(el.cvIdxGPU); // 1
 el.verletKernel->addArg(el.kappaGPU); // 2
 el.verletKernel->addArg(el.massSGPU); // 3
 el.verletKernel->addArg(el.sGPUArr); // 4
 el.verletKernel->addArg(el.pGPUArr); // 5
 el.verletKernel->addArg(0.0); // 6: dt (setArg per step)
 el.verletKernel->addArg(D); // 7
 el.verletKernel->addArg(); // 8: spare

 el.evalKernel = elProg->createKernel("extLagEvalBias");
 el.evalKernel->addArg(plan_.cvValues); // 0
 el.evalKernel->addArg(el.cvIdxGPU); // 1
 el.evalKernel->addArg(el.kappaGPU); // 2
 el.evalKernel->addArg(el.sGPUArr); // 3
 el.evalKernel->addArg(D); // 4
 el.evalKernel->addArg(plan_.cvBiasGradients); // 5
 el.evalKernel->addArg(plan_.biasEnergies); // 6
 el.evalKernel->addArg(el.biasEnergyIdx); // 7
 el.evalKernel->addArg(); // 8: spare (8.5.1 fix)

 } else if (bType == GluedForce::BIAS_EDS) {
 // EDS White-Voth: bParams = [target_0, max_range_0, ..., kbt]
 // bIntParams = [pace]
 edsBiases_.emplace_back();
 EdsBias& eds = edsBiases_.back();
 int D = (int)cvIndices.size();
 eds.cvIndices.assign(cvIndices.begin(), cvIndices.end());
 // params layout: D pairs of (target, max_range), then kbt
 for (int i = 0; i < D; i++) {
 eds.target.push_back(bParams[2*i]);
 eds.max_range.push_back(bParams[2*i + 1]);
 }
 eds.kbt = (bParams.size() > (size_t)(2*D)) ? bParams[2*D] : 2.479;  // default ~300K
 eds.lambda.assign(D, 0.0);
 eds.pace = (bIntParams.size() > 0) ? bIntParams[0] : 1;
 eds.initialized = true;  // no init kernel needed

 vector<double> zeros(D, 0.0);
 vector<int> izeros(D, 0);

 eds.cvIdxGPU.initialize<int>(cc_, D, "edsBias_cvIdx_" + to_string(bIdx));
 eds.cvIdxGPU.upload(vector<int>(cvIndices.begin(), cvIndices.end()));
 eds.lambdaGPU.initialize<double>(cc_, D, "edsBias_lambda_" + to_string(bIdx));
 eds.lambdaGPU.upload(zeros);
 eds.meanGPU.initialize<double>(cc_, D, "edsBias_mean_" + to_string(bIdx));
 eds.meanGPU.upload(zeros);
 eds.ssdGPU.initialize<double>(cc_, D, "edsBias_ssd_" + to_string(bIdx));
 eds.ssdGPU.upload(zeros);
 eds.accumGPU.initialize<double>(cc_, D, "edsBias_accum_" + to_string(bIdx));
 eds.accumGPU.upload(zeros);
 eds.countGPU.initialize<int>(cc_, D, "edsBias_count_" + to_string(bIdx));
 eds.countGPU.upload(izeros);
 eds.targetGPU.initialize<double>(cc_, D, "edsBias_target_" + to_string(bIdx));
 eds.targetGPU.upload(eds.target);
 eds.maxRangeGPU.initialize<double>(cc_, D, "edsBias_maxRange_" + to_string(bIdx));
 eds.maxRangeGPU.upload(eds.max_range);
 eds.biasEnergyIdx = biasEnergyCounter++;

 ComputeProgram edsProg = cc_.compileProgram(kEdsKernelSrc, {});

 eds.updateStateKernel = edsProg->createKernel("edsWVUpdateState");
 eds.updateStateKernel->addArg(plan_.cvValues);   // 0
 eds.updateStateKernel->addArg(eds.cvIdxGPU);     // 1
 eds.updateStateKernel->addArg(eds.meanGPU);      // 2
 eds.updateStateKernel->addArg(eds.ssdGPU);       // 3
 eds.updateStateKernel->addArg(eds.lambdaGPU);    // 4
 eds.updateStateKernel->addArg(eds.accumGPU);     // 5
 eds.updateStateKernel->addArg(eds.countGPU);     // 6
 eds.updateStateKernel->addArg(eds.targetGPU);    // 7
 eds.updateStateKernel->addArg(eds.maxRangeGPU);  // 8
 eds.updateStateKernel->addArg(eds.kbt);          // 9: constant double
 eds.updateStateKernel->addArg(0);                // 10: doUpdate (setArg per step)
 eds.updateStateKernel->addArg(D);                // 11
 eds.updateStateKernel->addArg();                 // 12: spare

 eds.evalKernel = edsProg->createKernel("edsEvalBias");
 eds.evalKernel->addArg(plan_.cvValues);          // 0
 eds.evalKernel->addArg(eds.cvIdxGPU);            // 1
 eds.evalKernel->addArg(eds.lambdaGPU);           // 2
 eds.evalKernel->addArg(D);                       // 3
 eds.evalKernel->addArg(plan_.cvBiasGradients);   // 4
 eds.evalKernel->addArg(plan_.biasEnergies);      // 5
 eds.evalKernel->addArg(eds.biasEnergyIdx);       // 6
 eds.evalKernel->addArg();                        // 7: spare

 } else if (bType == GluedForce::BIAS_MAXENT) {
 // MaxEnt: params = [kbt, sigma, alpha, at_0, kappa_0, tau_0, at_1, ...]
 //         intParams = [pace, type, errorType]
 // type: 0=EQUAL, 1=INEQUAL_GT, 2=INEQUAL_LT
 // errorType: 0=GAUSSIAN, 1=LAPLACE
 maxentBiases_.emplace_back();
 MaxentBias& mx = maxentBiases_.back();
 int D = (int)cvIndices.size();
 mx.cvIndices.assign(cvIndices.begin(), cvIndices.end());
 mx.kbt   = (bParams.size() > 0) ? bParams[0] : 2.479;
 mx.sigma = (bParams.size() > 1) ? bParams[1] : 0.0;
 mx.alpha = (bParams.size() > 2) ? bParams[2] : 1.0;
 // Per-CV params: groups of 3 starting at offset 3
 for (int i = 0; i < D; i++) {
  int base = 3 + 3 * i;
  mx.at.push_back   ((bParams.size() > (size_t)(base+0)) ? bParams[base+0] : 0.0);
  mx.kappa.push_back((bParams.size() > (size_t)(base+1)) ? bParams[base+1] : 0.01);
  mx.tau.push_back  ((bParams.size() > (size_t)(base+2)) ? bParams[base+2] : 1.0);
 }
 mx.lambda.assign(D, 0.0);
 mx.pace      = (bIntParams.size() > 0) ? bIntParams[0] : 1;
 mx.type      = (bIntParams.size() > 1) ? bIntParams[1] : 0;
 mx.errorType = (bIntParams.size() > 2) ? bIntParams[2] : 0;
 mx.biasEnergyIdx = biasEnergyCounter++;

 vector<double> zeros(D, 0.0);
 mx.cvIdxGPU.initialize<int>   (cc_, D, "maxent_cvIdx_"  + to_string(bIdx));
 mx.cvIdxGPU.upload(vector<int>(cvIndices.begin(), cvIndices.end()));
 mx.lambdaGPU.initialize<double>(cc_, D, "maxent_lambda_" + to_string(bIdx));
 mx.lambdaGPU.upload(zeros);
 mx.atGPU.initialize<double>   (cc_, D, "maxent_at_"     + to_string(bIdx));
 mx.atGPU.upload(mx.at);
 mx.kappaGPU.initialize<double>(cc_, D, "maxent_kappa_"  + to_string(bIdx));
 mx.kappaGPU.upload(mx.kappa);
 mx.tauGPU.initialize<double>  (cc_, D, "maxent_tau_"    + to_string(bIdx));
 mx.tauGPU.upload(mx.tau);

 ComputeProgram mxProg = cc_.compileProgram(kMaxEntKernelSrc, {});

 mx.updateKernel = mxProg->createKernel("maxentUpdateLambda");
 mx.updateKernel->addArg(plan_.cvValues);  // 0
 mx.updateKernel->addArg(mx.cvIdxGPU);    // 1
 mx.updateKernel->addArg(mx.lambdaGPU);   // 2
 mx.updateKernel->addArg(mx.atGPU);       // 3
 mx.updateKernel->addArg(mx.kappaGPU);    // 4
 mx.updateKernel->addArg(mx.tauGPU);      // 5
 mx.updateKernel->addArg(mx.sigma);       // 6: constant double
 mx.updateKernel->addArg(mx.alpha);       // 7: constant double
 mx.updateKernel->addArg(D);              // 8
 mx.updateKernel->addArg(mx.type);        // 9
 mx.updateKernel->addArg(mx.errorType);   // 10
 mx.updateKernel->addArg(0);              // 11: updateCount (setArg per update)
 mx.updateKernel->addArg();               // 12: spare

 mx.evalKernel = mxProg->createKernel("maxentEvalBias");
 mx.evalKernel->addArg(plan_.cvValues);         // 0
 mx.evalKernel->addArg(mx.cvIdxGPU);            // 1
 mx.evalKernel->addArg(mx.lambdaGPU);           // 2
 mx.evalKernel->addArg(mx.kbt);                 // 3: constant double
 mx.evalKernel->addArg(D);                      // 4
 mx.evalKernel->addArg(mx.type);                // 5
 mx.evalKernel->addArg(plan_.cvBiasGradients);  // 6
 mx.evalKernel->addArg(plan_.biasEnergies);     // 7
 mx.evalKernel->addArg(mx.biasEnergyIdx);       // 8
 mx.evalKernel->addArg();                       // 9: spare
 }
 }

 // Chain-rule scatter kernel
 if (plan_.numJacEntries > 0) {
 ComputeProgram scatProg = cc_.compileProgram(kScatterKernelSrc, {});
 scatterKernel_ = scatProg->createKernel("chainRuleScatter");
 scatterKernel_->addArg(plan_.cvBiasGradients); // 0
 scatterKernel_->addArg(plan_.jacobianAtomIdx); // 1
 scatterKernel_->addArg(plan_.jacobianGradsX); // 2
 scatterKernel_->addArg(plan_.jacobianGradsY); // 3
 scatterKernel_->addArg(plan_.jacobianGradsZ); // 4
 scatterKernel_->addArg(plan_.jacobianCvIdx); // 5
 scatterKernel_->addArg(cc_.getLongForceBuffer()); // 6
 scatterKernel_->addArg(cc_.getPaddedNumAtoms()); // 7
 scatterKernel_->addArg(plan_.numJacEntries); // 8 (constant)
 scatterKernel_->addArg(); // 9: spare (8.5.1 primitiveArgs/arrayArgs realignment)
 }
}
