#include "CommonGluedKernels.h"
#include "GluedForce.h"
#include "openmm/common/ContextSelector.h"
#include "openmm/common/ExpressionUtilities.h"
#include "lepton/Parser.h"
#include "lepton/ParsedExpression.h"
#include <cmath>
#include <map>
#include <string>
#include <utility>
#include <vector>

#ifdef GLUED_HAS_TORCH
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include <torch/script.h>
#pragma GCC diagnostic pop
#endif

using namespace GluedPlugin;
using namespace OpenMM;
using namespace std;

// Kernel source strings defined in CommonGluedKernels.cpp
extern const string kDistanceKernelSrc;
extern const string kAngleKernelSrc;
extern const string kDihedralKernelSrc;
extern const string kCOMDistanceKernelSrc;
extern const string kGyrationKernelSrc;
extern const string kPathKernelSrc;
extern const string kRMSDKernelSrc;
extern const string kCoordinationKernelSrc;
extern const string kPyTorchKernelSrc;
extern const string kPositionKernelSrc;
extern const string kDRMSDKernelSrc;
extern const string kContactMapKernelSrc;
extern const string kPlaneKernelSrc;
extern const string kProjectionKernelSrc;
extern const string kVolumeCellKernelSrc;
extern const string kDipoleKernelSrc;
extern const string kPCAKernelSrc;
extern const string kErmsdKernelSrc;
extern const string kPuckeringKernelSrc;
extern const string kSecondaryStructureKernelSrc;

// ---------------------------------------------------------------------------
// Generate the propagation kernel source for an expression CV.
// The kernel name is unique per expression CV to avoid NVRTC name conflicts
// when multiple expression CVs are present in the same simulation.
// ---------------------------------------------------------------------------
static string genExpressionPropKernelSrc(int uniqueId) {
 string uid = to_string(uniqueId);
 return R"(
KERNEL void cvExprProp_)" + uid + R"((
 GLOBAL const double* RESTRICT partials,
 GLOBAL const int* RESTRICT inputCVIdx,
 int D,
 int outputCVIdx,
 GLOBAL double* RESTRICT cvBiasGradients
) {
 if (GLOBAL_ID != 0) return;
 double g = cvBiasGradients[outputCVIdx];
 for (int d = 0; d < D; d++)
 cvBiasGradients[inputCVIdx[d]] += g * partials[d];
}
)";
}

// ---------------------------------------------------------------------------
// Generate the eval kernel source for an expression CV.
// The kernel runs on a single thread, reads cvValues for inputs, computes the
// expression value and all partial derivatives using Lepton, and stores them.
// ---------------------------------------------------------------------------
static string genExpressionEvalKernelSrc(OpenMM::ComputeContext& cc,
 const string& exprStr,
 int D, int uniqueId) {
 using Lepton::Parser;
 using Lepton::ParsedExpression;
 map<string, ParsedExpression> exprs;
 exprs["double cv_out = "] = Parser::parse(exprStr).optimize();
 for (int d = 0; d < D; d++) {
 exprs["double cv_pd" + to_string(d) + " = "] =
 Parser::parse(exprStr).differentiate("cv" + to_string(d)).optimize();
 }
 map<string, string> vars;
 for (int d = 0; d < D; d++)
 vars["cv" + to_string(d)] = "(double)cvValues[inputCVIdx[" + to_string(d) + "]]";
 string uid = to_string(uniqueId);
 vector<const TabulatedFunction*> noFunctions;
 vector<pair<string,string>> noFunctionNames;
 string body = cc.getExpressionUtilities().createExpressions(
 exprs, vars, noFunctions, noFunctionNames, "ev" + uid + "_", "double");
 string src = "KERNEL void cvExprEval_" + uid + "(\n"
 + " GLOBAL double* RESTRICT cvValues,\n"
 + " GLOBAL const int* RESTRICT inputCVIdx,\n"
 + " int outputCVIdx,\n"
 + " GLOBAL double* RESTRICT partials\n"
 + ") {\n if (GLOBAL_ID != 0) return;\n"
 + body
 + " cvValues[outputCVIdx] = cv_out;\n";
 for (int d = 0; d < D; d++)
 src += " partials[" + to_string(d) + "] = cv_pd" + to_string(d) + ";\n";
 src += "}\n";
 return src;
}

// ---------------------------------------------------------------------------
// Rebuild GPU-space atom index pairs from the user-space pairs in
// distanceUserAtoms_. Must be called after buildPlan() and whenever
// cc_.getAtomsWereReordered() returns true.
// ---------------------------------------------------------------------------

void CommonCalcGluedForceKernel::rebuildGpuAtomIndices() {
 int N = cc_.getNumAtoms();
 if (N == 0) return;

 vector<int> gpuToUser(cc_.getPaddedNumAtoms());
 cc_.getAtomIndexArray().download(gpuToUser);
 vector<int> userToGpu(N, -1);
 for (int g = 0; g < N; g++) {
 int u = gpuToUser[g];
 if (u >= 0 && u < N) userToGpu[u] = g;
 }

 if (plan_.numDistanceCVs > 0) {
 vector<int> pairs(2 * plan_.numDistanceCVs);
 for (int i = 0; i < plan_.numDistanceCVs; i++) {
 pairs[2*i] = userToGpu[distanceUserAtoms_[2*i]];
 pairs[2*i+1] = userToGpu[distanceUserAtoms_[2*i+1]];
 }
 plan_.distanceAtoms.upload(pairs);
 }
 if (plan_.numAngleCVs > 0) {
 vector<int> triplets(3 * plan_.numAngleCVs);
 for (int i = 0; i < plan_.numAngleCVs; i++) {
 triplets[3*i] = userToGpu[angleUserAtoms_[3*i]];
 triplets[3*i+1] = userToGpu[angleUserAtoms_[3*i+1]];
 triplets[3*i+2] = userToGpu[angleUserAtoms_[3*i+2]];
 }
 plan_.angleAtoms.upload(triplets);
 }
 if (plan_.numDihedralCVs > 0) {
 vector<int> quads(4 * plan_.numDihedralCVs);
 for (int i = 0; i < plan_.numDihedralCVs; i++) {
 quads[4*i] = userToGpu[dihedralUserAtoms_[4*i]];
 quads[4*i+1] = userToGpu[dihedralUserAtoms_[4*i+1]];
 quads[4*i+2] = userToGpu[dihedralUserAtoms_[4*i+2]];
 quads[4*i+3] = userToGpu[dihedralUserAtoms_[4*i+3]];
 }
 plan_.dihedralAtoms.upload(quads);
 }
 if (plan_.numCOMDistanceCVs > 0) {
 int total = (int)comDistanceUserAtoms_.size();
 vector<int> gpuAtoms(total);
 for (int j = 0; j < total; j++)
 gpuAtoms[j] = userToGpu[comDistanceUserAtoms_[j]];
 plan_.comDistanceAtoms.upload(gpuAtoms);
 }
 if (plan_.numGyrationCVs > 0) {
 int total = (int)gyrationUserAtoms_.size();
 vector<int> gpuAtoms(total);
 for (int j = 0; j < total; j++)
 gpuAtoms[j] = userToGpu[gyrationUserAtoms_[j]];
 plan_.gyrationAtoms.upload(gpuAtoms);
 }
 if (plan_.numCoordCVs > 0) {
 int total = (int)coordUserAtoms_.size();
 vector<int> gpuAtoms(total);
 for (int j = 0; j < total; j++)
 gpuAtoms[j] = userToGpu[coordUserAtoms_[j]];
 plan_.coordAtoms.upload(gpuAtoms);
 }
 if (plan_.numRMSDCVs > 0) {
 int total = (int)rmsdUserAtoms_.size();
 vector<int> gpuAtoms(total);
 for (int j = 0; j < total; j++)
 gpuAtoms[j] = userToGpu[rmsdUserAtoms_[j]];
 plan_.rmsdAtoms.upload(gpuAtoms);
 }
 if (plan_.numPathCVs > 0) {
 int total = (int)pathUserAtoms_.size();
 vector<int> gpuAtoms(total);
 for (int j = 0; j < total; j++)
 gpuAtoms[j] = userToGpu[pathUserAtoms_[j]];
 plan_.pathAtoms.upload(gpuAtoms);
 }

 // PyTorch CVs — update GPU atom indices and pre-fill static Jacobian entries.
 // jacobianAtomIdx and jacobianCvIdx don't change between reorders (same CV, same atoms);
 // only gpuAtomIdx changes (user→GPU mapping changes after Hilbert sort).
 if (!pytorchCVPlans_.empty() && plan_.numJacEntries > 0) {
 for (auto& pt : pytorchCVPlans_) {
 // Update CPU-side GPU-space atom indices
 for (int i = 0; i < pt.numAtoms; i++)
 pt.gpuAtomIdx[i] = userToGpu[pt.userAtoms[i]];

 // Upload to GPU buffer used by the extract kernel (GPU-native path)
 if (pt.atomIdxGPU.isInitialized())
 pt.atomIdxGPU.upload(pt.gpuAtomIdx);

 // Pre-fill static Jacobian index arrays (atom index and CV index per entry)
 plan_.jacobianAtomIdx.uploadSubArray(pt.gpuAtomIdx.data(),
 pt.jacOffset, pt.numAtoms);
 vector<int> cvIdxVec(pt.numAtoms, pt.outputCVIndex);
 plan_.jacobianCvIdx.uploadSubArray(cvIdxVec.data(),
 pt.jacOffset, pt.numAtoms);
 }
 }

 // Position CVs — single atom per CV
 if (plan_.numPositionCVs > 0) {
 int N = plan_.numPositionCVs;
 vector<int> gpuAtoms(N);
 for (int i = 0; i < N; i++)
 gpuAtoms[i] = userToGpu[positionUserAtoms_[i]];
 plan_.positionAtoms.upload(gpuAtoms);
 }

 // DRMSD CVs — flat pair list
 if (plan_.numDRMSDCVs > 0) {
 int total = (int)drmsdUserPairAtoms_.size();
 vector<int> gpuAtoms(total);
 for (int j = 0; j < total; j++)
 gpuAtoms[j] = userToGpu[drmsdUserPairAtoms_[j]];
 plan_.drmsdAtomPairs.upload(gpuAtoms);
 }

 // ContactMap CVs — flat pair list
 if (plan_.numContactMapCVs > 0) {
 int total = (int)contactMapUserPairAtoms_.size();
 vector<int> gpuAtoms(total);
 for (int j = 0; j < total; j++)
 gpuAtoms[j] = userToGpu[contactMapUserPairAtoms_[j]];
 plan_.contactMapAtomPairs.upload(gpuAtoms);
 }

 // Plane CVs — 3 atoms per CV
 if (plan_.numPlaneCVs > 0) {
 int total = (int)planeUserAtoms_.size();
 vector<int> gpuAtoms(total);
 for (int j = 0; j < total; j++)
 gpuAtoms[j] = userToGpu[planeUserAtoms_[j]];
 plan_.planeAtoms.upload(gpuAtoms);
 }

 // Projection CVs — 2 atoms per CV
 if (plan_.numProjectionCVs > 0) {
 int total = (int)projectionUserAtoms_.size();
 vector<int> gpuAtoms(total);
 for (int j = 0; j < total; j++)
 gpuAtoms[j] = userToGpu[projectionUserAtoms_[j]];
 plan_.projectionAtoms.upload(gpuAtoms);
 }

 // Dipole CVs — flat atom list
 if (plan_.numDipoleCVs > 0) {
 int total = (int)dipoleUserAtoms_.size();
 vector<int> gpuAtoms(total);
 for (int j = 0; j < total; j++)
 gpuAtoms[j] = userToGpu[dipoleUserAtoms_[j]];
 plan_.dipoleAtoms.upload(gpuAtoms);
 }

 // PCA CVs — flat atom list
 if (plan_.numPCACVs > 0) {
 int total = (int)pcaUserAtoms_.size();
 vector<int> gpuAtoms(total);
 for (int j = 0; j < total; j++)
 gpuAtoms[j] = userToGpu[pcaUserAtoms_[j]];
 plan_.pcaAtoms.upload(gpuAtoms);
 }

 // Secondary structure CVs — flat atom list
 if (plan_.numSecStrCVs > 0) {
 int total = (int)secStrUserAtoms_.size();
 vector<int> gpuAtoms(total);
 for (int j = 0; j < total; j++)
 gpuAtoms[j] = userToGpu[secStrUserAtoms_[j]];
 plan_.secStrAtoms.upload(gpuAtoms);
 }

 // Puckering CVs — flat ring atom list
 if (plan_.numPuckeringCVs > 0) {
 int total = (int)puckerUserAtoms_.size();
 vector<int> gpuAtoms(total);
 for (int j = 0; j < total; j++)
 gpuAtoms[j] = userToGpu[puckerUserAtoms_[j]];
 plan_.puckerAtoms.upload(gpuAtoms);
 }

 // eRMSD CVs — flat [P1,P2,P3,...] atom list (3 per residue)
 if (plan_.numErmsdCVs > 0) {
 int total = (int)ermsdUserAtoms_.size();
 vector<int> gpuAtoms(total);
 for (int j = 0; j < total; j++)
 gpuAtoms[j] = userToGpu[ermsdUserAtoms_[j]];
 plan_.ermsdAtoms.upload(gpuAtoms);
 }
}

// ---------------------------------------------------------------------------
// initialize / execute / misc
// ---------------------------------------------------------------------------


// Compile and bind all CV evaluation kernels.
// Called from initialize() after buildPlan() has allocated the GPU arrays.
void CommonCalcGluedForceKernel::compileCVKernels(const GluedForce& force) {
 // Distance CV kernel
 if (plan_.numDistanceCVs > 0) {
 ComputeProgram distProg = cc_.compileProgram(kDistanceKernelSrc, {});
 distanceKernel_ = distProg->createKernel("cvDistance");
 distanceKernel_->addArg(cc_.getPosq()); // 0
 distanceKernel_->addArg(plan_.distanceAtoms); // 1
 distanceKernel_->addArg(plan_.numDistanceCVs); // 2
 distanceKernel_->addArg(plan_.distanceFirstCVIndex); // 3
 distanceKernel_->addArg(plan_.distanceFirstJacEntry); // 4
 distanceKernel_->addArg(plan_.cvValues); // 5
 distanceKernel_->addArg(plan_.jacobianAtomIdx); // 6
 distanceKernel_->addArg(plan_.jacobianGradsX); // 7
 distanceKernel_->addArg(plan_.jacobianGradsY); // 8
 distanceKernel_->addArg(plan_.jacobianGradsZ); // 9
 distanceKernel_->addArg(plan_.jacobianCvIdx); // 10
 distanceKernel_->addArg(); // 11-15: PBC (setPeriodicBoxArgs)
 distanceKernel_->addArg();
 distanceKernel_->addArg();
 distanceKernel_->addArg();
 distanceKernel_->addArg();
 distanceKernel_->addArg(); // 16: periodic (setArg in execute)
 distanceKernel_->addArg(); // 17: spare (8.5.1 trailing-primitive fix)
 }

 // Angle CV kernel
 if (plan_.numAngleCVs > 0) {
 ComputeProgram angProg = cc_.compileProgram(kAngleKernelSrc, {});
 angleKernel_ = angProg->createKernel("cvAngle");
 angleKernel_->addArg(cc_.getPosq()); // 0
 angleKernel_->addArg(plan_.angleAtoms); // 1
 angleKernel_->addArg(plan_.numAngleCVs); // 2
 angleKernel_->addArg(plan_.angleFirstCVIndex); // 3
 angleKernel_->addArg(plan_.angleFirstJacEntry); // 4
 angleKernel_->addArg(plan_.cvValues); // 5
 angleKernel_->addArg(plan_.jacobianAtomIdx); // 6
 angleKernel_->addArg(plan_.jacobianGradsX); // 7
 angleKernel_->addArg(plan_.jacobianGradsY); // 8
 angleKernel_->addArg(plan_.jacobianGradsZ); // 9
 angleKernel_->addArg(plan_.jacobianCvIdx); // 10
 angleKernel_->addArg(); // 11-15: PBC (setPeriodicBoxArgs)
 angleKernel_->addArg();
 angleKernel_->addArg();
 angleKernel_->addArg();
 angleKernel_->addArg();
 angleKernel_->addArg(); // 16: periodic (setArg in execute)
 angleKernel_->addArg(); // 17: spare (8.5.1 trailing-primitive fix)
 }

 // Dihedral CV kernel
 if (plan_.numDihedralCVs > 0) {
 ComputeProgram dihProg = cc_.compileProgram(kDihedralKernelSrc, {});
 dihedralKernel_ = dihProg->createKernel("cvDihedral");
 dihedralKernel_->addArg(cc_.getPosq()); // 0
 dihedralKernel_->addArg(plan_.dihedralAtoms); // 1
 dihedralKernel_->addArg(plan_.numDihedralCVs); // 2
 dihedralKernel_->addArg(plan_.dihedralFirstCVIndex); // 3
 dihedralKernel_->addArg(plan_.dihedralFirstJacEntry); // 4
 dihedralKernel_->addArg(plan_.cvValues); // 5
 dihedralKernel_->addArg(plan_.jacobianAtomIdx); // 6
 dihedralKernel_->addArg(plan_.jacobianGradsX); // 7
 dihedralKernel_->addArg(plan_.jacobianGradsY); // 8
 dihedralKernel_->addArg(plan_.jacobianGradsZ); // 9
 dihedralKernel_->addArg(plan_.jacobianCvIdx); // 10
 dihedralKernel_->addArg(); // 11-15: PBC (setPeriodicBoxArgs)
 dihedralKernel_->addArg();
 dihedralKernel_->addArg();
 dihedralKernel_->addArg();
 dihedralKernel_->addArg();
 dihedralKernel_->addArg(); // 16: periodic (setArg in execute)
 dihedralKernel_->addArg(); // 17: spare (8.5.1 trailing-primitive fix)
 }

 // COM-distance CV kernel
 // Args 0-4: posq, atomOffsets, atoms, masses, totalMasses
 // Args 5-7: numCVs, firstCVIndex, firstJacEntry (primitives)
 // Args 8-13: cvValues, jacobianAtomIdx, jacobianGrads*, jacobianCvIdx
 // Args 14-18: PBC (setPeriodicBoxArgs at 14)
 // Arg 19: periodic (setArg); arg 20: spare (8.5.1 fix)
 if (plan_.numCOMDistanceCVs > 0) {
 ComputeProgram comProg = cc_.compileProgram(kCOMDistanceKernelSrc, {});
 comDistanceKernel_ = comProg->createKernel("cvCOMDistance");
 comDistanceKernel_->addArg(cc_.getPosq()); // 0
 comDistanceKernel_->addArg(plan_.comDistanceAtomOffsets); // 1
 comDistanceKernel_->addArg(plan_.comDistanceAtoms); // 2
 comDistanceKernel_->addArg(plan_.comDistanceMasses); // 3
 comDistanceKernel_->addArg(plan_.comDistanceTotalMasses); // 4
 comDistanceKernel_->addArg(plan_.numCOMDistanceCVs); // 5
 comDistanceKernel_->addArg(plan_.comDistanceFirstCVIndex); // 6
 comDistanceKernel_->addArg(plan_.comDistanceFirstJacEntry); // 7
 comDistanceKernel_->addArg(plan_.cvValues); // 8
 comDistanceKernel_->addArg(plan_.jacobianAtomIdx); // 9
 comDistanceKernel_->addArg(plan_.jacobianGradsX); // 10
 comDistanceKernel_->addArg(plan_.jacobianGradsY); // 11
 comDistanceKernel_->addArg(plan_.jacobianGradsZ); // 12
 comDistanceKernel_->addArg(plan_.jacobianCvIdx); // 13
 comDistanceKernel_->addArg(); // 14-18: PBC
 comDistanceKernel_->addArg();
 comDistanceKernel_->addArg();
 comDistanceKernel_->addArg();
 comDistanceKernel_->addArg();
 comDistanceKernel_->addArg(); // 19: periodic (setArg in execute)
 comDistanceKernel_->addArg(); // 20: spare
 }

 // Gyration (Rg) CV kernel — same arg layout as COM-distance
 if (plan_.numGyrationCVs > 0) {
 ComputeProgram rgProg = cc_.compileProgram(kGyrationKernelSrc, {});
 gyrationKernel_ = rgProg->createKernel("cvGyration");
 gyrationKernel_->addArg(cc_.getPosq()); // 0
 gyrationKernel_->addArg(plan_.gyrationAtomOffsets); // 1
 gyrationKernel_->addArg(plan_.gyrationAtoms); // 2
 gyrationKernel_->addArg(plan_.gyrationMasses); // 3
 gyrationKernel_->addArg(plan_.gyrationTotalMasses); // 4
 gyrationKernel_->addArg(plan_.numGyrationCVs); // 5
 gyrationKernel_->addArg(plan_.gyrationFirstCVIndex); // 6
 gyrationKernel_->addArg(plan_.gyrationFirstJacEntry); // 7
 gyrationKernel_->addArg(plan_.cvValues); // 8
 gyrationKernel_->addArg(plan_.jacobianAtomIdx); // 9
 gyrationKernel_->addArg(plan_.jacobianGradsX); // 10
 gyrationKernel_->addArg(plan_.jacobianGradsY); // 11
 gyrationKernel_->addArg(plan_.jacobianGradsZ); // 12
 gyrationKernel_->addArg(plan_.jacobianCvIdx); // 13
 gyrationKernel_->addArg(); // 14-18: PBC
 gyrationKernel_->addArg();
 gyrationKernel_->addArg();
 gyrationKernel_->addArg();
 gyrationKernel_->addArg();
 gyrationKernel_->addArg(); // 19: periodic (setArg in execute)
 gyrationKernel_->addArg(); // 20: spare
 }

 // Path CV kernel
 // Args 0-5: posq, atomOffsets, atoms, refOffsets, refPos, pathParams
 // Args 6-8: numCVs, firstCVIndex, firstJacEntry
 // Args 9-14: cvValues, jacobian arrays
 // Args 15-19: PBC (setPeriodicBoxArgs at 15)
 // Arg 20: periodic (setArg); arg 21: spare
 if (plan_.numPathCVs > 0) {
 ComputeProgram pathProg = cc_.compileProgram(kPathKernelSrc, {});
 pathKernel_ = pathProg->createKernel("cvPath");
 pathKernel_->addArg(cc_.getPosq()); // 0
 pathKernel_->addArg(plan_.pathAtomOffsets); // 1
 pathKernel_->addArg(plan_.pathAtoms); // 2
 pathKernel_->addArg(plan_.pathRefOffsets); // 3
 pathKernel_->addArg(plan_.pathRefPos); // 4
 pathKernel_->addArg(plan_.pathParams); // 5
 pathKernel_->addArg(plan_.numPathCVs); // 6
 pathKernel_->addArg(plan_.pathFirstCVIndex); // 7
 pathKernel_->addArg(plan_.pathFirstJacEntry); // 8
 pathKernel_->addArg(plan_.cvValues); // 9
 pathKernel_->addArg(plan_.jacobianAtomIdx); // 10
 pathKernel_->addArg(plan_.jacobianGradsX); // 11
 pathKernel_->addArg(plan_.jacobianGradsY); // 12
 pathKernel_->addArg(plan_.jacobianGradsZ); // 13
 pathKernel_->addArg(plan_.jacobianCvIdx); // 14
 pathKernel_->addArg(); // 15-19: PBC
 pathKernel_->addArg();
 pathKernel_->addArg();
 pathKernel_->addArg();
 pathKernel_->addArg();
 pathKernel_->addArg(); // 20: periodic (setArg in execute)
 pathKernel_->addArg(); // 21: spare (8.5.1 trailing-primitive fix)
 }

 // RMSD CV kernel — same arg layout as coordination
 // Args 0-3: posq, atomOffsets, atoms, refPos
 // Args 4-6: numCVs, firstCVIndex, firstJacEntry
 // Args 7-12: cvValues, jacobian arrays
 // Args 13-17: PBC; arg 18: periodic (setArg); arg 19: spare
 if (plan_.numRMSDCVs > 0) {
 ComputeProgram rmsdProg = cc_.compileProgram(kRMSDKernelSrc, {});
 rmsdKernel_ = rmsdProg->createKernel("cvRMSD");
 rmsdKernel_->addArg(cc_.getPosq()); // 0
 rmsdKernel_->addArg(plan_.rmsdAtomOffsets); // 1
 rmsdKernel_->addArg(plan_.rmsdAtoms); // 2
 rmsdKernel_->addArg(plan_.rmsdRefPos); // 3
 rmsdKernel_->addArg(plan_.numRMSDCVs); // 4
 rmsdKernel_->addArg(plan_.rmsdFirstCVIndex); // 5
 rmsdKernel_->addArg(plan_.rmsdFirstJacEntry); // 6
 rmsdKernel_->addArg(plan_.cvValues); // 7
 rmsdKernel_->addArg(plan_.jacobianAtomIdx); // 8
 rmsdKernel_->addArg(plan_.jacobianGradsX); // 9
 rmsdKernel_->addArg(plan_.jacobianGradsY); // 10
 rmsdKernel_->addArg(plan_.jacobianGradsZ); // 11
 rmsdKernel_->addArg(plan_.jacobianCvIdx); // 12
 rmsdKernel_->addArg(); // 13-17: PBC
 rmsdKernel_->addArg();
 rmsdKernel_->addArg();
 rmsdKernel_->addArg();
 rmsdKernel_->addArg();
 rmsdKernel_->addArg(); // 18: periodic (setArg in execute)
 rmsdKernel_->addArg(); // 19: spare (8.5.1 trailing-primitive fix)
 }

 // Coordination number CV kernel
 // Args 0-3: posq, atomOffsets, atoms, params
 // Args 4-6: numCVs, firstCVIndex, firstJacEntry
 // Args 7-12: cvValues, jacobianAtomIdx, jacobianGrads*, jacobianCvIdx
 // Args 13-17: PBC (setPeriodicBoxArgs at 13)
 // Arg 18: periodic (setArg); arg 19: spare (8.5.1 fix)
 if (plan_.numCoordCVs > 0) {
 ComputeProgram coordProg = cc_.compileProgram(kCoordinationKernelSrc, {});
 coordKernel_ = coordProg->createKernel("cvCoordination");
 coordKernel_->addArg(cc_.getPosq()); // 0
 coordKernel_->addArg(plan_.coordAtomOffsets); // 1
 coordKernel_->addArg(plan_.coordAtoms); // 2
 coordKernel_->addArg(plan_.coordParams); // 3
 coordKernel_->addArg(plan_.numCoordCVs); // 4
 coordKernel_->addArg(plan_.coordFirstCVIndex); // 5
 coordKernel_->addArg(plan_.coordFirstJacEntry); // 6
 coordKernel_->addArg(plan_.cvValues); // 7
 coordKernel_->addArg(plan_.jacobianAtomIdx); // 8
 coordKernel_->addArg(plan_.jacobianGradsX); // 9
 coordKernel_->addArg(plan_.jacobianGradsY); // 10
 coordKernel_->addArg(plan_.jacobianGradsZ); // 11
 coordKernel_->addArg(plan_.jacobianCvIdx); // 12
 coordKernel_->addArg(); // 13-17: PBC
 coordKernel_->addArg();
 coordKernel_->addArg();
 coordKernel_->addArg();
 coordKernel_->addArg();
 coordKernel_->addArg(); // 18: periodic (setArg in execute)
 coordKernel_->addArg(); // 19: spare (8.5.1 trailing-primitive fix)
 }

 // Position CV kernel
 if (plan_.numPositionCVs > 0) {
 ComputeProgram posProg = cc_.compileProgram(kPositionKernelSrc, {});
 positionKernel_ = posProg->createKernel("cvPosition");
 positionKernel_->addArg(cc_.getPosq()); // 0
 positionKernel_->addArg(plan_.positionAtoms); // 1
 positionKernel_->addArg(plan_.positionComponents); // 2
 positionKernel_->addArg(plan_.numPositionCVs); // 3
 positionKernel_->addArg(plan_.positionFirstCVIndex); // 4
 positionKernel_->addArg(plan_.positionFirstJacEntry); // 5
 positionKernel_->addArg(plan_.cvValues); // 6
 positionKernel_->addArg(plan_.jacobianAtomIdx); // 7
 positionKernel_->addArg(plan_.jacobianGradsX); // 8
 positionKernel_->addArg(plan_.jacobianGradsY); // 9
 positionKernel_->addArg(plan_.jacobianGradsZ); // 10
 positionKernel_->addArg(plan_.jacobianCvIdx); // 11
 positionKernel_->addArg(); // 12: spare (8.5.1 fix)
 }

 // DRMSD CV kernel
 if (plan_.numDRMSDCVs > 0) {
 ComputeProgram drmsdProg = cc_.compileProgram(kDRMSDKernelSrc, {});
 drmsdKernel_ = drmsdProg->createKernel("cvDRMSD");
 drmsdKernel_->addArg(cc_.getPosq()); // 0
 drmsdKernel_->addArg(plan_.drmsdPairOffsets); // 1
 drmsdKernel_->addArg(plan_.drmsdAtomPairs); // 2
 drmsdKernel_->addArg(plan_.drmsdRefDists); // 3
 drmsdKernel_->addArg(plan_.numDRMSDCVs); // 4
 drmsdKernel_->addArg(plan_.drmsdFirstCVIndex); // 5
 drmsdKernel_->addArg(plan_.drmsdFirstJacEntry); // 6
 drmsdKernel_->addArg(plan_.cvValues); // 7
 drmsdKernel_->addArg(plan_.jacobianAtomIdx); // 8
 drmsdKernel_->addArg(plan_.jacobianGradsX); // 9
 drmsdKernel_->addArg(plan_.jacobianGradsY); // 10
 drmsdKernel_->addArg(plan_.jacobianGradsZ); // 11
 drmsdKernel_->addArg(plan_.jacobianCvIdx); // 12
 drmsdKernel_->addArg(); // 13-17: PBC (setPeriodicBoxArgs)
 drmsdKernel_->addArg();
 drmsdKernel_->addArg();
 drmsdKernel_->addArg();
 drmsdKernel_->addArg();
 drmsdKernel_->addArg(); // 18: periodic (setArg in execute)
 drmsdKernel_->addArg(); // 19: spare (8.5.1 fix)
 }

 // ContactMap CV kernel
 if (plan_.numContactMapCVs > 0) {
 ComputeProgram cmProg = cc_.compileProgram(kContactMapKernelSrc, {});
 contactMapKernel_ = cmProg->createKernel("cvContactMap");
 contactMapKernel_->addArg(cc_.getPosq()); // 0
 contactMapKernel_->addArg(plan_.contactMapPairOffsets); // 1
 contactMapKernel_->addArg(plan_.contactMapAtomPairs); // 2
 contactMapKernel_->addArg(plan_.contactMapParams); // 3
 contactMapKernel_->addArg(plan_.numContactMapCVs); // 4
 contactMapKernel_->addArg(plan_.contactMapFirstCVIndex); // 5
 contactMapKernel_->addArg(plan_.contactMapFirstJacEntry); // 6
 contactMapKernel_->addArg(plan_.cvValues); // 7
 contactMapKernel_->addArg(plan_.jacobianAtomIdx); // 8
 contactMapKernel_->addArg(plan_.jacobianGradsX); // 9
 contactMapKernel_->addArg(plan_.jacobianGradsY); // 10
 contactMapKernel_->addArg(plan_.jacobianGradsZ); // 11
 contactMapKernel_->addArg(plan_.jacobianCvIdx); // 12
 contactMapKernel_->addArg(); // 13-17: PBC (setPeriodicBoxArgs)
 contactMapKernel_->addArg();
 contactMapKernel_->addArg();
 contactMapKernel_->addArg();
 contactMapKernel_->addArg();
 contactMapKernel_->addArg(); // 18: periodic (setArg in execute)
 contactMapKernel_->addArg(); // 19: spare (8.5.1 fix)
 }

 // Plane CV kernel
 if (plan_.numPlaneCVs > 0) {
 ComputeProgram planeProg = cc_.compileProgram(kPlaneKernelSrc, {});
 planeKernel_ = planeProg->createKernel("cvPlane");
 planeKernel_->addArg(cc_.getPosq()); // 0
 planeKernel_->addArg(plan_.planeAtoms); // 1
 planeKernel_->addArg(plan_.planeComponents); // 2
 planeKernel_->addArg(plan_.numPlaneCVs); // 3
 planeKernel_->addArg(plan_.planeFirstCVIndex); // 4
 planeKernel_->addArg(plan_.planeFirstJacEntry); // 5
 planeKernel_->addArg(plan_.cvValues); // 6
 planeKernel_->addArg(plan_.jacobianAtomIdx); // 7
 planeKernel_->addArg(plan_.jacobianGradsX); // 8
 planeKernel_->addArg(plan_.jacobianGradsY); // 9
 planeKernel_->addArg(plan_.jacobianGradsZ); // 10
 planeKernel_->addArg(plan_.jacobianCvIdx); // 11
 planeKernel_->addArg(); // 12-16: PBC (setPeriodicBoxArgs)
 planeKernel_->addArg();
 planeKernel_->addArg();
 planeKernel_->addArg();
 planeKernel_->addArg();
 planeKernel_->addArg(); // 17: periodic (setArg in execute)
 planeKernel_->addArg(); // 18: spare (8.5.1 fix)
 }

 // Projection CV kernel
 if (plan_.numProjectionCVs > 0) {
 ComputeProgram projProg = cc_.compileProgram(kProjectionKernelSrc, {});
 projectionKernel_ = projProg->createKernel("cvProjection");
 projectionKernel_->addArg(cc_.getPosq()); // 0
 projectionKernel_->addArg(plan_.projectionAtoms); // 1
 projectionKernel_->addArg(plan_.projectionDirs); // 2
 projectionKernel_->addArg(plan_.numProjectionCVs); // 3
 projectionKernel_->addArg(plan_.projectionFirstCVIndex); // 4
 projectionKernel_->addArg(plan_.projectionFirstJacEntry); // 5
 projectionKernel_->addArg(plan_.cvValues); // 6
 projectionKernel_->addArg(plan_.jacobianAtomIdx); // 7
 projectionKernel_->addArg(plan_.jacobianGradsX); // 8
 projectionKernel_->addArg(plan_.jacobianGradsY); // 9
 projectionKernel_->addArg(plan_.jacobianGradsZ); // 10
 projectionKernel_->addArg(plan_.jacobianCvIdx); // 11
 projectionKernel_->addArg(); // 12-16: PBC (setPeriodicBoxArgs)
 projectionKernel_->addArg();
 projectionKernel_->addArg();
 projectionKernel_->addArg();
 projectionKernel_->addArg();
 projectionKernel_->addArg(); // 17: periodic (setArg in execute)
 projectionKernel_->addArg(); // 18: spare (8.5.1 fix)
 }

 // +3.16: Volume/Cell combined kernel (always compiled when periodic)
 if (plan_.numVolumeCVs > 0 || plan_.numCellCVs > 0) {
 ComputeProgram vcProg = cc_.compileProgram(kVolumeCellKernelSrc, {});
 volumeCellKernel_ = vcProg->createKernel("cvVolumeCell");
 volumeCellKernel_->addArg(plan_.numVolumeCVs); // 0
 volumeCellKernel_->addArg(plan_.volumeFirstCVIndex); // 1
 volumeCellKernel_->addArg(plan_.numCellCVs); // 2
 volumeCellKernel_->addArg(plan_.cellFirstCVIndex); // 3
 volumeCellKernel_->addArg(plan_.cellComponents); // 4
 volumeCellKernel_->addArg(plan_.cvValues); // 5
 volumeCellKernel_->addArg(); // 6-10: PBC (setPeriodicBoxArgs)
 volumeCellKernel_->addArg();
 volumeCellKernel_->addArg();
 volumeCellKernel_->addArg();
 volumeCellKernel_->addArg();
 volumeCellKernel_->addArg(); // 11: spare (8.5.1 fix)
 }

 // Dipole CV kernel
 if (plan_.numDipoleCVs > 0) {
 ComputeProgram dipProg = cc_.compileProgram(kDipoleKernelSrc, {});
 dipoleKernel_ = dipProg->createKernel("cvDipole");
 dipoleKernel_->addArg(cc_.getPosq()); // 0
 dipoleKernel_->addArg(plan_.dipoleAtomOffsets); // 1
 dipoleKernel_->addArg(plan_.dipoleAtoms); // 2
 dipoleKernel_->addArg(plan_.dipoleCharges); // 3
 dipoleKernel_->addArg(plan_.dipoleComponents); // 4
 dipoleKernel_->addArg(plan_.numDipoleCVs); // 5
 dipoleKernel_->addArg(plan_.dipoleFirstCVIndex); // 6
 dipoleKernel_->addArg(plan_.dipoleFirstJacEntry); // 7
 dipoleKernel_->addArg(plan_.cvValues); // 8
 dipoleKernel_->addArg(plan_.jacobianAtomIdx); // 9
 dipoleKernel_->addArg(plan_.jacobianGradsX); // 10
 dipoleKernel_->addArg(plan_.jacobianGradsY); // 11
 dipoleKernel_->addArg(plan_.jacobianGradsZ); // 12
 dipoleKernel_->addArg(plan_.jacobianCvIdx); // 13
 dipoleKernel_->addArg(); // 14: spare (8.5.1 fix)
 }

 // PCA CV kernel
 if (plan_.numPCACVs > 0) {
 ComputeProgram pcaProg = cc_.compileProgram(kPCAKernelSrc, {});
 pcaKernel_ = pcaProg->createKernel("cvPCA");
 pcaKernel_->addArg(cc_.getPosq()); // 0
 pcaKernel_->addArg(plan_.pcaAtomOffsets); // 1
 pcaKernel_->addArg(plan_.pcaAtoms); // 2
 pcaKernel_->addArg(plan_.pcaRefPos); // 3
 pcaKernel_->addArg(plan_.pcaEigvec); // 4
 pcaKernel_->addArg(plan_.numPCACVs); // 5
 pcaKernel_->addArg(plan_.pcaFirstCVIndex); // 6
 pcaKernel_->addArg(plan_.pcaFirstJacEntry); // 7
 pcaKernel_->addArg(plan_.cvValues); // 8
 pcaKernel_->addArg(plan_.jacobianAtomIdx); // 9
 pcaKernel_->addArg(plan_.jacobianGradsX); // 10
 pcaKernel_->addArg(plan_.jacobianGradsY); // 11
 pcaKernel_->addArg(plan_.jacobianGradsZ); // 12
 pcaKernel_->addArg(plan_.jacobianCvIdx); // 13
 pcaKernel_->addArg(); // 14: spare (8.5.1 fix)
 }

 // Secondary structure CV kernel
 if (plan_.numSecStrCVs > 0) {
 ComputeProgram ssProg = cc_.compileProgram(kSecondaryStructureKernelSrc, {});
 secStrKernel_ = ssProg->createKernel("cvSecondaryStructure");
 secStrKernel_->addArg(cc_.getPosq()); // 0
 secStrKernel_->addArg(plan_.secStrAtoms); // 1
 secStrKernel_->addArg(plan_.secStrAtomOffsets); // 2
 secStrKernel_->addArg(plan_.secStrWindowOffsets); // 3
 secStrKernel_->addArg(plan_.secStrParams); // 4
 secStrKernel_->addArg(plan_.numSecStrCVs); // 5
 secStrKernel_->addArg(plan_.secStrFirstCVIndex); // 6
 secStrKernel_->addArg(plan_.secStrFirstJacEntry); // 7
 secStrKernel_->addArg(plan_.cvValues); // 8
 secStrKernel_->addArg(plan_.jacobianAtomIdx); // 9
 secStrKernel_->addArg(plan_.jacobianGradsX); // 10
 secStrKernel_->addArg(plan_.jacobianGradsY); // 11
 secStrKernel_->addArg(plan_.jacobianGradsZ); // 12
 secStrKernel_->addArg(plan_.jacobianCvIdx); // 13
 secStrKernel_->addArg(); // 14: spare (8.5.1 fix)
 }

 // Puckering CV kernel (Cremer-Pople ring puckering)
 if (plan_.numPuckeringCVs > 0) {
 ComputeProgram pkProg = cc_.compileProgram(kPuckeringKernelSrc, {});
 puckerKernel_ = pkProg->createKernel("cvPuckering");
 puckerKernel_->addArg(cc_.getPosq()); // 0
 puckerKernel_->addArg(plan_.puckerAtoms); // 1
 puckerKernel_->addArg(plan_.puckerAtomOffsets); // 2
 puckerKernel_->addArg(plan_.puckerParams); // 3
 puckerKernel_->addArg(plan_.numPuckeringCVs); // 4
 puckerKernel_->addArg(plan_.puckerFirstCVIndex); // 5
 puckerKernel_->addArg(plan_.puckerFirstJacEntry); // 6
 puckerKernel_->addArg(plan_.cvValues); // 7
 puckerKernel_->addArg(plan_.jacobianAtomIdx); // 8
 puckerKernel_->addArg(plan_.jacobianGradsX); // 9
 puckerKernel_->addArg(plan_.jacobianGradsY); // 10
 puckerKernel_->addArg(plan_.jacobianGradsZ); // 11
 puckerKernel_->addArg(plan_.jacobianCvIdx); // 12
 puckerKernel_->addArg(); // 13: spare (8.5.1 alignment fix)
 }

 // eRMSD CV kernel (Bottaro eRMSD for RNA)
 if (plan_.numErmsdCVs > 0) {
 ComputeProgram ep = cc_.compileProgram(kErmsdKernelSrc, {});
 ermsdKernel_ = ep->createKernel("cvErmsd");
 ermsdKernel_->addArg(cc_.getPosq()); // 0
 ermsdKernel_->addArg(plan_.ermsdAtoms); // 1
 ermsdKernel_->addArg(plan_.ermsdAtomOffsets); // 2
 ermsdKernel_->addArg(plan_.ermsdNRes); // 3
 ermsdKernel_->addArg(plan_.ermsdJacOffsets); // 4
 ermsdKernel_->addArg(plan_.ermsdRefGOffsets); // 5
 ermsdKernel_->addArg(plan_.ermsdRefG); // 6
 ermsdKernel_->addArg(plan_.ermsdCutoffs); // 7
 ermsdKernel_->addArg(plan_.numErmsdCVs); // 8
 ermsdKernel_->addArg(plan_.ermsdFirstCVIndex); // 9
 ermsdKernel_->addArg(plan_.ermsdFirstJacEntry); // 10
 ermsdKernel_->addArg(plan_.cvValues); // 11
 ermsdKernel_->addArg(plan_.jacobianAtomIdx); // 12
 ermsdKernel_->addArg(plan_.jacobianGradsX); // 13
 ermsdKernel_->addArg(plan_.jacobianGradsY); // 14
 ermsdKernel_->addArg(plan_.jacobianGradsZ); // 15
 ermsdKernel_->addArg(plan_.jacobianCvIdx); // 16
 ermsdKernel_->addArg(); // 17: spare (8.5.1 fix)
 }

 // Expression CV eval and prop kernels
 {
 // Pre-count expression CVs and reserve the vector to prevent reallocation.
 // Reallocation would invalidate pointers stored inside addArg() kernel arg lists.
 int numExprCVs = 0;
 for (int i = 0; i < force.getNumCollectiveVariableSpecs(); i++) {
 int t; vector<int> a; vector<double> p;
 force.getCollectiveVariableInfo(i, t, a, p);
 if (t == GluedForce::CV_EXPRESSION) numExprCVs++;
 }
 expressionCVPlans_.reserve(numExprCVs);

 int cvIdx = 0;
 for (int i = 0; i < force.getNumCollectiveVariableSpecs(); i++) {
 int cvType; vector<int> atoms; vector<double> params;
 force.getCollectiveVariableInfo(i, cvType, atoms, params);
 if (cvType == GluedForce::CV_EXPRESSION) {
 string exprStr; vector<int> inputCVs;
 force.getExpressionCVInfo(i, exprStr, inputCVs);
 int D = (int)inputCVs.size();
 int outputCVIdx = cvIdx;
 expressionCVPlans_.emplace_back();
 ExpressionCVPlan& ep = expressionCVPlans_.back();
 ep.numInputs = D;
 ep.outputCVIndex = outputCVIdx;
 ep.inputCVIdx.initialize<int>(cc_, max(D, 1), "exprCV_in_" + to_string(i));
 if (D > 0) ep.inputCVIdx.upload(inputCVs);
 ep.partials.initialize<double>(cc_, max(D, 1), "exprCV_pd_" + to_string(i));
 ComputeProgram evalProg = cc_.compileProgram(
 genExpressionEvalKernelSrc(cc_, exprStr, D, i), {});
 ep.evalKernel = evalProg->createKernel("cvExprEval_" + to_string(i));
 ep.evalKernel->addArg(plan_.cvValues); // 0
 ep.evalKernel->addArg(ep.inputCVIdx); // 1
 ep.evalKernel->addArg(outputCVIdx); // 2
 ep.evalKernel->addArg(ep.partials); // 3
 ep.evalKernel->addArg(); // 4: spare (8.5.1 fix)
 if (D > 0) {
 ComputeProgram propProg = cc_.compileProgram(genExpressionPropKernelSrc(i), {});
 ep.propKernel = propProg->createKernel("cvExprProp_" + to_string(i));
 ep.propKernel->addArg(ep.partials); // 0
 ep.propKernel->addArg(ep.inputCVIdx); // 1
 ep.propKernel->addArg(D); // 2
 ep.propKernel->addArg(outputCVIdx); // 3
 ep.propKernel->addArg(plan_.cvBiasGradients); // 4
 ep.propKernel->addArg(); // 5: spare (8.5.1 fix)
 }
 cvIdx += 1;
 } else {
 cvIdx += (cvType == GluedForce::CV_PATH) ? 2 : 1;
 }
 }
 }

 // PyTorch CV model loading.
 // pytorchCVPlans_ was populated in buildPlan(); just load the model files here.
#ifdef GLUED_HAS_TORCH
 {
 int numSpecs = force.getNumCollectiveVariableSpecs();
 int ptIdx = 0;
 for (int i = 0; i < numSpecs; i++) {
 int cvType; vector<int> atoms; vector<double> params;
 force.getCollectiveVariableInfo(i, cvType, atoms, params);
 if (cvType == GluedForce::CV_PYTORCH) {
 string modelPath = force.getPyTorchCVModelPath(i);
 auto* module = new torch::jit::script::Module(
 torch::jit::load(modelPath));
 module->eval();
 pytorchCVPlans_[ptIdx].model = shared_ptr<void>(module,
 [](void* p) {
 delete static_cast<torch::jit::script::Module*>(p);
 });
 ptIdx++;
 }
 }
 }

 // GPU-native inference buffers: allocate and compile NVRTC kernels when
 // running on CUDA (getNativeCudaStream returns non-null on that platform).
 if (!pytorchCVPlans_.empty() && getNativeCudaStream() != nullptr) {
 ComputeProgram ptProg = cc_.compileProgram(kPyTorchKernelSrc, {});
 for (auto& pt : pytorchCVPlans_) {
 string sfx = to_string(pt.outputCVIndex);
 pt.atomIdxGPU.initialize<int> (cc_, pt.numAtoms, "ptAtomIdx_" + sfx);
 pt.posBufGPU .initialize<float>(cc_, pt.numAtoms * 3, "ptPosBuf_" + sfx);
 pt.gradBufGPU.initialize<float>(cc_, pt.numAtoms * 3, "ptGradBuf_" + sfx);

 pt.extractKernel = ptProg->createKernel("pytorchExtractPos");
 pt.extractKernel->addArg(cc_.getPosq()); // 0: posq
 pt.extractKernel->addArg(pt.atomIdxGPU); // 1: atomIdx
 pt.extractKernel->addArg(pt.posBufGPU); // 2: posBuf
 pt.extractKernel->addArg(pt.numAtoms); // 3: numAtoms
 pt.extractKernel->addArg(); // 4: spare (8.5.1)

 pt.deinterleavKernel = ptProg->createKernel("pytorchDeinterleavGrad");
 pt.deinterleavKernel->addArg(pt.gradBufGPU); // 0: gradBuf
 pt.deinterleavKernel->addArg(plan_.jacobianGradsX); // 1: jacX
 pt.deinterleavKernel->addArg(plan_.jacobianGradsY); // 2: jacY
 pt.deinterleavKernel->addArg(plan_.jacobianGradsZ); // 3: jacZ
 pt.deinterleavKernel->addArg(pt.jacOffset); // 4: jacOffset
 pt.deinterleavKernel->addArg(pt.numAtoms); // 5: numAtoms
 pt.deinterleavKernel->addArg(); // 6: spare (8.5.1)
 }
 }
#endif
}
