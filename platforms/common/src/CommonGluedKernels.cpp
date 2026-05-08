#include "CommonGluedKernels.h"
#include "GluedForce.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/common/ContextSelector.h"
#include "openmm/common/CommonKernelUtilities.h"
#include "openmm/common/ExpressionUtilities.h"
#include "lepton/Parser.h"
#include "lepton/ParsedExpression.h"
#include <cmath>
#include <cstring>

#ifdef GLUED_HAS_TORCH
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include <torch/script.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAFunctions.h>
#include <cuda_runtime.h>
#pragma GCC diagnostic pop
#endif

using namespace GluedPlugin;
using namespace OpenMM;
using namespace std;

// ---------------------------------------------------------------------------
// Geometry device functions prepended to every kernel that needs PBC.
// Triclinic minimum-image order is strict: Z first, then Y, then X.
// ---------------------------------------------------------------------------
extern const string kGeometryFunctions = R"(
DEVICE inline real3 deltaPBC(
 real3 dr,
 real4 invBoxSize,
 real4 boxVecX, real4 boxVecY, real4 boxVecZ
) {
 real scale3 = floor(dr.z * invBoxSize.z + (real)0.5);
 dr.x -= scale3 * boxVecZ.x;
 dr.y -= scale3 * boxVecZ.y;
 dr.z -= scale3 * boxVecZ.z;
 real scale2 = floor(dr.y * invBoxSize.y + (real)0.5);
 dr.x -= scale2 * boxVecY.x;
 dr.y -= scale2 * boxVecY.y;
 real scale1 = floor(dr.x * invBoxSize.x + (real)0.5);
 dr.x -= scale1 * boxVecX.x;
 return dr;
}

DEVICE inline real3 deltaNoPBC(real3 a, real3 b) {
 return make_real3(b.x - a.x, b.y - a.y, b.z - a.z);
}

DEVICE inline real distancePBC(
 real4 posA, real4 posB,
 real4 invBoxSize,
 real4 bvX, real4 bvY, real4 bvZ
) {
 real3 dr = make_real3(posB.x - posA.x, posB.y - posA.y, posB.z - posA.z);
 dr = deltaPBC(dr, invBoxSize, bvX, bvY, bvZ);
 return sqrt(dr.x*dr.x + dr.y*dr.y + dr.z*dr.z);
}

DEVICE inline real distanceNoPBC(real4 posA, real4 posB) {
 real dx = posB.x - posA.x;
 real dy = posB.y - posA.y;
 real dz = posB.z - posA.z;
 return sqrt(dx*dx + dy*dy + dz*dz);
}
)";

// ---------------------------------------------------------------------------
// test kernel — exercises GPU position reads and force writes.
//
// mode 1: constant force (scale, 0, 0) on all atoms.
// mode 2: per-atom force (userAtomIdx * scale, 0, 0) — atom-reorder test.
// mode 3: PBC distance atoms 0 and 1, written as force.x of atom 0.
// ---------------------------------------------------------------------------
extern const string kTestForceKernelSrc = kGeometryFunctions + R"(
KERNEL void applyTestForce(
 GLOBAL const real4* RESTRICT posq,
 GLOBAL const int* RESTRICT atomIndexArray,
 GLOBAL mm_ulong* RESTRICT forceBuffer,
 int paddedNumAtoms,
 int numAtoms,
 int mode,
 double scale,
 real4 boxSize,
 real4 invBoxSize,
 real4 boxVecX,
 real4 boxVecY,
 real4 boxVecZ
) {
 if (mode == 3) {
 if (GLOBAL_ID == 0 && numAtoms >= 2) {
 int g0 = atomIndexArray[0];
 int g1 = atomIndexArray[1];
 real dist = distancePBC(posq[g0], posq[g1], invBoxSize,
 boxVecX, boxVecY, boxVecZ);
 mm_long fx = (mm_long)((double)dist * 0x100000000LL);
 atomicAdd(&forceBuffer[0], (mm_ulong)fx);
 atomicAdd(&forceBuffer[0 + paddedNumAtoms], (mm_ulong)0LL);
 atomicAdd(&forceBuffer[0 + 2*paddedNumAtoms], (mm_ulong)0LL);
 }
 return;
 }

 for (int gpuAtom = GLOBAL_ID; gpuAtom < numAtoms; gpuAtom += GLOBAL_SIZE) {
 double fx;
 if (mode == 1) {
 fx = scale;
 } else {
 int userAtom = atomIndexArray[gpuAtom];
 fx = (double)userAtom * scale;
 }
 mm_long fixedFx = (mm_long)(fx * 0x100000000LL);
 atomicAdd(&forceBuffer[gpuAtom], (mm_ulong)fixedFx);
 atomicAdd(&forceBuffer[gpuAtom + paddedNumAtoms], (mm_ulong)0LL);
 atomicAdd(&forceBuffer[gpuAtom + 2*paddedNumAtoms], (mm_ulong)0LL);
 }
}
)";

// ---------------------------------------------------------------------------
// Distance CV kernel.
//
// One thread per distance CV. distanceAtomPairs[2*i] and [2*i+1] are
// GPU-space atom indices; this avoids a search through atomIndexArray each
// step. The indices are rebuilt in rebuildGpuAtomIndices() whenever
// OpenMM reorders atoms (getAtomsWereReordered()).
//
// Jacobian gradients are stored as float (direction vectors, |val| ≤ 1);
// the scatter kernel promotes them to double for fixed-point accumulation.
// ---------------------------------------------------------------------------
extern const string kDistanceKernelSrc = kGeometryFunctions + R"(
KERNEL void cvDistance(
 GLOBAL const real4* RESTRICT posq,
 GLOBAL const int* RESTRICT distanceAtomPairs,
 int numCVs,
 int firstCVIndex,
 int firstJacEntry,
 GLOBAL double* RESTRICT cvValues,
 GLOBAL int* RESTRICT jacobianAtomIdx,
 GLOBAL float* RESTRICT jacobianGradsX,
 GLOBAL float* RESTRICT jacobianGradsY,
 GLOBAL float* RESTRICT jacobianGradsZ,
 GLOBAL int* RESTRICT jacobianCvIdx,
 real4 boxSize,
 real4 invBoxSize,
 real4 boxVecX,
 real4 boxVecY,
 real4 boxVecZ,
 int periodic
) {
 for (int i = GLOBAL_ID; i < numCVs; i += GLOBAL_SIZE) {
 int gpuA = distanceAtomPairs[2*i];
 int gpuB = distanceAtomPairs[2*i + 1];
 real4 posA = posq[gpuA];
 real4 posB = posq[gpuB];

 real3 dr = make_real3(posB.x - posA.x, posB.y - posA.y, posB.z - posA.z);
 if (periodic)
 dr = deltaPBC(dr, invBoxSize, boxVecX, boxVecY, boxVecZ);

 real dist = sqrt(dr.x*dr.x + dr.y*dr.y + dr.z*dr.z);
 real invDist = (dist > (real)1e-10) ? ((real)1.0 / dist) : (real)0.0;

 cvValues[firstCVIndex + i] = (double)dist;

 int jac0 = firstJacEntry + 2*i;

 // dDist/dPosA = -dr/dist
 jacobianAtomIdx[jac0] = gpuA;
 jacobianGradsX[jac0] = (float)(-dr.x * invDist);
 jacobianGradsY[jac0] = (float)(-dr.y * invDist);
 jacobianGradsZ[jac0] = (float)(-dr.z * invDist);
 jacobianCvIdx[jac0] = firstCVIndex + i;

 // dDist/dPosB = +dr/dist
 jacobianAtomIdx[jac0+1] = gpuB;
 jacobianGradsX[jac0+1] = (float)(dr.x * invDist);
 jacobianGradsY[jac0+1] = (float)(dr.y * invDist);
 jacobianGradsZ[jac0+1] = (float)(dr.z * invDist);
 jacobianCvIdx[jac0+1] = firstCVIndex + i;
 }
}
)";

// ---------------------------------------------------------------------------
// Angle CV kernel.
//
// One thread per angle CV. Atoms stored as triplets (A, B, C) where B is
// the vertex. Gradient: dTheta/dA = -(n2 - cosT*n1)/(r1*sinT), etc.
// Guard: sinT < 1e-6 zeroes the gradient (collinear atoms, theta ~ 0 or pi).
// ---------------------------------------------------------------------------
extern const string kAngleKernelSrc = kGeometryFunctions + R"(
KERNEL void cvAngle(
 GLOBAL const real4* RESTRICT posq,
 GLOBAL const int* RESTRICT angleAtoms,
 int numCVs,
 int firstCVIndex,
 int firstJacEntry,
 GLOBAL double* RESTRICT cvValues,
 GLOBAL int* RESTRICT jacobianAtomIdx,
 GLOBAL float* RESTRICT jacobianGradsX,
 GLOBAL float* RESTRICT jacobianGradsY,
 GLOBAL float* RESTRICT jacobianGradsZ,
 GLOBAL int* RESTRICT jacobianCvIdx,
 real4 boxSize,
 real4 invBoxSize,
 real4 boxVecX,
 real4 boxVecY,
 real4 boxVecZ,
 int periodic
) {
 for (int i = GLOBAL_ID; i < numCVs; i += GLOBAL_SIZE) {
 int gpuA = angleAtoms[3*i];
 int gpuB = angleAtoms[3*i + 1];
 int gpuC = angleAtoms[3*i + 2];
 real4 pA = posq[gpuA];
 real4 pB = posq[gpuB];
 real4 pC = posq[gpuC];

 // dr1 = A-B, dr2 = C-B
 real3 dr1 = make_real3(pA.x-pB.x, pA.y-pB.y, pA.z-pB.z);
 real3 dr2 = make_real3(pC.x-pB.x, pC.y-pB.y, pC.z-pB.z);
 if (periodic) {
 dr1 = deltaPBC(dr1, invBoxSize, boxVecX, boxVecY, boxVecZ);
 dr2 = deltaPBC(dr2, invBoxSize, boxVecX, boxVecY, boxVecZ);
 }

 real r1 = sqrt(dr1.x*dr1.x + dr1.y*dr1.y + dr1.z*dr1.z);
 real r2 = sqrt(dr2.x*dr2.x + dr2.y*dr2.y + dr2.z*dr2.z);
 real invR1 = (r1 > (real)1e-10) ? ((real)1.0/r1) : (real)0.0;
 real invR2 = (r2 > (real)1e-10) ? ((real)1.0/r2) : (real)0.0;

 real3 n1 = make_real3(dr1.x*invR1, dr1.y*invR1, dr1.z*invR1);
 real3 n2 = make_real3(dr2.x*invR2, dr2.y*invR2, dr2.z*invR2);

 real cosT = n1.x*n2.x + n1.y*n2.y + n1.z*n2.z;
 cosT = max((real)-1.0, min((real)1.0, cosT));
 real theta = acos(cosT);
 real sinT = sqrt((real)1.0 - cosT*cosT);
 real invSin = (sinT > (real)1e-6) ? ((real)1.0/sinT) : (real)0.0;

 cvValues[firstCVIndex + i] = (double)theta;

 // dTheta/dA = -(n2 - cosT*n1) / (r1*sinT)
 // dTheta/dC = -(n1 - cosT*n2) / (r2*sinT)
 // dTheta/dB = -(dTheta/dA + dTheta/dC)
 real s1 = invR1 * invSin;
 real s2 = invR2 * invSin;
 real3 jacA = make_real3(-(n2.x - cosT*n1.x)*s1,
 -(n2.y - cosT*n1.y)*s1,
 -(n2.z - cosT*n1.z)*s1);
 real3 jacC = make_real3(-(n1.x - cosT*n2.x)*s2,
 -(n1.y - cosT*n2.y)*s2,
 -(n1.z - cosT*n2.z)*s2);
 real3 jacB = make_real3(-(jacA.x+jacC.x), -(jacA.y+jacC.y), -(jacA.z+jacC.z));

 int jac0 = firstJacEntry + 3*i;
 int cvIdx = firstCVIndex + i;

 jacobianAtomIdx[jac0] = gpuA;
 jacobianGradsX[jac0] = (float)jacA.x;
 jacobianGradsY[jac0] = (float)jacA.y;
 jacobianGradsZ[jac0] = (float)jacA.z;
 jacobianCvIdx[jac0] = cvIdx;

 jacobianAtomIdx[jac0+1] = gpuB;
 jacobianGradsX[jac0+1] = (float)jacB.x;
 jacobianGradsY[jac0+1] = (float)jacB.y;
 jacobianGradsZ[jac0+1] = (float)jacB.z;
 jacobianCvIdx[jac0+1] = cvIdx;

 jacobianAtomIdx[jac0+2] = gpuC;
 jacobianGradsX[jac0+2] = (float)jacC.x;
 jacobianGradsY[jac0+2] = (float)jacC.y;
 jacobianGradsZ[jac0+2] = (float)jacC.z;
 jacobianCvIdx[jac0+2] = cvIdx;
 }
}
)";

// ---------------------------------------------------------------------------
// Dihedral (torsion) CV kernel.
//
// Blondel-Karplus formula (JCC 1996, 17, 1132).
// b1=B-A, b2=C-B, b3=D-C; t=b1×b2, u=b2×b3, d=|b2|.
// phi = atan2(d * dot(b1,u), dot(t,u))
// Guard: |t|^2 or |u|^2 < 1e-10 zeroes the respective Jacobian contribution.
// ---------------------------------------------------------------------------
extern const string kDihedralKernelSrc = kGeometryFunctions + R"(
KERNEL void cvDihedral(
 GLOBAL const real4* RESTRICT posq,
 GLOBAL const int* RESTRICT dihedralAtoms,
 int numCVs,
 int firstCVIndex,
 int firstJacEntry,
 GLOBAL double* RESTRICT cvValues,
 GLOBAL int* RESTRICT jacobianAtomIdx,
 GLOBAL float* RESTRICT jacobianGradsX,
 GLOBAL float* RESTRICT jacobianGradsY,
 GLOBAL float* RESTRICT jacobianGradsZ,
 GLOBAL int* RESTRICT jacobianCvIdx,
 real4 boxSize,
 real4 invBoxSize,
 real4 boxVecX,
 real4 boxVecY,
 real4 boxVecZ,
 int periodic
) {
 for (int i = GLOBAL_ID; i < numCVs; i += GLOBAL_SIZE) {
 int gpuA = dihedralAtoms[4*i];
 int gpuB = dihedralAtoms[4*i + 1];
 int gpuC = dihedralAtoms[4*i + 2];
 int gpuD = dihedralAtoms[4*i + 3];
 real4 pA = posq[gpuA];
 real4 pB = posq[gpuB];
 real4 pC = posq[gpuC];
 real4 pD = posq[gpuD];

 real3 b1 = make_real3(pB.x-pA.x, pB.y-pA.y, pB.z-pA.z);
 real3 b2 = make_real3(pC.x-pB.x, pC.y-pB.y, pC.z-pB.z);
 real3 b3 = make_real3(pD.x-pC.x, pD.y-pC.y, pD.z-pC.z);
 if (periodic) {
 b1 = deltaPBC(b1, invBoxSize, boxVecX, boxVecY, boxVecZ);
 b2 = deltaPBC(b2, invBoxSize, boxVecX, boxVecY, boxVecZ);
 b3 = deltaPBC(b3, invBoxSize, boxVecX, boxVecY, boxVecZ);
 }

 // t = b1 x b2, u = b2 x b3
 real3 t = make_real3(b1.y*b2.z - b1.z*b2.y,
 b1.z*b2.x - b1.x*b2.z,
 b1.x*b2.y - b1.y*b2.x);
 real3 u = make_real3(b2.y*b3.z - b2.z*b3.y,
 b2.z*b3.x - b2.x*b3.z,
 b2.x*b3.y - b2.y*b3.x);

 real d2 = b2.x*b2.x + b2.y*b2.y + b2.z*b2.z;
 real d = sqrt(d2);
 real t2 = t.x*t.x + t.y*t.y + t.z*t.z;
 real u2 = u.x*u.x + u.y*u.y + u.z*u.z;

 real tuDot = t.x*u.x + t.y*u.y + t.z*u.z;
 real b1uDot = b1.x*u.x + b1.y*u.y + b1.z*u.z;
 real phi = atan2(d * b1uDot, tuDot);

 cvValues[firstCVIndex + i] = (double)phi;

 // Blondel-Karplus Jacobian
 real invT2 = (t2 > (real)1e-10) ? ((real)1.0/t2) : (real)0.0;
 real invU2 = (u2 > (real)1e-10) ? ((real)1.0/u2) : (real)0.0;
 real c1 = (b1.x*b2.x + b1.y*b2.y + b1.z*b2.z) / d; // b1·b2 / d
 real c2 = (b3.x*b2.x + b3.y*b2.y + b3.z*b2.z) / d; // b3·b2 / d

 real aT = d * invT2;
 real aU = d * invU2;
 real bT_B = (d + c1) * invT2;
 real bU_B = c2 * invU2;

 real3 jacA = make_real3(-aT*t.x, -aT*t.y, -aT*t.z);
 real3 jacD = make_real3( aU*u.x, aU*u.y, aU*u.z);
 // BK atom B: +(d+c1)/t^2 * t + c2/u^2 * u  (both terms positive — see OpenMM reference)
 real3 jacB = make_real3(+bT_B*t.x + bU_B*u.x, +bT_B*t.y + bU_B*u.y, +bT_B*t.z + bU_B*u.z);
 // Newton's 3rd law: jacC = -(jacA + jacB + jacD) — avoids sign error in b2 expansion
 real3 jacC = make_real3(-(jacA.x+jacB.x+jacD.x),
 -(jacA.y+jacB.y+jacD.y),
 -(jacA.z+jacB.z+jacD.z));

 int jac0 = firstJacEntry + 4*i;
 int cvIdx = firstCVIndex + i;

 jacobianAtomIdx[jac0] = gpuA;
 jacobianGradsX[jac0] = (float)jacA.x;
 jacobianGradsY[jac0] = (float)jacA.y;
 jacobianGradsZ[jac0] = (float)jacA.z;
 jacobianCvIdx[jac0] = cvIdx;

 jacobianAtomIdx[jac0+1] = gpuB;
 jacobianGradsX[jac0+1] = (float)jacB.x;
 jacobianGradsY[jac0+1] = (float)jacB.y;
 jacobianGradsZ[jac0+1] = (float)jacB.z;
 jacobianCvIdx[jac0+1] = cvIdx;

 jacobianAtomIdx[jac0+2] = gpuC;
 jacobianGradsX[jac0+2] = (float)jacC.x;
 jacobianGradsY[jac0+2] = (float)jacC.y;
 jacobianGradsZ[jac0+2] = (float)jacC.z;
 jacobianCvIdx[jac0+2] = cvIdx;

 jacobianAtomIdx[jac0+3] = gpuD;
 jacobianGradsX[jac0+3] = (float)jacD.x;
 jacobianGradsY[jac0+3] = (float)jacD.y;
 jacobianGradsZ[jac0+3] = (float)jacD.z;
 jacobianCvIdx[jac0+3] = cvIdx;
 }
}
)";

// ---------------------------------------------------------------------------
// COM-distance CV kernel.
//
// One thread per CV. atomOffsets is a 2*numCVs+1 interleaved prefix-sum:
// group1 of CV i: [atomOffsets[2i], atomOffsets[2i+1])
// group2 of CV i: [atomOffsets[2i+1], atomOffsets[2i+2])
// Each offset is an index into atoms[] and masses[].
// Jacobian: dDist/dr_k = -dr_unit * (m_k/M1) for k in group1,
// = +dr_unit * (m_k/M2) for k in group2.
// ---------------------------------------------------------------------------
extern const string kCOMDistanceKernelSrc = kGeometryFunctions + R"(
KERNEL void cvCOMDistance(
 GLOBAL const real4* RESTRICT posq,
 GLOBAL const int* RESTRICT atomOffsets,
 GLOBAL const int* RESTRICT atoms,
 GLOBAL const float* RESTRICT masses,
 GLOBAL const float* RESTRICT totalMasses,
 int numCVs,
 int firstCVIndex,
 int firstJacEntry,
 GLOBAL double* RESTRICT cvValues,
 GLOBAL int* RESTRICT jacobianAtomIdx,
 GLOBAL float* RESTRICT jacobianGradsX,
 GLOBAL float* RESTRICT jacobianGradsY,
 GLOBAL float* RESTRICT jacobianGradsZ,
 GLOBAL int* RESTRICT jacobianCvIdx,
 real4 boxSize,
 real4 invBoxSize,
 real4 boxVecX,
 real4 boxVecY,
 real4 boxVecZ,
 int periodic
) {
 for (int i = GLOBAL_ID; i < numCVs; i += GLOBAL_SIZE) {
 int g1Start = atomOffsets[2*i];
 int g2Start = atomOffsets[2*i + 1];
 int g2End = atomOffsets[2*i + 2];
 int ng1 = g2Start - g1Start;
 int ng2 = g2End - g2Start;
 float m1 = totalMasses[2*i];
 float m2 = totalMasses[2*i + 1];

 real3 com1 = make_real3((real)0, (real)0, (real)0);
 for (int j = 0; j < ng1; j++) {
 real4 p = posq[atoms[g1Start + j]];
 float w = masses[g1Start + j];
 com1.x += (real)w * p.x;
 com1.y += (real)w * p.y;
 com1.z += (real)w * p.z;
 }
 real inv1 = (real)1.0 / (real)m1;
 com1.x *= inv1; com1.y *= inv1; com1.z *= inv1;

 real3 com2 = make_real3((real)0, (real)0, (real)0);
 for (int j = 0; j < ng2; j++) {
 real4 p = posq[atoms[g2Start + j]];
 float w = masses[g2Start + j];
 com2.x += (real)w * p.x;
 com2.y += (real)w * p.y;
 com2.z += (real)w * p.z;
 }
 real inv2 = (real)1.0 / (real)m2;
 com2.x *= inv2; com2.y *= inv2; com2.z *= inv2;

 real3 dr = make_real3(com2.x - com1.x, com2.y - com1.y, com2.z - com1.z);
 if (periodic)
 dr = deltaPBC(dr, invBoxSize, boxVecX, boxVecY, boxVecZ);

 real dist = sqrt(dr.x*dr.x + dr.y*dr.y + dr.z*dr.z);
 real invDist = (dist > (real)1e-10) ? ((real)1.0/dist) : (real)0.0;
 cvValues[firstCVIndex + i] = (double)dist;

 int cvIdx = firstCVIndex + i;
 for (int j = 0; j < ng1; j++) {
 int jac = firstJacEntry + g1Start + j;
 float w = masses[g1Start + j] / m1;
 jacobianAtomIdx[jac] = atoms[g1Start + j];
 jacobianGradsX[jac] = (float)(-dr.x * invDist * (real)w);
 jacobianGradsY[jac] = (float)(-dr.y * invDist * (real)w);
 jacobianGradsZ[jac] = (float)(-dr.z * invDist * (real)w);
 jacobianCvIdx[jac] = cvIdx;
 }
 for (int j = 0; j < ng2; j++) {
 int jac = firstJacEntry + g2Start + j;
 float w = masses[g2Start + j] / m2;
 jacobianAtomIdx[jac] = atoms[g2Start + j];
 jacobianGradsX[jac] = (float)(dr.x * invDist * (real)w);
 jacobianGradsY[jac] = (float)(dr.y * invDist * (real)w);
 jacobianGradsZ[jac] = (float)(dr.z * invDist * (real)w);
 jacobianCvIdx[jac] = cvIdx;
 }
 }
}
)";

// ---------------------------------------------------------------------------
// Gyration radius (Rg) CV kernel.
//
// Rg = sqrt( sum_i(m_i * |r_i - r_COM|^2) / M )
// Jacobian: dRg/dr_k = (m_k/M) * (r_k - r_COM) / Rg
//
// PBC is NOT applied when computing the COM (molecule assumed whole);
// only applied when measuring each atom's displacement from COM.
// ---------------------------------------------------------------------------
// cvGyration originally had 3 atom loops each re-loading posq (COM, Rg², Jacobian).
// The Rg² and Jacobian loops both compute dr = p - com from the same positions.
// Caching (drx, dry, drz) in the Rg² pass lets the Jacobian pass skip the
// re-load and PBC re-application.  MAX_GYR_ATOMS caps the cache; larger molecules
// fall back to re-loading (correctness preserved either way).
#define MAX_GYR_ATOMS 256
extern const string kGyrationKernelSrc = kGeometryFunctions + R"(
#define MAX_GYR_ATOMS 256
KERNEL void cvGyration(
 GLOBAL const real4* RESTRICT posq,
 GLOBAL const int* RESTRICT atomOffsets,
 GLOBAL const int* RESTRICT atoms,
 GLOBAL const float* RESTRICT masses,
 GLOBAL const float* RESTRICT totalMasses,
 int numCVs,
 int firstCVIndex,
 int firstJacEntry,
 GLOBAL double* RESTRICT cvValues,
 GLOBAL int* RESTRICT jacobianAtomIdx,
 GLOBAL float* RESTRICT jacobianGradsX,
 GLOBAL float* RESTRICT jacobianGradsY,
 GLOBAL float* RESTRICT jacobianGradsZ,
 GLOBAL int* RESTRICT jacobianCvIdx,
 real4 boxSize,
 real4 invBoxSize,
 real4 boxVecX,
 real4 boxVecY,
 real4 boxVecZ,
 int periodic
) {
 for (int i = GLOBAL_ID; i < numCVs; i += GLOBAL_SIZE) {
 int aStart = atomOffsets[i];
 int aEnd = atomOffsets[i + 1];
 int nAtoms = aEnd - aStart;
 float totalM = totalMasses[i];
 real invM = (real)1.0 / (real)totalM;
 bool useCache = (nAtoms <= MAX_GYR_ATOMS);

 // COM pass — unchanged, must precede Rg².
 real3 com = make_real3((real)0, (real)0, (real)0);
 for (int j = 0; j < nAtoms; j++) {
 real4 p = posq[atoms[aStart + j]];
 float m = masses[aStart + j];
 com.x += (real)m * p.x;
 com.y += (real)m * p.y;
 com.z += (real)m * p.z;
 }
 com.x *= invM; com.y *= invM; com.z *= invM;

 // Rg² pass: accumulate and cache (drx,dry,drz) when nAtoms fits.
 float cDrX[MAX_GYR_ATOMS], cDrY[MAX_GYR_ATOMS], cDrZ[MAX_GYR_ATOMS];
 real rg2 = (real)0;
 for (int j = 0; j < nAtoms; j++) {
 real4 p = posq[atoms[aStart + j]];
 float m = masses[aStart + j];
 real3 dr = make_real3(p.x - com.x, p.y - com.y, p.z - com.z);
 if (periodic)
 dr = deltaPBC(dr, invBoxSize, boxVecX, boxVecY, boxVecZ);
 rg2 += (real)m * (dr.x*dr.x + dr.y*dr.y + dr.z*dr.z);
 if (useCache) { cDrX[j] = (float)dr.x; cDrY[j] = (float)dr.y; cDrZ[j] = (float)dr.z; }
 }
 rg2 *= invM;
 real rg = sqrt(rg2);
 real invRg = (rg > (real)1e-10) ? ((real)1.0 / rg) : (real)0.0;
 cvValues[firstCVIndex + i] = (double)rg;

 // Jacobian pass: dRg/dr_k = (m_k/M) * dr_k / Rg — reuse cached dr when available.
 int cvIdx = firstCVIndex + i;
 for (int j = 0; j < nAtoms; j++) {
 float m = masses[aStart + j];
 real3 dr;
 if (useCache) {
  dr = make_real3((real)cDrX[j], (real)cDrY[j], (real)cDrZ[j]);
 } else {
  real4 p = posq[atoms[aStart + j]];
  dr = make_real3(p.x - com.x, p.y - com.y, p.z - com.z);
  if (periodic)
  dr = deltaPBC(dr, invBoxSize, boxVecX, boxVecY, boxVecZ);
 }
 float w = m / totalM;
 int jac = firstJacEntry + aStart + j;
 jacobianAtomIdx[jac] = atoms[aStart + j];
 jacobianGradsX[jac] = (float)((real)w * dr.x * invRg);
 jacobianGradsY[jac] = (float)((real)w * dr.y * invRg);
 jacobianGradsZ[jac] = (float)((real)w * dr.z * invRg);
 jacobianCvIdx[jac] = cvIdx;
 }
 }
}
)";

// ---------------------------------------------------------------------------
// Path CV kernel.
//
// Produces two CV values per call: s (progress, 0-indexed) and z (distance).
// wᵢ = exp(-lambda * RMSD²ᵢ), S = Σwᵢ
// s = Σ(i * wᵢ) / S
// z = -log(S) / lambda
//
// Jacobians (M = atoms per path, N = frames):
// ds/dr_k = (-2λ/(M·S)) · Σᵢ (i−s)·wᵢ·(r_k − r_ref_i_k)
// dz/dr_k = (2/(M·S)) · Σᵢ wᵢ·(r_k − r_ref_i_k)
//
// Uses a local float[MAX_PATH_FRAMES] to cache per-frame weights between
// passes; compile-time cap at 128 frames.
// Also caches atom positions (cPx/cPy/cPz) so Pass 1's inner atoms loop
// loads each position once instead of N times (one per frame).  The same
// cache is reused in Pass 2, eliminating its posq loads entirely.
//
// Args (same count as RMSD/coordination): 0=posq, 1=atomOffsets, 2=atoms,
// 3=refOffsets, 4=refPos, 5=pathParams([lambda,N_frames] per CV),
// 6=numCVs, 7=firstCVIndex, 8=firstJacEntry, 9-14=cvValues+jacobian,
// 15-19=PBC, 20=periodic, 21=spare.
// ---------------------------------------------------------------------------
#define MAX_PATH_FRAMES 128
#define MAX_PATH_ATOMS   64

extern const string kPathKernelSrc = kGeometryFunctions + R"(
#define MAX_PATH_FRAMES 128
#define MAX_PATH_ATOMS   64
KERNEL void cvPath(
 GLOBAL const real4* RESTRICT posq,
 GLOBAL const int* RESTRICT atomOffsets,
 GLOBAL const int* RESTRICT atoms,
 GLOBAL const int* RESTRICT refOffsets,
 GLOBAL const float* RESTRICT refPos,
 GLOBAL const float* RESTRICT pathParams,
 int numCVs,
 int firstCVIndex,
 int firstJacEntry,
 GLOBAL double* RESTRICT cvValues,
 GLOBAL int* RESTRICT jacobianAtomIdx,
 GLOBAL float* RESTRICT jacobianGradsX,
 GLOBAL float* RESTRICT jacobianGradsY,
 GLOBAL float* RESTRICT jacobianGradsZ,
 GLOBAL int* RESTRICT jacobianCvIdx,
 real4 boxSize,
 real4 invBoxSize,
 real4 boxVecX,
 real4 boxVecY,
 real4 boxVecZ,
 int periodic
) {
 for (int i = GLOBAL_ID; i < numCVs; i += GLOBAL_SIZE) {
 int aStart = atomOffsets[i];
 int aEnd = atomOffsets[i + 1];
 int M = aEnd - aStart;
 int refBase = refOffsets[i]; // start in refPos[], units of atoms
 int N = (int)pathParams[2*i + 1];
 real lambda = (real)pathParams[2*i];

 int cvIdxS = firstCVIndex + 2*i;
 int cvIdxZ = firstCVIndex + 2*i + 1;

 // Cache atom positions once to avoid N re-loads of posq per atom in Pass 1,
 // and to eliminate posq loads entirely in Pass 2.
 bool useAtomCache = (M <= MAX_PATH_ATOMS);
 float cPx[MAX_PATH_ATOMS], cPy[MAX_PATH_ATOMS], cPz[MAX_PATH_ATOMS];
 if (useAtomCache) {
 for (int j = 0; j < M; j++) {
  real4 p = posq[atoms[aStart + j]];
  cPx[j] = (float)p.x; cPy[j] = (float)p.y; cPz[j] = (float)p.z;
 }
 }

 // Pass 1: compute per-frame RMSD², cache weights w[f]
 // Frames are 1-indexed (k=1..N) per Branduardi et al. (2007) convention.
 float wBuf[MAX_PATH_FRAMES];
 real S = (real)0, sNum = (real)0;
 for (int f = 0; f < N; f++) {
 real rmsd2 = (real)0;
 for (int j = 0; j < M; j++) {
 real px, py, pz;
 if (useAtomCache) {
  px = (real)cPx[j]; py = (real)cPy[j]; pz = (real)cPz[j];
 } else {
  real4 p = posq[atoms[aStart + j]]; px = p.x; py = p.y; pz = p.z;
 }
 int ri = 3 * (refBase + f*M + j);
 real dx = px - (real)refPos[ri];
 real dy = py - (real)refPos[ri + 1];
 real dz = pz - (real)refPos[ri + 2];
 rmsd2 += dx*dx + dy*dy + dz*dz;
 }
 rmsd2 /= (real)M;
 real w = exp(-lambda * rmsd2);
 wBuf[f] = (float)w;
 S += w;
 sNum += (real)(f + 1) * w;
 }
 real invS = (S > (real)1e-300) ? ((real)1.0 / S) : (real)0.0;
 real s = sNum * invS;
 real z = (S > (real)1e-300) ? (-log(S) / lambda) : (real)0.0;

 cvValues[cvIdxS] = (double)s;
 cvValues[cvIdxZ] = (double)z;

 // Pass 2: accumulate Jacobians per atom
 real scaleS = (real)-2.0 * lambda / ((real)M * S);
 real scaleZ = (real)2.0 / ((real)M * S);

 for (int j = 0; j < M; j++) {
 int gpuAtom = atoms[aStart + j];
 real px, py, pz;
 if (useAtomCache) {
 px = (real)cPx[j]; py = (real)cPy[j]; pz = (real)cPz[j];
 } else {
 real4 p = posq[gpuAtom]; px = p.x; py = p.y; pz = p.z;
 }
 real jacSx = (real)0, jacSy = (real)0, jacSz = (real)0;
 real jacZx = (real)0, jacZy = (real)0, jacZz = (real)0;

 for (int f = 0; f < N; f++) {
 int ri = 3 * (refBase + f*M + j);
 real dx = px - (real)refPos[ri];
 real dy = py - (real)refPos[ri + 1];
 real dz = pz - (real)refPos[ri + 2];
 real w = (real)wBuf[f];

 jacSx += ((real)(f + 1) - s) * w * dx;
 jacSy += ((real)(f + 1) - s) * w * dy;
 jacSz += ((real)(f + 1) - s) * w * dz;
 jacZx += w * dx;
 jacZy += w * dy;
 jacZz += w * dz;
 }

 int jacS = firstJacEntry + 2*(aStart + j);
 int jacZ = firstJacEntry + 2*(aStart + j) + 1;

 jacobianAtomIdx[jacS] = gpuAtom;
 jacobianGradsX[jacS] = (float)(scaleS * jacSx);
 jacobianGradsY[jacS] = (float)(scaleS * jacSy);
 jacobianGradsZ[jacS] = (float)(scaleS * jacSz);
 jacobianCvIdx[jacS] = cvIdxS;

 jacobianAtomIdx[jacZ] = gpuAtom;
 jacobianGradsX[jacZ] = (float)(scaleZ * jacZx);
 jacobianGradsY[jacZ] = (float)(scaleZ * jacZy);
 jacobianGradsZ[jacZ] = (float)(scaleZ * jacZz);
 jacobianCvIdx[jacZ] = cvIdxZ;
 }
 }
}
)";

// ---------------------------------------------------------------------------
// RMSD CV kernel (no-fit / TYPE=SIMPLE).
//
// RMSD = sqrt( (1/N) * sum_i |r_i - r_ref_i|^2 )
// Jacobian: dRMSD/d(r_k) = (r_k - r_ref_k) / (N * RMSD)
//
// Reference positions are stored as float[3*total_atoms] in the same
// atom order as rmsdAtoms[]; they are constant (do not depend on the
// Hilbert-curve reorder of posq).
// ---------------------------------------------------------------------------
// Two-pass RMSD caches displacement (dx,dy,dz) per atom in Pass 1 so Pass 2
// avoids re-loading posq and re-reading refPos.  MAX_RMSD_ATOMS caps the cache;
// larger n falls back to re-loading (correctness preserved either way).
#define MAX_RMSD_ATOMS 512
extern const string kRMSDKernelSrc = kGeometryFunctions + R"(
#define MAX_RMSD_ATOMS 512
KERNEL void cvRMSD(
 GLOBAL const real4* RESTRICT posq,
 GLOBAL const int* RESTRICT atomOffsets,
 GLOBAL const int* RESTRICT atoms,
 GLOBAL const float* RESTRICT refPos,
 int numCVs,
 int firstCVIndex,
 int firstJacEntry,
 GLOBAL double* RESTRICT cvValues,
 GLOBAL int* RESTRICT jacobianAtomIdx,
 GLOBAL float* RESTRICT jacobianGradsX,
 GLOBAL float* RESTRICT jacobianGradsY,
 GLOBAL float* RESTRICT jacobianGradsZ,
 GLOBAL int* RESTRICT jacobianCvIdx,
 real4 boxSize,
 real4 invBoxSize,
 real4 boxVecX,
 real4 boxVecY,
 real4 boxVecZ,
 int periodic
) {
 for (int i = GLOBAL_ID; i < numCVs; i += GLOBAL_SIZE) {
 int aStart = atomOffsets[i];
 int aEnd = atomOffsets[i + 1];
 int n = aEnd - aStart;
 int cvIdx = firstCVIndex + i;
 bool useCache = (n <= MAX_RMSD_ATOMS);

 float cDx[MAX_RMSD_ATOMS], cDy[MAX_RMSD_ATOMS], cDz[MAX_RMSD_ATOMS];

 // Pass 1: accumulate sum of squared displacements; cache (dx,dy,dz) when n fits.
 real rmsd2 = (real)0;
 for (int j = 0; j < n; j++) {
 real4 p = posq[atoms[aStart + j]];
 int ri = 3 * (aStart + j);
 real dx = p.x - (real)refPos[ri];
 real dy = p.y - (real)refPos[ri + 1];
 real dz = p.z - (real)refPos[ri + 2];
 rmsd2 += dx*dx + dy*dy + dz*dz;
 if (useCache) { cDx[j] = (float)dx; cDy[j] = (float)dy; cDz[j] = (float)dz; }
 }
 rmsd2 /= (real)n;
 real rmsd = sqrt(rmsd2);
 real invRmsd = (rmsd > (real)1e-10) ? ((real)1.0 / rmsd) : (real)0.0;
 cvValues[cvIdx] = (double)rmsd;

 // Pass 2: write Jacobian — use cached displacements to skip position re-loads.
 real scale = invRmsd / (real)n;
 for (int j = 0; j < n; j++) {
 int gpuAtom = atoms[aStart + j];
 real dx, dy, dz;
 if (useCache) {
  dx = (real)cDx[j]; dy = (real)cDy[j]; dz = (real)cDz[j];
 } else {
  real4 p = posq[gpuAtom];
  int ri = 3 * (aStart + j);
  dx = p.x - (real)refPos[ri];
  dy = p.y - (real)refPos[ri + 1];
  dz = p.z - (real)refPos[ri + 2];
 }
 int jac = firstJacEntry + aStart + j;
 jacobianAtomIdx[jac] = gpuAtom;
 jacobianGradsX[jac] = (float)(dx * scale);
 jacobianGradsY[jac] = (float)(dy * scale);
 jacobianGradsZ[jac] = (float)(dz * scale);
 jacobianCvIdx[jac] = cvIdx;
 }
 }
}
)";

// ---------------------------------------------------------------------------
// Coordination number CV kernel.
//
// One thread per CV. Rational switching function:
// f(r) = (1 - x^n) / (1 - x^m), x = r/r0, clamped to 0 for r >= r0.
// CN = sum_{k in A, l in B} f(r_kl).
// Jacobian accumulated per-atom across all pairs involving that atom.
// dCN/d(r_k) = sum_l dfdr * (r_k - r_l)/r
// dCN/d(r_l) = -sum_k dfdr * (r_k - r_l)/r
// atomOffsets[2*N+1] is interleaved: [g1Start, g2Start, g2End, ...].
// ---------------------------------------------------------------------------
// Pre-load g2 atom positions before the k×l nested loop so each pL is fetched
// once from global memory instead of ng1 times.  Savings: ng2×(ng1-1) reads.
// MAX_COORD_G2 caps the cache; larger g2 groups fall back to per-iteration loads.
#define MAX_COORD_G2 256

extern const string kCoordinationKernelSrc = kGeometryFunctions + R"(
#define MAX_COORD_G2 256
KERNEL void cvCoordination(
 GLOBAL const real4* RESTRICT posq,
 GLOBAL const int* RESTRICT atomOffsets,
 GLOBAL const int* RESTRICT atoms,
 GLOBAL const float* RESTRICT params,
 int numCVs,
 int firstCVIndex,
 int firstJacEntry,
 GLOBAL double* RESTRICT cvValues,
 GLOBAL int* RESTRICT jacobianAtomIdx,
 GLOBAL float* RESTRICT jacobianGradsX,
 GLOBAL float* RESTRICT jacobianGradsY,
 GLOBAL float* RESTRICT jacobianGradsZ,
 GLOBAL int* RESTRICT jacobianCvIdx,
 real4 boxSize,
 real4 invBoxSize,
 real4 boxVecX,
 real4 boxVecY,
 real4 boxVecZ,
 int periodic
) {
 for (int i = GLOBAL_ID; i < numCVs; i += GLOBAL_SIZE) {
 int g1Start = atomOffsets[2*i];
 int g2Start = atomOffsets[2*i + 1];
 int g2End = atomOffsets[2*i + 2];
 int ng1 = g2Start - g1Start;
 int ng2 = g2End - g2Start;
 int cvIdx = firstCVIndex + i;

 real r0 = (real)params[3*i];
 real rn = (real)params[3*i + 1];
 real rm = (real)params[3*i + 2];

 // Zero-initialize Jacobian entries and set atom + CV indices
 for (int j = 0; j < ng1; j++) {
 int jac = firstJacEntry + g1Start + j;
 jacobianAtomIdx[jac] = atoms[g1Start + j];
 jacobianGradsX[jac] = (float)0;
 jacobianGradsY[jac] = (float)0;
 jacobianGradsZ[jac] = (float)0;
 jacobianCvIdx[jac] = cvIdx;
 }
 for (int j = 0; j < ng2; j++) {
 int jac = firstJacEntry + g2Start + j;
 jacobianAtomIdx[jac] = atoms[g2Start + j];
 jacobianGradsX[jac] = (float)0;
 jacobianGradsY[jac] = (float)0;
 jacobianGradsZ[jac] = (float)0;
 jacobianCvIdx[jac] = cvIdx;
 }

 // Taylor coefficients for (1−x^n)/(1−x^m) around x=1, derived from standard calculus.
 real preRes    = rn / rm;
 real preDfunc  = (real)0.5 * rn * (rn - rm) / rm;
 real preSecDev = rn * (rm*rm - (real)3.0*rm*(rn-(real)1.0)
                      + rn*((real)2.0*rn-(real)3.0)) / ((real)6.0*rm);

 // Cache g2 positions once; reused across all ng1 outer iterations.
 bool useG2Cache = (ng2 <= MAX_COORD_G2);
 float cLx[MAX_COORD_G2], cLy[MAX_COORD_G2], cLz[MAX_COORD_G2];
 if (useG2Cache) {
 for (int l = 0; l < ng2; l++) {
  real4 pL = posq[atoms[g2Start + l]];
  cLx[l] = (float)pL.x; cLy[l] = (float)pL.y; cLz[l] = (float)pL.z;
 }
 }

 real cn = (real)0;
 for (int k = 0; k < ng1; k++) {
 real4 pK = posq[atoms[g1Start + k]];
 for (int l = 0; l < ng2; l++) {
 real pLx, pLy, pLz;
 if (useG2Cache) {
  pLx = (real)cLx[l]; pLy = (real)cLy[l]; pLz = (real)cLz[l];
 } else {
  real4 pL = posq[atoms[g2Start + l]];
  pLx = pL.x; pLy = pL.y; pLz = pL.z;
 }
 real3 dr = make_real3(pK.x - pLx, pK.y - pLy, pK.z - pLz);
 if (periodic)
 dr = deltaPBC(dr, invBoxSize, boxVecX, boxVecY, boxVecZ);
 real r2 = dr.x*dr.x + dr.y*dr.y + dr.z*dr.z;
 if (r2 < (real)1e-20) continue;
 real r = sqrt(r2);
 real x = r / r0;
 // Skip only very distant pairs where pow() would overflow.
 if (x > (real)1000.0) continue;

 real f, dfdr;
 real dx1 = x - (real)1.0;
 if (fabs(dx1) < (real)1e-4) {
 // Taylor expansion around x=1
 f    = preRes + dx1 * (preDfunc + (real)0.5 * dx1 * preSecDev);
 real dfx = preDfunc + dx1 * preSecDev;
 dfdr = dfx / r0;
 } else if (x < (real)1e-10) {
 f    = (real)1;
 dfdr = (real)0;
 } else {
 real xn = pow(x, rn);
 real xm = pow(x, rm);
 real denom = (real)1 - xm;
 f = ((real)1 - xn) / denom;
 real xnm1 = xn / x;
 real xmm1 = xm / x;
 real xnpm1 = xn * xm / x;
 real dfx = (rm * xmm1 - rn * xnm1 + (rn - rm) * xnpm1)
 / (denom * denom);
 dfdr = dfx / r0;
 }
 cn += f;

 real invR = (real)1 / r;
 float gx = (float)(dfdr * dr.x * invR);
 float gy = (float)(dfdr * dr.y * invR);
 float gz = (float)(dfdr * dr.z * invR);

 int jacK = firstJacEntry + g1Start + k;
 int jacL = firstJacEntry + g2Start + l;
 jacobianGradsX[jacK] += gx;
 jacobianGradsY[jacK] += gy;
 jacobianGradsZ[jacK] += gz;
 jacobianGradsX[jacL] -= gx;
 jacobianGradsY[jacL] -= gy;
 jacobianGradsZ[jacL] -= gz;
 }
 }
 cvValues[cvIdx] = (double)cn;
 }
}
)";

// ---------------------------------------------------------------------------
// Chain-rule scatter kernel.
//
// One thread per Jacobian entry. Applies:
// force_i -= (dU/dCV_k) * (dCV_k/dr_i)
// in double arithmetic, then accumulates into the fixed-point force buffer.
//
// jacobianAtomIdx[j] is a GPU-space atom index; the buffer is field-major
// with stride paddedNumAtoms.
// ---------------------------------------------------------------------------
// PyTorch CV helper kernels.
// extractPos: gather atom positions from posq (real4) into a flat float[N*3] buffer for torch.
// deinterleavGrad: scatter interleaved float[N*3] torch gradient into separate XYZ Jacobian arrays.
extern const string kPyTorchKernelSrc = R"(
__kernel void pytorchExtractPos(
 __global const real4* posq,
 __global const int* atomIdx,
 __global float* posBuf,
 int numAtoms)
{
 int i = get_global_id(0);
 if (i >= numAtoms) return;
 real4 p = posq[atomIdx[i]];
 posBuf[3*i] = (float)p.x;
 posBuf[3*i+1] = (float)p.y;
 posBuf[3*i+2] = (float)p.z;
}

__kernel void pytorchDeinterleavGrad(
 __global const float* gradBuf,
 __global float* jacX,
 __global float* jacY,
 __global float* jacZ,
 int jacOffset,
 int numAtoms)
{
 int i = get_global_id(0);
 if (i >= numAtoms) return;
 jacX[jacOffset + i] = gradBuf[3*i];
 jacY[jacOffset + i] = gradBuf[3*i+1];
 jacZ[jacOffset + i] = gradBuf[3*i+2];
}
)";

// ---------------------------------------------------------------------------
// Position CV kernel.
//
// One thread per CV. Returns the x/y/z coordinate of a single atom.
// No PBC: position is reported in the lab frame.
// Jacobian: 1 entry per CV with gradient = canonical unit vector.
// ---------------------------------------------------------------------------
extern const string kPositionKernelSrc = R"(
KERNEL void cvPosition(
 GLOBAL const real4* RESTRICT posq,
 GLOBAL const int* RESTRICT positionAtoms, // int[N]: GPU atom idx
 GLOBAL const int* RESTRICT positionComponents, // int[N]: 0=x, 1=y, 2=z
 int numCVs,
 int firstCVIndex,
 int firstJacEntry,
 GLOBAL double* RESTRICT cvValues,
 GLOBAL int* RESTRICT jacobianAtomIdx,
 GLOBAL float* RESTRICT jacobianGradsX,
 GLOBAL float* RESTRICT jacobianGradsY,
 GLOBAL float* RESTRICT jacobianGradsZ,
 GLOBAL int* RESTRICT jacobianCvIdx
) {
 for (int i = GLOBAL_ID; i < numCVs; i += GLOBAL_SIZE) {
 int gpuA = positionAtoms[i];
 int comp = positionComponents[i];
 real4 pos = posq[gpuA];
 real val = (comp == 0) ? pos.x : (comp == 1) ? pos.y : pos.z;
 cvValues[firstCVIndex + i] = (double)val;

 int jac = firstJacEntry + i;
 jacobianAtomIdx[jac] = gpuA;
 jacobianGradsX[jac] = (comp == 0) ? (float)1.0 : (float)0.0;
 jacobianGradsY[jac] = (comp == 1) ? (float)1.0 : (float)0.0;
 jacobianGradsZ[jac] = (comp == 2) ? (float)1.0 : (float)0.0;
 jacobianCvIdx[jac] = firstCVIndex + i;
 }
}
)";

// ---------------------------------------------------------------------------
// DRMSD (distance RMSD) CV kernel.
//
// DRMSD = sqrt( 1/N * sum_p (d_p - d0_p)^2 )
// One thread per CV. pairOffsets[i..i+1] gives the half-open range into
// pairAtoms[] and refDists[] for CV i. Jacobian: 2 entries per pair.
// dCV/dr_A = -(d-d0)/(N*DRMSD*d) * dr (dr = r_B - r_A)
// dCV/dr_B = +(d-d0)/(N*DRMSD*d) * dr
//
// Two-pass structure caches (dr, dist^{-1}, dev) in local arrays between
// passes, eliminating redundant global loads and sqrt calls in Pass 2.
// MAX_DRMSD_PAIRS caps the cache; nPairs above this falls back to re-loading.
// ---------------------------------------------------------------------------
#define MAX_DRMSD_PAIRS 512
extern const string kDRMSDKernelSrc = kGeometryFunctions + R"(
#define MAX_DRMSD_PAIRS 512
KERNEL void cvDRMSD(
 GLOBAL const real4* RESTRICT posq,
 GLOBAL const int* RESTRICT pairOffsets,
 GLOBAL const int* RESTRICT pairAtoms,
 GLOBAL const float* RESTRICT refDists,
 int numCVs,
 int firstCVIndex,
 int firstJacEntry,
 GLOBAL double* RESTRICT cvValues,
 GLOBAL int* RESTRICT jacobianAtomIdx,
 GLOBAL float* RESTRICT jacobianGradsX,
 GLOBAL float* RESTRICT jacobianGradsY,
 GLOBAL float* RESTRICT jacobianGradsZ,
 GLOBAL int* RESTRICT jacobianCvIdx,
 real4 boxSize,
 real4 invBoxSize,
 real4 boxVecX,
 real4 boxVecY,
 real4 boxVecZ,
 int periodic
) {
 for (int i = GLOBAL_ID; i < numCVs; i += GLOBAL_SIZE) {
 int pStart = pairOffsets[i];
 int pEnd = pairOffsets[i + 1];
 int nPairs = pEnd - pStart;
 bool useCache = (nPairs <= MAX_DRMSD_PAIRS);

 float cDrX[MAX_DRMSD_PAIRS], cDrY[MAX_DRMSD_PAIRS], cDrZ[MAX_DRMSD_PAIRS];
 float cDev[MAX_DRMSD_PAIRS], cInvDist[MAX_DRMSD_PAIRS];

 // Pass 1: accumulate DRMSD²; cache (dr, dev, invDist) when nPairs fits.
 real sumSq = (real)0;
 for (int p = 0; p < nPairs; p++) {
 int gpuA = pairAtoms[2*(pStart+p)];
 int gpuB = pairAtoms[2*(pStart+p)+1];
 real4 pA = posq[gpuA]; real4 pB = posq[gpuB];
 real3 dr = make_real3(pB.x-pA.x, pB.y-pA.y, pB.z-pA.z);
 if (periodic) dr = deltaPBC(dr, invBoxSize, boxVecX, boxVecY, boxVecZ);
 real dist = sqrt(dr.x*dr.x + dr.y*dr.y + dr.z*dr.z);
 real dev = dist - (real)refDists[pStart+p];
 sumSq += dev * dev;
 if (useCache) {
  cDrX[p] = (float)dr.x; cDrY[p] = (float)dr.y; cDrZ[p] = (float)dr.z;
  cDev[p] = (float)dev;
  cInvDist[p] = (dist > (real)1e-10) ? (float)((real)1.0/dist) : 0.0f;
 }
 }
 real drmsd = sqrt(sumSq / (real)nPairs);
 cvValues[firstCVIndex + i] = (double)drmsd;

 real invDRMSD = (drmsd > (real)1e-10) ? ((real)1.0 / drmsd) : (real)0.0;
 real invN = (real)1.0 / (real)nPairs;
 int cvIdx = firstCVIndex + i;

 // Pass 2: write Jacobian — use cached values to skip position re-loads and sqrt.
 for (int p = 0; p < nPairs; p++) {
 real3 dr; real invDist; real dev;
 if (useCache) {
  dr = make_real3((real)cDrX[p], (real)cDrY[p], (real)cDrZ[p]);
  dev = (real)cDev[p];
  invDist = (real)cInvDist[p];
 } else {
  int gpuA2 = pairAtoms[2*(pStart+p)];
  int gpuB2 = pairAtoms[2*(pStart+p)+1];
  real4 pA2 = posq[gpuA2]; real4 pB2 = posq[gpuB2];
  dr = make_real3(pB2.x-pA2.x, pB2.y-pA2.y, pB2.z-pA2.z);
  if (periodic) dr = deltaPBC(dr, invBoxSize, boxVecX, boxVecY, boxVecZ);
  real dist2 = sqrt(dr.x*dr.x + dr.y*dr.y + dr.z*dr.z);
  invDist = (dist2 > (real)1e-10) ? ((real)1.0/dist2) : (real)0.0;
  dev = dist2 - (real)refDists[pStart+p];
 }
 real coeff = dev * invN * invDRMSD * invDist;

 int gpuA = pairAtoms[2*(pStart+p)];
 int gpuB = pairAtoms[2*(pStart+p)+1];
 int jacA = firstJacEntry + 2*(pStart+p);
 int jacB = firstJacEntry + 2*(pStart+p)+1;

 jacobianAtomIdx[jacA] = gpuA;
 jacobianGradsX[jacA] = (float)(-coeff * dr.x);
 jacobianGradsY[jacA] = (float)(-coeff * dr.y);
 jacobianGradsZ[jacA] = (float)(-coeff * dr.z);
 jacobianCvIdx[jacA] = cvIdx;

 jacobianAtomIdx[jacB] = gpuB;
 jacobianGradsX[jacB] = (float)(coeff * dr.x);
 jacobianGradsY[jacB] = (float)(coeff * dr.y);
 jacobianGradsZ[jacB] = (float)(coeff * dr.z);
 jacobianCvIdx[jacB] = cvIdx;
 }
 }
}
)";

// ---------------------------------------------------------------------------
// Contact-map CV kernel.
//
// CV = sum_p w_p * s(d_p; r0_p, nn_p, mm_p)
// s(d; r0, n, m) = (1 - (d/r0)^n) / (1 - (d/r0)^m) [rational switch]
// At d = r0 the fraction is 0/0 → L'Hopital gives s = n/m.
// Jacobian: 2 entries per pair.
// dCV/dr_A = w_p * (ds/dd) * (-unit_dr)
// dCV/dr_B = w_p * (ds/dd) * (+unit_dr)
// ---------------------------------------------------------------------------
extern const string kContactMapKernelSrc = kGeometryFunctions + R"(
DEVICE inline real ipow_cm(real x, int n) {
 real r = (real)1.0;
 for (int i = 0; i < n; i++) r *= x;
 return r;
}
KERNEL void cvContactMap(
 GLOBAL const real4* RESTRICT posq,
 GLOBAL const int* RESTRICT pairOffsets,
 GLOBAL const int* RESTRICT pairAtoms,
 GLOBAL const float* RESTRICT pairParams, // [r0, nn, mm, w] per pair (4 floats)
 int numCVs,
 int firstCVIndex,
 int firstJacEntry,
 GLOBAL double* RESTRICT cvValues,
 GLOBAL int* RESTRICT jacobianAtomIdx,
 GLOBAL float* RESTRICT jacobianGradsX,
 GLOBAL float* RESTRICT jacobianGradsY,
 GLOBAL float* RESTRICT jacobianGradsZ,
 GLOBAL int* RESTRICT jacobianCvIdx,
 real4 boxSize,
 real4 invBoxSize,
 real4 boxVecX,
 real4 boxVecY,
 real4 boxVecZ,
 int periodic
) {
 for (int i = GLOBAL_ID; i < numCVs; i += GLOBAL_SIZE) {
 int pStart = pairOffsets[i];
 int pEnd = pairOffsets[i + 1];
 int nPairs = pEnd - pStart;

 real cvVal = (real)0;
 for (int p = 0; p < nPairs; p++) {
 int gpuA = pairAtoms[2*(pStart+p)];
 int gpuB = pairAtoms[2*(pStart+p)+1];
 real r0 = (real)pairParams[4*(pStart+p)];
 int nn = (int) pairParams[4*(pStart+p)+1];
 int mm = (int) pairParams[4*(pStart+p)+2];
 real w = (real)pairParams[4*(pStart+p)+3];

 real4 pA = posq[gpuA]; real4 pB = posq[gpuB];
 real3 dr = make_real3(pB.x-pA.x, pB.y-pA.y, pB.z-pA.z);
 if (periodic) dr = deltaPBC(dr, invBoxSize, boxVecX, boxVecY, boxVecZ);
 real dist = sqrt(dr.x*dr.x + dr.y*dr.y + dr.z*dr.z);
 real invDist = (dist > (real)1e-10) ? ((real)1.0/dist) : (real)0.0;
 real invR0 = ((real)1.0) / r0;

 real x = dist * invR0;
 real xnn = ipow_cm(x, nn);
 real xmm = ipow_cm(x, mm);
 real num = (real)1.0 - xnn;
 real den = (real)1.0 - xmm;

 real s, dsdd;
 if (fabs(den) < (real)1e-8) {
 s = (real)nn / (real)mm;
 dsdd = (real)0;
 } else {
 s = num / den;
 real dxnn = (nn > 0) ? (real)nn * ipow_cm(x, nn-1) : (real)0;
 real dxmm = (mm > 0) ? (real)mm * ipow_cm(x, mm-1) : (real)0;
 real dsdx = (-dxnn * den + dxmm * num) / (den * den);
 dsdd = dsdx * invR0;
 }
 cvVal += w * s;

 real coeff = w * dsdd * invDist;
 int jacA = firstJacEntry + 2*(pStart+p);
 int jacB = firstJacEntry + 2*(pStart+p)+1;
 int cvIdx = firstCVIndex + i;

 jacobianAtomIdx[jacA] = gpuA;
 jacobianGradsX[jacA] = (float)(-coeff * dr.x);
 jacobianGradsY[jacA] = (float)(-coeff * dr.y);
 jacobianGradsZ[jacA] = (float)(-coeff * dr.z);
 jacobianCvIdx[jacA] = cvIdx;

 jacobianAtomIdx[jacB] = gpuB;
 jacobianGradsX[jacB] = (float)(coeff * dr.x);
 jacobianGradsY[jacB] = (float)(coeff * dr.y);
 jacobianGradsZ[jacB] = (float)(coeff * dr.z);
 jacobianCvIdx[jacB] = cvIdx;
 }
 cvValues[firstCVIndex + i] = (double)cvVal;
 }
}
)";

// ---------------------------------------------------------------------------
// Plane CV kernel.
//
// CV = n_hat[component] where n = (b-a) × (c-a), n_hat = n / |n|.
// 3 Jacobian entries per CV: grad_a = -(grad_b + grad_c),
// grad_b = (1/|n|) * (v×e_k - n_hat[k]*(v×n_hat))
// grad_c = (1/|n|) * (e_k×u - n_hat[k]*(n_hat×u))
// where u=b-a, v=c-a, k=component.
// ---------------------------------------------------------------------------
extern const string kPlaneKernelSrc = kGeometryFunctions + R"(
KERNEL void cvPlane(
 GLOBAL const real4* RESTRICT posq,
 GLOBAL const int* RESTRICT planeAtoms,
 GLOBAL const int* RESTRICT planeComponents,
 int numCVs,
 int firstCVIndex,
 int firstJacEntry,
 GLOBAL double* RESTRICT cvValues,
 GLOBAL int* RESTRICT jacobianAtomIdx,
 GLOBAL float* RESTRICT jacobianGradsX,
 GLOBAL float* RESTRICT jacobianGradsY,
 GLOBAL float* RESTRICT jacobianGradsZ,
 GLOBAL int* RESTRICT jacobianCvIdx,
 real4 boxSize,
 real4 invBoxSize,
 real4 boxVecX,
 real4 boxVecY,
 real4 boxVecZ,
 int periodic,
 int spare
) {
 for (int i = GLOBAL_ID; i < numCVs; i += GLOBAL_SIZE) {
 int gpuA = planeAtoms[3*i];
 int gpuB = planeAtoms[3*i + 1];
 int gpuC = planeAtoms[3*i + 2];
 int k = planeComponents[i];

 real4 pA = posq[gpuA], pB = posq[gpuB], pC = posq[gpuC];
 real3 u = make_real3(pB.x-pA.x, pB.y-pA.y, pB.z-pA.z);
 real3 v = make_real3(pC.x-pA.x, pC.y-pA.y, pC.z-pA.z);
 if (periodic) {
 u = deltaPBC(u, invBoxSize, boxVecX, boxVecY, boxVecZ);
 v = deltaPBC(v, invBoxSize, boxVecX, boxVecY, boxVecZ);
 }

 real3 n = make_real3(u.y*v.z - u.z*v.y,
 u.z*v.x - u.x*v.z,
 u.x*v.y - u.y*v.x);
 real len = sqrt(n.x*n.x + n.y*n.y + n.z*n.z);
 real invLen = (len > (real)1e-10) ? ((real)1.0/len) : (real)0.0;
 real3 nh = make_real3(n.x*invLen, n.y*invLen, n.z*invLen);

 real cvVal = (k == 0) ? nh.x : ((k == 1) ? nh.y : nh.z);
 cvValues[firstCVIndex + i] = (double)cvVal;

 // Row k of ∂n/∂u (= v × e_k)
 real3 vXek;
 if (k == 0) vXek = make_real3((real)0, v.z, -v.y);
 else if (k == 1) vXek = make_real3(-v.z, (real)0, v.x);
 else vXek = make_real3( v.y, -v.x, (real)0);

 // Row k of ∂n/∂v (= e_k × u)
 real3 ekXu;
 if (k == 0) ekXu = make_real3((real)0, -u.z, u.y);
 else if (k == 1) ekXu = make_real3( u.z, (real)0, -u.x);
 else ekXu = make_real3(-u.y, u.x, (real)0);

 // v × n_hat and n_hat × u (for the projection term)
 real3 vXnh = make_real3(v.y*nh.z - v.z*nh.y,
 v.z*nh.x - v.x*nh.z,
 v.x*nh.y - v.y*nh.x);
 real3 nhXu = make_real3(nh.y*u.z - nh.z*u.y,
 nh.z*u.x - nh.x*u.z,
 nh.x*u.y - nh.y*u.x);

 real3 grad_b = make_real3(invLen*(vXek.x - cvVal*vXnh.x),
 invLen*(vXek.y - cvVal*vXnh.y),
 invLen*(vXek.z - cvVal*vXnh.z));
 real3 grad_c = make_real3(invLen*(ekXu.x - cvVal*nhXu.x),
 invLen*(ekXu.y - cvVal*nhXu.y),
 invLen*(ekXu.z - cvVal*nhXu.z));
 real3 grad_a = make_real3(-(grad_b.x + grad_c.x),
 -(grad_b.y + grad_c.y),
 -(grad_b.z + grad_c.z));

 int jac0 = firstJacEntry + 3*i;
 int cvIdx = firstCVIndex + i;

 jacobianAtomIdx[jac0] = gpuA;
 jacobianGradsX[jac0] = (float)grad_a.x;
 jacobianGradsY[jac0] = (float)grad_a.y;
 jacobianGradsZ[jac0] = (float)grad_a.z;
 jacobianCvIdx[jac0] = cvIdx;

 jacobianAtomIdx[jac0+1] = gpuB;
 jacobianGradsX[jac0+1] = (float)grad_b.x;
 jacobianGradsY[jac0+1] = (float)grad_b.y;
 jacobianGradsZ[jac0+1] = (float)grad_b.z;
 jacobianCvIdx[jac0+1] = cvIdx;

 jacobianAtomIdx[jac0+2] = gpuC;
 jacobianGradsX[jac0+2] = (float)grad_c.x;
 jacobianGradsY[jac0+2] = (float)grad_c.y;
 jacobianGradsZ[jac0+2] = (float)grad_c.z;
 jacobianCvIdx[jac0+2] = cvIdx;
 }
}
)";

// ---------------------------------------------------------------------------
// Projection CV kernel.
//
// CV = dot(r_b - r_a, d_hat) where d_hat is a pre-normalized direction.
// Jacobian: grad_a = -d_hat, grad_b = +d_hat.
// ---------------------------------------------------------------------------
extern const string kProjectionKernelSrc = kGeometryFunctions + R"(
KERNEL void cvProjection(
 GLOBAL const real4* RESTRICT posq,
 GLOBAL const int* RESTRICT projAtoms,
 GLOBAL const float* RESTRICT projDirs, // [nx,ny,nz] per CV (pre-normalized)
 int numCVs,
 int firstCVIndex,
 int firstJacEntry,
 GLOBAL double* RESTRICT cvValues,
 GLOBAL int* RESTRICT jacobianAtomIdx,
 GLOBAL float* RESTRICT jacobianGradsX,
 GLOBAL float* RESTRICT jacobianGradsY,
 GLOBAL float* RESTRICT jacobianGradsZ,
 GLOBAL int* RESTRICT jacobianCvIdx,
 real4 boxSize,
 real4 invBoxSize,
 real4 boxVecX,
 real4 boxVecY,
 real4 boxVecZ,
 int periodic,
 int spare
) {
 for (int i = GLOBAL_ID; i < numCVs; i += GLOBAL_SIZE) {
 int gpuA = projAtoms[2*i];
 int gpuB = projAtoms[2*i + 1];
 real4 pA = posq[gpuA], pB = posq[gpuB];
 real3 dr = make_real3(pB.x-pA.x, pB.y-pA.y, pB.z-pA.z);
 if (periodic)
 dr = deltaPBC(dr, invBoxSize, boxVecX, boxVecY, boxVecZ);

 float dx = projDirs[3*i],
 dy = projDirs[3*i + 1],
 dz = projDirs[3*i + 2];

 cvValues[firstCVIndex + i] = (double)(dr.x*dx + dr.y*dy + dr.z*dz);

 int jac0 = firstJacEntry + 2*i;
 int cvIdx = firstCVIndex + i;

 jacobianAtomIdx[jac0] = gpuA;
 jacobianGradsX[jac0] = -dx;
 jacobianGradsY[jac0] = -dy;
 jacobianGradsZ[jac0] = -dz;
 jacobianCvIdx[jac0] = cvIdx;

 jacobianAtomIdx[jac0+1] = gpuB;
 jacobianGradsX[jac0+1] = dx;
 jacobianGradsY[jac0+1] = dy;
 jacobianGradsZ[jac0+1] = dz;
 jacobianCvIdx[jac0+1] = cvIdx;
 }
}
)";

// ---------------------------------------------------------------------------
// +3.16 — Volume and Cell CVs kernel (single-thread).
//
// Volume: V = boxVecX.x * boxVecY.y * boxVecZ.z
// Cell lengths: |a| = boxVecX.x
// |b| = sqrt(boxVecY.x^2 + boxVecY.y^2)
// |c| = sqrt(boxVecZ.x^2 + boxVecZ.y^2 + boxVecZ.z^2)
// No atom Jacobian entries for either kind.
// ---------------------------------------------------------------------------
extern const string kVolumeCellKernelSrc = R"(
KERNEL void cvVolumeCell(
 int numVolumeCVs,
 int volumeFirstCVIndex,
 int numCellCVs,
 int cellFirstCVIndex,
 GLOBAL const int* RESTRICT cellComponents,
 GLOBAL double* RESTRICT cvValues,
 real4 boxSize,
 real4 invBoxSize,
 real4 boxVecX,
 real4 boxVecY,
 real4 boxVecZ,
 int spare
) {
 if (GLOBAL_ID != 0) return;
 real vol = boxVecX.x * boxVecY.y * boxVecZ.z;
 for (int i = 0; i < numVolumeCVs; i++)
 cvValues[volumeFirstCVIndex + i] = (double)vol;

 for (int i = 0; i < numCellCVs; i++) {
 int comp = cellComponents[i];
 real val;
 if (comp == 0)
 val = boxVecX.x;
 else if (comp == 1)
 val = sqrt(boxVecY.x*boxVecY.x + boxVecY.y*boxVecY.y);
 else
 val = sqrt(boxVecZ.x*boxVecZ.x + boxVecZ.y*boxVecZ.y + boxVecZ.z*boxVecZ.z);
 cvValues[cellFirstCVIndex + i] = (double)val;
 }
}
)";

// ---------------------------------------------------------------------------
// Dipole CV kernel.
//
// CV = μ_k (k=0→x, 1→y, 2→z) or CV = |μ| (k=3)
// μ = Σ_i q_i * r_i (sum over the selected atom group)
// Jacobian for component k: grad_i = q_i * e_k
// Jacobian for magnitude: grad_i = q_i * μ_hat (guarded: 0 if |μ| < 1e-10)
// N Jacobian entries per CV.
// ---------------------------------------------------------------------------
extern const string kDipoleKernelSrc = R"(
KERNEL void cvDipole(
 GLOBAL const real4* RESTRICT posq,
 GLOBAL const int* RESTRICT atomOffsets,
 GLOBAL const int* RESTRICT atoms,
 GLOBAL const float* RESTRICT charges,
 GLOBAL const int* RESTRICT dipoleComponents,
 int numCVs,
 int firstCVIndex,
 int firstJacEntry,
 GLOBAL double* RESTRICT cvValues,
 GLOBAL int* RESTRICT jacobianAtomIdx,
 GLOBAL float* RESTRICT jacobianGradsX,
 GLOBAL float* RESTRICT jacobianGradsY,
 GLOBAL float* RESTRICT jacobianGradsZ,
 GLOBAL int* RESTRICT jacobianCvIdx,
 int spare
) {
 for (int i = GLOBAL_ID; i < numCVs; i += GLOBAL_SIZE) {
 int aStart = atomOffsets[i];
 int aEnd = atomOffsets[i + 1];
 int nAtoms = aEnd - aStart;
 int comp = dipoleComponents[i];
 int cvIdx = firstCVIndex + i;

 real3 mu = make_real3((real)0, (real)0, (real)0);
 for (int j = 0; j < nAtoms; j++) {
 real4 p = posq[atoms[aStart + j]];
 real q = (real)charges[aStart + j];
 mu.x += q * p.x;
 mu.y += q * p.y;
 mu.z += q * p.z;
 }

 real cvVal;
 if (comp == 0) cvVal = mu.x;
 else if (comp == 1) cvVal = mu.y;
 else if (comp == 2) cvVal = mu.z;
 else {
 cvVal = sqrt(mu.x*mu.x + mu.y*mu.y + mu.z*mu.z);
 }
 cvValues[cvIdx] = (double)cvVal;

 real invMag = (comp == 3 && cvVal > (real)1e-10) ? ((real)1.0 / cvVal) : (real)0.0;

 for (int j = 0; j < nAtoms; j++) {
 int jac = firstJacEntry + aStart + j;
 real q = (real)charges[aStart + j];
 jacobianAtomIdx[jac] = atoms[aStart + j];
 jacobianCvIdx[jac] = cvIdx;
 if (comp == 0) {
 jacobianGradsX[jac] = (float)q;
 jacobianGradsY[jac] = (float)0;
 jacobianGradsZ[jac] = (float)0;
 } else if (comp == 1) {
 jacobianGradsX[jac] = (float)0;
 jacobianGradsY[jac] = (float)q;
 jacobianGradsZ[jac] = (float)0;
 } else if (comp == 2) {
 jacobianGradsX[jac] = (float)0;
 jacobianGradsY[jac] = (float)0;
 jacobianGradsZ[jac] = (float)q;
 } else {
 jacobianGradsX[jac] = (float)(q * mu.x * invMag);
 jacobianGradsY[jac] = (float)(q * mu.y * invMag);
 jacobianGradsZ[jac] = (float)(q * mu.z * invMag);
 }
 }
 }
}
)";

// ---------------------------------------------------------------------------
// PCA CV kernel.
//
// CV = dot(r_flat - mean_flat, ev_flat)
// = Σ_i [ (r_i - ref_i) · ev_i ]
// Jacobian: grad_i = ev_i (the eigenvector slice for atom i)
// N Jacobian entries per CV.
// ---------------------------------------------------------------------------
extern const string kPCAKernelSrc = R"(
KERNEL void cvPCA(
 GLOBAL const real4* RESTRICT posq,
 GLOBAL const int* RESTRICT atomOffsets,
 GLOBAL const int* RESTRICT atoms,
 GLOBAL const float* RESTRICT refPos, // [x,y,z] per atom (3*total floats)
 GLOBAL const float* RESTRICT eigvec, // [x,y,z] per atom (3*total floats)
 int numCVs,
 int firstCVIndex,
 int firstJacEntry,
 GLOBAL double* RESTRICT cvValues,
 GLOBAL int* RESTRICT jacobianAtomIdx,
 GLOBAL float* RESTRICT jacobianGradsX,
 GLOBAL float* RESTRICT jacobianGradsY,
 GLOBAL float* RESTRICT jacobianGradsZ,
 GLOBAL int* RESTRICT jacobianCvIdx,
 int spare
) {
 for (int i = GLOBAL_ID; i < numCVs; i += GLOBAL_SIZE) {
 int aStart = atomOffsets[i];
 int aEnd = atomOffsets[i + 1];
 int nAtoms = aEnd - aStart;
 int cvIdx = firstCVIndex + i;

 double cv = 0.0;
 for (int j = 0; j < nAtoms; j++) {
 int flat = aStart + j;
 real4 p = posq[atoms[flat]];
 real dx = p.x - (real)refPos[3*flat];
 real dy = p.y - (real)refPos[3*flat + 1];
 real dz = p.z - (real)refPos[3*flat + 2];
 cv += (double)(dx * (real)eigvec[3*flat]
 + dy * (real)eigvec[3*flat + 1]
 + dz * (real)eigvec[3*flat + 2]);
 }
 cvValues[cvIdx] = cv;

 for (int j = 0; j < nAtoms; j++) {
 int flat = aStart + j;
 int jac = firstJacEntry + flat;
 jacobianAtomIdx[jac] = atoms[flat];
 jacobianGradsX[jac] = eigvec[3*flat];
 jacobianGradsY[jac] = eigvec[3*flat + 1];
 jacobianGradsZ[jac] = eigvec[3*flat + 2];
 jacobianCvIdx[jac] = cvIdx;
 }
 }
}
)";

// ---------------------------------------------------------------------------
// eRMSD for RNA structure comparison (Bottaro et al. 2014, Nucleic Acids Res. 42:13306).
//
// One thread per ERMSD CV. Supports up to N=64 residues per CV (local array
// limit). Each residue is defined by 3 atoms whose centroid forms the frame.
//
// Implements the full 4-D G-vector with form_factor=[2,2,1/0.3] and
// user-supplied cutoff (default 2.4 nm). All N*(N-1) ordered pairs
// (i,j) with i≠j contribute; normalization divides by N.
//
// Gradient: centroid-only approximation (frame-rotation terms omitted).
// Reference: Bottaro et al. (2014) Nucleic Acids Res. 42:13306.
//
// Two-pass cache: when N ≤ MAX_ERMSD_CACHE_N, stores (sc, co, dG0–3) for
// each ordered pair from Pass 1.  Pass 2 then skips sin/cos and the 4
// refG global reads per pair.  MAX_ERMSD_PAIRS = 24*23 = 552.
// ---------------------------------------------------------------------------
#define MAX_ERMSD_CACHE_N    24
#define MAX_ERMSD_PAIRS     (MAX_ERMSD_CACHE_N * (MAX_ERMSD_CACHE_N - 1))
extern const string kErmsdKernelSrc = R"(
#define MAX_ERMSD_CACHE_N    24
#define MAX_ERMSD_PAIRS     (MAX_ERMSD_CACHE_N * (MAX_ERMSD_CACHE_N - 1))
KERNEL void cvErmsd(
 GLOBAL const real4* RESTRICT posq,
 GLOBAL const int* RESTRICT ermsdAtoms,
 GLOBAL const int* RESTRICT atomOffsets,
 GLOBAL const int* RESTRICT nResArray,
 GLOBAL const int* RESTRICT jacOffsets,
 GLOBAL const int* RESTRICT refGOffsets,
 GLOBAL const float* RESTRICT refG,
 GLOBAL const float* RESTRICT cutoffs,
 int numCVs,
 int firstCVIndex,
 int firstJacEntry,
 GLOBAL double* RESTRICT cvValues,
 GLOBAL int* RESTRICT jacobianAtomIdx,
 GLOBAL float* RESTRICT jacobianGradsX,
 GLOBAL float* RESTRICT jacobianGradsY,
 GLOBAL float* RESTRICT jacobianGradsZ,
 GLOBAL int* RESTRICT jacobianCvIdx,
 int spare
) {
 // Bottaro (2014) form factors f=(2, 2, 1/0.3) — eq. (2) of the paper.
 const float ff0 = 2.0f, ff1 = 2.0f, ff2 = (float)(1.0/0.3);
 const float PI = (float)(3.14159265358979323846);

 for (int cvI = GLOBAL_ID; cvI < numCVs; cvI += GLOBAL_SIZE) {
 int N = nResArray[cvI];
 int aStart = atomOffsets[cvI];
 int jBase  = jacOffsets[cvI];
 int gBase  = refGOffsets[cvI];
 float cutoff  = cutoffs[cvI];
 float gamma   = PI / cutoff;
 float maxdist = cutoff / ff0;
 int   cvIdx   = firstCVIndex + cvI;

 float cx[64], cy[64], cz[64];
 float e1x[64], e1y[64], e1z[64];
 float e2x[64], e2y[64], e2z[64];
 float e3x[64], e3y[64], e3z[64];

 // Build Bottaro local frames: e1=(atom0-center)/|...|, e3=(a×b)/|...|, e2=e3×e1
 for (int i = 0; i < N; i++) {
 int ai = aStart + 3*i;
 real4 p0 = posq[ermsdAtoms[ai  ]];
 real4 p1 = posq[ermsdAtoms[ai+1]];
 real4 p2 = posq[ermsdAtoms[ai+2]];
 cx[i] = (p0.x+p1.x+p2.x)*(1.0f/3.0f);
 cy[i] = (p0.y+p1.y+p2.y)*(1.0f/3.0f);
 cz[i] = (p0.z+p1.z+p2.z)*(1.0f/3.0f);

 float ax = p0.x-cx[i], ay_v = p0.y-cy[i], az_v = p0.z-cz[i];
 float la = sqrt(ax*ax+ay_v*ay_v+az_v*az_v);
 if (la < 1e-8f) la = 1e-8f;
 e1x[i]=ax/la; e1y[i]=ay_v/la; e1z[i]=az_v/la;

 float bx = p1.x-cx[i], by_v = p1.y-cy[i], bz_v = p1.z-cz[i];
 float dx = ay_v*bz_v - az_v*by_v;
 float dy = az_v*bx   - ax*bz_v;
 float dz = ax*by_v   - ay_v*bx;
 float ld = sqrt(dx*dx+dy*dy+dz*dz);
 if (ld < 1e-8f) ld = 1e-8f;
 e3x[i]=dx/ld; e3y[i]=dy/ld; e3z[i]=dz/ld;

 e2x[i]=e3y[i]*e1z[i]-e3z[i]*e1y[i];
 e2y[i]=e3z[i]*e1x[i]-e3x[i]*e1z[i];
 e2z[i]=e3x[i]*e1y[i]-e3y[i]*e1x[i];
 }

 // Pass 1: accumulate sumSq over all ordered pairs (i,j), i≠j
 // Ordered pair index: pIdx = i*(N-1) + (j < i ? j : j-1)
 // Two-pass cache: when useCache, store (sc,co,dG0-3) in Pass 1 so Pass 2
 // can skip sin/cos and the 4 refG global reads.  cSC=-1 is the sentinel
 // for non-contributing pairs (gradient=0).  sc is always ≥0 in [0,π) range.
 bool useCache = (N <= MAX_ERMSD_CACHE_N);
 float cSC[MAX_ERMSD_PAIRS], cCO[MAX_ERMSD_PAIRS];
 float cDG0[MAX_ERMSD_PAIRS], cDG1[MAX_ERMSD_PAIRS];
 float cDG2[MAX_ERMSD_PAIRS], cDG3[MAX_ERMSD_PAIRS];
 double sumSq = 0.0;
 for (int i = 0; i < N; i++) {
 for (int j = 0; j < N; j++) {
 if (j == i) continue;
 int pIdx = i*(N-1) + (j < i ? j : j-1);
 float drx = cx[j]-cx[i], dry = cy[j]-cy[i], drz = cz[j]-cz[i];
 float dist = sqrt(drx*drx+dry*dry+drz*drz);
 float dG0, dG1, dG2, dG3;
 if (dist < maxdist) {
  float rt0 = (drx*e1x[i]+dry*e1y[i]+drz*e1z[i])*ff0;
  float rt1 = (drx*e2x[i]+dry*e2y[i]+drz*e2z[i])*ff1;
  float rt2 = (drx*e3x[i]+dry*e3y[i]+drz*e3z[i])*ff2;
  float rtn = sqrt(rt0*rt0+rt1*rt1+rt2*rt2);
  float G0, G1, G2, G3;
  if (rtn > 1e-8f && rtn < cutoff) {
   float sc = sin(gamma*rtn)/(rtn*gamma);
   float co = cos(gamma*rtn);
   G0=sc*rt0; G1=sc*rt1; G2=sc*rt2; G3=(1.0f+co)/gamma;
   dG0=G0-refG[gBase+4*pIdx  ]; dG1=G1-refG[gBase+4*pIdx+1];
   dG2=G2-refG[gBase+4*pIdx+2]; dG3=G3-refG[gBase+4*pIdx+3];
   if (useCache) {
    cSC[pIdx]=sc; cCO[pIdx]=co;
    cDG0[pIdx]=dG0; cDG1[pIdx]=dG1;
    cDG2[pIdx]=dG2; cDG3[pIdx]=dG3;
   }
  } else if (rtn <= 1e-8f) {
   G0=rt0; G1=rt1; G2=rt2; G3=2.0f/gamma;
   dG0=G0-refG[gBase+4*pIdx  ]; dG1=G1-refG[gBase+4*pIdx+1];
   dG2=G2-refG[gBase+4*pIdx+2]; dG3=G3-refG[gBase+4*pIdx+3];
   if (useCache) cSC[pIdx] = -1.0f; // rtn too small: gradient=0
  } else {
   G0=G1=G2=G3=0.0f;
   dG0=-refG[gBase+4*pIdx  ]; dG1=-refG[gBase+4*pIdx+1];
   dG2=-refG[gBase+4*pIdx+2]; dG3=-refG[gBase+4*pIdx+3];
   if (useCache) cSC[pIdx] = -1.0f; // rtn >= cutoff: gradient=0
  }
 } else {
  dG0=-refG[gBase+4*pIdx  ]; dG1=-refG[gBase+4*pIdx+1];
  dG2=-refG[gBase+4*pIdx+2]; dG3=-refG[gBase+4*pIdx+3];
  if (useCache) cSC[pIdx] = -1.0f; // dist >= maxdist: gradient=0
 }
 sumSq += (double)(dG0*dG0+dG1*dG1+dG2*dG2+dG3*dG3);
 }
 }
 double ermsd = sqrt(sumSq / N);
 cvValues[cvIdx] = ermsd;

 // Pass 2: Jacobian entries (centroid-only; frame-rotation terms omitted)
 // d(ermsd)/d(diff_ij) = iermsd*(S1 + scal2*J) where:
 //   S1[k] = A*(dG0*ff0*e1[k]+dG1*ff1*e2[k]+dG2*ff2*e3[k])
 //   J[k]  = ff0*rt0*e1[k]+ff1*rt1*e2[k]+ff2*rt2*e3[k]  (= r̃·∂r̃/∂diff)
 //   scal2 = B*dotDGrt - C*dG3,  A=sc, B=(co-sc)/rtn², C=sc*gamma
 float iermsd = (ermsd > 1e-10) ? (float)(1.0/(ermsd*(double)N)) : 0.0f;

 for (int i = 0; i < N; i++) {
 for (int j = 0; j < N; j++) {
 if (j == i) continue;
 int pIdx = i*(N-1) + (j < i ? j : j-1);
 float gFx = 0.0f, gFy = 0.0f, gFz = 0.0f;

 if (useCache) {
  if (cSC[pIdx] >= 0.0f) {
   float sc = cSC[pIdx], co = cCO[pIdx];
   float dG0 = cDG0[pIdx], dG1 = cDG1[pIdx];
   float dG2 = cDG2[pIdx], dG3 = cDG3[pIdx];
   float drx = cx[j]-cx[i], dry = cy[j]-cy[i], drz = cz[j]-cz[i];
   float rt0 = (drx*e1x[i]+dry*e1y[i]+drz*e1z[i])*ff0;
   float rt1 = (drx*e2x[i]+dry*e2y[i]+drz*e2z[i])*ff1;
   float rt2 = (drx*e3x[i]+dry*e3y[i]+drz*e3z[i])*ff2;
   float rtn2 = rt0*rt0+rt1*rt1+rt2*rt2;
   float irnorm2 = (rtn2 > 1e-12f) ? 1.0f/rtn2 : 0.0f;
   float A = sc;
   float B = (co-sc)*irnorm2;
   float C = sc*gamma;
   float dotDGrt = dG0*rt0+dG1*rt1+dG2*rt2;
   float scal2 = B*dotDGrt - C*dG3;
   float S1x = A*(dG0*ff0*e1x[i]+dG1*ff1*e2x[i]+dG2*ff2*e3x[i]);
   float S1y = A*(dG0*ff0*e1y[i]+dG1*ff1*e2y[i]+dG2*ff2*e3y[i]);
   float S1z = A*(dG0*ff0*e1z[i]+dG1*ff1*e2z[i]+dG2*ff2*e3z[i]);
   float Jx = ff0*rt0*e1x[i]+ff1*rt1*e2x[i]+ff2*rt2*e3x[i];
   float Jy = ff0*rt0*e1y[i]+ff1*rt1*e2y[i]+ff2*rt2*e3y[i];
   float Jz = ff0*rt0*e1z[i]+ff1*rt1*e2z[i]+ff2*rt2*e3z[i];
   gFx = iermsd*(S1x+scal2*Jx);
   gFy = iermsd*(S1y+scal2*Jy);
   gFz = iermsd*(S1z+scal2*Jz);
  }
  // cSC<0: non-contributing pair, gF stays 0
 } else {
  float drx = cx[j]-cx[i], dry = cy[j]-cy[i], drz = cz[j]-cz[i];
  float dist = sqrt(drx*drx+dry*dry+drz*drz);
  if (dist < maxdist) {
   float rt0 = (drx*e1x[i]+dry*e1y[i]+drz*e1z[i])*ff0;
   float rt1 = (drx*e2x[i]+dry*e2y[i]+drz*e2z[i])*ff1;
   float rt2 = (drx*e3x[i]+dry*e3y[i]+drz*e3z[i])*ff2;
   float rtn = sqrt(rt0*rt0+rt1*rt1+rt2*rt2);
   if (rtn > 1e-6f && rtn < cutoff) {
    float sc = sin(gamma*rtn)/(rtn*gamma);
    float co = cos(gamma*rtn);
    float dG0 = sc*rt0-refG[gBase+4*pIdx  ];
    float dG1 = sc*rt1-refG[gBase+4*pIdx+1];
    float dG2 = sc*rt2-refG[gBase+4*pIdx+2];
    float dG3 = (1.0f+co)/gamma-refG[gBase+4*pIdx+3];
    float irnorm2 = 1.0f/(rtn*rtn);
    float A = sc;
    float B = (co-sc)*irnorm2;
    float C = sc*gamma;
    float dotDGrt = dG0*rt0+dG1*rt1+dG2*rt2;
    float scal2 = B*dotDGrt - C*dG3;
    float S1x = A*(dG0*ff0*e1x[i]+dG1*ff1*e2x[i]+dG2*ff2*e3x[i]);
    float S1y = A*(dG0*ff0*e1y[i]+dG1*ff1*e2y[i]+dG2*ff2*e3y[i]);
    float S1z = A*(dG0*ff0*e1z[i]+dG1*ff1*e2z[i]+dG2*ff2*e3z[i]);
    float Jx = ff0*rt0*e1x[i]+ff1*rt1*e2x[i]+ff2*rt2*e3x[i];
    float Jy = ff0*rt0*e1y[i]+ff1*rt1*e2y[i]+ff2*rt2*e3y[i];
    float Jz = ff0*rt0*e1z[i]+ff1*rt1*e2z[i]+ff2*rt2*e3z[i];
    gFx = iermsd*(S1x+scal2*Jx);
    gFy = iermsd*(S1y+scal2*Jy);
    gFz = iermsd*(S1z+scal2*Jz);
   }
  }
 }

 int jacPairBase = firstJacEntry + jBase + 6*pIdx;
 // Residue i atoms: center_i = -diff_ij direction
 for (int m = 0; m < 3; m++) {
 jacobianAtomIdx[jacPairBase+m] = ermsdAtoms[aStart+3*i+m];
 jacobianGradsX [jacPairBase+m] = -gFx*(1.0f/3.0f);
 jacobianGradsY [jacPairBase+m] = -gFy*(1.0f/3.0f);
 jacobianGradsZ [jacPairBase+m] = -gFz*(1.0f/3.0f);
 jacobianCvIdx  [jacPairBase+m] = cvIdx;
 }
 // Residue j atoms: center_j = +diff_ij direction
 for (int m = 0; m < 3; m++) {
 jacobianAtomIdx[jacPairBase+3+m] = ermsdAtoms[aStart+3*j+m];
 jacobianGradsX [jacPairBase+3+m] = +gFx*(1.0f/3.0f);
 jacobianGradsY [jacPairBase+3+m] = +gFy*(1.0f/3.0f);
 jacobianGradsZ [jacPairBase+3+m] = +gFz*(1.0f/3.0f);
 jacobianCvIdx  [jacPairBase+3+m] = cvIdx;
 }
 }
 }
 }
}
)";

// ---------------------------------------------------------------------------
// Cremer-Pople ring puckering CV kernel.
//
// One thread per puckering CV. Supports 5- and 6-membered rings.
// Fixed-normal Jacobian approximation (treat n̂ as constant — dominant term).
//
// For N=5: output components — 0=Q, 1=φ
// For N=6: output components — 0=Q, 1=θ, 2=φ
// ---------------------------------------------------------------------------
extern const string kPuckeringKernelSrc = R"(
KERNEL void cvPuckering(
 GLOBAL const real4* RESTRICT posq,
 GLOBAL const int* RESTRICT pkAtoms,
 GLOBAL const int* RESTRICT atomOffsets,
 GLOBAL const float* RESTRICT pkParams,
 int numCVs,
 int firstCVIndex,
 int firstJacEntry,
 GLOBAL double* RESTRICT cvValues,
 GLOBAL int* RESTRICT jacobianAtomIdx,
 GLOBAL float* RESTRICT jacobianGradsX,
 GLOBAL float* RESTRICT jacobianGradsY,
 GLOBAL float* RESTRICT jacobianGradsZ,
 GLOBAL int* RESTRICT jacobianCvIdx,
 int spare
) {
 for (int cvI = GLOBAL_ID; cvI < numCVs; cvI += GLOBAL_SIZE) {
 int N = (int)(pkParams[2*cvI] + 0.5f); // ring size (5 or 6)
 int comp = (int)(pkParams[2*cvI+1] + 0.5f); // output component
 int aStart = atomOffsets[cvI];
 int cvIdx = firstCVIndex + cvI;
 int jacBase = firstJacEntry + aStart; // N consecutive Jac entries

 // Load ring atom positions
 real px[6], py[6], pz[6];
 for (int j = 0; j < N; j++) {
 real4 p = posq[pkAtoms[aStart + j]];
 px[j] = p.x; py[j] = p.y; pz[j] = p.z;
 }

 // Centroid
 real cx = 0, cy = 0, cz = 0;
 for (int j = 0; j < N; j++) { cx += px[j]; cy += py[j]; cz += pz[j]; }
 real invN = (real)(1.0) / (real)N;
 cx *= invN; cy *= invN; cz *= invN;

 // Centered positions
 real dx[6], dy[6], dz[6];
 for (int j = 0; j < N; j++) { dx[j]=px[j]-cx; dy[j]=py[j]-cy; dz[j]=pz[j]-cz; }

 // S = sum_j r'_j sin(2*pi*j/N), C = sum_j r'_j cos(2*pi*j/N)
 real Sx=0,Sy=0,Sz=0, Cx2=0,Cy2=0,Cz2=0;
 for (int j = 0; j < N; j++) {
 real ang = (real)(2.0*3.14159265358979323846) * (real)j / (real)N;
 real sj = sin(ang), cj2 = cos(ang);
 Sx += dx[j]*sj; Sy += dy[j]*sj; Sz += dz[j]*sj;
 Cx2 += dx[j]*cj2; Cy2 += dy[j]*cj2; Cz2 += dz[j]*cj2;
 }

 // Normal n = S x C, then normalize
 real nx = Sy*Cz2-Sz*Cy2, ny = Sz*Cx2-Sx*Cz2, nz = Sx*Cy2-Sy*Cx2;
 real nlen = sqrt(nx*nx+ny*ny+nz*nz);
 real invNlen = (nlen > (real)1e-10) ? ((real)1.0/nlen) : (real)0.0;
 nx *= invNlen; ny *= invNlen; nz *= invNlen;

 // Out-of-plane displacements: z_j = r'_j . n_hat
 real zj[6];
 for (int j = 0; j < N; j++)
 zj[j] = dx[j]*nx + dy[j]*ny + dz[j]*nz;

 // Puckering coordinates and full 3D Jacobian gradients
 double cv = 0.0;
 float jacGradX[6], jacGradY[6], jacGradZ[6];
 for (int j=0; j<6; j++) { jacGradX[j]=0.0f; jacGradY[j]=0.0f; jacGradZ[j]=0.0f; }

 if (N == 5) {
 // Huang et al. (2014) torsion-based puckering for 5-membered rings.
 // v1 = torsion(r1,r2,r3,r4), v3 = torsion(r3,r4,r0,r1)
 // Zx=(v1+v3)/(2cos(4π/5)), Zy=(v1-v3)/(2sin(4π/5))
 // amp=sqrt(Zx²+Zy²),  phi=atan2(Zy,Zx)

 // torsion v1: atoms r1(A),r2(B),r3(C),r4(D)
 real b1x=px[2]-px[1], b1y=py[2]-py[1], b1z=pz[2]-pz[1];
 real b2x=px[3]-px[2], b2y=py[3]-py[2], b2z=pz[3]-pz[2];
 real b3x=px[4]-px[3], b3y=py[4]-py[3], b3z=pz[4]-pz[3];
 real t1x=b1y*b2z-b1z*b2y, t1y=b1z*b2x-b1x*b2z, t1z=b1x*b2y-b1y*b2x;
 real u1x=b2y*b3z-b2z*b3y, u1y=b2z*b3x-b2x*b3z, u1z=b2x*b3y-b2y*b3x;
 real d1sq=b2x*b2x+b2y*b2y+b2z*b2z, d1=sqrt(d1sq);
 real v1=(real)atan2((double)(d1*(b1x*u1x+b1y*u1y+b1z*u1z)),(double)(t1x*u1x+t1y*u1y+t1z*u1z));
 real tt1=t1x*t1x+t1y*t1y+t1z*t1z, uu1=u1x*u1x+u1y*u1y+u1z*u1z;
 real invT1=(tt1>(real)1e-10)?((real)1.0/tt1):(real)0.0;
 real invU1=(uu1>(real)1e-10)?((real)1.0/uu1):(real)0.0;
 real c1_1=(b1x*b2x+b1y*b2y+b1z*b2z)/d1, c2_1=(b3x*b2x+b3y*b2y+b3z*b2z)/d1;
 real aT1=d1*invT1, aU1=d1*invU1;
 real bTB1=(d1+c1_1)*invT1, bUB1=c2_1*invU1;
 real jv1Ax=-aT1*t1x, jv1Ay=-aT1*t1y, jv1Az=-aT1*t1z;
 real jv1Dx= aU1*u1x, jv1Dy= aU1*u1y, jv1Dz= aU1*u1z;
 real jv1Bx=bTB1*t1x+bUB1*u1x, jv1By=bTB1*t1y+bUB1*u1y, jv1Bz=bTB1*t1z+bUB1*u1z;
 real jv1Cx=-(jv1Ax+jv1Bx+jv1Dx), jv1Cy=-(jv1Ay+jv1By+jv1Dy), jv1Cz=-(jv1Az+jv1Bz+jv1Dz);

 // torsion v3: atoms r3(A),r4(B),r0(C),r1(D)
 real e1x=px[4]-px[3], e1y=py[4]-py[3], e1z=pz[4]-pz[3];
 real e2x=px[0]-px[4], e2y=py[0]-py[4], e2z=pz[0]-pz[4];
 real e3x=px[1]-px[0], e3y=py[1]-py[0], e3z=pz[1]-pz[0];
 real t3x=e1y*e2z-e1z*e2y, t3y=e1z*e2x-e1x*e2z, t3z=e1x*e2y-e1y*e2x;
 real u3x=e2y*e3z-e2z*e3y, u3y=e2z*e3x-e2x*e3z, u3z=e2x*e3y-e2y*e3x;
 real d3sq=e2x*e2x+e2y*e2y+e2z*e2z, d3=sqrt(d3sq);
 real v3=(real)atan2((double)(d3*(e1x*u3x+e1y*u3y+e1z*u3z)),(double)(t3x*u3x+t3y*u3y+t3z*u3z));
 real tt3=t3x*t3x+t3y*t3y+t3z*t3z, uu3=u3x*u3x+u3y*u3y+u3z*u3z;
 real invT3=(tt3>(real)1e-10)?((real)1.0/tt3):(real)0.0;
 real invU3=(uu3>(real)1e-10)?((real)1.0/uu3):(real)0.0;
 real c1_3=(e1x*e2x+e1y*e2y+e1z*e2z)/d3, c2_3=(e3x*e2x+e3y*e2y+e3z*e2z)/d3;
 real aT3=d3*invT3, aU3=d3*invU3;
 real bTB3=(d3+c1_3)*invT3, bUB3=c2_3*invU3;
 real jv3Ax=-aT3*t3x, jv3Ay=-aT3*t3y, jv3Az=-aT3*t3z;
 real jv3Dx= aU3*u3x, jv3Dy= aU3*u3y, jv3Dz= aU3*u3z;
 real jv3Bx=bTB3*t3x+bUB3*u3x, jv3By=bTB3*t3y+bUB3*u3y, jv3Bz=bTB3*t3z+bUB3*u3z;
 real jv3Cx=-(jv3Ax+jv3Bx+jv3Dx), jv3Cy=-(jv3Ay+jv3By+jv3Dy), jv3Cz=-(jv3Az+jv3Bz+jv3Dz);

 const real cos4pi5 = (real)(-0.8090169943749474);
 const real sin4pi5 = (real)( 0.5877852522924731);
 real Zx = (v1+v3) / ((real)2.0*cos4pi5);
 real Zy = (v1-v3) / ((real)2.0*sin4pi5);
 real amp5 = sqrt(Zx*Zx+Zy*Zy);

 if (comp == 0) { // amplitude
  cv = (double)amp5;
  if (amp5 > (real)1e-10) {
  real invA = (real)1.0/amp5;
  real alpha = Zx*invA / ((real)2.0*cos4pi5);
  real beta  = Zy*invA / ((real)2.0*sin4pi5);
  real dv1 = alpha + beta, dv3 = alpha - beta;
  // v1: r1=A, r2=B, r3=C, r4=D
  jacGradX[1]+=(float)(dv1*jv1Ax); jacGradY[1]+=(float)(dv1*jv1Ay); jacGradZ[1]+=(float)(dv1*jv1Az);
  jacGradX[2]+=(float)(dv1*jv1Bx); jacGradY[2]+=(float)(dv1*jv1By); jacGradZ[2]+=(float)(dv1*jv1Bz);
  jacGradX[3]+=(float)(dv1*jv1Cx); jacGradY[3]+=(float)(dv1*jv1Cy); jacGradZ[3]+=(float)(dv1*jv1Cz);
  jacGradX[4]+=(float)(dv1*jv1Dx); jacGradY[4]+=(float)(dv1*jv1Dy); jacGradZ[4]+=(float)(dv1*jv1Dz);
  // v3: r3=A, r4=B, r0=C, r1=D
  jacGradX[3]+=(float)(dv3*jv3Ax); jacGradY[3]+=(float)(dv3*jv3Ay); jacGradZ[3]+=(float)(dv3*jv3Az);
  jacGradX[4]+=(float)(dv3*jv3Bx); jacGradY[4]+=(float)(dv3*jv3By); jacGradZ[4]+=(float)(dv3*jv3Bz);
  jacGradX[0]+=(float)(dv3*jv3Cx); jacGradY[0]+=(float)(dv3*jv3Cy); jacGradZ[0]+=(float)(dv3*jv3Cz);
  jacGradX[1]+=(float)(dv3*jv3Dx); jacGradY[1]+=(float)(dv3*jv3Dy); jacGradZ[1]+=(float)(dv3*jv3Dz);
  }
 } else { // phi = atan2(Zy, Zx)
  cv = (double)atan2((double)Zy,(double)Zx);
  real amp2_5 = amp5*amp5;
  if (amp2_5 > (real)1e-20) {
  real invA2 = (real)1.0/amp2_5;
  real gamma = -Zy*invA2 / ((real)2.0*cos4pi5);
  real delta  =  Zx*invA2 / ((real)2.0*sin4pi5);
  real dv1 = gamma + delta, dv3 = gamma - delta;
  jacGradX[1]+=(float)(dv1*jv1Ax); jacGradY[1]+=(float)(dv1*jv1Ay); jacGradZ[1]+=(float)(dv1*jv1Az);
  jacGradX[2]+=(float)(dv1*jv1Bx); jacGradY[2]+=(float)(dv1*jv1By); jacGradZ[2]+=(float)(dv1*jv1Bz);
  jacGradX[3]+=(float)(dv1*jv1Cx); jacGradY[3]+=(float)(dv1*jv1Cy); jacGradZ[3]+=(float)(dv1*jv1Cz);
  jacGradX[4]+=(float)(dv1*jv1Dx); jacGradY[4]+=(float)(dv1*jv1Dy); jacGradZ[4]+=(float)(dv1*jv1Dz);
  jacGradX[3]+=(float)(dv3*jv3Ax); jacGradY[3]+=(float)(dv3*jv3Ay); jacGradZ[3]+=(float)(dv3*jv3Az);
  jacGradX[4]+=(float)(dv3*jv3Bx); jacGradY[4]+=(float)(dv3*jv3By); jacGradZ[4]+=(float)(dv3*jv3Bz);
  jacGradX[0]+=(float)(dv3*jv3Cx); jacGradY[0]+=(float)(dv3*jv3Cy); jacGradZ[0]+=(float)(dv3*jv3Cz);
  jacGradX[1]+=(float)(dv3*jv3Dx); jacGradY[1]+=(float)(dv3*jv3Dy); jacGradZ[1]+=(float)(dv3*jv3Dz);
  }
 }

 } else { // N == 6
 const real SQRT1_3 = (real)0.5773502691896258; // sqrt(1/3)
 const real SQRT1_6 = (real)0.4082482904638631; // sqrt(1/6)
 real A2=0, B2=0, q3=0;
 for (int j = 0; j < 6; j++) {
 real ang2 = (real)(2.0*3.14159265358979323846/3.0) * (real)j; // 2*pi*j/3 for mode m=2
 A2 += zj[j] * cos(ang2);
 B2 -= zj[j] * sin(ang2);
 q3 += zj[j] * (j%2==0 ? (real)1.0 : (real)-1.0);
 }
 A2 *= SQRT1_3; B2 *= SQRT1_3; q3 *= SQRT1_6;
 real rho = sqrt(A2*A2 + B2*B2);
 real Q = sqrt(rho*rho + q3*q3);
 float jacS6[6];

 if (comp == 0) { // Q
 cv = (double)Q;
 if (Q > (real)1e-10) {
 real invQ = (real)1.0/Q;
 for (int j = 0; j < 6; j++) {
 real ang2 = (real)(2.0*3.14159265358979323846/3.0) * (real)j;
 real sign3 = (j%2==0) ? (real)1.0 : (real)-1.0;
 jacS6[j] = (float)(invQ * (SQRT1_3*(A2*cos(ang2) - B2*sin(ang2)) + SQRT1_6*sign3*q3));
 }
 } else {
 for (int j = 0; j < 6; j++) jacS6[j] = 0.0f;
 }
 } else if (comp == 1) { // theta = atan2(rho, q3)
 cv = (double)atan2(rho, q3);
 real Q2 = Q*Q;
 if (Q2 > (real)1e-20 && rho > (real)1e-10) {
 real invQ2 = (real)1.0/Q2;
 real invRho = (real)1.0/rho;
 for (int j = 0; j < 6; j++) {
 real ang2 = (real)(2.0*3.14159265358979323846/3.0) * (real)j;
 real sign3 = (j%2==0) ? (real)1.0 : (real)-1.0;
 real termMode2 = SQRT1_3 * (A2*cos(ang2) - B2*sin(ang2));
 real termMode3 = SQRT1_6 * sign3;
 jacS6[j] = (float)(invQ2 * (q3*invRho*termMode2 - rho*termMode3));
 }
 } else {
 for (int j = 0; j < 6; j++) jacS6[j] = 0.0f;
 }
 } else { // phi = atan2(B2, A2)
 cv = (double)atan2(B2, A2);
 real rho2 = rho*rho;
 if (rho2 > (real)1e-20) {
 real invRho2 = (real)1.0/rho2;
 for (int j = 0; j < 6; j++) {
 real ang2 = (real)(2.0*3.14159265358979323846/3.0) * (real)j;
 jacS6[j] = (float)(-SQRT1_3 * invRho2 * (A2*sin(ang2) + B2*cos(ang2)));
 }
 } else {
 for (int j = 0; j < 6; j++) jacS6[j] = 0.0f;
 }
 }
 for (int j = 0; j < 6; j++) {
 jacGradX[j] = jacS6[j]*(float)nx;
 jacGradY[j] = jacS6[j]*(float)ny;
 jacGradZ[j] = jacS6[j]*(float)nz;
 }
 }

 cvValues[cvIdx] = cv;

 for (int j = 0; j < N; j++) {
 jacobianAtomIdx[jacBase + j] = pkAtoms[aStart + j];
 jacobianGradsX [jacBase + j] = jacGradX[j];
 jacobianGradsY [jacBase + j] = jacGradY[j];
 jacobianGradsZ [jacBase + j] = jacGradZ[j];
 jacobianCvIdx [jacBase + j] = cvIdx;
 }
 }
}
)";

// ---------------------------------------------------------------------------
// Secondary structure CV kernel (ALPHARMSD, ANTIBETARMSD, PARABETARMSD).
//
// 5 backbone atoms per residue (N, CA, CB, C, O) × 6 residues = 30 atoms per window.
// Templates: ideal backbone geometry, pre-centered to origin (nm).
// Derived from Engh & Huber (1991) Acta Cryst. A47:392 bond lengths/angles
// and Pauling & Corey (1951) backbone dihedrals, via tools/gen_secstr_templates.py.
//
// Subtype 0 = alpha helix:  N*5 backbone atoms, numWindows = N/5−5, stride = 5
// Subtype 1 = antibeta:     30*W backbone atoms (strand pairs), stride = 30
// Subtype 2 = parabeta:     30*W backbone atoms (strand pairs), stride = 30
//   Parabeta has two reference structures; minimum RMSD is used per window.
//
// Jacobian: 30 entries per window at firstJacEntry + (wBase+w)*30 + k.
// ---------------------------------------------------------------------------
extern const string kSecondaryStructureKernelSrc = R"(
KERNEL void cvSecondaryStructure(
 GLOBAL const real4* RESTRICT posq,
 GLOBAL const int* RESTRICT ssAtoms,
 GLOBAL const int* RESTRICT atomOffsets,
 GLOBAL const int* RESTRICT windowOffsets,
 GLOBAL const float* RESTRICT ssParams,
 int numSsCVs,
 int firstCVIndex,
 int firstJacEntry,
 GLOBAL double* RESTRICT cvValues,
 GLOBAL int* RESTRICT jacobianAtomIdx,
 GLOBAL float* RESTRICT jacobianGradsX,
 GLOBAL float* RESTRICT jacobianGradsY,
 GLOBAL float* RESTRICT jacobianGradsZ,
 GLOBAL int* RESTRICT jacobianCvIdx,
 int spare
) {
 for (int cvI = GLOBAL_ID; cvI < numSsCVs; cvI += GLOBAL_SIZE) {
 int subtype = (int)ssParams[2*cvI];
 float r0 = ssParams[2*cvI + 1];

 // Load template 1 (pre-centered, nm)
 float tx1[30], ty1[30], tz1[30];
 if (subtype == 0) {
  // Alpha helix — Engh & Huber (1991) Acta Cryst. A47:392 geometry,
  // Pauling & Corey (1951) phi=-57 deg, psi=-47 deg.
  // Generated by tools/gen_secstr_templates.py via NERF algorithm (Parsons 2005).
  tx1[ 0]=-0.296458f; ty1[ 0]=-0.401214f; tz1[ 0]=-0.210973f;
  tx1[ 1]=-0.150658f; ty1[ 1]=-0.401214f; tz1[ 1]=-0.210973f;
  tx1[ 2]=-0.097391f; ty1[ 2]=-0.472448f; tz1[ 2]=-0.334354f;
  tx1[ 3]=-0.095510f; ty1[ 3]=-0.259035f; tz1[ 3]=-0.210973f;
  tx1[ 4]=-0.005629f; ty1[ 4]=-0.226399f; tz1[ 4]=-0.133767f;
  tx1[ 5]=-0.150113f; ty1[ 5]=-0.174920f; tz1[ 5]=-0.298184f;
  tx1[ 6]=-0.106516f; ty1[ 6]=-0.036167f; tz1[ 6]=-0.308404f;
  tx1[ 7]=-0.190650f; ty1[ 7]=0.038711f; tz1[ 7]=-0.410625f;
  tx1[ 8]=-0.119637f; ty1[ 8]=0.035762f; tz1[ 8]=-0.174574f;
  tx1[ 9]=-0.027661f; ty1[ 9]=0.104531f; tz1[ 9]=-0.130809f;
  tx1[10]=-0.234697f; ty1[10]=0.019284f; tz1[10]=-0.110137f;
  tx1[11]=-0.260076f; ty1[11]=0.082562f; tz1[11]=0.018741f;
  tx1[12]=-0.396021f; ty1[12]=0.040845f; tz1[12]=0.072713f;
  tx1[13]=-0.154352f; ty1[13]=0.043111f; tz1[13]=0.121319f;
  tx1[14]=-0.100415f; ty1[14]=0.128013f; tz1[14]=0.191937f;
  tx1[15]=-0.125318f; ty1[15]=-0.086361f; tz1[15]=0.128833f;
  tx1[16]=-0.026069f; ty1[16]=-0.137298f; tz1[16]=0.222709f;
  tx1[17]=-0.007769f; ty1[17]=-0.287221f; tz1[17]=0.204741f;
  tx1[18]=0.109286f; ty1[18]=-0.070287f; tz1[18]=0.201617f;
  tx1[19]=0.173494f; ty1[19]=-0.026764f; tz1[19]=0.296946f;
  tx1[20]=0.151930f; ty1[20]=-0.062432f; tz1[20]=0.075990f;
  tx1[21]=0.279636f; ty1[21]=-0.000760f; tz1[21]=0.042147f;
  tx1[22]=0.299600f; ty1[22]=0.001091f; tz1[22]=-0.108625f;
  tx1[23]=0.286513f; ty1[23]=0.143018f; tz1[23]=0.092514f;
  tx1[24]=0.385421f; ty1[24]=0.183861f; tz1[24]=0.152957f;
  tx1[25]=0.180638f; ty1[25]=0.219221f; tz1[25]=0.067096f;
  tx1[26]=0.174465f; ty1[26]=0.358497f; tz1[26]=0.109778f;
  tx1[27]=0.039642f; ty1[27]=0.419209f; tz1[27]=0.074122f;
  tx1[28]=0.193480f; ty1[28]=0.370855f; tz1[28]=0.260583f;
  tx1[29]=0.270836f; ty1[29]=0.453949f; tz1[29]=0.307657f;
 } else if (subtype == 1) {
  // Antiparallel beta sheet — Engh & Huber (1991) geometry,
  // Pauling & Corey (1951) phi=-139 deg, psi=135 deg; interstrand ~4.72 A.
  // Generated by tools/gen_secstr_templates.py via NERF algorithm (Parsons 2005).
  tx1[ 0]=-0.425640f; ty1[ 0]=-0.251248f; tz1[ 0]=-0.266352f;
  tx1[ 1]=-0.279840f; ty1[ 1]=-0.251248f; tz1[ 1]=-0.266352f;
  tx1[ 2]=-0.226574f; ty1[ 2]=-0.322481f; tz1[ 2]=-0.389733f;
  tx1[ 3]=-0.224693f; ty1[ 3]=-0.109068f; tz1[ 3]=-0.266352f;
  tx1[ 4]=-0.271530f; ty1[ 4]=-0.023403f; tz1[ 4]=-0.340999f;
  tx1[ 5]=-0.124861f; ty1[ 5]=-0.084855f; tz1[ 5]=-0.182033f;
  tx1[ 6]=-0.063166f; ty1[ 6]=0.046879f; tz1[ 6]=-0.172151f;
  tx1[ 7]=-0.115927f; ty1[ 7]=0.121419f; tz1[ 7]=-0.050519f;
  tx1[ 8]=0.088359f; ty1[ 8]=0.035953f; tz1[ 8]=-0.158845f;
  tx1[ 9]=0.139075f; ty1[ 9]=-0.045405f; tz1[ 9]=-0.081949f;
  tx1[10]=0.160002f; ty1[10]=0.118559f; tz1[10]=-0.234384f;
  tx1[11]=0.305766f; ty1[11]=0.118334f; tz1[11]=-0.231173f;
  tx1[12]=0.361565f; ty1[12]=0.042782f; tz1[12]=-0.350809f;
  tx1[13]=0.361228f; ty1[13]=0.260343f; tz1[13]=-0.234884f;
  tx1[14]=0.316237f; ty1[14]=0.343440f; tz1[14]=-0.313467f;
  tx1[15]=-0.160002f; ty1[15]=0.118559f; tz1[15]=0.237616f;
  tx1[16]=-0.305766f; ty1[16]=0.118334f; tz1[16]=0.240827f;
  tx1[17]=-0.361565f; ty1[17]=0.042782f; tz1[17]=0.121191f;
  tx1[18]=-0.361228f; ty1[18]=0.260343f; tz1[18]=0.237116f;
  tx1[19]=-0.316237f; ty1[19]=0.343440f; tz1[19]=0.158533f;
  tx1[20]=0.124861f; ty1[20]=-0.084855f; tz1[20]=0.289967f;
  tx1[21]=0.063166f; ty1[21]=0.046879f; tz1[21]=0.299849f;
  tx1[22]=0.115927f; ty1[22]=0.121419f; tz1[22]=0.421481f;
  tx1[23]=-0.088359f; ty1[23]=0.035953f; tz1[23]=0.313155f;
  tx1[24]=-0.139075f; ty1[24]=-0.045405f; tz1[24]=0.390051f;
  tx1[25]=0.425640f; ty1[25]=-0.251248f; tz1[25]=0.205648f;
  tx1[26]=0.279840f; ty1[26]=-0.251248f; tz1[26]=0.205648f;
  tx1[27]=0.226574f; ty1[27]=-0.322481f; tz1[27]=0.082267f;
  tx1[28]=0.224693f; ty1[28]=-0.109068f; tz1[28]=0.205648f;
  tx1[29]=0.271530f; ty1[29]=-0.023403f; tz1[29]=0.131001f;
 } else {
  // Parallel beta sheet template 1 — Engh & Huber (1991) geometry,
  // Pauling & Corey (1951) phi=-119 deg, psi=113 deg; interstrand ~4.85 A.
  // Generated by tools/gen_secstr_templates.py via NERF algorithm (Parsons 2005).
  tx1[ 0]=-0.404229f; ty1[ 0]=-0.251661f; tz1[ 0]=-0.285656f;
  tx1[ 1]=-0.258429f; ty1[ 1]=-0.251661f; tz1[ 1]=-0.285656f;
  tx1[ 2]=-0.205163f; ty1[ 2]=-0.322895f; tz1[ 2]=-0.409037f;
  tx1[ 3]=-0.203282f; ty1[ 3]=-0.109482f; tz1[ 3]=-0.285656f;
  tx1[ 4]=-0.218981f; ty1[ 4]=-0.035894f; tz1[ 4]=-0.382830f;
  tx1[ 5]=-0.138623f; ty1[ 5]=-0.071626f; tz1[ 5]=-0.175890f;
  tx1[ 6]=-0.081050f; ty1[ 6]=0.061707f; tz1[ 6]=-0.163026f;
  tx1[ 7]=-0.140756f; ty1[ 7]=0.134192f; tz1[ 7]=-0.043379f;
  tx1[ 8]=0.070052f; ty1[ 8]=0.054890f; tz1[ 8]=-0.143591f;
  tx1[ 9]=0.118230f; ty1[ 9]=0.007037f; tz1[ 9]=-0.041154f;
  tx1[10]=0.144204f; ty1[10]=0.103144f; tz1[10]=-0.242765f;
  tx1[11]=0.289880f; ty1[11]=0.102619f; tz1[11]=-0.236768f;
  tx1[12]=0.347698f; ty1[12]=0.023593f; tz1[12]=-0.353157f;
  tx1[13]=0.345858f; ty1[13]=0.244319f; tz1[13]=-0.243392f;
  tx1[14]=0.334593f; ty1[14]=0.311720f; tz1[14]=-0.345542f;
  tx1[15]=-0.404229f; ty1[15]=-0.251661f; tz1[15]=0.199344f;
  tx1[16]=-0.258429f; ty1[16]=-0.251661f; tz1[16]=0.199344f;
  tx1[17]=-0.205163f; ty1[17]=-0.322895f; tz1[17]=0.075963f;
  tx1[18]=-0.203282f; ty1[18]=-0.109482f; tz1[18]=0.199344f;
  tx1[19]=-0.218981f; ty1[19]=-0.035894f; tz1[19]=0.102170f;
  tx1[20]=-0.138623f; ty1[20]=-0.071626f; tz1[20]=0.309110f;
  tx1[21]=-0.081050f; ty1[21]=0.061707f; tz1[21]=0.321974f;
  tx1[22]=-0.140756f; ty1[22]=0.134192f; tz1[22]=0.441621f;
  tx1[23]=0.070052f; ty1[23]=0.054890f; tz1[23]=0.341409f;
  tx1[24]=0.118230f; ty1[24]=0.007037f; tz1[24]=0.443846f;
  tx1[25]=0.144204f; ty1[25]=0.103144f; tz1[25]=0.242235f;
  tx1[26]=0.289880f; ty1[26]=0.102619f; tz1[26]=0.248232f;
  tx1[27]=0.347698f; ty1[27]=0.023593f; tz1[27]=0.131843f;
  tx1[28]=0.345858f; ty1[28]=0.244319f; tz1[28]=0.241608f;
  tx1[29]=0.334593f; ty1[29]=0.311720f; tz1[29]=0.139458f;
 }

 // Parallel beta sheet template 2 — Engh & Huber (1991) geometry,
 // register-shifted (strand 2 displaced one residue rise ~3.4 A along strand).
 // Generated by tools/gen_secstr_templates.py via NERF algorithm (Parsons 2005).
 float tx2[30], ty2[30], tz2[30];
 if (subtype == 2) {
  tx2[ 0]=-0.574229f; ty2[ 0]=-0.251661f; tz2[ 0]=-0.285656f;
  tx2[ 1]=-0.428429f; ty2[ 1]=-0.251661f; tz2[ 1]=-0.285656f;
  tx2[ 2]=-0.375163f; ty2[ 2]=-0.322895f; tz2[ 2]=-0.409037f;
  tx2[ 3]=-0.373282f; ty2[ 3]=-0.109482f; tz2[ 3]=-0.285656f;
  tx2[ 4]=-0.388981f; ty2[ 4]=-0.035894f; tz2[ 4]=-0.382830f;
  tx2[ 5]=-0.308623f; ty2[ 5]=-0.071626f; tz2[ 5]=-0.175890f;
  tx2[ 6]=-0.251050f; ty2[ 6]=0.061707f; tz2[ 6]=-0.163026f;
  tx2[ 7]=-0.310756f; ty2[ 7]=0.134192f; tz2[ 7]=-0.043379f;
  tx2[ 8]=-0.099948f; ty2[ 8]=0.054890f; tz2[ 8]=-0.143591f;
  tx2[ 9]=-0.051770f; ty2[ 9]=0.007037f; tz2[ 9]=-0.041154f;
  tx2[10]=-0.025796f; ty2[10]=0.103144f; tz2[10]=-0.242765f;
  tx2[11]=0.119880f; ty2[11]=0.102619f; tz2[11]=-0.236768f;
  tx2[12]=0.177698f; ty2[12]=0.023593f; tz2[12]=-0.353157f;
  tx2[13]=0.175858f; ty2[13]=0.244319f; tz2[13]=-0.243392f;
  tx2[14]=0.164593f; ty2[14]=0.311720f; tz2[14]=-0.345542f;
  tx2[15]=-0.234229f; ty2[15]=-0.251661f; tz2[15]=0.199344f;
  tx2[16]=-0.088429f; ty2[16]=-0.251661f; tz2[16]=0.199344f;
  tx2[17]=-0.035163f; ty2[17]=-0.322895f; tz2[17]=0.075963f;
  tx2[18]=-0.033282f; ty2[18]=-0.109482f; tz2[18]=0.199344f;
  tx2[19]=-0.048981f; ty2[19]=-0.035894f; tz2[19]=0.102170f;
  tx2[20]=0.031377f; ty2[20]=-0.071626f; tz2[20]=0.309110f;
  tx2[21]=0.088950f; ty2[21]=0.061707f; tz2[21]=0.321974f;
  tx2[22]=0.029244f; ty2[22]=0.134192f; tz2[22]=0.441621f;
  tx2[23]=0.240052f; ty2[23]=0.054890f; tz2[23]=0.341409f;
  tx2[24]=0.288230f; ty2[24]=0.007037f; tz2[24]=0.443846f;
  tx2[25]=0.314204f; ty2[25]=0.103144f; tz2[25]=0.242235f;
  tx2[26]=0.459880f; ty2[26]=0.102619f; tz2[26]=0.248232f;
  tx2[27]=0.517698f; ty2[27]=0.023593f; tz2[27]=0.131843f;
  tx2[28]=0.515858f; ty2[28]=0.244319f; tz2[28]=0.241608f;
  tx2[29]=0.504593f; ty2[29]=0.311720f; tz2[29]=0.139458f;
 }

 int aStart = atomOffsets[cvI];
 int aEnd = atomOffsets[cvI + 1];
 int N = aEnd - aStart;
 // Alpha: stride 5 (one residue), numWindows = N/5 - 5
 // Beta:  stride 30 (one pair of 3-residue strands), numWindows = N/30
 int stride = (subtype == 0) ? 5 : 30;
 int numWindows = (subtype == 0) ? (N/5 - 5) : (N / 30);
 int wBase = windowOffsets[cvI];
 int cvIdx = firstCVIndex + cvI;

 double cvVal = 0.0;

 for (int w = 0; w < numWindows; w++) {
 int wStart = aStart + w * stride;

 // Load 30 atom positions and compute COM
 float px[30], py[30], pz[30];
 float cx = 0.0f, cy = 0.0f, cz = 0.0f;
 for (int k = 0; k < 30; k++) {
  real4 p = posq[ssAtoms[wStart + k]];
  px[k] = (float)p.x; py[k] = (float)p.y; pz[k] = (float)p.z;
  cx += px[k]; cy += py[k]; cz += pz[k];
 }
 cx *= (1.0f/30.0f); cy *= (1.0f/30.0f); cz *= (1.0f/30.0f);
 for (int k = 0; k < 30; k++) { px[k] -= cx; py[k] -= cy; pz[k] -= cz; }

 // MSD vs template 1
 float msd1 = 0.0f;
 for (int k = 0; k < 30; k++) {
  float dx = px[k] - tx1[k], dy = py[k] - ty1[k], dz = pz[k] - tz1[k];
  msd1 += dx*dx + dy*dy + dz*dz;
 }
 msd1 *= (1.0f/30.0f);

 // MSD vs template 2 (parabeta only)
 float msd2 = msd1 + 1.0f;  // default: template1 wins
 if (subtype == 2) {
  msd2 = 0.0f;
  for (int k = 0; k < 30; k++) {
  float dx = px[k] - tx2[k], dy = py[k] - ty2[k], dz = pz[k] - tz2[k];
  msd2 += dx*dx + dy*dy + dz*dz;
  }
  msd2 *= (1.0f/30.0f);
 }

 int useT2 = (msd2 < msd1) ? 1 : 0;
 float msd = useT2 ? msd2 : msd1;
 real rmsd = sqrt((real)msd);

 // Rational switching function: s = (1-x^6)/(1-x^12), x = rmsd/r0
 double sw = 0.0;
 float jacFactor = 0.0f;
 if (rmsd < (real)1e-6) {
  sw = 1.0;
 } else {
  real x = rmsd / (real)r0;
  if (x < (real)0.9999) {
  real x2 = x*x, x6 = x2*x2*x2, x12 = x6*x6;
  real num = (real)1.0 - x6, den = (real)1.0 - x12;
  sw = (double)(num / den);
  real x5 = x2*x2*x;
  real dsdx = (real)-6.0 * x5 * num * num / (den * den);
  jacFactor = (float)(dsdx / ((real)r0 * (real)30.0 * rmsd));
  }
 }

 cvVal += sw;

 // Write 30 Jacobian entries for this window
 int jacBase = firstJacEntry + (wBase + w) * 30;
 for (int k = 0; k < 30; k++) {
  float dkx = px[k] - (useT2 ? tx2[k] : tx1[k]);
  float dky = py[k] - (useT2 ? ty2[k] : ty1[k]);
  float dkz = pz[k] - (useT2 ? tz2[k] : tz1[k]);
  jacobianAtomIdx[jacBase + k] = ssAtoms[wStart + k];
  jacobianGradsX [jacBase + k] = jacFactor * dkx;
  jacobianGradsY [jacBase + k] = jacFactor * dky;
  jacobianGradsZ [jacBase + k] = jacFactor * dkz;
  jacobianCvIdx  [jacBase + k] = cvIdx;
 }
 }

 cvValues[cvIdx] = cvVal;
 }
}
)";

extern const string kScatterKernelSrc = R"(
KERNEL void chainRuleScatter(
 GLOBAL const double* RESTRICT cvBiasGradients,
 GLOBAL const int* RESTRICT jacobianAtomIdx,
 GLOBAL const float* RESTRICT jacobianGradsX,
 GLOBAL const float* RESTRICT jacobianGradsY,
 GLOBAL const float* RESTRICT jacobianGradsZ,
 GLOBAL const int* RESTRICT jacobianCvIdx,
 GLOBAL mm_ulong* RESTRICT forceBuffer,
 int paddedNumAtoms,
 int numEntries
) {
 for (int j = GLOBAL_ID; j < numEntries; j += GLOBAL_SIZE) {
 int gpuAtom = jacobianAtomIdx[j];
 int cvIdx = jacobianCvIdx[j];
 double dUdCV = cvBiasGradients[cvIdx];

 mm_long fx = (mm_long)(-(dUdCV * (double)jacobianGradsX[j]) * 0x100000000LL);
 mm_long fy = (mm_long)(-(dUdCV * (double)jacobianGradsY[j]) * 0x100000000LL);
 mm_long fz = (mm_long)(-(dUdCV * (double)jacobianGradsZ[j]) * 0x100000000LL);

 atomicAdd(&forceBuffer[gpuAtom], (mm_ulong)fx);
 atomicAdd(&forceBuffer[gpuAtom + paddedNumAtoms], (mm_ulong)fy);
 atomicAdd(&forceBuffer[gpuAtom + 2*paddedNumAtoms], (mm_ulong)fz);
 }
}
)";

// ---------------------------------------------------------------------------
// Harmonic restraint bias kernel (single thread).
//
// V = Σ_d k_d/2 * (s_d − s0_d)²
// dV/ds_d = k_d * (s_d − s0_d) accumulated into cvBiasGradients.
//
// params layout: [k_0, s0_0, k_1, s0_1, ...] (2 floats per CV)
// ---------------------------------------------------------------------------
extern const string kHarmonicKernelSrc = R"(
KERNEL void harmonicEvalBias(
 GLOBAL const double* RESTRICT cvValues,
 GLOBAL const int* RESTRICT cvIdxList,
 GLOBAL const float* RESTRICT params,
 int numCVsBias,
 GLOBAL double* RESTRICT cvBiasGradients,
 GLOBAL double* RESTRICT biasEnergies,
 int biasIdx
) {
 if (GLOBAL_ID != 0) return;
 double V = 0.0;
 for (int d = 0; d < numCVsBias; d++) {
 double s = cvValues[cvIdxList[d]];
 double k = (double)params[2*d];
 double s0 = (double)params[2*d + 1];
 double ds = s - s0;
 V += 0.5 * k * ds * ds;
 cvBiasGradients[cvIdxList[d]] += k * ds;
 }
 biasEnergies[biasIdx] = V;
}
)";

// ---------------------------------------------------------------------------
// Linear coupling bias kernel (single thread).
//
// V = sum_d k_d * cv_d
// dV/dcv_d = k_d (accumulated into cvBiasGradients)
// params layout: [k_0, k_1, ...] (1 float per CV dim)
// ---------------------------------------------------------------------------
extern const string kLinearKernelSrc = R"(
KERNEL void linearEvalBias(
 GLOBAL const double* RESTRICT cvValues,
 GLOBAL const int* RESTRICT cvIdxList,
 GLOBAL const float* RESTRICT params, // [k_0, k_1, ...]
 int numCVsBias,
 GLOBAL double* RESTRICT cvBiasGradients,
 GLOBAL double* RESTRICT biasEnergies,
 int biasIdx
) {
 if (GLOBAL_ID != 0) return;
 double V = 0.0;
 for (int d = 0; d < numCVsBias; d++) {
 double s = cvValues[cvIdxList[d]];
 double k = (double)params[d];
 V += k * s;
 cvBiasGradients[cvIdxList[d]] += k;
 }
 biasEnergies[biasIdx] = V;
}
)";

// ---------------------------------------------------------------------------
// One-sided polynomial wall bias kernel (single thread).
//
// wallType=0 UPPER: delta = s - at; activates when delta > 0
// wallType=1 LOWER: delta = at - s; activates when delta > 0
// V = kappa * delta^n * exp(eps * delta)
// dV/ds = kappa * delta^(n-1) * (n + eps*delta) * exp(eps*delta) * sign
// where sign = +1 for upper, -1 for lower
//
// params layout per dim: [at, kappa, eps, n] (4 floats)
// ---------------------------------------------------------------------------
extern const string kWallKernelSrc = R"(
KERNEL void wallEvalBias(
 GLOBAL const double* RESTRICT cvValues,
 GLOBAL const int* RESTRICT cvIdxList,
 GLOBAL const float* RESTRICT params, // [at,kappa,eps,n] per dim
 int numCVsBias,
 int wallType,
 GLOBAL double* RESTRICT cvBiasGradients,
 GLOBAL double* RESTRICT biasEnergies,
 int biasIdx
) {
 if (GLOBAL_ID != 0) return;
 double V = 0.0;
 for (int d = 0; d < numCVsBias; d++) {
 double s = cvValues[cvIdxList[d]];
 double at = (double)params[4*d];
 double kappa = (double)params[4*d + 1];
 double eps = (double)params[4*d + 2];
 double n = (double)params[4*d + 3];

 double delta = (wallType == 0) ? (s - at) : (at - s);
 if (delta <= 0.0) continue;

 // n is typically integer; handle common cases fast to avoid pow()
 double dPow, powered;
 if (n == 1.0) { powered = delta; dPow = 1.0; }
 else if (n == 2.0) { powered = delta*delta; dPow = 2.0*delta; }
 else { powered = pow(delta, n); dPow = n * pow(delta, n - 1.0); }

 double expFac = exp(eps * delta);
 V += kappa * powered * expFac;

 double dV = kappa * (dPow + eps * powered) * expFac;
 double sign = (wallType == 0) ? 1.0 : -1.0;
 cvBiasGradients[cvIdxList[d]] += sign * dV;
 }
 biasEnergies[biasIdx] = V;
}
)";

// ---------------------------------------------------------------------------
// OPES bias evaluation kernel (single thread).
// Algorithm: Invernizzi & Parrinello (2020) J. Phys. Chem. Lett. 11:2731.
//
//   evaluateKernel(s,k) = h_k * (G_unnorm(s|c_k) - ε)   if norm2 < cutoff2, else 0
//   prob_unnorm = Σ_k evaluateKernel(s,k)
//   Zed = sum_uprob / KDNorm / nker
//   p/Zed = prob_unnorm * nker / sum_uprob
//   V(s) = kT·invGF·log(p/Zed + ε)
//
//   cutoff2  = 2·γ/invGF  = 2·γ²/(γ-1)
//   ε        = exp(-1/(invGF·(1-invGF))) = exp(-γ/invGF) = exp(-γ²/(γ-1))
//   invGF    = (γ-1)/γ  = bias_prefactor
//   sum_uprob = Σ_{j,k} h_k*(G_jk - ε)   (maintained by gatherDepositKernel)
// ---------------------------------------------------------------------------
extern const string kOPESKernelSrc = R"(
#define MAX_OPES_CVS 16
// ---------------------------------------------------------------------------
// opesEvalBias — OPES bias potential (Invernizzi & Parrinello 2020)
// ---------------------------------------------------------------------------
KERNEL void opesEvalBias(
 GLOBAL const double* RESTRICT cvValues,         // 0
 GLOBAL const int* RESTRICT cvIdxList,           // 1
 GLOBAL const float* RESTRICT centers,           // 2
 GLOBAL const float* RESTRICT sigmas,            // 3
 GLOBAL const float* RESTRICT logWeights,        // 4
 GLOBAL const int* RESTRICT numKernelsGPU,       // 5
 int numCVsBias,                                 // 6
 double invGammaFactor,                          // 7
 double kT,                                      // 8
 GLOBAL const double* RESTRICT sumUprobGPU,      // 9: Σ_{j,k} h_k*(G_jk-ε) (direct value)
 GLOBAL const double* RESTRICT sumWeightsGPU,    // 10: Σ h_k (KDNorm) — unused in formula but kept for API compat
 double cutoff2,                                 // 11: 2·γ/invGF = 2·γ²/(γ-1)
 GLOBAL double* RESTRICT cvBiasGradients,        // 12
 GLOBAL double* RESTRICT biasEnergies,           // 13
 int biasIdx                                     // 14
) {
 if (GLOBAL_ID != 0) return;
 double barrier = (invGammaFactor < 1.0) ? (kT / (1.0 - invGammaFactor)) : 0.0;
 // ε = exp(-γ/invGF) = exp(-1/(invGF·(1-invGF)))
 double eps = (invGammaFactor < 1.0)
  ? exp(-1.0 / (invGammaFactor * (1.0 - invGammaFactor)))
  : 0.0;

 int numKernels = numKernelsGPU[0];
 if (numKernels == 0) {
  biasEnergies[biasIdx] = -barrier;
  return;
 }

 double sVals[MAX_OPES_CVS];
 for (int d = 0; d < numCVsBias; d++)
  sVals[d] = cvValues[cvIdxList[d]];

 // Kernel contribution h_k * (G_k - ε) with hard cutoff at cutoff2.
 // G_k - ε > 0 for all norm2 < cutoff2 (since G_k=exp(-norm2/2) > exp(-cutoff2/2)=ε).
 // dV/ds_d uses d(G_k)/ds_d = G_k * (-dc_d/sigma_d^2); the ε term is constant → no gradient.
 double prob_unnorm = 0.0;
 double dAccum[MAX_OPES_CVS];
 for (int d = 0; d < numCVsBias; d++) dAccum[d] = 0.0;

 for (int k = 0; k < numKernels; k++) {
  double norm2 = 0.0;
  double dc_d[MAX_OPES_CVS], sig_d[MAX_OPES_CVS];
  bool skip = false;
  for (int d = 0; d < numCVsBias; d++) {
   dc_d[d] = sVals[d] - (double)centers[k*numCVsBias + d];
   sig_d[d] = (double)sigmas[k*numCVsBias + d];
   norm2 += dc_d[d]*dc_d[d] / (sig_d[d]*sig_d[d]);
   if (norm2 >= cutoff2) { skip = true; break; }
  }
  if (skip) continue;
  double h_k = exp((double)logWeights[k]);
  double G_k = exp(-0.5*norm2);
  double fk  = h_k * (G_k - eps);
  prob_unnorm += fk;
  for (int d = 0; d < numCVsBias; d++)
   dAccum[d] += h_k * G_k * (-dc_d[d] / (sig_d[d]*sig_d[d]));
 }

 // p/Zed = prob_unnorm * nker / sum_uprob
 double sum_uprob = sumUprobGPU[0];
 double pz = (sum_uprob > 0.0) ? prob_unnorm * (double)numKernels / sum_uprob : 0.0;

 biasEnergies[biasIdx] = invGammaFactor * kT * log(pz + eps);

 // dV/ds_d = invGF·kT · pz/(pz+ε) / prob_unnorm · dAccum[d]
 if (prob_unnorm > 0.0) {
  double dVfactor = invGammaFactor * kT * (pz / (pz + eps)) / prob_unnorm;
  for (int d = 0; d < numCVsBias; d++)
   cvBiasGradients[cvIdxList[d]] += dVfactor * dAccum[d];
 }
}

// Running weight accumulation for neff = (1+sum_w)^2 / (1+sum_w2).
// weight = exp(V/kT); sumWArr[0] = Σ w_i, sumWArr[1] = Σ w_i^2  (linear, not log).
// NOTE: this kernel is superseded by inline logic in opesGatherDeposit;
// kept as fallback/reference but not called from updateState().
KERNEL void opesUpdateNeff(
 GLOBAL const double* RESTRICT biasEnergies,
 int biasEnergyIdx,
 double invKT,
 GLOBAL double* RESTRICT sumWArr, // double[2]: {sum_w, sum_w2} (linear)
 GLOBAL int* RESTRICT stepCountArr
) {
 if (GLOBAL_ID != 0) return;
 double w = exp(biasEnergies[biasEnergyIdx] * invKT);
 sumWArr[0] += w;
 sumWArr[1] += w * w;
 stepCountArr[0]++;
}

// Deposit one OPES kernel and update sum_uprob — fully GPU-resident.
// Algorithm: Invernizzi & Parrinello (2020) J. Phys. Chem. Lett. 11:2731.
//
//  - log_weight = V/kT (full bias / kT, NOT invGF*V/kT)
//  - neff = (1+sum_w)^2/(1+sum_w2)  — 1+ terms give neff≈1 for a single early sample
//  - Silverman: sigma_dep = sigma0*(neff*(D+2)/4)^{-1/(D+4)}
//  - Height correction: stored_height = exp(V/kT) * Π(sigma0/sigma_dep)
//  - sumWArr stores {sum_w, sum_w2} (LINEAR, initialized with sentinel exp(-gamma))
//  - sum_uprob = Σ_{j,k} h_k_stored*(G_jk - ε)  (recomputed after each deposit)
KERNEL void opesGatherDeposit(
 GLOBAL const double* RESTRICT cvValues,          // 0
 GLOBAL const int* RESTRICT cvIdxList,            // 1
 GLOBAL double* RESTRICT runningMean,             // 2
 GLOBAL double* RESTRICT runningM2,              // 3
 GLOBAL int* RESTRICT nSamples,                  // 4
 GLOBAL const double* RESTRICT sigma0,           // 5
 double sigmaMin,                                // 6
 GLOBAL float* RESTRICT kCenters,               // 7
 GLOBAL float* RESTRICT kSigmas,                // 8
 GLOBAL float* RESTRICT kLogWeights,            // 9
 GLOBAL double* RESTRICT logSumPairwiseGPU,     // 10: sum_uprob (direct, not log)
 GLOBAL int* RESTRICT numKernelsGPU,            // 11: committed kernel count (read by eval)
 int D,                                         // 12
 int variant,                                   // 13
 int adaptiveSigmaStride,                       // 14
 GLOBAL const double* RESTRICT biasEnergies,   // 15
 int biasEnergyIdx,                             // 16
 double invKT,                                  // 17: 1/kT — weight = exp(V/kT) (full bias)
 GLOBAL double* RESTRICT sumWeightsGPU,        // 18: Σ h_raw (KDNorm, uncorrected)
 double cutoff2,                                // 19: 2·γ/invGF = 2·γ²/(γ-1)
 GLOBAL double* RESTRICT sumWArr,              // 20: {sum_w, sum_w2} (LINEAR)
 GLOBAL int* RESTRICT stepCountArr,            // 21: deposit count (for diagnostics)
 GLOBAL int* RESTRICT numAllocatedGPU,         // 22: B2 multiwalker slot-claim counter
 int maxKernels                                 // 23: buffer capacity limit
) {
 if (GLOBAL_ID != 0) return;
 if (adaptiveSigmaStride > 0 && nSamples[0] < adaptiveSigmaStride) return;
 int nBefore = numKernelsGPU[0];
 if (variant == 0 && adaptiveSigmaStride == 0) nSamples[0]++;
 int n = nSamples[0];

 // Compute new kernel center from current CV values.
 double c_new[MAX_OPES_CVS];
 for (int d = 0; d < D; d++)
  c_new[d] = cvValues[cvIdxList[d]];

 // weight = exp(V/kT)
 double lw_raw = biasEnergies[biasEnergyIdx] * invKT;
 double h_raw  = exp(lw_raw);

 // Linear weight accumulators {sum_w, sum_w2}.
 // Initialized with sentinel {exp(-gamma), exp(-2*gamma)} before any deposits.
 sumWArr[0] += h_raw;
 sumWArr[1] += h_raw * h_raw;
 stepCountArr[0]++;

 // neff = (1+sum_w)^2/(1+sum_w2) — 1+ terms give neff≈1
 // for very small early weights (exp(-gamma)), keeping all early deposits equal.
 double sw  = sumWArr[0];
 double sw2 = sumWArr[1];
 double neff_deposit = (1.0 + sw) * (1.0 + sw) / (1.0 + sw2);

 // Compute new kernel sigma.
 double sig_new[MAX_OPES_CVS];
 for (int d = 0; d < D; d++) {
  double sig;
  if (variant == 1) {
   // FIXED_SIGMA: no Silverman adaptation.
   sig = sigma0[d];
  } else if (adaptiveSigmaStride > 0) {
   double var = (n > 1) ? runningM2[d] / (n - 1) : 0.0;
   sig = (var > 0.0) ? sqrt(var) : 1.0;
   if (sig < sigmaMin) sig = sigmaMin;
  } else {
   // Default well-tempered: apply Silverman's rule (Scott 1992) — applied
   // even when sigma0 is provided, unless FIXED_SIGMA or adaptive stride is set.
   double s_rescaling = pow(neff_deposit * (D + 2) / 4.0, -1.0 / (D + 4.0));
   sig = sigma0[d] * s_rescaling;
   if (sig < sigmaMin) sig = sigmaMin;
  }
  sig_new[d] = sig;
 }

 // Height correction: stored_height = exp(V/kT) * Π(sigma0/sigma_dep).
 // Only applied for Silverman path (variant=0, no adaptive stride) where sigma0>0.
 // For adaptive sigma (sigma0=0 sentinel), skip to avoid log(0)=-inf.
 // For variant=1 (EXPLORE), sigma0/sig_new=1 so log=0 — correction is harmless but skip for clarity.
 double lw_new = lw_raw;
 if (variant == 0 && adaptiveSigmaStride == 0) {
  for (int d = 0; d < D; d++)
   lw_new += log(sigma0[d] / sig_new[d]);
 }
 double h_new = exp(lw_new);

 // Compression: find nearest existing kernel by Mahalanobis distance
 // using that existing kernel's sigma.
 const double compressionThresh2 = 1.0;
 int merge_k  = -1;
 double min_n2 = compressionThresh2;
 for (int kp = 0; kp < nBefore; kp++) {
  double n2 = 0.0;
  bool over = false;
  for (int d = 0; d < D; d++) {
   double dc = c_new[d] - (double)kCenters[kp*D+d];
   double s  = (double)kSigmas[kp*D+d];
   double x  = dc / s;
   n2 += x*x;
   if (n2 >= min_n2) { over = true; break; }
  }
  if (!over) { min_n2 = n2; merge_k = kp; }
 }

 int nAfter;
 if (merge_k >= 0) {
  // Merge: weighted Gaussian mixture merge (standard GMM statistics, Bishop 2006 §2.3).
  // NOTE: In B2 multiwalker mode, concurrent merges to the same slot can race.
  // This is a known limitation; compression is optional and rare in practice.
  double lw_old = (double)kLogWeights[merge_k];
  double h_old  = exp(lw_old);
  double h_tot  = h_old + h_new;
  double lw_max = lw_old > lw_new ? lw_old : lw_new;
  double lw_mrg = lw_max + log1p(exp((lw_old < lw_new ? lw_old : lw_new) - lw_max));
  double f1 = h_old / h_tot, f2 = h_new / h_tot;
  for (int d = 0; d < D; d++) {
   double c1 = (double)kCenters[merge_k*D+d];
   double s1 = (double)kSigmas[merge_k*D+d];
   double c2 = c_new[d], s2 = sig_new[d];
   double cm = f1*c1 + f2*c2;
   double ss = f1*(s1*s1+c1*c1) + f2*(s2*s2+c2*c2) - cm*cm;
   if (ss < 0.0) ss = 0.0;
   kCenters[merge_k*D+d] = (float)cm;
   kSigmas[merge_k*D+d]  = (float)sqrt(ss);
  }
  kLogWeights[merge_k] = (float)lw_mrg;
  nAfter = nBefore;
 } else {
  // B2 multiwalker: claim a slot atomically so concurrent walkers don't collide.
  // numAllocatedGPU is the slot-claim counter; numKernelsGPU is committed (visible to eval).
  int k = atomicAdd(numAllocatedGPU, 1);
  if (k < maxKernels) {
   // Write kernel data into the claimed slot.
   for (int d = 0; d < D; d++) {
    kCenters[k*D+d] = (float)c_new[d];
    kSigmas[k*D+d]  = (float)sig_new[d];
   }
   kLogWeights[k] = (float)lw_new;
   // Atomically increment committed count; eval kernel reads numKernelsGPU[0].
   // NOTE: In B2 multiwalker mode there is a small window where the eval kernel on
   // another context might read this slot before data is fully visible. This is an
   // accepted limitation of the first B2 implementation — the eval kernel would
   // simply use slightly stale (still-valid) sigma/center values for that slot.
   // For strict cross-context ordering, __threadfence_system() would be needed,
   // but it is not available in NVRTC-compiled kernels.
   atomicAdd(numKernelsGPU, 1);
   nAfter = nBefore + 1;
  } else {
   // Buffer full: roll back the allocation claim and skip.
   atomicAdd(numAllocatedGPU, -1);
   nAfter = nBefore;
  }
 }

 // KDNorm update: add uncorrected weight (Σ exp(bias_i/kT)).
 sumWeightsGPU[0] += h_raw;

 // Update sum_uprob = Σ_{j,k'} h_{k'}*(G_jk' - ε).
 // ε = exp(-cutoff2/2).  G_jk' uses sigma of k' (asymmetric).
 //
 // Append case (O(N)): when a new kernel k_new is added at index nBefore,
 // only two new cross-sections contribute: the new column (all j vs k_new)
 // and the new row (k_new vs all prior kp).  The existing pairs are unchanged.
 //
 // Merge case (O(N²)): merged kernel changes h, c, and σ at one index,
 // which affects all cross-pairs involving that index.  Merges are rare
 // (occur only when a new deposit lands inside an existing kernel), so full
 // recomputation is acceptable.
 double eps_dep = exp(-0.5 * cutoff2);
 if (merge_k < 0 && nAfter > nBefore) {
  // Append: O(N) incremental update.
  // New column k_new (index nBefore, Σ_j h_new*(G(c_j, c_new, s_new)-ε)):
  //   diagonal j=k_new: G=1, contribute h_new*(1-ε).
  double delta = h_new * (1.0 - eps_dep);
  for (int j = 0; j < nBefore; j++) {
   double n2 = 0.0; bool skip = false;
   for (int d = 0; d < D; d++) {
    double dc = (double)kCenters[j*D+d] - c_new[d];
    n2 += dc*dc / (sig_new[d]*sig_new[d]);
    if (n2 >= cutoff2) { skip = true; break; }
   }
   if (!skip) delta += h_new * (exp(-0.5*n2) - eps_dep);
  }
  // New row j=k_new (Σ_{kp=0..nBefore-1} h_kp*(G(c_new, c_kp, s_kp)-ε)):
  for (int kp = 0; kp < nBefore; kp++) {
   double n2 = 0.0; bool skip = false;
   for (int d = 0; d < D; d++) {
    double dc = c_new[d] - (double)kCenters[kp*D+d];
    double s  = (double)kSigmas[kp*D+d];
    n2 += dc*dc / (s*s);
    if (n2 >= cutoff2) { skip = true; break; }
   }
   if (!skip) delta += exp((double)kLogWeights[kp]) * (exp(-0.5*n2) - eps_dep);
  }
  logSumPairwiseGPU[0] += delta;
 } else {
  // Merge (or buffer-full no-op): full O(N²) recomputation.
  // Merges are rare; correctness is paramount.
  double sumUprob = 0.0;
  for (int j = 0; j < nAfter; j++) {
   for (int kp = 0; kp < nAfter; kp++) {
    double lw  = (double)kLogWeights[kp];
    double n2  = 0.0;
    bool skip  = false;
    for (int d = 0; d < D; d++) {
     double dc = (double)kCenters[j*D+d] - (double)kCenters[kp*D+d];
     double s  = (double)kSigmas[kp*D+d];
     n2 += dc*dc / (s*s);
     if (n2 >= cutoff2) { skip = true; break; }
    }
    if (skip) continue;
    double h_kp = exp(lw);
    double G_jkp = exp(-0.5*n2);
    sumUprob += h_kp * (G_jkp - eps_dep);
   }
  }
  logSumPairwiseGPU[0] = sumUprob;
 }
}

// Every-step Welford accumulator for fully-adaptive sigma mode.
// Called from execute() (once per step, before bias eval) whenever
// adaptiveSigmaStride > 0.  The deposit kernel reads the accumulated
// variance instead of computing it at deposition time.
KERNEL void opesAccumulateWelford(
 GLOBAL const double* RESTRICT cvValues,
 GLOBAL const int* RESTRICT cvIdxList,
 GLOBAL double* RESTRICT runningMean,
 GLOBAL double* RESTRICT runningM2,
 GLOBAL int* RESTRICT nSamples,
 int D
) {
 if (GLOBAL_ID != 0) return;
 nSamples[0]++;
 int n = nSamples[0];
 for (int d = 0; d < D; d++) {
 double cv = cvValues[cvIdxList[d]];
 double delta = cv - runningMean[d];
 runningMean[d] += delta / n;
 double delta2 = cv - runningMean[d];
 runningM2[d] += delta * delta2;
 }
}
)";

// ---------------------------------------------------------------------------
// OPES Expanded ensemble bias kernel (single thread).
//
// Target: uniform coverage across D thermodynamic states, each described by
// an energy collective variable (ECV) ecv_λ and a pre-normalized target weight w_λ.
//
// logQ(x) = log Σ_λ exp(logW_λ − ecv_λ(x)·invKT)
// V(x) = −kT · (logQ(x) − logZ)
// p_λ(x) = exp(logW_λ − ecv_λ·invKT − logQ) [softmax]
// dV/d(ecv_λ) = p_λ [bias gradient]
//
// logWeights = log(w_λ) are stored as doubles on the GPU.
// logZ is the running log-mean of Q updated on the CPU every pace steps.
// ---------------------------------------------------------------------------
// lw[l] = logWeights[l] - ecv_l * invKT is computed identically in all three
// passes.  Cache it to eliminate the repeated cvValues/logWeights reads.
#define MAX_OPES_EXP_STATES 256

extern const string kOPESExpandedKernelSrc = R"(
#define MAX_OPES_EXP_STATES 256
KERNEL void opesExpandedEvalBias(
 GLOBAL const double* RESTRICT cvValues,
 GLOBAL const int* RESTRICT cvIdxList,
 GLOBAL const double* RESTRICT logWeights, // log(w_λ) per state
 int numStates,
 double invKT,
 GLOBAL const double* RESTRICT logZGPU,
 GLOBAL double* RESTRICT cvBiasGradients,
 GLOBAL double* RESTRICT biasEnergies,
 int biasIdx
) {
 if (GLOBAL_ID != 0) return;

 // Cache lw[l] = logWeights[l] - ecv_l*invKT to avoid 3× re-reads.
 bool useCache = (numStates <= MAX_OPES_EXP_STATES);
 double lwBuf[MAX_OPES_EXP_STATES];

 // Pass 1: log-max for numerical stability
 double logMax = -1e300;
 for (int l = 0; l < numStates; l++) {
 double lw = logWeights[l] - cvValues[cvIdxList[l]] * invKT;
 if (useCache) lwBuf[l] = lw;
 if (lw > logMax) logMax = lw;
 }

 // Pass 2: sum exp via softmax
 double sumExp = 0.0;
 for (int l = 0; l < numStates; l++) {
 double lw = useCache ? lwBuf[l] : (logWeights[l] - cvValues[cvIdxList[l]] * invKT);
 sumExp += exp(lw - logMax);
 }
 double logQ = logMax + log(sumExp);

 biasEnergies[biasIdx] = -(1.0 / invKT) * (logQ - logZGPU[0]);

 for (int l = 0; l < numStates; l++) {
 double lw = useCache ? lwBuf[l] : (logWeights[l] - cvValues[cvIdxList[l]] * invKT);
 double p = exp(lw - logQ);
 cvBiasGradients[cvIdxList[l]] += p;
 }
}

// Running log-mean logZ update for OPES_EXPANDED — fully GPU-resident.
// logZ = log((n*exp(logZ) + exp(logQ)) / (n+1)) via numerically stable LSE.
KERNEL void opesExpandedUpdateLogZ(
 GLOBAL const double* RESTRICT cvValues,
 GLOBAL const int* RESTRICT cvIdxList,
 GLOBAL const double* RESTRICT logWeights,
 GLOBAL double* RESTRICT logZGPU,
 GLOBAL int* RESTRICT numUpdatesGPU,
 int numStates,
 double invKT
) {
 if (GLOBAL_ID != 0) return;
 double logMax = -1e300;
 for (int l = 0; l < numStates; l++) {
 double lw = logWeights[l] - cvValues[cvIdxList[l]] * invKT;
 if (lw > logMax) logMax = lw;
 }
 double sumExp = 0.0;
 for (int l = 0; l < numStates; l++) {
 double lw = logWeights[l] - cvValues[cvIdxList[l]] * invKT;
 sumExp += exp(lw - logMax);
 }
 double logQ = logMax + log(sumExp);
 int n = numUpdatesGPU[0];
 if (n == 0) {
 logZGPU[0] = logQ;
 } else {
 double A = logZGPU[0] + log((double)n);
 double B = logQ;
 double mx = (A > B) ? A : B;
 logZGPU[0] = mx + log(exp(A - mx) + exp(B - mx)) - log((double)(n + 1));
 }
 numUpdatesGPU[0] = n + 1;
}
)";

// ---------------------------------------------------------------------------
// Moving restraint bias kernel (single thread).
//
// schedule layout (stride = 1+2*D per anchor):
// schedule[i*stride+0] = step_i (double)
// schedule[i*stride+1+2*d] = k_{i,d} (double)
// schedule[i*stride+2+2*d] = at_{i,d}(double)
//
// Finds the interpolation segment containing currentStep and evaluates
// V = Σ_d 0.5*k_d*(cv_d - at_d)^2 with linearly interpolated k and at.
// ---------------------------------------------------------------------------
extern const string kMovingRestraintKernelSrc = R"(
KERNEL void movingRestraintEvalBias(
 GLOBAL const double* RESTRICT cvValues,
 GLOBAL const int* RESTRICT cvIdxList,
 GLOBAL const double* RESTRICT schedule,
 int M, int D,
 double currentStep,
 GLOBAL double* RESTRICT cvBiasGradients,
 GLOBAL double* RESTRICT biasEnergies,
 int biasIdx
) {
 if (GLOBAL_ID != 0) return;
 int stride = 1 + 2*D;

 int i0 = 0, i1 = 0;
 double alpha = 0.0;
 if (M > 1 && currentStep > schedule[0]) {
 if (currentStep >= schedule[(M-1)*stride]) {
 i0 = M-1; i1 = M-1;
 } else {
 for (int i = 0; i < M-1; i++) {
 double s1 = schedule[(i+1)*stride];
 if (currentStep < s1) {
 double s0 = schedule[i*stride];
 i0 = i; i1 = i+1;
 alpha = (currentStep - s0) / (s1 - s0);
 break;
 }
 }
 }
 }

 double V = 0.0;
 for (int d = 0; d < D; d++) {
 double k0 = schedule[i0*stride + 1 + 2*d];
 double at0 = schedule[i0*stride + 2 + 2*d];
 double k = k0, at = at0;
 if (i1 != i0) {
 double k1 = schedule[i1*stride + 1 + 2*d];
 double at1 = schedule[i1*stride + 2 + 2*d];
 k = k0 + alpha * (k1 - k0);
 at = at0 + alpha * (at1 - at0);
 }
 double cv = cvValues[cvIdxList[d]];
 double ds = cv - at;
 V += 0.5 * k * ds * ds;
 cvBiasGradients[cvIdxList[d]] += k * ds;
 }
 biasEnergies[biasIdx] = V;
}
)";

// ---------------------------------------------------------------------------
// ABMD ratchet bias kernel (single thread).
//
// Ratchet-and-pawl bias — Marchi & Ballone, J. Chem. Phys. 110:3697 (1999):
//   rho   = (cv - TO)^2          (squared distance from target)
//   rhoMin = running minimum of rho over the simulation
//   V = 0.5 * kappa * (rho - rhoMin)^2   when rho > rhoMin
//   V = 0                                  when rho <= rhoMin
//
// rhoMin is initialized to -1 (sentinel: first call always updates).
// rhoMin is updated inside evalBias (once per execute() call) — deposition-free.
// ---------------------------------------------------------------------------
extern const string kABMDKernelSrc = R"(
KERNEL void abmdEvalBias(
 GLOBAL const double* RESTRICT cvValues,
 GLOBAL const int* RESTRICT cvIdxList,
 GLOBAL const double* RESTRICT toValues,
 GLOBAL double* RESTRICT rhoMin,
 GLOBAL const float* RESTRICT kappa,
 int D,
 GLOBAL double* RESTRICT cvBiasGradients,
 GLOBAL double* RESTRICT biasEnergies,
 int biasIdx
) {
 if (GLOBAL_ID != 0) return;
 double V = 0.0;
 for (int d = 0; d < D; d++) {
 double cv = cvValues[cvIdxList[d]];
 double to = toValues[d];
 double cvDist = cv - to;
 double rho = cvDist * cvDist;
 double rhoM = rhoMin[d];
 if (rhoM < 0.0 || rho < rhoM) {
 rhoMin[d] = rho;
 } else {
 double diff = rho - rhoM;
 double k = (double)kappa[d];
 V += 0.5 * k * diff * diff;
 // dV/d(cv) = 2*k*(rho-rhoMin)*(cv-TO) — stored as gradient; scatter negates it
 cvBiasGradients[cvIdxList[d]] += 2.0 * k * diff * cvDist;
 }
 }
 biasEnergies[biasIdx] = V;
}
)";

// ---------------------------------------------------------------------------
// Extended Lagrangian (AFED) bias kernels (single thread each).
//
// extLagInitS: one-shot initialisation — copies cvValues into sArr at step 0.
// extLagVerlet: velocity Verlet for the auxiliary coordinate s.
// Coupling potential V = Σ_i κ_i/2*(cv_i - s_i)²
// extLagEvalBias: evaluates V and dV/d(cv_i) using the current sArr (GPU).
// ---------------------------------------------------------------------------
extern const string kExtLagKernelSrc = R"(
KERNEL void extLagInitS(
 GLOBAL const double* RESTRICT cvValues,
 GLOBAL const int* RESTRICT cvIdxList,
 GLOBAL double* RESTRICT sArr,
 int D
) {
 if (GLOBAL_ID != 0) return;
 for (int i = 0; i < D; i++)
 sArr[i] = cvValues[cvIdxList[i]];
}

KERNEL void extLagVerlet(
 GLOBAL const double* RESTRICT cvValues,
 GLOBAL const int* RESTRICT cvIdxList,
 GLOBAL const double* RESTRICT kappaArr,
 GLOBAL const double* RESTRICT massSArr,
 GLOBAL double* RESTRICT sArr,
 GLOBAL double* RESTRICT pArr,
 double dt,
 int D
) {
 if (GLOBAL_ID != 0) return;
 for (int i = 0; i < D; i++) {
 double cv = cvValues[cvIdxList[i]];
 double F = -kappaArr[i] * (sArr[i] - cv);
 pArr[i] += F * (0.5 * dt);
 sArr[i] += pArr[i] * dt / massSArr[i];
 double F2 = -kappaArr[i] * (sArr[i] - cv);
 pArr[i] += F2 * (0.5 * dt);
 }
}

KERNEL void extLagEvalBias(
 GLOBAL const double* RESTRICT cvValues,
 GLOBAL const int* RESTRICT cvIdxList,
 GLOBAL const double* RESTRICT kappaArr,
 GLOBAL const double* RESTRICT sArr,
 int D,
 GLOBAL double* RESTRICT cvBiasGradients,
 GLOBAL double* RESTRICT biasEnergies,
 int biasIdx
) {
 if (GLOBAL_ID != 0) return;
 double E = 0.0;
 for (int i = 0; i < D; i++) {
 int ci = cvIdxList[i];
 double cv = cvValues[ci];
 double s = sArr[i];
 double F = kappaArr[i] * (cv - s);
 cvBiasGradients[ci] += F;
 E += 0.5 * kappaArr[i] * (cv - s) * (cv - s);
 }
 biasEnergies[biasIdx] = E;
}
)";

// ---------------------------------------------------------------------------
// EDS adaptive bias kernels (single thread).
// Algorithm: White & Voth, J. Chem. Theory Comput. 10:3023 (2014).
//   - Welford online mean+SSD accumulation every step
//   - AdaGrad coupling update every PERIOD steps:
//       step  = 2*(mean−target)*ssd/(N−1)/kbt/scale
//       accum += step²
//       lambda += step * max_range / sqrt(accum)   (lambda = −coupling)
//   - Bias: V = −Σ lambda_i * cv_i,  force = lambda_i on cv_i
//
// edsWVUpdateState: Welford accumulation every step; AdaGrad update + reset
//   when doUpdate != 0.
// edsEvalBias: evaluates the linear bias.
// ---------------------------------------------------------------------------
extern const string kEdsKernelSrc = R"(
KERNEL void edsWVUpdateState(
 GLOBAL const double* RESTRICT cvValues,
 GLOBAL const int* RESTRICT cvIdxList,
 GLOBAL double* RESTRICT eds_mean,
 GLOBAL double* RESTRICT eds_ssd,
 GLOBAL double* RESTRICT eds_lambda,
 GLOBAL double* RESTRICT eds_accum,
 GLOBAL int* RESTRICT eds_count,
 GLOBAL const double* RESTRICT target,
 GLOBAL const double* RESTRICT max_range,
 double kbt,
 int doUpdate,
 int D
) {
 if (GLOBAL_ID != 0) return;
 for (int i = 0; i < D; i++) {
 double cv = cvValues[cvIdxList[i]];
 int n = eds_count[i] + 1;
 eds_count[i] = n;
 double delta = cv - eds_mean[i];
 eds_mean[i] += delta / (double)n;
 eds_ssd[i] += delta * (cv - eds_mean[i]);

 if (doUpdate) {
  double tgt = target[i];
  double mr = max_range[i];
  double scale = tgt < 0.0 ? -tgt : tgt;
  if (scale < 1e-10) scale = 1.0;
  double Nm1 = (n > 1) ? (double)(n - 1) : 1.0;
  // step = 2*(mean-target)*ssd/(N-1)/kbt/scale
  double step = 2.0 * (eds_mean[i] - tgt) * eds_ssd[i] / Nm1 / kbt / scale;
  // clip step to max_range
  if (step > mr) step = mr;
  else if (step < -mr) step = -mr;
  eds_accum[i] += step * step;
  double sqrtAccum = sqrt(eds_accum[i]);
  if (sqrtAccum < 1e-300) sqrtAccum = 1e-300;
  // lambda = -coupling; coupling += step*mr/sqrt(accum)
  // => lambda -= step*mr/sqrt(accum)
  eds_lambda[i] -= step * mr / sqrtAccum;
  // Reset statistics for next period
  eds_mean[i] = 0.0;
  eds_ssd[i] = 0.0;
  eds_count[i] = 0;
 }
 }
}

KERNEL void edsEvalBias(
 GLOBAL const double* RESTRICT cvValues,
 GLOBAL const int* RESTRICT cvIdxList,
 GLOBAL const double* RESTRICT lambdaArr,
 int D,
 GLOBAL double* RESTRICT cvBiasGradients,
 GLOBAL double* RESTRICT biasEnergies,
 int biasIdx
) {
 if (GLOBAL_ID != 0) return;
 double E = 0.0;
 for (int i = 0; i < D; i++) {
 int ci = cvIdxList[i];
 double cv = cvValues[ci];
 double lam = lambdaArr[i];
 cvBiasGradients[ci] -= lam;
 E -= lam * cv;
 }
 biasEnergies[biasIdx] = E;
}
)";

// ---------------------------------------------------------------------------
// MaxEnt: Lagrange-multiplier linear bias (Cesari et al. JCTC 2016).
//
// maxentUpdateLambda: single-thread kernel, runs every PACE steps.
//   lambda_i += [kappa_i / (1 + t/tau_i)] * (cv_i + xi_i - at_i)
//   xi_i (GAUSSIAN): -lambda_i * sigma^2
//   xi_i (LAPLACE):  -lambda_i*sigma^2 / (1 - lambda_i^2*sigma^2/(alpha+1))
//   convert_lambda(type, lam): EQUAL→lam, INEQUAL_GT→min(lam,0), INEQUAL_LT→max(lam,0)
//
// maxentEvalBias: single-thread kernel, runs every execute().
//   dV/dcv_i = kT * convert_lambda(type, lambda_i)
// ---------------------------------------------------------------------------
extern const string kMaxEntKernelSrc = R"(
KERNEL void maxentUpdateLambda(
 GLOBAL const double* RESTRICT cvValues,
 GLOBAL const int*    RESTRICT cvIdxList,
 GLOBAL double*       RESTRICT lambdaArr,
 GLOBAL const double* RESTRICT atArr,
 GLOBAL const double* RESTRICT kappaArr,
 GLOBAL const double* RESTRICT tauArr,
 double sigma,
 double alpha,
 int D,
 int type,
 int errorType,
 int updateCount
) {
 if (GLOBAL_ID != 0) return;
 double t = (double)updateCount;
 for (int i = 0; i < D; i++) {
  double cv  = cvValues[cvIdxList[i]];
  double lam = lambdaArr[i];
  double at  = atArr[i];
  double kap = kappaArr[i];
  double tau = tauArr[i];

  // Decaying learning rate
  double lr = kap / (1.0 + t / tau);

  // Error model correction
  double xi = 0.0;
  if (sigma > 0.0) {
   if (errorType == 0) {
    // GAUSSIAN
    xi = -lam * sigma * sigma;
   } else {
    // LAPLACE: clamp lambda so denominator stays positive
    double s2 = sigma * sigma;
    double ap1 = alpha + 1.0;
    double denom = 1.0 - lam * lam * s2 / ap1;
    if (denom < 0.01) denom = 0.01;
    xi = -lam * s2 / denom;
    // Hard clamp: keep |lam| < sqrt(alpha+1)/sigma - 0.01
    double lim = sqrt(ap1) / sigma - 0.01;
    if (lim < 0.0) lim = 0.0;
    if (lam >  lim) lam =  lim;
    if (lam < -lim) lam = -lim;
   }
  }

  lam += lr * (cv + xi - at);
  lambdaArr[i] = lam;
 }
}

KERNEL void maxentEvalBias(
 GLOBAL const double* RESTRICT cvValues,
 GLOBAL const int*    RESTRICT cvIdxList,
 GLOBAL const double* RESTRICT lambdaArr,
 double kbt,
 int D,
 int type,
 GLOBAL double*       RESTRICT cvBiasGradients,
 GLOBAL double*       RESTRICT biasEnergies,
 int biasIdx
) {
 if (GLOBAL_ID != 0) return;
 double E = 0.0;
 for (int i = 0; i < D; i++) {
  int ci = cvIdxList[i];
  double cv  = cvValues[ci];
  double lam = lambdaArr[i];
  // convert_lambda
  double leff;
  if      (type == 1) leff = (lam < 0.0) ? lam : 0.0;  // INEQUAL_GT
  else if (type == 2) leff = (lam > 0.0) ? lam : 0.0;  // INEQUAL_LT
  else                leff = lam;                        // EQUAL
  double grad = kbt * leff;
  cvBiasGradients[ci] += grad;   // scatter: F = -(dV/dCV * jac); store dV/dCV = kbt*leff
  E += grad * cv;                // V = kT * leff * cv
 }
 biasEnergies[biasIdx] = E;
}
)";

// ---------------------------------------------------------------------------
// MetaD grid evaluation and Gaussian deposition.
//
// metaDEvalBias: single-thread multilinear interpolation over the bias grid.
// V(cv) and dV/dcv accumulated into cvBiasGradients (+=).
//
// metaDDeposit: parallel kernel (one thread per grid point).
// Deposits a Gaussian of given height and sigma at center[] into grid[].
// No atomicAdd needed — each thread owns exactly one grid cell.
//
// Grid layout: column-major, strides[0]=1, strides[d]=strides[d-1]*actualPoints[d-1].
// actualPoints[d] = numBins[d] + (periodic[d] ? 0 : 1).
// ---------------------------------------------------------------------------
extern const string kMetaDKernelSrc = R"(
KERNEL void metaDEvalBias(
 GLOBAL const double* RESTRICT cvValues,
 GLOBAL const int* RESTRICT cvIdxList,
 GLOBAL const double* RESTRICT grid,
 GLOBAL const double* RESTRICT origin,
 GLOBAL const double* RESTRICT invSpacing,
 GLOBAL const int* RESTRICT actualPoints,
 GLOBAL const int* RESTRICT strides,
 GLOBAL const int* RESTRICT periodic,
 int D,
 GLOBAL double* RESTRICT cvBiasGradients,
 GLOBAL double* RESTRICT biasEnergies,
 int biasIdx
) {
 if (GLOBAL_ID != 0) return;
 int lo[3], hi[3];
 double alpha[3];
 for (int d = 0; d < D; d++) {
 double frac = (cvValues[cvIdxList[d]] - origin[d]) * invSpacing[d];
 int N = actualPoints[d];
 if (N <= 1) {
 lo[d] = 0; hi[d] = 0; alpha[d] = 0.0;
 } else if (periodic[d]) {
 frac = frac - floor(frac / (double)N) * (double)N;
 if (frac < 0.0) frac += N;
 lo[d] = (int)frac;
 if (lo[d] >= N) lo[d] = N - 1;
 alpha[d] = frac - lo[d];
 hi[d] = (lo[d] + 1) % N;
 } else {
 if (frac < 0.0) frac = 0.0;
 if (frac > (double)(N-1)) frac = (double)(N-1);
 lo[d] = (int)frac;
 if (lo[d] >= N-1) lo[d] = N-2;
 alpha[d] = frac - lo[d];
 hi[d] = lo[d] + 1;
 }
 }
 double V = 0.0, dV[3] = {0.0, 0.0, 0.0};
 int numCorners = 1 << D;
 for (int c = 0; c < numCorners; c++) {
 int flatIdx = 0;
 for (int d = 0; d < D; d++)
 flatIdx += (((c >> d) & 1) ? hi[d] : lo[d]) * strides[d];
 double val = grid[flatIdx];
 double w = 1.0;
 for (int d = 0; d < D; d++)
 w *= ((c >> d) & 1) ? alpha[d] : (1.0 - alpha[d]);
 V += w * val;
 for (int d = 0; d < D; d++) {
 double sign = ((c >> d) & 1) ? 1.0 : -1.0;
 double wo = 1.0;
 for (int d2 = 0; d2 < D; d2++) {
 if (d2 == d) continue;
 wo *= ((c >> d2) & 1) ? alpha[d2] : (1.0 - alpha[d2]);
 }
 dV[d] += sign * wo * val;
 }
 }
 biasEnergies[biasIdx] = V;
 for (int d = 0; d < D; d++)
 cvBiasGradients[cvIdxList[d]] += dV[d] * invSpacing[d];
}

// metaDDeposit reads V = biasEnergies[biasEnergyIdx] from the shared energy buffer
// (written by metaDEvalBias in the same execute() before this updateState() call)
// and computes height = h0 * exp(-V * heightInvFactor), where
// heightInvFactor = 0 → flat (γ≤1), height = h0
// heightInvFactor = 1/((γ-1)*kT) → well-tempered
// atomicAdd is used so that concurrent deposits from multiple walkers (B2 multiwalker)
// are race-free. Requires sm_60+ for double atomicAdd.
KERNEL void metaDDeposit(
 GLOBAL double* RESTRICT grid,
 GLOBAL const double* RESTRICT origin,
 GLOBAL const double* RESTRICT invSpacing,
 GLOBAL const int* RESTRICT actualPoints,
 GLOBAL const int* RESTRICT strides,
 GLOBAL const int* RESTRICT periodic,
 int D,
 int totalPoints,
 GLOBAL const double* RESTRICT center,
 GLOBAL const double* RESTRICT sigma,
 GLOBAL const double* RESTRICT biasEnergies,
 int biasEnergyIdx,
 double h0,
 double heightInvFactor
) {
 double V = biasEnergies[biasEnergyIdx];
 double height = (heightInvFactor > 0.0) ? h0 * exp(-V * heightInvFactor) : h0;
 for (int idx = GLOBAL_ID; idx < totalPoints; idx += GLOBAL_SIZE) {
 double exponent = 0.0;
 for (int d = 0; d < D; d++) {
 int bi = (idx / strides[d]) % actualPoints[d];
 double pos = origin[d] + (double)bi / invSpacing[d];
 double dc = pos - center[d];
 if (periodic[d]) {
 double period = (double)actualPoints[d] / invSpacing[d];
 dc -= round(dc / period) * period;
 }
 double s = sigma[d];
 exponent += dc * dc / (2.0 * s * s);
 }
 if (exponent <= 8.0)
 atomicAdd(&grid[idx], height * exp(-exponent));
 }
}

// Gather CV values into the deposition center buffer — replaces the D2H+H2D round-trip.
KERNEL void metaDGatherCVs(
 GLOBAL const double* RESTRICT cvValues,
 GLOBAL const int* RESTRICT cvIdxList,
 GLOBAL double* RESTRICT center,
 int D
) {
 if (GLOBAL_ID != 0) return;
 for (int d = 0; d < D; d++)
 center[d] = cvValues[cvIdxList[d]];
}
)";

// ---------------------------------------------------------------------------
// PBMETAD-specific kernels: log-sum-exp combining and per-sub-grid evaluation.
// ---------------------------------------------------------------------------
extern const string kPBMetaDKernelSrc = R"(
// Per-sub-grid eval: 1-D linear interpolation writing to localEnergy/localGrad.
// Does NOT write to biasEnergies or cvBiasGradients (combineKernel does that).
KERNEL void pbmetaDSubgridEval(
 GLOBAL const double* RESTRICT cvValues,
 GLOBAL const int* RESTRICT cvIdxList,
 GLOBAL const double* RESTRICT grid,
 GLOBAL const double* RESTRICT origin,
 GLOBAL const double* RESTRICT invSpacing,
 GLOBAL const int* RESTRICT actualPoints,
 GLOBAL const int* RESTRICT periodic,
 GLOBAL double* RESTRICT localEnergy,
 GLOBAL double* RESTRICT localGrad,
 int subgridIdx
) {
 if (GLOBAL_ID != 0) return;
 double frac = (cvValues[cvIdxList[0]] - origin[0]) * invSpacing[0];
 int N = actualPoints[0];
 int lo, hi;
 double alpha;
 if (periodic[0]) {
  frac = frac - floor(frac / (double)N) * (double)N;
  if (frac < 0.0) frac += (double)N;
  lo = (int)frac; if (lo >= N) lo = N - 1;
  alpha = frac - lo;
  hi = (lo + 1) % N;
 } else {
  if (frac < 0.0) frac = 0.0;
  if (frac > (double)(N - 1)) frac = (double)(N - 1);
  lo = (int)frac; if (lo >= N - 1) lo = N - 2;
  alpha = frac - lo;
  hi = lo + 1;
 }
 double vlo = grid[lo];
 double vhi = grid[hi];
 localEnergy[subgridIdx] = vlo * (1.0 - alpha) + vhi * alpha;
 localGrad[subgridIdx]   = (vhi - vlo) * invSpacing[0];
}

// Combine N sub-grid values into log-sum-exp PBMETAD bias.
// Writes V_total to biasEnergies[combinedSlot] and softmax-weighted gradients
// to cvBiasGradients. softmaxWeights[i] is written for use by pbmetaDDeposit.
KERNEL void pbmetaDCombine(
 GLOBAL const double* RESTRICT localEnergy,
 GLOBAL const double* RESTRICT localGrad,
 GLOBAL const int* RESTRICT cvIdxList,
 int N,
 double kT,
 GLOBAL double* RESTRICT biasEnergies,
 int combinedSlot,
 GLOBAL double* RESTRICT cvBiasGradients,
 GLOBAL double* RESTRICT softmaxWeights
) {
 if (GLOBAL_ID != 0) return;
 double bmin = localEnergy[0];
 for (int i = 1; i < N; i++)
  if (localEnergy[i] < bmin) bmin = localEnergy[i];
 double ene = 0.0;
 for (int i = 0; i < N; i++)
  ene += exp((bmin - localEnergy[i]) / kT);
 biasEnergies[combinedSlot] = -kT * (log(ene) - log((double)N)) + bmin;
 for (int i = 0; i < N; i++) {
  double wi = exp((bmin - localEnergy[i]) / kT) / ene;
  softmaxWeights[i] = wi;
  cvBiasGradients[cvIdxList[i]] += wi * localGrad[i];
 }
}

// PBMETAD deposit: height = softmaxWeights[subgridIdx] * h0 * exp(-V_i * heightInvFactor).
KERNEL void pbmetaDDeposit(
 GLOBAL double* RESTRICT grid,
 GLOBAL const double* RESTRICT origin,
 GLOBAL const double* RESTRICT invSpacing,
 GLOBAL const int* RESTRICT actualPoints,
 GLOBAL const int* RESTRICT periodic,
 int totalPoints,
 GLOBAL const double* RESTRICT center,
 GLOBAL const double* RESTRICT sigma,
 GLOBAL const double* RESTRICT localEnergy,
 int subgridIdx,
 double h0,
 double heightInvFactor,
 GLOBAL const double* RESTRICT softmaxWeights
) {
 double Vi = localEnergy[subgridIdx];
 double wi = softmaxWeights[subgridIdx];
 double height = (heightInvFactor > 0.0) ? wi * h0 * exp(-Vi * heightInvFactor) : wi * h0;
 for (int idx = GLOBAL_ID; idx < totalPoints; idx += GLOBAL_SIZE) {
  int bi = idx;
  double pos = origin[0] + (double)bi / invSpacing[0];
  double dc = pos - center[0];
  if (periodic[0]) {
   double period = (double)actualPoints[0] / invSpacing[0];
   dc -= round(dc / period) * period;
  }
  double s = sigma[0];
  double exponent = dc * dc / (2.0 * s * s);
  if (exponent <= 8.0)
   grid[idx] += height * exp(-exponent);
 }
}
)";


// ---------------------------------------------------------------------------
// Helper: build GpuPlan from Force descriptor and allocate GPU buffers.
// ---------------------------------------------------------------------------

void CommonCalcGluedForceKernel::buildPlan(const System& system,
 const GluedForce& force) {
 int numCVs = force.getNumCollectiveVariables(); // total CV values
 int numSpecs = force.getNumCollectiveVariableSpecs(); // number of addCV calls
 plan_.numCVs = numCVs;
 plan_.periodic = force.usesPeriodicBoundaryConditions();
 plan_.numDistanceCVs = 0;
 plan_.numAngleCVs = 0;
 plan_.numDihedralCVs = 0;
 plan_.numCOMDistanceCVs = 0;
 plan_.numGyrationCVs = 0;
 plan_.numCoordCVs = 0;
 plan_.numRMSDCVs = 0;
 plan_.numPathCVs = 0;
 plan_.numPositionCVs = 0;
 plan_.numDRMSDCVs = 0;
 plan_.numContactMapCVs = 0;
 plan_.numPlaneCVs = 0;
 plan_.numProjectionCVs = 0;
 plan_.numVolumeCVs = 0;
 plan_.numCellCVs = 0;
 plan_.numDipoleCVs = 0;
 plan_.numPCACVs = 0;
 distanceUserAtoms_.clear();
 angleUserAtoms_.clear();
 dihedralUserAtoms_.clear();
 comDistanceUserAtoms_.clear();
 comDistanceNGroup1_.clear();
 comDistanceCVAtomCount_.clear();
 comDistanceMassData_.clear();
 gyrationUserAtoms_.clear();
 gyrationNAtoms_.clear();
 gyrationMassData_.clear();
 coordUserAtoms_.clear();
 coordNGroup1_.clear();
 coordCVAtomCount_.clear();
 coordParamData_.clear();
 rmsdUserAtoms_.clear();
 rmsdNAtoms_.clear();
 rmsdRefPosData_.clear();
 pathUserAtoms_.clear();
 pathNAtoms_.clear();
 pathNFrames_.clear();
 pathLambdaData_.clear();
 pathRefPosData_.clear();
 pytorchCVPlans_.clear();
 pytorchNAtoms_.clear();
 positionUserAtoms_.clear();
 positionComponentData_.clear();
 drmsdUserPairAtoms_.clear();
 drmsdNPairs_.clear();
 drmsdRefDistData_.clear();
 contactMapUserPairAtoms_.clear();
 contactMapNPairs_.clear();
 contactMapParamData_.clear();
 planeUserAtoms_.clear();
 planeComponentData_.clear();
 projectionUserAtoms_.clear();
 projectionDirData_.clear();
 cellComponentData_.clear();
 dipoleUserAtoms_.clear();
 dipoleNAtoms_.clear();
 dipoleChargeData_.clear();
 dipoleComponentData_.clear();
 pcaUserAtoms_.clear();
 pcaNAtoms_.clear();
 pcaRefPosData_.clear();
 pcaEigvecData_.clear();

 int cvValueIdx = 0; // running CV value index (2 per PATH, 1 per others)
 for (int i = 0; i < numSpecs; i++) {
 int cvType;
 vector<int> atoms;
 vector<double> params;
 force.getCollectiveVariableInfo(i, cvType, atoms, params);
 if (cvType == GluedForce::CV_DISTANCE) {
 if (plan_.numDistanceCVs == 0) plan_.distanceFirstCVIndex = cvValueIdx;
 distanceUserAtoms_.push_back(atoms[0]);
 distanceUserAtoms_.push_back(atoms[1]);
 plan_.numDistanceCVs++;
 cvValueIdx += 1;
 } else if (cvType == GluedForce::CV_ANGLE) {
 if (plan_.numAngleCVs == 0) plan_.angleFirstCVIndex = cvValueIdx;
 angleUserAtoms_.push_back(atoms[0]);
 angleUserAtoms_.push_back(atoms[1]);
 angleUserAtoms_.push_back(atoms[2]);
 plan_.numAngleCVs++;
 cvValueIdx += 1;
 } else if (cvType == GluedForce::CV_DIHEDRAL) {
 if (plan_.numDihedralCVs == 0) plan_.dihedralFirstCVIndex = cvValueIdx;
 dihedralUserAtoms_.push_back(atoms[0]);
 dihedralUserAtoms_.push_back(atoms[1]);
 dihedralUserAtoms_.push_back(atoms[2]);
 dihedralUserAtoms_.push_back(atoms[3]);
 plan_.numDihedralCVs++;
 cvValueIdx += 1;
 } else if (cvType == GluedForce::CV_COM_DISTANCE) {
 if (plan_.numCOMDistanceCVs == 0) plan_.comDistanceFirstCVIndex = cvValueIdx;
 int ng1 = atoms[0];
 int totalAtoms = (int)atoms.size() - 1;
 comDistanceNGroup1_.push_back(ng1);
 comDistanceCVAtomCount_.push_back(totalAtoms);
 for (int a = 1; a <= totalAtoms; a++)
 comDistanceUserAtoms_.push_back(atoms[a]);
 for (float m : vector<float>(params.begin(), params.end()))
 comDistanceMassData_.push_back(m);
 plan_.numCOMDistanceCVs++;
 cvValueIdx += 1;
 } else if (cvType == GluedForce::CV_GYRATION) {
 if (plan_.numGyrationCVs == 0) plan_.gyrationFirstCVIndex = cvValueIdx;
 gyrationNAtoms_.push_back((int)atoms.size());
 for (int a : atoms) gyrationUserAtoms_.push_back(a);
 for (double m : params) gyrationMassData_.push_back((float)m);
 plan_.numGyrationCVs++;
 cvValueIdx += 1;
 } else if (cvType == GluedForce::CV_COORDINATION) {
 if (plan_.numCoordCVs == 0) plan_.coordFirstCVIndex = cvValueIdx;
 int nA = atoms[0];
 int totalAtoms = (int)atoms.size() - 1;
 coordNGroup1_.push_back(nA);
 coordCVAtomCount_.push_back(totalAtoms);
 for (int a = 1; a <= totalAtoms; a++)
 coordUserAtoms_.push_back(atoms[a]);
 for (int p = 0; p < 3; p++)
 coordParamData_.push_back((float)params[p]);
 plan_.numCoordCVs++;
 cvValueIdx += 1;
 } else if (cvType == GluedForce::CV_RMSD) {
 if (plan_.numRMSDCVs == 0) plan_.rmsdFirstCVIndex = cvValueIdx;
 int n = (int)atoms.size();
 rmsdNAtoms_.push_back(n);
 for (int a : atoms) rmsdUserAtoms_.push_back(a);
 for (double v : params) rmsdRefPosData_.push_back((float)v);
 plan_.numRMSDCVs++;
 cvValueIdx += 1;
 } else if (cvType == GluedForce::CV_PATH) {
 // params[0]=lambda, params[1]=N_frames, params[2..]=N*M*3 ref coords
 // Produces 2 CV values: s (progress) at cvValueIdx, z (dist) at cvValueIdx+1
 if (plan_.numPathCVs == 0) plan_.pathFirstCVIndex = cvValueIdx;
 int M = (int)atoms.size();
 int N = (int)params[1];
 pathNAtoms_.push_back(M);
 pathNFrames_.push_back(N);
 pathLambdaData_.push_back((float)params[0]);
 for (int a : atoms) pathUserAtoms_.push_back(a);
 for (int p = 2; p < (int)params.size(); p++)
 pathRefPosData_.push_back((float)params[p]);
 plan_.numPathCVs++;
 cvValueIdx += 2;
 } else if (cvType == GluedForce::CV_EXPRESSION) {
 // No atom Jacobian — gradient propagates through CV-level chain rule in execute()
 cvValueIdx += 1;
 } else if (cvType == GluedForce::CV_PYTORCH) {
 // PyTorch CV — CPU-intermediary, populates jacobian at runtime
 PyTorchCVPlan pt;
 pt.outputCVIndex = cvValueIdx;
 pt.numAtoms = (int)atoms.size();
 pt.userAtoms = atoms;
 pt.gpuAtomIdx.resize(pt.numAtoms, 0);
 pytorchCVPlans_.push_back(std::move(pt));
 pytorchNAtoms_.push_back((int)atoms.size());
 cvValueIdx += 1;
 } else if (cvType == GluedForce::CV_POSITION) {
 // single atom, single component
 if (plan_.numPositionCVs == 0) plan_.positionFirstCVIndex = cvValueIdx;
 positionUserAtoms_.push_back(atoms[0]);
 positionComponentData_.push_back((int)params[0]);
 plan_.numPositionCVs++;
 cvValueIdx += 1;
 } else if (cvType == GluedForce::CV_DRMSD) {
 // flat pair list; params = reference distances
 if (plan_.numDRMSDCVs == 0) plan_.drmsdFirstCVIndex = cvValueIdx;
 int nPairs = (int)atoms.size() / 2;
 drmsdNPairs_.push_back(nPairs);
 for (int a : atoms) drmsdUserPairAtoms_.push_back(a);
 for (double d : params) drmsdRefDistData_.push_back((float)d);
 plan_.numDRMSDCVs++;
 cvValueIdx += 1;
 } else if (cvType == GluedForce::CV_CONTACTMAP) {
 // flat pair list; params per pair = [r0, nn, mm, w]
 if (plan_.numContactMapCVs == 0) plan_.contactMapFirstCVIndex = cvValueIdx;
 int nPairs = (int)atoms.size() / 2;
 contactMapNPairs_.push_back(nPairs);
 for (int a : atoms) contactMapUserPairAtoms_.push_back(a);
 for (int p = 0; p < nPairs; p++) {
 contactMapParamData_.push_back((float)params[4*p]); // r0
 contactMapParamData_.push_back((float)params[4*p + 1]); // nn
 contactMapParamData_.push_back((float)params[4*p + 2]); // mm
 contactMapParamData_.push_back((float)params[4*p + 3]); // w
 }
 plan_.numContactMapCVs++;
 cvValueIdx += 1;
 } else if (cvType == GluedForce::CV_PLANE) {
 // 3 atoms [a,b,c]; params[0]=component (0/1/2)
 if (plan_.numPlaneCVs == 0) plan_.planeFirstCVIndex = cvValueIdx;
 planeUserAtoms_.push_back(atoms[0]);
 planeUserAtoms_.push_back(atoms[1]);
 planeUserAtoms_.push_back(atoms[2]);
 planeComponentData_.push_back((int)params[0]);
 plan_.numPlaneCVs++;
 cvValueIdx += 1;
 } else if (cvType == GluedForce::CV_PROJECTION) {
 // 2 atoms [a,b]; params[0..2]=[nx,ny,nz] direction
 if (plan_.numProjectionCVs == 0) plan_.projectionFirstCVIndex = cvValueIdx;
 projectionUserAtoms_.push_back(atoms[0]);
 projectionUserAtoms_.push_back(atoms[1]);
 // Store pre-normalized direction
 double nx = params[0], ny = params[1], nz = params[2];
 double nlen = sqrt(nx*nx + ny*ny + nz*nz);
 double invNlen = (nlen > 1e-10) ? (1.0/nlen) : 1.0;
 projectionDirData_.push_back((float)(nx * invNlen));
 projectionDirData_.push_back((float)(ny * invNlen));
 projectionDirData_.push_back((float)(nz * invNlen));
 plan_.numProjectionCVs++;
 cvValueIdx += 1;
 } else if (cvType == GluedForce::CV_VOLUME) {
 // no atoms; periodic system only; 0 Jacobian entries
 if (plan_.numVolumeCVs == 0) plan_.volumeFirstCVIndex = cvValueIdx;
 plan_.numVolumeCVs++;
 cvValueIdx += 1;
 } else if (cvType == GluedForce::CV_CELL) {
 // no atoms; params[0]=component (0=|a|, 1=|b|, 2=|c|)
 if (plan_.numCellCVs == 0) plan_.cellFirstCVIndex = cvValueIdx;
 cellComponentData_.push_back((int)params[0]);
 plan_.numCellCVs++;
 cvValueIdx += 1;
 } else if (cvType == GluedForce::CV_DIPOLE) {
 // atoms=group; params=[q0..qN-1, component (0=x,1=y,2=z,3=|μ|)]
 if (plan_.numDipoleCVs == 0) plan_.dipoleFirstCVIndex = cvValueIdx;
 int N = (int)atoms.size();
 dipoleNAtoms_.push_back(N);
 for (int a : atoms) dipoleUserAtoms_.push_back(a);
 for (int j = 0; j < N; j++) dipoleChargeData_.push_back((float)params[j]);
 dipoleComponentData_.push_back((int)params[N]);
 plan_.numDipoleCVs++;
 cvValueIdx += 1;
 } else if (cvType == GluedForce::CV_PCA) {
 // atoms=group; params=[mean_x0,y0,z0..., ev_x0,y0,z0...] (6N floats)
 if (plan_.numPCACVs == 0) plan_.pcaFirstCVIndex = cvValueIdx;
 int N = (int)atoms.size();
 pcaNAtoms_.push_back(N);
 for (int a : atoms) pcaUserAtoms_.push_back(a);
 for (int j = 0; j < 3*N; j++) pcaRefPosData_.push_back((float)params[j]);
 for (int j = 0; j < 3*N; j++) pcaEigvecData_.push_back((float)params[3*N + j]);
 plan_.numPCACVs++;
 cvValueIdx += 1;
 } else if (cvType == GluedForce::CV_SECONDARY_STRUCTURE) {
 // atoms=backbone list (5 per residue: N,CA,CB,C,O); params[0]=subtype, params[1]=r0 (nm)
 if (plan_.numSecStrCVs == 0) plan_.secStrFirstCVIndex = cvValueIdx;
 int N = (int)atoms.size();
 secStrNAtoms_.push_back(N);
 for (int a : atoms) secStrUserAtoms_.push_back(a);
 secStrParamData_.push_back((float)params[0]); // subtype
 secStrParamData_.push_back((float)params[1]); // r0
 plan_.numSecStrCVs++;
 cvValueIdx += 1;
 } else if (cvType == GluedForce::CV_PUCKERING) {
 // ring atoms in sequential order (5 or 6); params[0]=ring_size, params[1]=component
 if (plan_.numPuckeringCVs == 0) plan_.puckerFirstCVIndex = cvValueIdx;
 puckerNAtoms_.push_back((int)atoms.size());
 for (int a : atoms) puckerUserAtoms_.push_back(a);
 puckerParamData_.push_back((float)params[0]); // ring size
 puckerParamData_.push_back((float)params[1]); // component
 plan_.numPuckeringCVs++;
 cvValueIdx += 1;
 } else if (cvType == GluedForce::CV_ERMSD) {
 // eRMSD for RNA — 3*N atoms, params=[N, cutoff, G_ref_4D...]
 // G_ref has 4 values per ordered pair (i,j), i≠j: total 4*N*(N-1) values.
 if (plan_.numErmsdCVs == 0) plan_.ermsdFirstCVIndex = cvValueIdx;
 int N = (int)(params[0] + 0.5);
 ermsdNRes_.push_back(N);
 for (int a : atoms) ermsdUserAtoms_.push_back(a); // 3*N atoms
 ermsdCutoffData_.push_back((float)params[1]); // cutoff
 int nPairs = N*(N-1); // all ordered pairs
 for (int p = 0; p < 4*nPairs; p++)
 ermsdRefGData_.push_back((float)params[2 + p]);
 plan_.numErmsdCVs++;
 cvValueIdx += 1;
 }
 }

 // Jacobian entry counts:
 // distance: 2/CV, angle: 3/CV, dihedral: 4/CV
 // COM-distance / gyration / coordination: sum(atoms per CV)
 int comDistanceTotalAtoms = (int)comDistanceUserAtoms_.size();
 int gyrationTotalAtoms = (int)gyrationUserAtoms_.size();
 int coordTotalAtoms = (int)coordUserAtoms_.size();
 int rmsdTotalAtoms = (int)rmsdUserAtoms_.size();
 // Path: 2 Jacobian entries per atom per CV (one for s, one for z)
 int pathTotalAtoms = (int)pathUserAtoms_.size();

 int pytorchTotalAtoms = 0;
 for (int n : pytorchNAtoms_) pytorchTotalAtoms += n;

 int drmsdTotalPairs = (int)drmsdUserPairAtoms_.size() / 2;
 int contactMapTotalPairs = (int)contactMapUserPairAtoms_.size() / 2;

 plan_.distanceFirstJacEntry = 0;
 plan_.angleFirstJacEntry = 2 * plan_.numDistanceCVs;
 plan_.dihedralFirstJacEntry = plan_.angleFirstJacEntry + 3 * plan_.numAngleCVs;
 plan_.comDistanceFirstJacEntry = plan_.dihedralFirstJacEntry + 4 * plan_.numDihedralCVs;
 plan_.gyrationFirstJacEntry = plan_.comDistanceFirstJacEntry + comDistanceTotalAtoms;
 plan_.coordFirstJacEntry = plan_.gyrationFirstJacEntry + gyrationTotalAtoms;
 plan_.rmsdFirstJacEntry = plan_.coordFirstJacEntry + coordTotalAtoms;
 plan_.pathFirstJacEntry = plan_.rmsdFirstJacEntry + rmsdTotalAtoms;
 plan_.pytorchFirstJacEntry = plan_.pathFirstJacEntry + 2 * pathTotalAtoms;
 plan_.positionFirstJacEntry = plan_.pytorchFirstJacEntry + pytorchTotalAtoms;
 plan_.drmsdFirstJacEntry = plan_.positionFirstJacEntry + plan_.numPositionCVs;
 plan_.contactMapFirstJacEntry = plan_.drmsdFirstJacEntry + 2 * drmsdTotalPairs;
 plan_.planeFirstJacEntry = plan_.contactMapFirstJacEntry + 2 * contactMapTotalPairs;
 int dipoleTotalAtoms = (int)dipoleUserAtoms_.size();
 int pcaTotalAtoms = (int)pcaUserAtoms_.size();

 plan_.projectionFirstJacEntry = plan_.planeFirstJacEntry + 3 * plan_.numPlaneCVs;
 plan_.dipoleFirstJacEntry = plan_.projectionFirstJacEntry + 2 * plan_.numProjectionCVs;
 plan_.pcaFirstJacEntry = plan_.dipoleFirstJacEntry + dipoleTotalAtoms;

 // Compute SS total windows (30 Jacobian entries per window)
 int secStrTotalWindows = 0;
 for (int c = 0; c < plan_.numSecStrCVs; c++) {
 int subtype = (int)secStrParamData_[2*c];
 int N = secStrNAtoms_[c];
 // Alpha: N/5 backbone atoms per residue, sliding by 1 residue
 // Beta: stride-30 windows
 secStrTotalWindows += (subtype == 0) ? (N/5 - 5) : (N / 30);
 }
 plan_.secStrTotalWindows = secStrTotalWindows;
 plan_.secStrFirstJacEntry = plan_.pcaFirstJacEntry + pcaTotalAtoms;
 // puckering — N ring atoms per CV (N=5 or 6)
 int puckerTotalAtoms = (int)puckerUserAtoms_.size();
 plan_.puckerFirstJacEntry = plan_.secStrFirstJacEntry + secStrTotalWindows * 30;
 // eRMSD — 6*N*(N-1) Jacobian entries per CV (all ordered pairs)
 int ermsdTotalJac = 0;
 for (int c = 0; c < plan_.numErmsdCVs; c++) {
 int N = ermsdNRes_[c];
 ermsdTotalJac += 6 * N * (N-1);
 }
 plan_.ermsdFirstJacEntry = plan_.puckerFirstJacEntry + puckerTotalAtoms;
 plan_.numJacEntries = plan_.ermsdFirstJacEntry + ermsdTotalJac;
 // Volume and Cell have no atom Jacobian entries

 // Assign jacOffset to each PyTorchCVPlan
 {
 int offset = plan_.pytorchFirstJacEntry;
 for (auto& pt : pytorchCVPlans_) {
 pt.jacOffset = offset;
 offset += pt.numAtoms;
 }
 }

 plan_.cvValues.initialize<double>(cc_, numCVs, "cvValues");
 plan_.cvBiasGradients.initialize<double>(cc_, numCVs, "cvBiasGradients");
 cc_.addAutoclearBuffer(plan_.cvBiasGradients);

 if (plan_.numJacEntries > 0) {
 plan_.jacobianAtomIdx.initialize<int>(cc_, plan_.numJacEntries, "jacobianAtomIdx");
 plan_.jacobianGradsX.initialize<float>(cc_, plan_.numJacEntries, "jacobianGradsX");
 plan_.jacobianGradsY.initialize<float>(cc_, plan_.numJacEntries, "jacobianGradsY");
 plan_.jacobianGradsZ.initialize<float>(cc_, plan_.numJacEntries, "jacobianGradsZ");
 plan_.jacobianCvIdx.initialize<int>(cc_, plan_.numJacEntries, "jacobianCvIdx");
 }

 if (plan_.numDistanceCVs > 0)
 plan_.distanceAtoms.initialize<int>(cc_, 2 * plan_.numDistanceCVs, "distanceAtoms");
 if (plan_.numAngleCVs > 0)
 plan_.angleAtoms.initialize<int>(cc_, 3 * plan_.numAngleCVs, "angleAtoms");
 if (plan_.numDihedralCVs > 0)
 plan_.dihedralAtoms.initialize<int>(cc_, 4 * plan_.numDihedralCVs, "dihedralAtoms");

 if (plan_.numCOMDistanceCVs > 0) {
 int N = plan_.numCOMDistanceCVs;
 plan_.comDistanceAtomOffsets.initialize<int>(cc_, 2*N+1, "comDistanceAtomOffsets");
 plan_.comDistanceAtoms.initialize<int>(cc_, comDistanceTotalAtoms, "comDistanceAtoms");
 plan_.comDistanceMasses.initialize<float>(cc_, comDistanceTotalAtoms, "comDistanceMasses");
 plan_.comDistanceTotalMasses.initialize<float>(cc_, 2*N, "comDistanceTotalMasses");

 // Build interleaved offsets and compute total masses (masses don't change with reorder)
 vector<int> offsets(2*N+1);
 vector<float> totMasses(2*N);
 int cursor = 0;
 for (int c = 0; c < N; c++) {
 int ng1 = comDistanceNGroup1_[c];
 int total = comDistanceCVAtomCount_[c];
 int ng2 = total - ng1;
 offsets[2*c] = cursor;
 offsets[2*c+1] = cursor + ng1;
 float m1 = 0, m2 = 0;
 for (int j = 0; j < ng1; j++) m1 += comDistanceMassData_[cursor + j];
 for (int j = ng1; j < total; j++) m2 += comDistanceMassData_[cursor + j];
 totMasses[2*c] = m1;
 totMasses[2*c+1] = m2;
 cursor += total;
 }
 offsets[2*N] = cursor;
 plan_.comDistanceAtomOffsets.upload(offsets);
 plan_.comDistanceMasses.upload(comDistanceMassData_);
 plan_.comDistanceTotalMasses.upload(totMasses);
 }

 if (plan_.numGyrationCVs > 0) {
 int N = plan_.numGyrationCVs;
 plan_.gyrationAtomOffsets.initialize<int>(cc_, N+1, "gyrationAtomOffsets");
 plan_.gyrationAtoms.initialize<int>(cc_, gyrationTotalAtoms, "gyrationAtoms");
 plan_.gyrationMasses.initialize<float>(cc_, gyrationTotalAtoms, "gyrationMasses");
 plan_.gyrationTotalMasses.initialize<float>(cc_, N, "gyrationTotalMasses");

 vector<int> offsets(N+1);
 vector<float> totMasses(N);
 int cursor = 0;
 for (int c = 0; c < N; c++) {
 offsets[c] = cursor;
 float m = 0;
 for (int j = 0; j < gyrationNAtoms_[c]; j++)
 m += gyrationMassData_[cursor + j];
 totMasses[c] = m;
 cursor += gyrationNAtoms_[c];
 }
 offsets[N] = cursor;
 plan_.gyrationAtomOffsets.upload(offsets);
 plan_.gyrationMasses.upload(gyrationMassData_);
 plan_.gyrationTotalMasses.upload(totMasses);
 }

 if (plan_.numCoordCVs > 0) {
 int N = plan_.numCoordCVs;
 plan_.coordAtomOffsets.initialize<int>(cc_, 2*N+1, "coordAtomOffsets");
 plan_.coordAtoms.initialize<int>(cc_, coordTotalAtoms, "coordAtoms");
 plan_.coordParams.initialize<float>(cc_, 3*N, "coordParams");

 vector<int> offsets(2*N+1);
 int cursor = 0;
 for (int c = 0; c < N; c++) {
 int nA = coordNGroup1_[c];
 int total = coordCVAtomCount_[c];
 offsets[2*c] = cursor;
 offsets[2*c+1] = cursor + nA;
 cursor += total;
 }
 offsets[2*N] = cursor;
 plan_.coordAtomOffsets.upload(offsets);
 plan_.coordParams.upload(coordParamData_);
 }

 if (plan_.numRMSDCVs > 0) {
 int N = plan_.numRMSDCVs;
 plan_.rmsdAtomOffsets.initialize<int>(cc_, N+1, "rmsdAtomOffsets");
 plan_.rmsdAtoms.initialize<int>(cc_, rmsdTotalAtoms, "rmsdAtoms");
 plan_.rmsdRefPos.initialize<float>(cc_, 3*rmsdTotalAtoms, "rmsdRefPos");

 vector<int> offsets(N+1);
 int cursor = 0;
 for (int c = 0; c < N; c++) {
 offsets[c] = cursor;
 cursor += rmsdNAtoms_[c];
 }
 offsets[N] = cursor;
 plan_.rmsdAtomOffsets.upload(offsets);
 plan_.rmsdRefPos.upload(rmsdRefPosData_);
 }

 if (plan_.numPathCVs > 0) {
 int N = plan_.numPathCVs;
 plan_.pathAtomOffsets.initialize<int>(cc_, N+1, "pathAtomOffsets");
 plan_.pathAtoms.initialize<int>(cc_, pathTotalAtoms, "pathAtoms");

 // refOffsets[c] = cumulative N_i*M_i before CV c (in atoms, not floats)
 vector<int> atomOffsets(N+1), refOffsets(N+1);
 vector<float> paramsVec;
 int aCursor = 0, rCursor = 0;
 for (int c = 0; c < N; c++) {
 atomOffsets[c] = aCursor;
 refOffsets[c] = rCursor;
 paramsVec.push_back(pathLambdaData_[c]);
 paramsVec.push_back((float)pathNFrames_[c]);
 aCursor += pathNAtoms_[c];
 rCursor += pathNFrames_[c] * pathNAtoms_[c];
 }
 atomOffsets[N] = aCursor;
 refOffsets[N] = rCursor;

 int totalRefAtoms = rCursor;
 plan_.pathRefOffsets.initialize<int>(cc_, N+1, "pathRefOffsets");
 plan_.pathRefPos.initialize<float>(cc_, 3*totalRefAtoms, "pathRefPos");
 plan_.pathParams.initialize<float>(cc_, 2*N, "pathParams");

 plan_.pathAtomOffsets.upload(atomOffsets);
 plan_.pathRefOffsets.upload(refOffsets);
 plan_.pathRefPos.upload(pathRefPosData_);
 plan_.pathParams.upload(paramsVec);
 }

 // Position CVs
 if (plan_.numPositionCVs > 0) {
 int N = plan_.numPositionCVs;
 plan_.positionAtoms.initialize<int>(cc_, N, "positionAtoms");
 plan_.positionComponents.initialize<int>(cc_, N, "positionComponents");
 plan_.positionComponents.upload(positionComponentData_);
 // positionAtoms are GPU-space indices; uploaded in rebuildGpuAtomIndices()
 }

 // DRMSD CVs
 if (plan_.numDRMSDCVs > 0) {
 int N = plan_.numDRMSDCVs;
 int totalPairs = (int)drmsdUserPairAtoms_.size() / 2;
 plan_.drmsdPairOffsets.initialize<int>(cc_, N+1, "drmsdPairOffsets");
 plan_.drmsdAtomPairs.initialize<int>(cc_, 2*totalPairs, "drmsdAtomPairs");
 plan_.drmsdRefDists.initialize<float>(cc_, totalPairs, "drmsdRefDists");

 vector<int> offsets(N+1);
 int cursor = 0;
 for (int c = 0; c < N; c++) {
 offsets[c] = cursor;
 cursor += drmsdNPairs_[c];
 }
 offsets[N] = cursor;
 plan_.drmsdPairOffsets.upload(offsets);
 plan_.drmsdRefDists.upload(drmsdRefDistData_);
 // drmsdAtomPairs uploaded in rebuildGpuAtomIndices()
 }

 // ContactMap CVs
 if (plan_.numContactMapCVs > 0) {
 int N = plan_.numContactMapCVs;
 int totalPairs = (int)contactMapUserPairAtoms_.size() / 2;
 plan_.contactMapPairOffsets.initialize<int>(cc_, N+1, "contactMapPairOffsets");
 plan_.contactMapAtomPairs.initialize<int>(cc_, 2*totalPairs, "contactMapAtomPairs");
 plan_.contactMapParams.initialize<float>(cc_, 4*totalPairs, "contactMapParams");

 vector<int> offsets(N+1);
 int cursor = 0;
 for (int c = 0; c < N; c++) {
 offsets[c] = cursor;
 cursor += contactMapNPairs_[c];
 }
 offsets[N] = cursor;
 plan_.contactMapPairOffsets.upload(offsets);
 plan_.contactMapParams.upload(contactMapParamData_);
 // contactMapAtomPairs uploaded in rebuildGpuAtomIndices()
 }

 // Plane CVs
 if (plan_.numPlaneCVs > 0) {
 int N = plan_.numPlaneCVs;
 plan_.planeAtoms.initialize<int>(cc_, 3*N, "planeAtoms");
 plan_.planeComponents.initialize<int>(cc_, N, "planeComponents");
 plan_.planeComponents.upload(planeComponentData_);
 // planeAtoms uploaded in rebuildGpuAtomIndices()
 }

 // Projection CVs
 if (plan_.numProjectionCVs > 0) {
 int N = plan_.numProjectionCVs;
 plan_.projectionAtoms.initialize<int>(cc_, 2*N, "projectionAtoms");
 plan_.projectionDirs.initialize<float>(cc_, 3*N, "projectionDirs");
 plan_.projectionDirs.upload(projectionDirData_);
 // projectionAtoms uploaded in rebuildGpuAtomIndices()
 }

 // Cell CVs
 if (plan_.numCellCVs > 0) {
 int N = plan_.numCellCVs;
 plan_.cellComponents.initialize<int>(cc_, N, "cellComponents");
 plan_.cellComponents.upload(cellComponentData_);
 } else {
 // Always allocate at least 1 to avoid zero-size buffer in the combined kernel
 plan_.cellComponents.initialize<int>(cc_, 1, "cellComponents");
 }

 // Dipole CVs
 if (plan_.numDipoleCVs > 0) {
 int N = plan_.numDipoleCVs;
 vector<int> offsets(N + 1);
 int cursor = 0;
 for (int c = 0; c < N; c++) { offsets[c] = cursor; cursor += dipoleNAtoms_[c]; }
 offsets[N] = cursor;
 plan_.dipoleAtomOffsets.initialize<int>(cc_, N+1, "dipoleAtomOffsets");
 plan_.dipoleAtomOffsets.upload(offsets);
 plan_.dipoleAtoms.initialize<int>(cc_, dipoleTotalAtoms, "dipoleAtoms");
 plan_.dipoleCharges.initialize<float>(cc_, dipoleTotalAtoms, "dipoleCharges");
 plan_.dipoleCharges.upload(dipoleChargeData_);
 plan_.dipoleComponents.initialize<int>(cc_, N, "dipoleComponents");
 plan_.dipoleComponents.upload(dipoleComponentData_);
 // dipoleAtoms uploaded in rebuildGpuAtomIndices()
 }

 // PCA CVs
 if (plan_.numPCACVs > 0) {
 int N = plan_.numPCACVs;
 vector<int> offsets(N + 1);
 int cursor = 0;
 for (int c = 0; c < N; c++) { offsets[c] = cursor; cursor += pcaNAtoms_[c]; }
 offsets[N] = cursor;
 plan_.pcaAtomOffsets.initialize<int>(cc_, N+1, "pcaAtomOffsets");
 plan_.pcaAtomOffsets.upload(offsets);
 plan_.pcaAtoms.initialize<int>(cc_, pcaTotalAtoms, "pcaAtoms");
 plan_.pcaRefPos.initialize<float>(cc_, 3*pcaTotalAtoms, "pcaRefPos");
 plan_.pcaRefPos.upload(pcaRefPosData_);
 plan_.pcaEigvec.initialize<float>(cc_, 3*pcaTotalAtoms, "pcaEigvec");
 plan_.pcaEigvec.upload(pcaEigvecData_);
 // pcaAtoms uploaded in rebuildGpuAtomIndices()
 }

 // Secondary structure CVs
 if (plan_.numSecStrCVs > 0) {
 int numSsCVs = plan_.numSecStrCVs;
 int totalAtoms = (int)secStrUserAtoms_.size();

 // Build atom offsets (prefix sum of atom counts per CV)
 vector<int> atomOff(numSsCVs + 1);
 {
 int cursor = 0;
 for (int c = 0; c < numSsCVs; c++) {
 atomOff[c] = cursor;
 cursor += secStrNAtoms_[c];
 }
 atomOff[numSsCVs] = cursor;
 }

 // Build window offsets (prefix sum of window counts per CV)
 vector<int> winOff(numSsCVs + 1);
 {
 int cursor = 0;
 for (int c = 0; c < numSsCVs; c++) {
 winOff[c] = cursor;
 int subtype = (int)secStrParamData_[2*c];
 int N = secStrNAtoms_[c];
 cursor += (subtype == 0) ? (N - 5) : (N / 6);
 }
 winOff[numSsCVs] = cursor;
 }

 plan_.secStrAtomOffsets.initialize<int>(cc_, numSsCVs + 1, "secStrAtomOffsets");
 plan_.secStrAtomOffsets.upload(atomOff);
 plan_.secStrWindowOffsets.initialize<int>(cc_, numSsCVs + 1, "secStrWindowOffsets");
 plan_.secStrWindowOffsets.upload(winOff);
 plan_.secStrAtoms.initialize<int>(cc_, totalAtoms, "secStrAtoms");
 plan_.secStrParams.initialize<float>(cc_, 2 * numSsCVs, "secStrParams");
 plan_.secStrParams.upload(secStrParamData_);
 // secStrAtoms uploaded in rebuildGpuAtomIndices()
 }

 // Puckering CVs
 if (plan_.numPuckeringCVs > 0) {
 int numP = plan_.numPuckeringCVs;
 vector<int> aOff(numP + 1, 0);
 for (int c = 0; c < numP; c++) aOff[c+1] = aOff[c] + puckerNAtoms_[c];
 plan_.puckerAtomOffsets.initialize<int>(cc_, numP + 1, "puckerAtomOffsets");
 plan_.puckerAtomOffsets.upload(aOff);
 plan_.puckerAtoms.initialize<int>(cc_, (int)puckerUserAtoms_.size(), "puckerAtoms");
 plan_.puckerParams.initialize<float>(cc_, 2 * numP, "puckerParams");
 plan_.puckerParams.upload(puckerParamData_);
 // puckerAtoms uploaded in rebuildGpuAtomIndices()
 }

 // eRMSD CVs
 if (plan_.numErmsdCVs > 0) {
 int numE = plan_.numErmsdCVs;
 vector<int> aOff(numE+1,0), jOff(numE+1,0), gOff(numE+1,0), nRes(numE);
 for (int c = 0; c < numE; c++) {
 int N = ermsdNRes_[c];
 nRes[c] = N;
 aOff[c+1] = aOff[c] + 3*N;
 jOff[c+1] = jOff[c] + 6*N*(N-1);  // all ordered pairs
 gOff[c+1] = gOff[c] + 4*N*(N-1);  // 4D G-vectors, all ordered pairs
 }
 plan_.ermsdAtomOffsets.initialize<int>(cc_, numE+1, "ermsdAtomOffsets");
 plan_.ermsdAtomOffsets.upload(aOff);
 plan_.ermsdJacOffsets.initialize<int>(cc_, numE+1, "ermsdJacOffsets");
 plan_.ermsdJacOffsets.upload(jOff);
 plan_.ermsdRefGOffsets.initialize<int>(cc_, numE+1, "ermsdRefGOffsets");
 plan_.ermsdRefGOffsets.upload(gOff);
 plan_.ermsdNRes.initialize<int>(cc_, numE, "ermsdNRes");
 plan_.ermsdNRes.upload(nRes);
 plan_.ermsdAtoms.initialize<int>(cc_, aOff[numE], "ermsdAtoms");
 plan_.ermsdRefG.initialize<float>(cc_, (int)ermsdRefGData_.size(), "ermsdRefG");
 plan_.ermsdRefG.upload(ermsdRefGData_);
 plan_.ermsdCutoffs.initialize<float>(cc_, numE, "ermsdCutoffs");
 plan_.ermsdCutoffs.upload(ermsdCutoffData_);
 // ermsdAtoms uploaded in rebuildGpuAtomIndices()
 }

 rebuildGpuAtomIndices();
}




void CommonCalcGluedForceKernel::initialize(const System& system,
 const GluedForce& force) {
 ContextSelector selector(cc_);
 forceGroupFlag_ = 1 << force.getForceGroup();
 testForceMode_ = force.getTestForceMode();
 testForceScale_ = force.getTestForceScale();
 testBiasGradients_ = force.getTestBiasGradients();

 // test kernel (always compiled — cheap NVRTC call at startup)
 ComputeProgram testProg = cc_.compileProgram(kTestForceKernelSrc, {});
 testForceKernel_ = testProg->createKernel("applyTestForce");
 testForceKernel_->addArg(cc_.getPosq());
 testForceKernel_->addArg(cc_.getAtomIndexArray());
 testForceKernel_->addArg(cc_.getLongForceBuffer());
 testForceKernel_->addArg(cc_.getPaddedNumAtoms());
 testForceKernel_->addArg(cc_.getNumAtoms());
 testForceKernel_->addArg(); // 5: mode
 testForceKernel_->addArg(); // 6: scale
 testForceKernel_->addArg(); // 7: boxSize \
 testForceKernel_->addArg(); // 8: invBoxSize | setPeriodicBoxArgs(cc_,k,7)
 testForceKernel_->addArg(); // 9: boxVecX |
 testForceKernel_->addArg(); // 10: boxVecY |
 testForceKernel_->addArg(); // 11: boxVecZ /
 // Note: OpenMM 8.5.1 requires one extra placeholder beyond the 11 PBC
 // args (indices 0-11 = 12 params) because addPrimitiveArg in 8.5.1 uses
 // arrayArgs.size() as the index, and arrayArgs and primitiveArgs can
 // diverge by one element under certain kernel compilation paths.
 testForceKernel_->addArg(); // 12: spare (ensures primitiveArgs.size()>=12)

 // CV kernels only when CVs are present
 if (force.getNumCollectiveVariableSpecs() > 0) {
 buildPlan(system, force);

 compileCVKernels(force);
 setupBiases(force);
 }
}

double CommonCalcGluedForceKernel::execute(ContextImpl& context,
 bool includeForces,
 bool includeEnergy) {
 // test path — short-circuits the CV pipeline
 if (testForceMode_ != 0) {
 ContextSelector selector(cc_);
 testForceKernel_->setArg(5, testForceMode_);
 testForceKernel_->setArg(6, testForceScale_);
 setPeriodicBoxArgs(cc_, testForceKernel_, 7);
 testForceKernel_->execute(cc_.getNumAtoms());
 return 0.0;
 }

 if (plan_.numCVs == 0)
 return 0.0;

 ContextSelector selector(cc_);

 // Refresh GPU atom indices if OpenMM reordered atoms (Hilbert sort)
 if (cc_.getAtomsWereReordered())
 rebuildGpuAtomIndices();

 // cvBiasGradients is registered with addAutoclearBuffer — already zeroed by OpenMM.
 // Only upload when test mode supplies explicit gradient values.
 if (!testBiasGradients_.empty())
 plan_.cvBiasGradients.upload(testBiasGradients_);

 // Push box vectors to CV kernels only when the box has actually changed.
 // For NVE/NVT (no barostat) this fires exactly once (first step); for NPT it
 // fires at most once per barostat interval (typically every 25 steps).
 {
 Vec3 a, b, c;
 cc_.getPeriodicBoxVectors(a, b, c);
 if (boxArgsNeedUpdate_ || a != lastBoxA_ || b != lastBoxB_ || c != lastBoxC_) {
 lastBoxA_ = a; lastBoxB_ = b; lastBoxC_ = c;
 boxArgsNeedUpdate_ = false;
 int periodic = plan_.periodic ? 1 : 0;
 if (plan_.numDistanceCVs > 0) {
 setPeriodicBoxArgs(cc_, distanceKernel_, 11);
 distanceKernel_->setArg(16, periodic);
 }
 if (plan_.numAngleCVs > 0) {
 setPeriodicBoxArgs(cc_, angleKernel_, 11);
 angleKernel_->setArg(16, periodic);
 }
 if (plan_.numDihedralCVs > 0) {
 setPeriodicBoxArgs(cc_, dihedralKernel_, 11);
 dihedralKernel_->setArg(16, periodic);
 }
 if (plan_.numCOMDistanceCVs > 0) {
 setPeriodicBoxArgs(cc_, comDistanceKernel_, 14);
 comDistanceKernel_->setArg(19, periodic);
 }
 if (plan_.numGyrationCVs > 0) {
 setPeriodicBoxArgs(cc_, gyrationKernel_, 14);
 gyrationKernel_->setArg(19, periodic);
 }
 if (plan_.numCoordCVs > 0) {
 setPeriodicBoxArgs(cc_, coordKernel_, 13);
 coordKernel_->setArg(18, periodic);
 }
 if (plan_.numRMSDCVs > 0) {
 setPeriodicBoxArgs(cc_, rmsdKernel_, 13);
 rmsdKernel_->setArg(18, periodic);
 }
 if (plan_.numPathCVs > 0) {
 setPeriodicBoxArgs(cc_, pathKernel_, 15);
 pathKernel_->setArg(20, periodic);
 }
 if (plan_.numDRMSDCVs > 0) {
 setPeriodicBoxArgs(cc_, drmsdKernel_, 13);
 drmsdKernel_->setArg(18, periodic);
 }
 if (plan_.numContactMapCVs > 0) {
 setPeriodicBoxArgs(cc_, contactMapKernel_, 13);
 contactMapKernel_->setArg(18, periodic);
 }
 if (plan_.numPlaneCVs > 0) {
 setPeriodicBoxArgs(cc_, planeKernel_, 12);
 planeKernel_->setArg(17, periodic);
 }
 if (plan_.numProjectionCVs > 0) {
 setPeriodicBoxArgs(cc_, projectionKernel_, 12);
 projectionKernel_->setArg(17, periodic);
 }
 if (plan_.numVolumeCVs > 0 || plan_.numCellCVs > 0)
 setPeriodicBoxArgs(cc_, volumeCellKernel_, 6);
 }
 }

 if (plan_.numDistanceCVs > 0)
 distanceKernel_->execute(plan_.numDistanceCVs);
 if (plan_.numAngleCVs > 0)
 angleKernel_->execute(plan_.numAngleCVs);
 if (plan_.numDihedralCVs > 0)
 dihedralKernel_->execute(plan_.numDihedralCVs);
 if (plan_.numCOMDistanceCVs > 0)
 comDistanceKernel_->execute(plan_.numCOMDistanceCVs);
 if (plan_.numGyrationCVs > 0)
 gyrationKernel_->execute(plan_.numGyrationCVs);
 if (plan_.numCoordCVs > 0)
 coordKernel_->execute(plan_.numCoordCVs);
 if (plan_.numRMSDCVs > 0)
 rmsdKernel_->execute(plan_.numRMSDCVs);
 if (plan_.numPathCVs > 0)
 pathKernel_->execute(plan_.numPathCVs);
 if (plan_.numPositionCVs > 0)
 positionKernel_->execute(plan_.numPositionCVs);
 if (plan_.numDRMSDCVs > 0)
 drmsdKernel_->execute(plan_.numDRMSDCVs);
 if (plan_.numContactMapCVs > 0)
 contactMapKernel_->execute(plan_.numContactMapCVs);
 if (plan_.numPlaneCVs > 0)
 planeKernel_->execute(plan_.numPlaneCVs);
 if (plan_.numProjectionCVs > 0)
 projectionKernel_->execute(plan_.numProjectionCVs);
 if (plan_.numVolumeCVs > 0 || plan_.numCellCVs > 0)
 volumeCellKernel_->execute(1);
 if (plan_.numDipoleCVs > 0) {
 dipoleKernel_->execute(plan_.numDipoleCVs);
 }
 if (plan_.numPCACVs > 0) {
 pcaKernel_->execute(plan_.numPCACVs);
 }
 if (plan_.numSecStrCVs > 0) {
 secStrKernel_->execute(plan_.numSecStrCVs);
 }
 if (plan_.numPuckeringCVs > 0) {
 puckerKernel_->execute(plan_.numPuckeringCVs);
 }
 if (plan_.numErmsdCVs > 0) {
 ermsdKernel_->execute(plan_.numErmsdCVs);
 }

 // expression CV eval (reads cvValues of inputs, writes cvValues of output + partials)
 for (auto& ep : expressionCVPlans_)
 ep.evalKernel->execute(1);

 // PyTorch CV evaluation.
 // GPU-native path (primary): extract kernel gathers positions into a GPU buffer;
 // torch runs entirely on the OpenMM CUDA stream via CUDAStreamGuard; backward
 // gradient is D2D-copied to gradBufGPU; deinterleave kernel scatters it into jacXYZ.
 // CPU-intermediary fallback: used when the GPU-native buffers aren't initialized
 // (non-CUDA platform) — downloads posq, runs torch on CPU, uploads results.
 if (!pytorchCVPlans_.empty()) {
#ifdef GLUED_HAS_TORCH
 void* rawStream = getNativeCudaStream();
 if (rawStream != nullptr && pytorchCVPlans_[0].posBufGPU.isInitialized()) {
 // === GPU-native path ===
 // All operations are enqueued on the same CUDA stream (OpenMM's queue
 // and torch ops via CUDAStreamGuard), so they are automatically serialized.
 cudaStream_t stream = static_cast<cudaStream_t>(rawStream);
 auto torchStream = c10::cuda::getStreamFromExternal(
 stream, c10::cuda::current_device());
 c10::cuda::CUDAStreamGuard guard(torchStream);

 for (auto& pt : pytorchCVPlans_) {
 // 1. Gather positions into flat float buffer on GPU
 pt.extractKernel->execute(pt.numAtoms);

 // 2. Zero-copy CUDA tensor wrapping posBufGPU
 void* posBufPtr = getComputeArrayDevPtr(pt.posBufGPU);
 auto opts = torch::TensorOptions()
 .dtype(torch::kFloat32)
 .device(torch::kCUDA, c10::cuda::current_device());
 auto input = torch::from_blob(posBufPtr, {pt.numAtoms, 3}, opts)
 .clone() // D2D copy → leaf tensor
 .requires_grad_(true);

 // 3. Forward pass on the shared CUDA stream
 auto* module = static_cast<torch::jit::script::Module*>(pt.model.get());
 auto output = module->forward({input}).toTensor().squeeze();

 // 4. Write scalar CV value D2D: cast to float64 on GPU then
 // cudaMemcpyAsync directly into the cvValues slot — no CPU touch,
 // no stream sync. .to(kFloat64) is a no-op if the model already
 // outputs double; otherwise it enqueues a cast kernel on the stream.
 {
 auto output_d = output.to(torch::kFloat64);
 char* cvBase = reinterpret_cast<char*>(getComputeArrayDevPtr(plan_.cvValues));
 cudaMemcpyAsync(cvBase + pt.outputCVIndex * sizeof(double),
                 output_d.data_ptr<double>(),
                 sizeof(double), cudaMemcpyDeviceToDevice, stream);
 }

 // 5. Backward pass on the shared CUDA stream
 output.backward();
 auto grad = input.grad().contiguous();

 // 6. D2D copy interleaved gradient into gradBufGPU on the same stream.
 // Stream serialization ensures deinterleavKernel (step 7) sees the
 // completed copy; no explicit sync needed.
 void* gradBufPtr = getComputeArrayDevPtr(pt.gradBufGPU);
 cudaMemcpyAsync(gradBufPtr, grad.data_ptr<float>(),
 static_cast<size_t>(pt.numAtoms) * 3 * sizeof(float),
 cudaMemcpyDeviceToDevice, stream);

 // 7. Deinterleave gradient into separate XYZ Jacobian arrays
 pt.deinterleavKernel->execute(pt.numAtoms);
 }
 } else {
 // === CPU-intermediary fallback ===
 // Flush GPU queue so posq reflects the current step's positions.
 cc_.flushQueue();
 vector<mm_float4> posqAll(cc_.getPaddedNumAtoms());
 cc_.getPosq().download(posqAll);

 for (auto& pt : pytorchCVPlans_) {
 vector<float> posCPU(pt.numAtoms * 3);
 for (int i = 0; i < pt.numAtoms; i++) {
 int g = pt.gpuAtomIdx[i];
 posCPU[3*i] = posqAll[g].x;
 posCPU[3*i+1] = posqAll[g].y;
 posCPU[3*i+2] = posqAll[g].z;
 }
 auto input = torch::from_blob(posCPU.data(), {pt.numAtoms, 3},
 torch::TensorOptions().dtype(torch::kFloat32))
 .clone()
 .requires_grad_(true);
 auto* module = static_cast<torch::jit::script::Module*>(pt.model.get());
 auto output = module->forward({input}).toTensor().squeeze();
 double cvVal = output.item<double>();
 plan_.cvValues.uploadSubArray(&cvVal, pt.outputCVIndex, 1);
 output.backward();
 auto grad = input.grad().contiguous();
 const float* gdata = grad.data_ptr<float>();
 vector<float> gx(pt.numAtoms), gy(pt.numAtoms), gz(pt.numAtoms);
 for (int i = 0; i < pt.numAtoms; i++) {
 gx[i] = gdata[3*i];
 gy[i] = gdata[3*i+1];
 gz[i] = gdata[3*i+2];
 }
 plan_.jacobianGradsX.uploadSubArray(gx.data(), pt.jacOffset, pt.numAtoms);
 plan_.jacobianGradsY.uploadSubArray(gy.data(), pt.jacOffset, pt.numAtoms);
 plan_.jacobianGradsZ.uploadSubArray(gz.data(), pt.jacOffset, pt.numAtoms);
 }
 }
#else
 throw OpenMMException(
 "Glued: PyTorch CVs require the CUDA platform compiled with libtorch "
 "(GLUED_HAS_TORCH). Use a CUDA-capable system with PyTorch installed.");
#endif
 }

 // evaluate bias potentials and accumulate into cvBiasGradients.
 // Must run after CV kernels (need cvValues) and before scatter (fills gradients).
 // Skip when testBiasGradients_ is set (unit tests supply gradients directly).
 double biasEnergy = 0.0;
 if (testBiasGradients_.empty()) {
 for (auto& h : harmonicBiases_) {
 h.evalKernel->execute(1);
 }
 // Use context.getStepCount() rather than lastKnownStep_.
 // getState(getEnergy=True) calls execute() WITHOUT calling updateContextState()
 // first, so lastKnownStep_ would be stale (set at step N-1, not N) after
 // the step count is incremented by the Verlet kernel's execute.
 double currentStepForBias = (double)context.getStepCount();
 for (auto& mv : movingRestraintBiases_) {
 if (currentStepForBias != mv.lastStep) {
 mv.lastStep = currentStepForBias;
 mv.evalKernel->setArg(5, currentStepForBias);
 }
 mv.evalKernel->execute(1);
 }
 for (auto& o : opesBiases_) {
 // Fully-adaptive mode: accumulate Welford stats every step so the deposit
 // kernel has a good variance estimate when it fires at the next pace boundary.
 if (o.welfordKernel) {
 o.welfordKernel->execute(1);
 o.nSamplesCPU++;  // mirrors GPU nSamples; used in updateState() guard
 }
 // numKernels is GPU-resident in o.numKernelsGPU — no setArg needed.
 o.evalKernel->execute(1);
 }
 for (auto& a : abmdBiases_) {
 a.evalKernel->execute(1);
 }
 for (auto& m : metaDGridBiases_) {
 m.evalKernel->execute(1);
 }
 for (auto& pb : pbmetaDGridBiases_) {
 for (auto& m : pb.subGrids)
 m.evalKernel->execute(1);  // pbmetaDSubgridEval → localEnergy[d], localGrad[d]
 pb.combineKernel->execute(1); // log-sum-exp → biasEnergies[combinedSlot], softmaxWeights
 }
 for (auto& m : externalGridBiases_) {
 m.evalKernel->execute(1);
 }
 for (auto& lin : linearBiases_) {
 lin.evalKernel->execute(1);
 }
 for (auto& wall : wallBiases_) {
 wall.evalKernel->execute(1);
 }
 for (auto& oe : opesExpandedBiases_)
 oe.evalKernel->execute(1);
 // Extended Lagrangian: evaluate coupling (s is GPU-resident).
 // On first execute() call, s is uninitialized — seed it from current cvValues.
 for (auto& el : extLagBiases_) {
 if (!el.initialized) {
 el.initSKernel->execute(1);
 el.initialized = true;
 }
 el.evalKernel->execute(1);
 }
 // EDS: evaluate linear bias (lambda is GPU-resident).
 for (auto& eds : edsBiases_)
 eds.evalKernel->execute(1);
 // MaxEnt: evaluate linear bias (lambda is GPU-resident).
 for (auto& mx : maxentBiases_)
 mx.evalKernel->execute(1);

 // D2H only when energy reporting is requested — MetaD height and OPES neff
 // are now fully GPU-resident and no longer need this mirror every step.
 if (includeEnergy && !plan_.biasEnergiesCPU.empty()) {
 plan_.biasEnergies.download(plan_.biasEnergiesCPU);
 for (double e : plan_.biasEnergiesCPU) biasEnergy += e;
 }
 }

 // expression CV gradient propagation (reverse dep order).
 // Must run unconditionally — even in testBiasGradients_ mode, the test sets
 // gradients on the expression CV output and expects them to propagate to inputs.
 for (int ei = (int)expressionCVPlans_.size()-1; ei >= 0; ei--)
 if (expressionCVPlans_[ei].numInputs > 0)
 expressionCVPlans_[ei].propKernel->execute(1);

 if (plan_.numJacEntries > 0)
 scatterKernel_->execute(plan_.numJacEntries);

 cvValuesReady_ = true;
 return biasEnergy;
}

void CommonCalcGluedForceKernel::updateState(ContextImpl& context,
 int step) {
 // Always track the step count — used by moving restraint in execute().
 lastKnownStep_ = step;

 // updateState() is called BEFORE execute() within the same step.
 // cvValues holds results from the PREVIOUS execute(); skip if uninitialized.
 if (!cvValuesReady_) return;
 if (step == lastUpdateStep_) return;
 if (opesBiases_.empty() && abmdBiases_.empty() && metaDGridBiases_.empty() && pbmetaDGridBiases_.empty() && opesExpandedBiases_.empty() && extLagBiases_.empty() && edsBiases_.empty() && maxentBiases_.empty()) return;
 lastUpdateStep_ = step;

 ContextSelector selector(cc_);

 // Extended Lagrangian: GPU velocity Verlet every step.
 if (!extLagBiases_.empty()) {
 double dt = context.getIntegrator().getStepSize();
 for (auto& el : extLagBiases_) {
 if (!el.initialized) {
 el.initSKernel->execute(1);
 el.initialized = true;
 continue;
 }
 if (dt != el.lastDt) {
 el.lastDt = dt;
 el.verletKernel->setArg(6, dt);
 }
 el.verletKernel->execute(1);
 }
 }

 // EDS White-Voth: Welford accumulation every step; AdaGrad update every pace steps.
 if (!edsBiases_.empty()) {
 for (auto& eds : edsBiases_) {
 int doUpdate = (step % eds.pace == 0) ? 1 : 0;
 if (doUpdate != eds.lastDoUpdate) {
 eds.lastDoUpdate = doUpdate;
 eds.updateStateKernel->setArg(10, doUpdate);
 }
 eds.updateStateKernel->execute(1);
 }
 }

 // MaxEnt: update Lagrange multipliers every pace steps.
 if (!maxentBiases_.empty()) {
 for (auto& mx : maxentBiases_) {
 if (step % mx.pace != 0) continue;
 int uc = step / mx.pace;   // update count t
 mx.updateKernel->setArg(11, uc);
 mx.updateKernel->execute(1);
 }
 }

 // OPES: GPU-resident Welford + logZ update + kernel deposition (pace-gated).
 // gatherDepositKernel reads cvValues, updates runningMean/runningM2/logZ on GPU,
 // and writes the new kernel row into kernelCenters/Sigmas/LogWeights.
 // neff accumulation reads biasEnergies[biasEnergyIdx] directly on GPU.
 for (auto& o : opesBiases_) {
 if (step == 0 || step % o.pace != 0) continue;
 if (o.numKernels >= o.maxKernels) continue;

 // In fully-adaptive mode the GPU deposit kernel skips until enough samples
 // have been accumulated by welfordKernel.  Mirror that guard on the CPU so
 // numKernels stays in sync (o.nSamplesCPU is incremented by execute()).
 bool gpuWillDeposit = (o.adaptiveSigmaStride == 0) ||
 (o.nSamplesCPU >= o.adaptiveSigmaStride);

 o.gatherDepositKernel->execute(1); // reads+increments numKernelsGPU; also updates neff accumulators
 if (gpuWillDeposit)
 o.numKernels++; // CPU mirror for the maxKernels guard
 }

 // OPES_EXPANDED: update logZ every pace steps — fully GPU-resident.
 for (auto& oe : opesExpandedBiases_) {
 if (step == 0 || step % oe.pace != 0) continue;
 oe.updateLogZKernel->execute(1);
 }

 // MetaD deposition (pace-gated) — fully GPU-resident.
 // gatherCVsKernel populates centerGPU; depositKernel reads biasEnergies[biasEnergyIdx]
 // to compute the well-tempered height internally — no CPU round-trip.
 for (auto& m : metaDGridBiases_) {
 if (step == 0 || step % m.pace != 0) continue;
 m.gatherCVsKernel->execute(1);
 m.depositKernel->execute(m.totalGridPoints);
 m.numDeposited++;
 }

 // PBMetaD deposition (pace-gated) — fully GPU-resident.
 for (auto& pb : pbmetaDGridBiases_) {
 if (step == 0 || step % pb.pace != 0) continue;
 for (auto& m : pb.subGrids) {
 m.gatherCVsKernel->execute(1);
 m.depositKernel->execute(m.totalGridPoints);
 m.numDeposited++;
 }
 }

}

void CommonCalcGluedForceKernel::getCurrentCVs(ContextImpl& context,
 vector<double>& values) {
 values = downloadCVValues();
}

vector<double> CommonCalcGluedForceKernel::getOPESMetrics(int biasIndex) {
 if (biasIndex < 0 || biasIndex >= (int)opesBiases_.size())
 throw OpenMM::OpenMMException("getOPESMetrics: biasIndex out of range");
 ContextSelector selector(cc_);
 OpesBias& o = opesBiases_[biasIndex];
 vector<double> logZVec(1, 0.0);
 if (o.numKernels > 0)
 o.logZGPU.download(logZVec);
 double logZ = logZVec[0];
 double zed = std::exp(logZ);
 double rct = o.kT * logZ;
 // Download the GPU-resident committed kernel count so that multiwalker B2
 // shared deposits from other walkers are reflected in the returned value.
 // (o.numKernels is only the CPU mirror for this walker's own deposits.)
 vector<int> nkVec(1, 0);
 o.numKernelsGPU.download(nkVec);
 double nker = (double)nkVec[0];
 // neff = (1+sum_w)^2/(1+sum_w2) — 1+ terms give neff≈1 for a single early sample.
 vector<double> swVec(2, 0.0);
 o.logSumWGPU.download(swVec);
 double sw = swVec[0], sw2 = swVec[1];
 double neff = (1.0 + sw) * (1.0 + sw) / (1.0 + sw2);
 return {zed, rct, nker, neff};
}

void CommonCalcGluedForceKernel::redirectToPrimaryBias(
    int biasType, int localIdx, const std::vector<long long>& ptrs) {
    if (biasType == GluedForce::BIAS_METAD && localIdx < (int)metaDGridBiases_.size()) {
        auto& m = metaDGridBiases_[localIdx];
        unsigned long long gridPtr = (unsigned long long)ptrs[0];
        m.sharedGridPtr = gridPtr;
        // Redirect evalKernel arg 2 (grid) and depositKernel arg 0 (grid).
        m.evalKernel->setArg(2, gridPtr);
        m.depositKernel->setArg(0, gridPtr);
    } else if (biasType == GluedForce::BIAS_OPES && localIdx < (int)opesBiases_.size()) {
        auto& o = opesBiases_[localIdx];
        o.sharedCentersPtr    = (unsigned long long)ptrs[0];
        o.sharedSigmasPtr     = (unsigned long long)ptrs[1];
        o.sharedLogWeightsPtr = (unsigned long long)ptrs[2];
        o.sharedNumKernelsPtr = (unsigned long long)ptrs[3];
        o.sharedNumAllocPtr   = (unsigned long long)ptrs[4];
        // Redirect eval kernel args (centers=2, sigmas=3, logWeights=4, numKernels=5).
        o.evalKernel->setArg(2, (unsigned long long)ptrs[0]);
        o.evalKernel->setArg(3, (unsigned long long)ptrs[1]);
        o.evalKernel->setArg(4, (unsigned long long)ptrs[2]);
        o.evalKernel->setArg(5, (unsigned long long)ptrs[3]);
        // Redirect deposit kernel args (centers=7, sigmas=8, logWeights=9, numKernels=11, numAlloc=22).
        o.gatherDepositKernel->setArg(7,  (unsigned long long)ptrs[0]);
        o.gatherDepositKernel->setArg(8,  (unsigned long long)ptrs[1]);
        o.gatherDepositKernel->setArg(9,  (unsigned long long)ptrs[2]);
        o.gatherDepositKernel->setArg(11, (unsigned long long)ptrs[3]);
        o.gatherDepositKernel->setArg(22, (unsigned long long)ptrs[4]);
    }
}

vector<double> CommonCalcGluedForceKernel::downloadCVValues() {
 if (plan_.numCVs == 0)
 return {};
 ContextSelector selector(cc_);
 vector<double> values(plan_.numCVs);
 plan_.cvValues.download(values);
 return values;
}

