#include "CommonGluedKernels.h"
#include "openmm/common/ContextSelector.h"
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

using namespace GluedPlugin;
using namespace OpenMM;
using namespace std;

// Serialization format (little-endian):
// int32 numOpesBiases
// per bias: int32 numKernels, double logZ, int32 nSamples,
// double[D] runningMean, double[D] runningM2,
// float[K*D] centers, float[K*D] sigmas, float[K] logWeights
// int32 numAbmdBiases
// per bias: double[D] rhoMinCPU
vector<char> CommonCalcGluedForceKernel::getBiasStateBytes() {
 ContextSelector selector(cc_);
 vector<char> buf;
 auto write = [&](const void* src, size_t n) {
 const char* p = reinterpret_cast<const char*>(src);
 buf.insert(buf.end(), p, p + n);
 };
 // Header: magic 'GPUS' (4 bytes) + version = 1 (int32 little-endian)
 const char magic[4] = {'G','P','U','S'};
 write(magic, 4);
 int32_t version = 1;
 write(&version, 4);

 int32_t nb = (int32_t)opesBiases_.size();
 write(&nb, 4);
 for (auto& o : opesBiases_) {
 int D = o.numCVsBias;
 // Use GPU-resident count (after compression) not CPU deposit counter.
 {
 vector<int> tmp(1);
 o.numKernelsGPU.download(tmp);
 o.numKernels = tmp[0]; // sync CPU mirror
 }
 int nk = o.numKernels;
 // Download GPU-resident Welford state and logZ on demand for serialization.
 {
 vector<double> tmp(1);
 o.logZGPU.download(tmp);
 o.logZCPU = tmp[0];
 }
 {
 vector<int> tmp(1);
 o.nSamplesGPU.download(tmp);
 o.nSamples = tmp[0];
 }
 o.runningMeanGPU.download(o.runningMean);
 o.runningM2GPU.download(o.runningM2);
 // Download kernel data (full buffer; write only nk rows).
 if (nk > 0) {
 o.kernelCenters.download(o.kernelCentersCPU);
 o.kernelSigmas.download(o.kernelSigmasCPU);
 o.kernelLogWeights.download(o.kernelLogWeightsCPU);
 }
 int32_t nk32 = nk;
 write(&nk32, 4);
 write(&o.logZCPU, 8);
 int32_t ns = o.nSamples;
 write(&ns, 4);
 write(o.runningMean.data(), D * 8);
 write(o.runningM2.data(), D * 8);
 write(o.kernelCentersCPU.data(), nk * D * 4);
 write(o.kernelSigmasCPU.data(), nk * D * 4);
 write(o.kernelLogWeightsCPU.data(), nk * 4);
 }
 int32_t na = (int32_t)abmdBiases_.size();
 write(&na, 4);
 for (auto& a : abmdBiases_) {
 a.rhoMin.download(a.rhoMinCPU);
 write(a.rhoMinCPU.data(), a.numCVsBias * 8);
 }
 int32_t nm = (int32_t)metaDGridBiases_.size();
 write(&nm, 4);
 for (auto& m : metaDGridBiases_) {
 // Grid lives on GPU; download into gridCPU buffer on demand for serialization.
 m.grid.download(m.gridCPU);
 int32_t nd = m.numDeposited;
 write(&nd, 4);
 write(m.gridCPU.data(), m.totalGridPoints * 8);
 }
 int32_t npb = (int32_t)pbmetaDGridBiases_.size();
 write(&npb, 4);
 for (auto& pb : pbmetaDGridBiases_) {
 int32_t nsub = (int32_t)pb.subGrids.size();
 write(&nsub, 4);
 for (auto& m : pb.subGrids) {
 m.grid.download(m.gridCPU);
 int32_t nd = m.numDeposited;
 write(&nd, 4);
 write(m.gridCPU.data(), m.totalGridPoints * 8);
 }
 }
 // External bias: grid is fixed at construction — write count only for version tagging.
 int32_t nex = (int32_t)externalGridBiases_.size();
 write(&nex, 4);
 // Linear and wall biases are stateless — write counts only for version tagging.
 int32_t nlin = (int32_t)linearBiases_.size();
 int32_t nwall = (int32_t)wallBiases_.size();
 write(&nlin, 4);
 write(&nwall, 4);
 // OPES_EXPANDED: logZ and numUpdates are the only mutable state (GPU-resident).
 int32_t noe = (int32_t)opesExpandedBiases_.size();
 write(&noe, 4);
 for (auto& oe : opesExpandedBiases_) {
 {
 vector<double> tmp(1);
 oe.logZGPU.download(tmp);
 oe.logZCPU = tmp[0];
 }
 write(&oe.logZCPU, 8);
 {
 vector<int> tmp(1);
 oe.numUpdatesGPU.download(tmp);
 int32_t nu = tmp[0];
 write(&nu, 4);
 }
 }
 // Extended Lagrangian: s[D] and p[D] per bias (GPU-resident, download on demand).
 int32_t nel = (int32_t)extLagBiases_.size();
 write(&nel, 4);
 for (auto& el : extLagBiases_) {
 int D = (int)el.cvIndices.size();
 if (el.initialized) {
 el.sGPUArr.download(el.s);
 el.pGPUArr.download(el.p);
 }
 write(el.s.data(), D * 8);
 write(el.p.data(), D * 8);
 }
 // EDS White-Voth: lambda, mean, ssd, accum per bias; count as int.
 int32_t neds = (int32_t)edsBiases_.size();
 write(&neds, 4);
 for (auto& eds : edsBiases_) {
 int D = (int)eds.cvIndices.size();
 eds.lambdaGPU.download(eds.lambda);
 vector<double> mean(D), ssd(D), accum(D);
 eds.meanGPU.download(mean);
 eds.ssdGPU.download(ssd);
 eds.accumGPU.download(accum);
 vector<int> cnt(D);
 eds.countGPU.download(cnt);
 write(eds.lambda.data(), D * 8);
 write(mean.data(), D * 8);
 write(ssd.data(), D * 8);
 write(accum.data(), D * 8);
 for (int k = 0; k < D; k++) { int32_t c = cnt[k]; write(&c, 4); }
 }
 // MaxEnt: lambda per bias.
 int32_t nmxe = (int32_t)maxentBiases_.size();
 write(&nmxe, 4);
 for (auto& mx : maxentBiases_) {
 int D = (int)mx.cvIndices.size();
 mx.lambdaGPU.download(mx.lambda);
 write(mx.lambda.data(), D * 8);
 }
 return buf;
}

void CommonCalcGluedForceKernel::setBiasStateBytes(
 const vector<char>& bytes) {
 if (bytes.empty()) return;
 const char* p = bytes.data();
 auto read = [&](void* dst, size_t n) {
 std::memcpy(dst, p, n);
 p += n;
 };
 // Detect versioned header: magic 'GPUS' + int32 version.
 // Legacy blobs (no header) start with int32 numOpesBiases — typically a small
 // non-negative integer whose first byte is never 'G', so detection is safe.
 const char expected[4] = {'G','P','U','S'};
 if (bytes.size() >= 8 && std::memcmp(p, expected, 4) == 0) {
 p += 4; // skip magic
 int32_t version; read(&version, 4); // consume version (reserved for future use)
 }
 int32_t nb;
 read(&nb, 4);
 if (nb != (int32_t)opesBiases_.size()) return; // mismatch — ignore
 ContextSelector selector(cc_);
 for (auto& o : opesBiases_) {
 int32_t nk, ns;
 read(&nk, 4);
 read(&o.logZCPU, 8);
 read(&ns, 4);
 o.nSamples = ns;
 int D = o.numCVsBias;
 read(o.runningMean.data(), D * 8);
 read(o.runningM2.data(), D * 8);
 o.kernelCentersCPU.resize(nk * D);
 o.kernelSigmasCPU.resize(nk * D);
 o.kernelLogWeightsCPU.resize(nk);
 read(o.kernelCentersCPU.data(), nk * D * 4);
 read(o.kernelSigmasCPU.data(), nk * D * 4);
 read(o.kernelLogWeightsCPU.data(), nk * 4);
 o.numKernels = nk;
 o.numKernelsGPU.upload(vector<int>{nk});
 if (nk > 0) {
 o.kernelCenters.uploadSubArray(o.kernelCentersCPU.data(), 0, nk * D);
 o.kernelSigmas.uploadSubArray(o.kernelSigmasCPU.data(), 0, nk * D);
 o.kernelLogWeights.uploadSubArray(o.kernelLogWeightsCPU.data(), 0, nk);
 }
 // Restore GPU-resident Welford state and logZ.
 o.logZGPU.upload(vector<double>{o.logZCPU});
 o.runningMeanGPU.upload(o.runningMean);
 o.runningM2GPU.upload(o.runningM2);
 o.nSamplesGPU.upload(vector<int>{o.nSamples});
 // Recompute KDNorm from loaded kernels: each compressed kernel's logW equals the
 // sum of all merged heights, so Σ exp(lw_k) + exp(-gamma) = sum_weights_ exactly.
 {
 double sumW = exp(-o.gamma);
 for (int k = 0; k < nk; k++) sumW += exp((double)o.kernelLogWeightsCPU[k]);
 o.sumWeightsGPU.upload(vector<double>{sumW});
 }
 }
 // ABMD rhoMin state (may be absent in states saved before )
 if (p < bytes.data() + bytes.size()) {
 int32_t na;
 read(&na, 4);
 if (na == (int32_t)abmdBiases_.size()) {
 for (auto& a : abmdBiases_) {
 read(a.rhoMinCPU.data(), a.numCVsBias * 8);
 a.rhoMin.upload(a.rhoMinCPU);
 }
 }
 }
 // MetaD grid state (may be absent in states saved before )
 if (p < bytes.data() + bytes.size()) {
 int32_t nm;
 read(&nm, 4);
 if (nm == (int32_t)metaDGridBiases_.size()) {
 for (auto& m : metaDGridBiases_) {
 int32_t nd;
 read(&nd, 4);
 m.numDeposited = nd;
 read(m.gridCPU.data(), m.totalGridPoints * 8);
 m.grid.upload(m.gridCPU);
 }
 }
 }
 // PBMetaD grid state (may be absent in states saved before )
 if (p < bytes.data() + bytes.size()) {
 int32_t npb;
 read(&npb, 4);
 if (npb == (int32_t)pbmetaDGridBiases_.size()) {
 for (auto& pb : pbmetaDGridBiases_) {
 int32_t nsub;
 read(&nsub, 4);
 if (nsub != (int32_t)pb.subGrids.size()) continue;
 for (auto& m : pb.subGrids) {
 int32_t nd;
 read(&nd, 4);
 m.numDeposited = nd;
 read(m.gridCPU.data(), m.totalGridPoints * 8);
 m.grid.upload(m.gridCPU);
 }
 }
 }
 }
 // External bias count (may be absent in states saved before ).
 // Grid is fixed at construction — nothing to restore.
 if (p < bytes.data() + bytes.size()) {
 int32_t nex; read(&nex, 4); (void)nex;
 }
 // Linear and wall bias counts (stateless — nothing to restore).
 if (p < bytes.data() + bytes.size()) {
 int32_t nlin; read(&nlin, 4); (void)nlin;
 }
 if (p < bytes.data() + bytes.size()) {
 int32_t nwall; read(&nwall, 4); (void)nwall;
 }
 // OPES_EXPANDED logZ and numUpdates
 if (p < bytes.data() + bytes.size()) {
 int32_t noe; read(&noe, 4);
 if (noe == (int32_t)opesExpandedBiases_.size()) {
 for (auto& oe : opesExpandedBiases_) {
 read(&oe.logZCPU, 8);
 int32_t nu; read(&nu, 4);
 oe.logZGPU.upload(vector<double>{oe.logZCPU});
 oe.numUpdatesGPU.upload(vector<int>{(int)nu});
 }
 }
 }
 // Extended Lagrangian: s and p state
 if (p < bytes.data() + bytes.size()) {
 int32_t nel; read(&nel, 4);
 if (nel == (int32_t)extLagBiases_.size()) {
 for (auto& el : extLagBiases_) {
 int D = (int)el.cvIndices.size();
 read(el.s.data(), D * 8);
 read(el.p.data(), D * 8);
 el.initialized = true;
 el.sGPUArr.upload(el.s);
 el.pGPUArr.upload(el.p);
 }
 }
 }
 // EDS White-Voth state: lambda, mean, ssd, accum, count
 if (p < bytes.data() + bytes.size()) {
 int32_t neds; read(&neds, 4);
 if (neds == (int32_t)edsBiases_.size()) {
 for (auto& eds : edsBiases_) {
 int D = (int)eds.cvIndices.size();
 vector<double> mean(D), ssd(D), accum(D);
 vector<int> cnt(D);
 read(eds.lambda.data(), D * 8);
 read(mean.data(), D * 8);
 read(ssd.data(), D * 8);
 read(accum.data(), D * 8);
 for (int k = 0; k < D; k++) { int32_t c; read(&c, 4); cnt[k] = c; }
 eds.lambdaGPU.upload(eds.lambda);
 eds.meanGPU.upload(mean);
 eds.ssdGPU.upload(ssd);
 eds.accumGPU.upload(accum);
 eds.countGPU.upload(cnt);
 }
 }
 }
 // MaxEnt: lambda state
 if (p < bytes.data() + bytes.size()) {
 int32_t nmxe; read(&nmxe, 4);
 if (nmxe == (int32_t)maxentBiases_.size()) {
 ContextSelector selector(cc_);
 for (auto& mx : maxentBiases_) {
 int D = (int)mx.cvIndices.size();
 read(mx.lambda.data(), D * 8);
 mx.lambdaGPU.upload(mx.lambda);
 }
 }
 }
}
