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

// Bias-state checkpoint, version 2 (native little-endian; all sections mandatory).
// The Python parser in python/MultiGPUManager.py mirrors this layout exactly.
//
//   char[4] 'GPUS', int32 version = 2
//   uint64 configurationHash, int64 lastUpdateStep
//   int32 numOpes; per bias (D = numCVs, K = numKernels):
//       int32 K, double sumUprob, int32 numSamples, int32 depositCount
//       double sumWeights, double[2] {sumW, sumW2}
//       double[D] runningMean, double[D] runningM2, double[D] sigma0
//       double[K*D] centers, double[K*D] sigmas, double[K] logWeights
//   int32 numAbmd;         per bias: double[D] rhoMin
//   int32 numMetaD;        per bias: int32 numDeposited, double[G] grid
//   int32 numPBMetaD;      per bias: int32 numSubGrids, then per sub-grid as MetaD
//   int32 numExternal, int32 numLinear, int32 numWall   (stateless: counts only)
//   int32 numOpesExpanded; per bias: double logZ, int32 numUpdates
//   int32 numExtLag;       per bias: int32 initialized, double[D] s, double[D] p
//   int32 numEds;          per bias: double[D] lambda, mean, ssd, accum; int32[D] count
//   int32 numMaxEnt;       per bias: double[D] lambda
//   int32 numMultithermal; per bias: int32 N, double[N] deltaF, double rct, double counter

static const int32_t kCheckpointVersion = 2;

vector<char> CommonCalcGluedForceKernel::getBiasStateBytes() {
    ContextSelector selector(cc_);
    vector<char> buf;
    auto write = [&](const void* src, size_t n) {
        const char* p = reinterpret_cast<const char*>(src);
        buf.insert(buf.end(), p, p + n);
    };
    auto writeInt = [&](int32_t v) { write(&v, 4); };
    auto writeDouble = [&](double v) { write(&v, 8); };

    write("GPUS", 4);
    writeInt(kCheckpointVersion);
    write(&configurationHash_, 8);
    int64_t lastUpdateStep = lastUpdateStep_;
    write(&lastUpdateStep, 8);

    writeInt((int32_t)opesBiases_.size());
    for (auto& o : opesBiases_) {
        int D = o.numCVsBias;
        vector<int> numKernels(1), numSamples(1), depositCount(1);
        vector<double> sumUprob(1), sumWeights(1), sumW(2), sigma0(D);
        o.numKernelsGPU.download(numKernels);
        o.nSamplesGPU.download(numSamples);
        o.stepCountGPU.download(depositCount);
        o.logZGPU.download(sumUprob);
        o.sumWeightsGPU.download(sumWeights);
        o.logSumWGPU.download(sumW);
        o.sigma0GPU.download(sigma0);
        o.runningMeanGPU.download(o.runningMean);
        o.runningM2GPU.download(o.runningM2);
        int nk = numKernels[0];
        o.numKernels = nk;
        if (nk > 0) {
            o.kernelCenters.download(o.kernelCentersCPU);
            o.kernelSigmas.download(o.kernelSigmasCPU);
            o.kernelLogWeights.download(o.kernelLogWeightsCPU);
        }
        writeInt(nk);
        writeDouble(sumUprob[0]);
        writeInt(numSamples[0]);
        writeInt(depositCount[0]);
        writeDouble(sumWeights[0]);
        write(sumW.data(), 16);
        write(o.runningMean.data(), (size_t)D * 8);
        write(o.runningM2.data(), (size_t)D * 8);
        write(sigma0.data(), (size_t)D * 8);
        write(o.kernelCentersCPU.data(), (size_t)nk * D * 8);
        write(o.kernelSigmasCPU.data(), (size_t)nk * D * 8);
        write(o.kernelLogWeightsCPU.data(), (size_t)nk * 8);
    }

    writeInt((int32_t)abmdBiases_.size());
    for (auto& a : abmdBiases_) {
        a.rhoMin.download(a.rhoMinCPU);
        write(a.rhoMinCPU.data(), (size_t)a.numCVsBias * 8);
    }

    writeInt((int32_t)metaDGridBiases_.size());
    for (auto& m : metaDGridBiases_) {
        m.grid.download(m.gridCPU);
        writeInt(m.numDeposited);
        write(m.gridCPU.data(), (size_t)m.totalGridPoints * 8);
    }

    writeInt((int32_t)pbmetaDGridBiases_.size());
    for (auto& pb : pbmetaDGridBiases_) {
        writeInt((int32_t)pb.subGrids.size());
        for (auto& m : pb.subGrids) {
            m.grid.download(m.gridCPU);
            writeInt(m.numDeposited);
            write(m.gridCPU.data(), (size_t)m.totalGridPoints * 8);
        }
    }

    // Stateless biases: counts only, so a mismatched configuration is detected.
    writeInt((int32_t)externalGridBiases_.size());
    writeInt((int32_t)linearBiases_.size());
    writeInt((int32_t)wallBiases_.size());

    writeInt((int32_t)opesExpandedBiases_.size());
    for (auto& oe : opesExpandedBiases_) {
        vector<double> logZ(1);
        vector<int> numUpdates(1);
        oe.logZGPU.download(logZ);
        oe.numUpdatesGPU.download(numUpdates);
        oe.logZCPU = logZ[0];
        writeDouble(logZ[0]);
        writeInt(numUpdates[0]);
    }

    writeInt((int32_t)extLagBiases_.size());
    for (auto& el : extLagBiases_) {
        int D = (int)el.cvIndices.size();
        if (el.initialized) {
            el.sGPUArr.download(el.s);
            el.pGPUArr.download(el.p);
        }
        writeInt(el.initialized ? 1 : 0);
        write(el.s.data(), (size_t)D * 8);
        write(el.p.data(), (size_t)D * 8);
    }

    writeInt((int32_t)edsBiases_.size());
    for (auto& eds : edsBiases_) {
        int D = (int)eds.cvIndices.size();
        vector<double> mean(D), ssd(D), accum(D);
        vector<int> count(D);
        eds.lambdaGPU.download(eds.lambda);
        eds.meanGPU.download(mean);
        eds.ssdGPU.download(ssd);
        eds.accumGPU.download(accum);
        eds.countGPU.download(count);
        write(eds.lambda.data(), (size_t)D * 8);
        write(mean.data(), (size_t)D * 8);
        write(ssd.data(), (size_t)D * 8);
        write(accum.data(), (size_t)D * 8);
        for (int c : count)
            writeInt(c);
    }

    writeInt((int32_t)maxentBiases_.size());
    for (auto& mx : maxentBiases_) {
        mx.lambdaGPU.download(mx.lambda);
        write(mx.lambda.data(), mx.cvIndices.size() * 8);
    }

    writeInt((int32_t)multithermalBiases_.size());
    for (auto& mt : multithermalBiases_) {
        int N = (int)mt.betaMinusBeta0.size();
        vector<double> rct(1), counter(1);
        mt.deltaFGPU.download(mt.deltaF);
        mt.rctGPU.download(rct);
        mt.counterGPU.download(counter);
        mt.rct = rct[0];
        mt.counter = (long long)counter[0];
        writeInt(N);
        write(mt.deltaF.data(), (size_t)N * 8);
        writeDouble(rct[0]);
        writeDouble(counter[0]);
    }
    return buf;
}

void CommonCalcGluedForceKernel::setBiasStateBytes(const vector<char>& bytes) {
    // Sections are uploaded as they are parsed, so a blob that fails validation
    // part-way would leave the Context half restored. Roll back to the state
    // captured before the attempt.
    const vector<char> previous = getBiasStateBytes();
    try {
        loadBiasState(bytes);
    } catch (...) {
        loadBiasState(previous);
        throw;
    }
}

void CommonCalcGluedForceKernel::loadBiasState(const vector<char>& bytes) {
    const char* p = bytes.data();
    const char* const end = bytes.data() + bytes.size();
    auto fail = [](const string& what) {
        throw OpenMMException("GLUED bias-state: " + what);
    };
    // Bounds-checked reader: every read must fit in the remaining input.
    auto read = [&](void* dst, size_t n) {
        if ((size_t)(end - p) < n)
            fail("truncated blob");
        std::memcpy(dst, p, n);
        p += n;
    };
    auto readInt = [&]() { int32_t v; read(&v, 4); return v; };
    auto readDouble = [&]() { double v; read(&v, 8); return v; };
    auto readCount = [&](const char* section, size_t expected) {
        int32_t n = readInt();
        if (n != (int32_t)expected)
            fail(string(section) + " bias count mismatch (expected " + to_string(expected) +
                 ", got " + to_string((long long)n) + ")");
    };
    auto requireFinite = [&](const vector<double>& values, const char* what) {
        for (double v : values)
            if (!std::isfinite(v))
                fail(string("nonfinite ") + what);
    };

    char magic[4];
    read(magic, 4);
    if (std::memcmp(magic, "GPUS", 4) != 0)
        fail("missing versioned header");
    if (readInt() != kCheckpointVersion)
        fail("unsupported version; only version " + to_string(kCheckpointVersion) +
             " checkpoints can be restored (older ones lack the state needed to continue)");
    unsigned long long hash;
    read(&hash, 8);
    if (hash != configurationHash_)
        fail("configuration mismatch: the checkpoint was written by a differently "
             "configured GluedForce");
    int64_t lastUpdateStep;
    read(&lastUpdateStep, 8);

    ContextSelector selector(cc_);

    readCount("OPES", opesBiases_.size());
    for (auto& o : opesBiases_) {
        int D = o.numCVsBias;
        int nk = readInt();
        if (nk < 0 || nk > o.maxKernels)
            fail("OPES kernel count out of range");
        vector<double> sumUprob{readDouble()};
        vector<int> numSamples{readInt()};
        vector<int> depositCount{readInt()};
        vector<double> sumWeights{readDouble()};
        vector<double> sumW(2), sigma0(D);
        read(sumW.data(), 16);
        read(o.runningMean.data(), (size_t)D * 8);
        read(o.runningM2.data(), (size_t)D * 8);
        read(sigma0.data(), (size_t)D * 8);
        o.kernelCentersCPU.resize((size_t)nk * D);
        o.kernelSigmasCPU.resize((size_t)nk * D);
        o.kernelLogWeightsCPU.resize(nk);
        read(o.kernelCentersCPU.data(), (size_t)nk * D * 8);
        read(o.kernelSigmasCPU.data(), (size_t)nk * D * 8);
        read(o.kernelLogWeightsCPU.data(), (size_t)nk * 8);
        if (numSamples[0] < 0 || depositCount[0] < 0 || sumWeights[0] < 0 || sumW[0] < 0 ||
            sumW[1] < 0)
            fail("negative OPES counter");
        requireFinite(sumUprob, "OPES normalization");
        requireFinite(sumWeights, "OPES weight sum");
        requireFinite(sumW, "OPES weight sum");
        requireFinite(o.runningMean, "OPES running mean");
        requireFinite(o.runningM2, "OPES running variance");
        requireFinite(sigma0, "OPES sigma");
        requireFinite(o.kernelCentersCPU, "OPES kernel center");
        requireFinite(o.kernelSigmasCPU, "OPES kernel width");
        requireFinite(o.kernelLogWeightsCPU, "OPES kernel weight");
        for (double s : o.kernelSigmasCPU)
            if (s <= 0)
                fail("nonpositive OPES kernel width");

        o.numKernels = nk;
        o.nSamples = numSamples[0];
        o.logZCPU = sumUprob[0];
        o.numKernelsGPU.upload(vector<int>{nk});
        o.numAllocatedGPU.upload(vector<int>{nk});
        o.nSamplesGPU.upload(numSamples);
        o.stepCountGPU.upload(depositCount);
        o.logZGPU.upload(sumUprob);
        o.sumWeightsGPU.upload(sumWeights);
        o.logSumWGPU.upload(sumW);
        o.runningMeanGPU.upload(o.runningMean);
        o.runningM2GPU.upload(o.runningM2);
        o.sigma0GPU.upload(sigma0);
        if (nk > 0) {
            o.kernelCenters.uploadSubArray(o.kernelCentersCPU.data(), 0, nk * D);
            o.kernelSigmas.uploadSubArray(o.kernelSigmasCPU.data(), 0, nk * D);
            o.kernelLogWeights.uploadSubArray(o.kernelLogWeightsCPU.data(), 0, nk);
        }
    }

    readCount("ABMD", abmdBiases_.size());
    for (auto& a : abmdBiases_) {
        read(a.rhoMinCPU.data(), (size_t)a.numCVsBias * 8);
        a.rhoMin.upload(a.rhoMinCPU);
    }

    readCount("MetaD", metaDGridBiases_.size());
    for (auto& m : metaDGridBiases_) {
        m.numDeposited = readInt();
        read(m.gridCPU.data(), (size_t)m.totalGridPoints * 8);
        m.grid.upload(m.gridCPU);
    }

    readCount("PBMetaD", pbmetaDGridBiases_.size());
    for (auto& pb : pbmetaDGridBiases_) {
        readCount("PBMetaD sub-grid", pb.subGrids.size());
        for (auto& m : pb.subGrids) {
            m.numDeposited = readInt();
            read(m.gridCPU.data(), (size_t)m.totalGridPoints * 8);
            m.grid.upload(m.gridCPU);
        }
    }

    readCount("external", externalGridBiases_.size());
    readCount("linear", linearBiases_.size());
    readCount("wall", wallBiases_.size());

    readCount("OPES_EXPANDED", opesExpandedBiases_.size());
    for (auto& oe : opesExpandedBiases_) {
        oe.logZCPU = readDouble();
        int numUpdates = readInt();
        oe.logZGPU.upload(vector<double>{oe.logZCPU});
        oe.numUpdatesGPU.upload(vector<int>{numUpdates});
    }

    readCount("extended-Lagrangian", extLagBiases_.size());
    for (auto& el : extLagBiases_) {
        int D = (int)el.cvIndices.size();
        el.initialized = readInt() != 0;
        read(el.s.data(), (size_t)D * 8);
        read(el.p.data(), (size_t)D * 8);
        el.sGPUArr.upload(el.s);
        el.pGPUArr.upload(el.p);
    }

    readCount("EDS", edsBiases_.size());
    for (auto& eds : edsBiases_) {
        int D = (int)eds.cvIndices.size();
        vector<double> mean(D), ssd(D), accum(D);
        vector<int> count(D);
        read(eds.lambda.data(), (size_t)D * 8);
        read(mean.data(), (size_t)D * 8);
        read(ssd.data(), (size_t)D * 8);
        read(accum.data(), (size_t)D * 8);
        for (int& c : count)
            c = readInt();
        eds.lambdaGPU.upload(eds.lambda);
        eds.meanGPU.upload(mean);
        eds.ssdGPU.upload(ssd);
        eds.accumGPU.upload(accum);
        eds.countGPU.upload(count);
    }

    readCount("MaxEnt", maxentBiases_.size());
    for (auto& mx : maxentBiases_) {
        read(mx.lambda.data(), mx.cvIndices.size() * 8);
        mx.lambdaGPU.upload(mx.lambda);
    }

    readCount("OPES multithermal", multithermalBiases_.size());
    for (auto& mt : multithermalBiases_) {
        int N = (int)mt.betaMinusBeta0.size();
        readCount("OPES multithermal state", N);
        mt.deltaF.resize(N);
        read(mt.deltaF.data(), (size_t)N * 8);
        mt.rct = readDouble();
        double counter = readDouble();
        mt.counter = (long long)counter;
        mt.deltaFGPU.upload(mt.deltaF);
        mt.rctGPU.upload(vector<double>{mt.rct});
        mt.counterGPU.upload(vector<double>{counter});
    }

    if (p != end)
        fail("unexpected trailing bytes");
    lastUpdateStep_ = lastUpdateStep;
}
