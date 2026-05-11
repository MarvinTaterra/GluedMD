#include "internal/GluedForceImpl.h"
#include "GluedKernels.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/OpenMMException.h"
#include <map>
#include <string>
#include <vector>

using namespace GluedPlugin;
using namespace OpenMM;
using namespace std;

GluedForceImpl::GluedForceImpl(const GluedForce& owner)
    : owner_(owner) {}

GluedForceImpl::~GluedForceImpl() {}

void GluedForceImpl::initialize(ContextImpl& context) {
    kernel_ = context.getPlatform().createKernel(
        CalcGluedForceKernel::Name(), context);
    kernel_.getAs<CalcGluedForceKernel>().initialize(
        context.getSystem(), owner_);
    // Register this impl so getBiasStateBytes() / setBiasStateBytes() can
    // delegate without requiring a Context at call time.
    owner_.impl_ = this;
}

void GluedForceImpl::updateContextState(ContextImpl& context,
                                             bool& forcesInvalid) {
    // Guard: bias deposition must fire exactly once per step, even when
    // execute() is called multiple times (minimization, constraint iteration).
    int step = context.getStepCount();
    if (step != lastStepIndex_) {
        kernel_.getAs<CalcGluedForceKernel>().updateState(context, step);
        lastStepIndex_ = step;
    }
}

double GluedForceImpl::calcForcesAndEnergy(ContextImpl& context,
                                                bool includeForces,
                                                bool includeEnergy,
                                                int groups) {
    if ((groups & (1 << owner_.getForceGroup())) == 0)
        return 0.0;
    return kernel_.getAs<CalcGluedForceKernel>().execute(
        context, includeForces, includeEnergy);
}

map<string, double> GluedForceImpl::getDefaultParameters() {
    return map<string, double>();
}

vector<string> GluedForceImpl::getKernelNames() {
    return {CalcGluedForceKernel::Name()};
}

vector<char> GluedForceImpl::getBiasStateBytes() const {
    return const_cast<GluedForceImpl*>(this)->kernel_
        .getAs<CalcGluedForceKernel>().getBiasStateBytes();
}

void GluedForceImpl::setBiasStateBytes(const vector<char>& bytes) {
    kernel_.getAs<CalcGluedForceKernel>().setBiasStateBytes(bytes);
}

void GluedForceImpl::getCurrentCVs(vector<double>& values) {
    values = downloadCVValues();
}

vector<double> GluedForceImpl::downloadCVValues() {
    return kernel_.getAs<CalcGluedForceKernel>().downloadCVValues();
}

vector<double> GluedForceImpl::getOPESMetrics(int biasIndex) {
    return kernel_.getAs<CalcGluedForceKernel>().getOPESMetrics(biasIndex);
}

vector<float> GluedForceImpl::getKernelSigmas(int biasIndex) {
    return kernel_.getAs<CalcGluedForceKernel>().getKernelSigmas(biasIndex);
}

vector<long long> GluedForceImpl::getMultiWalkerPtrs(int biasType, int localIdx) {
    return kernel_.getAs<CalcGluedForceKernel>().getMultiWalkerPtrs(biasType, localIdx);
}

void GluedForceImpl::redirectToPrimaryBias(int biasType, int localIdx,
                                                const vector<long long>& ptrs) {
    kernel_.getAs<CalcGluedForceKernel>().redirectToPrimaryBias(biasType, localIdx, ptrs);
}
