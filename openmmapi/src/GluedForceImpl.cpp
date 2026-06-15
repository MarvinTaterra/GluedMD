#include "internal/GluedForceImpl.h"
#include "GluedKernels.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/OpenMMException.h"
#include "openmm/Context.h"
#include "openmm/System.h"
#include "openmm/Force.h"
#include "openmm/VerletIntegrator.h"
#include "openmm/Vec3.h"
#include "openmm/serialization/XmlSerializer.h"
#include <map>
#include <string>
#include <vector>

using namespace GluedPlugin;
using namespace OpenMM;
using namespace std;

GluedForceImpl::GluedForceImpl(const GluedForce& owner)
    : owner_(owner) {}

GluedForceImpl::~GluedForceImpl() {
    // H27: owner_.impl_ was set to point at this impl in initialize(). If we are
    // that impl, clear the back-pointer so getBiasStateBytes()/setBiasStateBytes()
    // on the owning Force after Context destruction throw cleanly instead of
    // dereferencing a dangling pointer (use-after-free).
    if (owner_.impl_ == this)
        owner_.impl_ = nullptr;
    // Tear down the CV_ENERGY inner context (context first; it references the
    // integrator + system we own).
    delete innerContext_;
    delete innerIntegrator_;
    delete innerSystem_;
}

void GluedForceImpl::initialize(ContextImpl& context) {
    kernel_ = context.getPlatform().createKernel(
        CalcGluedForceKernel::Name(), context);
    kernel_.getAs<CalcGluedForceKernel>().initialize(
        context.getSystem(), owner_);
    // Register this impl so getBiasStateBytes() / setBiasStateBytes() can
    // delegate without requiring a Context at call time.
    owner_.impl_ = this;

    // CV_ENERGY (OPES multithermal): if any CV is a total-energy CV, build a linked
    // inner Context holding all of the System's forces EXCEPT this GluedForce. The
    // inner context then yields the UNBIASED potential energy U (and forces F) — the
    // GluedForce's own bias is absent by construction, so U excludes it (ATMForce
    // pattern). Skipped entirely when no energy CV is present (zero overhead).
    bool hasEnergyCV = false;
    for (int i = 0; i < owner_.getNumCollectiveVariableSpecs(); i++) {
        int type; std::vector<int> atoms; std::vector<double> params;
        owner_.getCollectiveVariableInfo(i, type, atoms, params);
        if (type == GluedForce::CV_ENERGY) { hasEnergyCV = true; break; }
    }
    if (hasEnergyCV) {
        const System& sys = context.getSystem();
        innerSystem_ = new System();
        for (int i = 0; i < sys.getNumParticles(); i++)
            innerSystem_->addParticle(sys.getParticleMass(i));
        Vec3 a, b, c;
        sys.getDefaultPeriodicBoxVectors(a, b, c);
        innerSystem_->setDefaultPeriodicBoxVectors(a, b, c);
        // Clone every force except this GluedForce. (Constraints/virtual sites are not
        // copied: constraints don't contribute to PE/forces from getState; virtual-site
        // force redistribution is a documented Stage-1 limitation — validate on
        // vsite-free systems. Mirrors ATMForceImpl::copySystem.)
        for (int i = 0; i < sys.getNumForces(); i++) {
            const Force& f = sys.getForce(i);
            if (&f == &owner_)
                continue;
            innerSystem_->addForce(XmlSerializer::clone<Force>(f));
        }
        innerIntegrator_ = new VerletIntegrator(0.001);  // dummy; never stepped
        innerContext_ = context.createLinkedContext(*innerSystem_, *innerIntegrator_);
        // Seed positions once so the inner context's hasSetPositions flag is set
        // (calcForcesAndEnergy throws otherwise). The device copyState kernel then
        // overwrites posq in place every step, bypassing setPositions. (CustomCVForce
        // pattern.) The kernel needs the inner ContextImpl* too — Context::getImpl()
        // is private to the kernel but accessible here via ForceImpl::getContextImpl.
        std::vector<Vec3> zeros(sys.getNumParticles(), Vec3());
        innerContext_->setPositions(zeros);
        kernel_.getAs<CalcGluedForceKernel>().setInnerContext(
            innerContext_, &getContextImpl(*innerContext_));
    }
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

double GluedForceImpl::downloadLastBias() {
    return kernel_.getAs<CalcGluedForceKernel>().downloadLastBias();
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
