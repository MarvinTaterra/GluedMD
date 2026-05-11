#include "ReferenceGluedKernels.h"
#include "GluedForce.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/ContextImpl.h"

using namespace GluedPlugin;
using namespace OpenMM;
using namespace std;

void ReferenceCalcGluedForceKernel::initialize(const System& system,
                                                    const GluedForce& force) {
    numParticles_ = system.getNumParticles();
}

double ReferenceCalcGluedForceKernel::execute(ContextImpl& context,
                                                   bool includeForces,
                                                   bool includeEnergy) {
    // Stage 1.3 stub: no CVs, no biases — return zero energy.
    // Stages 3+ will populate CV evaluation and bias kernels here.
    return 0.0;
}

void ReferenceCalcGluedForceKernel::updateState(ContextImpl& context,
                                                     int step) {
    // Stage 5+ will trigger bias deposition here.
}

void ReferenceCalcGluedForceKernel::getCurrentCVs(ContextImpl& context,
                                                       vector<double>& values) {
    values.assign(0, 0.0); // Stage 6
}

vector<double> ReferenceCalcGluedForceKernel::downloadCVValues() {
    return {}; // Reference platform: CV evaluation not implemented
}

vector<char> ReferenceCalcGluedForceKernel::getBiasStateBytes() {
    return {}; // Stage 6
}

void ReferenceCalcGluedForceKernel::setBiasStateBytes(
    const vector<char>& bytes) {
    // Stage 6
}

vector<double> ReferenceCalcGluedForceKernel::getOPESMetrics(int biasIndex) {
    return {1.0, 0.0, 0.0, 1.0}; // zed=1, rct=0, nker=0, neff=1 (no OPES on Reference)
}

vector<float> ReferenceCalcGluedForceKernel::getKernelSigmas(int biasIndex) {
    return {};   // Reference doesn't implement OPES — no kernels to report.
}
