#include "GluedForce.h"
#include "internal/GluedForceImpl.h"
#include "openmm/Context.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/OpenMMException.h"
#include <sstream>
#include <cmath>
#include <limits>

using namespace GluedPlugin;
using namespace OpenMM;
using namespace std;

GluedForce::GluedForce() {}
GluedForce::~GluedForce() {}

// L24: copy all CV/bias/config state but never alias the per-Context impl
// back-pointer. A copied Force is "not yet bound to a Context".
GluedForce::GluedForce(const GluedForce& other)
    : Force(other),
      cvs_(other.cvs_),
      numCVValues_(other.numCVValues_),
      biases_(other.biases_),
      temperature_(other.temperature_),
      usesPBC_(other.usesPBC_),
      testForceMode_(other.testForceMode_),
      testForceScale_(other.testForceScale_),
      testBiasGradients_(other.testBiasGradients_) {}

GluedForce& GluedForce::operator=(const GluedForce& other) {
    if (this == &other)
        return *this;
    Force::operator=(other);
    cvs_              = other.cvs_;
    numCVValues_     = other.numCVValues_;
    biases_          = other.biases_;
    temperature_     = other.temperature_;
    usesPBC_         = other.usesPBC_;
    testForceMode_   = other.testForceMode_;
    testForceScale_  = other.testForceScale_;
    testBiasGradients_ = other.testBiasGradients_;
    impls_.clear();  // a copy is not bound to any Context
    return *this;
}

int GluedForce::addCollectiveVariable(int type, const vector<int>& atoms,
                                           const vector<double>& parameters) {
    // Cheap structural validation. Full atom/param validation (which requires the
    // System) is performed in the common kernel layer at initialize() time.
    if (type < CV_DISTANCE || type > CV_ENERGY || type == CV_EXPRESSION || type == CV_PYTORCH) {
        std::stringstream ss;
        ss << "GluedForce::addCollectiveVariable: invalid CV type " << type
           << " (use the dedicated expression/PyTorch methods for those types)";
        throw OpenMMException(ss.str());
    }
    int firstIdx = numCVValues_;
    numCVValues_ += (type == CV_PATH) ? 2 : 1;
    CV cv;
    cv.type = type;
    cv.atoms = atoms;
    cv.params = parameters;
    cvs_.push_back(cv);
    return firstIdx;
}

int GluedForce::getNumCollectiveVariables() const {
    return numCVValues_;
}

int GluedForce::getNumCollectiveVariableSpecs() const {
    return static_cast<int>(cvs_.size());
}

void GluedForce::getCollectiveVariableInfo(int idx, int& type,
                                                vector<int>& atoms,
                                                vector<double>& parameters) const {
    if (idx < 0 || idx >= static_cast<int>(cvs_.size()))
        throw OpenMMException("GluedForce: CV index out of range");
    type = cvs_[idx].type;
    atoms = cvs_[idx].atoms;
    parameters = cvs_[idx].params;
}

// Validates the parameter layout of a bias. Everything checked here is
// independent of the System; atom and mass checks happen at Context creation
// in the kernel layer. Layouts are documented in docs/bias_methods.md.
static void validateBiasLayout(int type, size_t D,
                               const vector<double>& p, const vector<int>& ip) {
    auto require = [](bool ok, const char* message) {
        if (!ok)
            throw OpenMMException(string("GluedForce::addBias: ") + message);
    };
    auto requireSizes = [&](size_t numParams, size_t numIntParams) {
        require(p.size() >= numParams && ip.size() >= numIntParams,
                "incomplete parameter arrays");
    };
    auto requirePace = [&](size_t index) {
        require(ip[index] > 0, "pace must be positive");
    };
    // Checks one grid axis per CV and returns the total number of grid points.
    auto validateGrid = [&](size_t bins, size_t flags, size_t origin, size_t upper) {
        const size_t maxPoints = size_t(std::numeric_limits<int>::max());
        size_t points = 1;
        for (size_t d = 0; d < D; ++d) {
            require(ip[bins + d] > 0, "invalid grid bin count");
            require(ip[flags + d] == 0 || ip[flags + d] == 1, "periodic flags must be 0 or 1");
            require(p[upper + d] > p[origin + d], "grid upper bound must exceed lower bound");
            size_t n = size_t(ip[bins + d]) + (ip[flags + d] ? 0 : 1);
            require(points <= maxPoints / n, "grid is too large");
            points *= n;
        }
        return points;
    };

    require(D > 0, "bias must reference at least one CV");
    for (double value : p)
        require(std::isfinite(value), "parameters must be finite");

    switch (type) {
    case GluedForce::BIAS_HARMONIC:
    case GluedForce::BIAS_ABMD:
        requireSizes(2 * D, 0);
        for (size_t d = 0; d < D; ++d)
            require(p[2 * d] >= 0, "spring constants must be nonnegative");
        break;
    case GluedForce::BIAS_MOVING_RESTRAINT: {
        const size_t rowSize = 1 + 2 * D;
        int n = ip.empty() ? 1 : ip[0];
        require(n > 0 && size_t(n) <= size_t(std::numeric_limits<int>::max()) / rowSize,
                "invalid schedule size");
        require(p.size() == size_t(n) * rowSize, "schedule size mismatch");
        for (size_t row = 0; row < p.size(); row += rowSize) {
            require(p[row] >= 0 && (row == 0 || p[row] > p[row - rowSize]),
                    "schedule times must increase");
            for (size_t d = 0; d < D; ++d)
                require(p[row + 1 + 2 * d] >= 0, "spring constants must be nonnegative");
        }
        break;
    }
    case GluedForce::BIAS_METAD:
        require(D <= 3, "MetaD supports at most 3 CVs");
        requireSizes(3 + 3 * D, 1 + 2 * D);
        requirePace(0);
        require(p[0] >= 0 && (p[1 + D] == 0 || p[1 + D] >= 1) && p[2 + D] > 0,
                "invalid height, gamma or kT");
        for (size_t d = 0; d < D; ++d)
            require(p[1 + d] > 0, "sigma must be positive");
        validateGrid(1, 1 + D, 3 + D, 3 + 2 * D);
        break;
    case GluedForce::BIAS_EXTERNAL: {
        require(D <= 3, "external grids support at most 3 CVs");
        requireSizes(2 * D, 2 * D);
        size_t points = validateGrid(0, D, 0, D);
        require(p.size() == 2 * D + points, "external grid size mismatch");
        break;
    }
    case GluedForce::BIAS_PBMETAD:
        requireSizes(3 + 3 * D, 1 + 2 * D);
        requirePace(0);
        require(p[0] >= 0 && (p[1] == 0 || p[1] >= 1) && p[2] > 0,
                "invalid PBMetaD height, gamma or kT");
        for (size_t d = 0; d < D; ++d) {
            require(p[3 + 3 * d] > 0 && p[5 + 3 * d] > p[4 + 3 * d],
                    "invalid PBMetaD grid or sigma");
            require(ip[1 + 2 * d] > 0, "invalid grid bin count");
            require(ip[2 + 2 * d] == 0 || ip[2 + 2 * d] == 1, "periodic flags must be 0 or 1");
        }
        break;
    case GluedForce::BIAS_OPES: {
        require(D <= 16, "OPES supports at most 16 CVs");
        requireSizes(3 + D, 0);
        require(p[0] > 0 && p[1] > 1 && p[2 + D] >= 0,
                "invalid OPES kT, gamma or sigma minimum");
        // Sigma is either given for every CV or zero for every CV (fully adaptive).
        bool adaptive = p[2] <= 0;
        for (size_t d = 0; d < D; ++d)
            require((p[2 + d] <= 0) == adaptive, "mixed fixed/adaptive sigmas unsupported");
        require(ip.empty() || (ip[0] >= 0 && ip[0] <= 2), "invalid OPES variant");
        if (ip.size() > 1)
            requirePace(1);
        if (ip.size() > 2)
            require(ip[2] > 0 && size_t(ip[2]) <= size_t(std::numeric_limits<int>::max()) / D,
                    "invalid OPES capacity");
        if (ip.size() > 3)
            requirePace(3);
        break;
    }
    case GluedForce::BIAS_LINEAR:
        requireSizes(D, 0);
        break;
    case GluedForce::BIAS_UPPER_WALL:
    case GluedForce::BIAS_LOWER_WALL:
        requireSizes(4 * D, 0);
        for (size_t d = 0; d < D; ++d)
            require(p[4 * d + 1] >= 0 && p[4 * d + 3] >= 1, "invalid wall parameters");
        break;
    case GluedForce::BIAS_OPES_EXPANDED:
        requireSizes(1 + D, 0);
        require(p[0] > 0, "kT must be positive");
        for (size_t d = 0; d < D; ++d)
            require(p[1 + d] > 0, "expanded weights must be positive");
        if (!ip.empty())
            requirePace(0);
        break;
    case GluedForce::BIAS_OPES_MULTITHERMAL:
        require(D == 1, "multithermal requires one energy CV");
        requireSizes(2, 0);
        for (double x : p)
            require(x > 0, "temperatures and inverse temperatures must be positive");
        if (!ip.empty())
            requirePace(0);
        break;
    case GluedForce::BIAS_EXT_LAGRANGIAN:
        requireSizes(2 * D, 0);
        for (size_t d = 0; d < D; ++d)
            require(p[2 * d] >= 0 && p[2 * d + 1] > 0, "invalid auxiliary spring or mass");
        break;
    case GluedForce::BIAS_EDS:
        requireSizes(2 * D, 0);
        for (size_t d = 0; d < D; ++d)
            require(p[2 * d + 1] > 0, "EDS range must be positive");
        if (p.size() > 2 * D)
            require(p[2 * D] > 0, "kT must be positive");
        if (!ip.empty())
            requirePace(0);
        break;
    case GluedForce::BIAS_MAXENT:
        requireSizes(3 + 3 * D, 0);
        require(p[0] > 0 && p[1] >= 0 && p[2] > 0, "invalid MaxEnt kT, sigma or alpha");
        for (size_t d = 0; d < D; ++d)
            require(p[4 + 3 * d] >= 0 && p[5 + 3 * d] > 0, "invalid MaxEnt kappa or tau");
        if (!ip.empty())
            requirePace(0);
        if (ip.size() > 1)
            require(ip[1] >= 0 && ip[1] <= 2, "invalid MaxEnt type");
        if (ip.size() > 2)
            require(ip[2] >= 0 && ip[2] <= 2, "invalid MaxEnt error type");
        break;
    default:
        require(false, "unknown bias type");
    }
}

int GluedForce::addBias(int type, const vector<int>& cvIndices,
                             const vector<double>& parameters,
                             const vector<int>& integerParameters) {
    validateBiasLayout(type, cvIndices.size(), parameters, integerParameters);
    // Validate that every referenced CV value index is in range. cvIndices refer
    // to CV *value* slots (0 .. getNumCollectiveVariables()-1), so they must be
    // bounded by numCVValues_ — not by the number of addCollectiveVariable calls.
    const int numCVs = getNumCollectiveVariables();
    const int biasIdx = static_cast<int>(biases_.size());
    for (size_t i = 0; i < cvIndices.size(); i++) {
        int cvIndex = cvIndices[i];
        if (cvIndex < 0 || cvIndex >= numCVs) {
            std::stringstream ss;
            ss << "GluedForce::addBias: bias " << biasIdx << " (type " << type
               << ") references CV value index " << cvIndex
               << ", which is out of range [0, " << numCVs << ")";
            throw OpenMMException(ss.str());
        }
    }
    Bias bias;
    bias.type = type;
    bias.cvIndices = cvIndices;
    bias.params = parameters;
    bias.intParams = integerParameters;
    biases_.push_back(bias);
    return static_cast<int>(biases_.size()) - 1;
}

int GluedForce::getNumBiases() const {
    return static_cast<int>(biases_.size());
}

void GluedForce::getBiasInfo(int idx, int& type,
                                  vector<int>& cvIndices,
                                  vector<double>& parameters,
                                  vector<int>& integerParameters) const {
    if (idx < 0 || idx >= static_cast<int>(biases_.size()))
        throw OpenMMException("GluedForce: bias index out of range");
    type             = biases_[idx].type;
    cvIndices        = biases_[idx].cvIndices;
    parameters       = biases_[idx].params;
    integerParameters = biases_[idx].intParams;
}

void GluedForce::setTemperature(double kelvin) {
    if (!std::isfinite(kelvin) || kelvin <= 0)
        throw OpenMMException("GluedForce::setTemperature: temperature must be finite and positive");
    temperature_ = kelvin;
}

double GluedForce::getTemperature() const {
    return temperature_;
}

// The Context-free checkpoint calls are only well defined for a Force bound to
// exactly one Context.
static GluedForceImpl& singleImpl(const std::set<GluedForceImpl*>& impls, const char* method) {
    if (impls.size() != 1)
        throw OpenMMException(string("GluedForce::") + method +
            ": the Force must be bound to exactly one Context; pass the Context "
            "explicitly when a System is shared between Contexts");
    return **impls.begin();
}

vector<char> GluedForce::getBiasStateBytes() const {
    return singleImpl(impls_, "getBiasStateBytes").getBiasStateBytes();
}

void GluedForce::setBiasStateBytes(const vector<char>& bytes) {
    singleImpl(impls_, "setBiasStateBytes").setBiasStateBytes(bytes);
}

vector<char> GluedForce::getBiasStateBytes(Context& context) const {
    return dynamic_cast<GluedForceImpl&>(getImplInContext(context)).getBiasStateBytes();
}

void GluedForce::setBiasStateBytes(Context& context, const vector<char>& bytes) {
    dynamic_cast<GluedForceImpl&>(getImplInContext(context)).setBiasStateBytes(bytes);
}

void GluedForce::getCurrentCollectiveVariables(Context& context,
                                                    vector<double>& values) const {
    // Force a fresh evaluation so cvValues reflect the current positions.
    // Uses only this force's group so other forces are not recomputed.
    getContextImpl(context).calcForcesAndEnergy(true, false, 1 << getForceGroup());
    dynamic_cast<GluedForceImpl&>(getImplInContext(context)).getCurrentCVs(values);
}

void GluedForce::setTestForce(int mode, double scale) {
    testForceMode_ = mode;
    testForceScale_ = scale;
}

int GluedForce::getTestForceMode() const {
    return testForceMode_;
}

double GluedForce::getTestForceScale() const {
    return testForceScale_;
}

void GluedForce::setTestBiasGradients(const vector<double>& gradients) {
    testBiasGradients_ = gradients;
}

int GluedForce::addExpressionCV(const string& expression, const vector<int>& inputCVIndices) {
    if (expression.empty())
        throw OpenMMException("GluedForce::addExpressionCV: expression cannot be empty");
    for (int index : inputCVIndices)
        if (index < 0 || index >= numCVValues_)
            throw OpenMMException("GluedForce::addExpressionCV: inputs must refer to "
                                  "previously added CV value indices");
    int firstIdx = numCVValues_;
    numCVValues_ += 1;
    CV cv;
    cv.type = CV_EXPRESSION;
    cv.atoms = inputCVIndices;
    cv.exprString = expression;
    cvs_.push_back(cv);
    return firstIdx;
}

void GluedForce::getExpressionCVInfo(int specIdx, string& expression, vector<int>& inputCVIndices) const {
    if (specIdx < 0 || specIdx >= (int)cvs_.size())
        throw OpenMMException("GluedForce: spec index out of range");
    if (cvs_[specIdx].type != CV_EXPRESSION)
        throw OpenMMException("GluedForce: CV spec is not an expression CV");
    expression = cvs_[specIdx].exprString;
    inputCVIndices = cvs_[specIdx].atoms;
}

int GluedForce::addPyTorchCV(const string& torchScriptPath,
                                   const vector<int>& atomIndices,
                                   const vector<double>& parameters) {
    int firstIdx = numCVValues_;
    numCVValues_ += 1;
    CV cv;
    cv.type = CV_PYTORCH;
    cv.atoms = atomIndices;
    cv.params = parameters;
    cv.exprString = torchScriptPath;
    cvs_.push_back(cv);
    return firstIdx;
}

string GluedForce::getPyTorchCVModelPath(int specIdx) const {
    if (specIdx < 0 || specIdx >= (int)cvs_.size())
        throw OpenMMException("GluedForce: spec index out of range");
    if (cvs_[specIdx].type != CV_PYTORCH)
        throw OpenMMException("GluedForce: CV spec is not a PyTorch CV");
    return cvs_[specIdx].exprString;
}

vector<double> GluedForce::getTestBiasGradients() const {
    return testBiasGradients_;
}

vector<double> GluedForce::getLastCVValues(OpenMM::Context& context) const {
    return dynamic_cast<GluedForceImpl&>(
        getImplInContext(context)).downloadCVValues();
}

double GluedForce::getLastBias(OpenMM::Context& context) const {
    return dynamic_cast<GluedForceImpl&>(
        getImplInContext(context)).downloadLastBias();
}

// OPES diagnostics take a global bias index (as returned by addBias) but the
// kernel stores OPES biases in their own list, so map to the position within it.
static int opesLocalIndex(const GluedForce& force, int biasIndex) {
    int type;
    vector<int> cvIndices, intParams;
    vector<double> params;
    force.getBiasInfo(biasIndex, type, cvIndices, params, intParams);
    if (type != GluedForce::BIAS_OPES)
        throw OpenMMException("GluedForce: biasIndex must identify an OPES bias");
    int local = 0;
    for (int b = 0; b < biasIndex; ++b) {
        force.getBiasInfo(b, type, cvIndices, params, intParams);
        if (type == GluedForce::BIAS_OPES)
            ++local;
    }
    return local;
}

vector<double> GluedForce::getOPESMetrics(OpenMM::Context& context, int biasIndex) const {
    return dynamic_cast<GluedForceImpl&>(
        getImplInContext(context)).getOPESMetrics(opesLocalIndex(*this, biasIndex));
}

vector<float> GluedForce::getKernelSigmas(OpenMM::Context& context, int biasIndex) const {
    return dynamic_cast<GluedForceImpl&>(
        getImplInContext(context)).getKernelSigmas(opesLocalIndex(*this, biasIndex));
}

ForceImpl* GluedForce::createImpl() const {
    return new GluedForceImpl(*this);
}
