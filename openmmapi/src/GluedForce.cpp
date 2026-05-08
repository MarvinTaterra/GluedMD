#include "GluedForce.h"
#include "internal/GluedForceImpl.h"
#include "openmm/Context.h"
#include "openmm/OpenMMException.h"
#include <sstream>

using namespace GluedPlugin;
using namespace OpenMM;
using namespace std;

GluedForce::GluedForce() {}
GluedForce::~GluedForce() {}

int GluedForce::addCollectiveVariable(int type, const vector<int>& atoms,
                                           const vector<double>& parameters) {
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

int GluedForce::addBias(int type, const vector<int>& cvIndices,
                             const vector<double>& parameters,
                             const vector<int>& integerParameters) {
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
    temperature_ = kelvin;
}

double GluedForce::getTemperature() const {
    return temperature_;
}

vector<char> GluedForce::getBiasStateBytes() const {
    if (impl_ == nullptr)
        throw OpenMMException("GluedForce::getBiasStateBytes: no Context created yet");
    return impl_->getBiasStateBytes();
}

void GluedForce::setBiasStateBytes(const vector<char>& bytes) {
    if (impl_ == nullptr)
        throw OpenMMException("GluedForce::setBiasStateBytes: no Context created yet");
    impl_->setBiasStateBytes(bytes);
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

vector<double> GluedForce::getOPESMetrics(OpenMM::Context& context, int biasIndex) const {
    return dynamic_cast<GluedForceImpl&>(
        getImplInContext(context)).getOPESMetrics(biasIndex);
}

ForceImpl* GluedForce::createImpl() const {
    return new GluedForceImpl(*this);
}

// Helper: given a global bias index, return the bias type and the local index within
// that bias type (i.e., how many biases of the same type precede it).
static void getBiasTypeAndLocalIdx(const GluedForce& force, int biasIdx,
                                    int& bType, int& localIdx) {
    if (biasIdx < 0 || biasIdx >= force.getNumBiases())
        throw OpenMMException("GluedForce: bias index out of range");
    vector<int> ci, ip;
    vector<double> p;
    force.getBiasInfo(biasIdx, bType, ci, p, ip);
    localIdx = 0;
    for (int i = 0; i < biasIdx; i++) {
        int t;
        force.getBiasInfo(i, t, ci, p, ip);
        if (t == bType) localIdx++;
    }
}

vector<long long> GluedForce::getMultiWalkerPtrs(OpenMM::Context& context, int biasIdx) const {
    int bType, localIdx;
    getBiasTypeAndLocalIdx(*this, biasIdx, bType, localIdx);
    return dynamic_cast<GluedForceImpl&>(
        getImplInContext(context)).getMultiWalkerPtrs(bType, localIdx);
}

void GluedForce::setMultiWalkerPtrs(OpenMM::Context& context, int biasIdx,
                                          const vector<long long>& ptrs) {
    int bType, localIdx;
    getBiasTypeAndLocalIdx(*this, biasIdx, bType, localIdx);
    dynamic_cast<GluedForceImpl&>(
        getImplInContext(context)).redirectToPrimaryBias(bType, localIdx, ptrs);
}
