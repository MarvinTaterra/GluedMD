%module gluedplugin

%import(module="openmm") "swig/OpenMMSwigHeaders.i"
%include "swig/typemaps.i"
%include "std_string.i"
%include "std_vector.i"

namespace std {
  %template(vectori) vector<int>;
  %template(vectord) vector<double>;
  %template(vectorc) vector<char>;
  %template(vectorll) vector<long long>;
}

%{
#include "GluedForce.h"
#include "OpenMM.h"
#include "OpenMMAmoeba.h"
#include "OpenMMDrude.h"
#include "openmm/RPMDIntegrator.h"
#include "openmm/RPMDMonteCarloBarostat.h"
%}

%pythoncode %{
import openmm as mm
__version__ = "@GLUED_VERSION@"
# Use OpenMM's vector types so SWIG type-checking works regardless of import order.
vectori = mm.vectori
vectord = mm.vectord
%}

namespace GluedPlugin {

class GluedForce : public OpenMM::Force {
public:
    enum CVType {
        CV_DISTANCE            = 1,
        CV_ANGLE               = 2,
        CV_DIHEDRAL            = 3,
        CV_COM_DISTANCE        = 4,
        CV_GYRATION            = 5,
        CV_COORDINATION        = 6,
        CV_RMSD                = 7,
        CV_EXPRESSION          = 8,
        CV_PYTORCH             = 9,
        CV_PATH                = 10,
        CV_POSITION            = 11,
        CV_DRMSD               = 12,
        CV_CONTACTMAP          = 13,
        CV_PLANE               = 14,
        CV_PROJECTION          = 15,
        CV_PUCKERING           = 16,
        CV_DIPOLE              = 17,
        CV_VOLUME              = 18,
        CV_CELL                = 19,
        CV_SECONDARY_STRUCTURE = 20,
        CV_PCA                 = 21,
        CV_ERMSD               = 22
    };

    enum BiasType {
        BIAS_HARMONIC         = 1,
        BIAS_MOVING_RESTRAINT = 2,
        BIAS_METAD            = 3,
        BIAS_PBMETAD          = 4,
        BIAS_OPES             = 5,
        BIAS_EXTERNAL         = 6,
        BIAS_ABMD             = 7,
        BIAS_UPPER_WALL       = 8,
        BIAS_LOWER_WALL       = 9,
        BIAS_LINEAR           = 10,
        BIAS_OPES_EXPANDED    = 11,
        BIAS_EXT_LAGRANGIAN   = 12,
        BIAS_MAXENT           = 13,
        BIAS_EDS              = 14
    };

    GluedForce();

    int addCollectiveVariable(int type, const std::vector<int>& atoms,
                              const std::vector<double>& parameters);
    int getNumCollectiveVariables() const;
    int getNumCollectiveVariableSpecs() const;
    void getCollectiveVariableInfo(int idx, int& type,
                                   std::vector<int>& atoms,
                                   std::vector<double>& parameters) const;

    int addBias(int type, const std::vector<int>& cvIndices,
                const std::vector<double>& parameters,
                const std::vector<int>& integerParameters);
    int getNumBiases() const;
    void getBiasInfo(int idx, int& type, std::vector<int>& cvIndices,
                     std::vector<double>& parameters,
                     std::vector<int>& integerParameters) const;

    void setTemperature(double kelvin);
    double getTemperature() const;

    static double barrierToGamma(double barrier_kJ, double kT_kJ);
    static double kTFromTemperature(double temperatureKelvin);

    bool usesPeriodicBoundaryConditions() const;
    void setUsesPeriodicBoundaryConditions(bool yes);

    %extend {
        // Python-friendly CV query: returns current values as a list rather than
        // filling an output vector.  getCurrentCollectiveVariables(ctx, vec&) is the
        // C++ form; this overload is more natural from Python.
        std::vector<double> getCurrentCVValues(OpenMM::Context& context) const {
            std::vector<double> values;
            self->getCurrentCollectiveVariables(context, values);
            return values;
        }
    }
    %extend {
        // Python-friendly checkpoint API: returns/accepts Python bytes objects.
        // The raw C++ methods return a SWIG tuple, which SWIG won't accept back
        // as std::vector<char> without an explicit conversion.
        PyObject* getBiasState() const {
            std::vector<char> raw = self->getBiasStateBytes();
            return PyBytes_FromStringAndSize(raw.data(), raw.size());
        }
        void setBiasState(PyObject* b) {
            if (!PyBytes_Check(b) && !PyByteArray_Check(b))
                throw OpenMM::OpenMMException("setBiasState: expected bytes or bytearray");
            char* buf; Py_ssize_t len;
            if (PyBytes_Check(b)) { buf = PyBytes_AS_STRING(b); len = PyBytes_GET_SIZE(b); }
            else                  { buf = PyByteArray_AS_STRING(b); len = PyByteArray_GET_SIZE(b); }
            std::vector<char> vec(buf, buf + len);
            self->setBiasStateBytes(vec);
        }
    }
    std::vector<char> getBiasStateBytes() const;
    void setBiasStateBytes(const std::vector<char>& bytes);

    void getCurrentCollectiveVariables(OpenMM::Context& context,
                                       std::vector<double>& values) const;

    void setTestForce(int mode, double scale);
    int getTestForceMode() const;
    double getTestForceScale() const;

    void setTestBiasGradients(const std::vector<double>& gradients);
    std::vector<double> getTestBiasGradients() const;

    std::vector<double> getLastCVValues(OpenMM::Context& context) const;

    %extend {
        // Returns [zed, rct, nker, neff] for the biasIndex-th OPES bias.
        // zed=exp(logZ), rct=kT*logZ (convergence indicator), nker=kernel count, neff=effective samples.
        std::vector<double> getOPESMetrics(OpenMM::Context& context, int biasIndex) const {
            return self->getOPESMetrics(context, biasIndex);
        }
    }

    %extend {
        // Multiwalker B2: get raw device pointers for sharing with secondary walkers.
        // biasIdx: 0-based index of the bias (order of addBias calls).
        // Returns platform-specific raw device pointers as a list of long long.
        // biasType=BIAS_METAD: [grid_ptr]; biasType=BIAS_OPES: [centers, sigmas, logweights, numKernels, numAllocated].
        // Only valid for CUDA platform.
        std::vector<long long> getMultiWalkerPtrs(OpenMM::Context& context, int biasIdx) const {
            return self->getMultiWalkerPtrs(context, biasIdx);
        }
        // Set up this walker to share bias arrays with a primary walker.
        // Must be called AFTER context creation. ptrs comes from primary's getMultiWalkerPtrs().
        void setMultiWalkerPtrs(OpenMM::Context& context, int biasIdx, const std::vector<long long>& ptrs) {
            self->setMultiWalkerPtrs(context, biasIdx, ptrs);
        }
    }

    int addExpressionCV(const std::string& expression, const std::vector<int>& inputCVIndices);
    void getExpressionCVInfo(int specIdx, std::string& expression, std::vector<int>& inputCVIndices) const;

    int addPyTorchCV(const std::string& torchScriptPath,
                     const std::vector<int>& atomIndices,
                     const std::vector<double>& parameters);
    std::string getPyTorchCVModelPath(int specIdx) const;

    %extend {
        // Python-friendly getters that avoid output-reference arguments.
        // getCollectiveVariableInfo returns (type, [atoms], [params]).
        PyObject* getCollectiveVariableParameters(int idx) const {
            int type;
            std::vector<int> atoms;
            std::vector<double> params;
            self->getCollectiveVariableInfo(idx, type, atoms, params);
            PyObject* tup = PyTuple_New(3);
            PyTuple_SET_ITEM(tup, 0, PyLong_FromLong(type));
            PyObject* atomList = PyList_New(atoms.size());
            for (size_t i = 0; i < atoms.size(); ++i)
                PyList_SET_ITEM(atomList, i, PyLong_FromLong(atoms[i]));
            PyTuple_SET_ITEM(tup, 1, atomList);
            PyObject* paramList = PyList_New(params.size());
            for (size_t i = 0; i < params.size(); ++i)
                PyList_SET_ITEM(paramList, i, PyFloat_FromDouble(params[i]));
            PyTuple_SET_ITEM(tup, 2, paramList);
            return tup;
        }
        // getBiasParameters returns (type, [cvIndices], [params], [intParams]).
        PyObject* getBiasParameters(int idx) const {
            int type;
            std::vector<int> cvIndices, intParams;
            std::vector<double> params;
            self->getBiasInfo(idx, type, cvIndices, params, intParams);
            PyObject* tup = PyTuple_New(4);
            PyTuple_SET_ITEM(tup, 0, PyLong_FromLong(type));
            PyObject* cviList = PyList_New(cvIndices.size());
            for (size_t i = 0; i < cvIndices.size(); ++i)
                PyList_SET_ITEM(cviList, i, PyLong_FromLong(cvIndices[i]));
            PyTuple_SET_ITEM(tup, 1, cviList);
            PyObject* paramList = PyList_New(params.size());
            for (size_t i = 0; i < params.size(); ++i)
                PyList_SET_ITEM(paramList, i, PyFloat_FromDouble(params[i]));
            PyTuple_SET_ITEM(tup, 2, paramList);
            PyObject* ipList = PyList_New(intParams.size());
            for (size_t i = 0; i < intParams.size(); ++i)
                PyList_SET_ITEM(ipList, i, PyLong_FromLong(intParams[i]));
            PyTuple_SET_ITEM(tup, 3, ipList);
            return tup;
        }
        // getExpressionCVParameters returns (expression, [inputCVIndices]).
        PyObject* getExpressionCVParameters(int specIdx) const {
            std::string expr;
            std::vector<int> inputs;
            self->getExpressionCVInfo(specIdx, expr, inputs);
            PyObject* tup = PyTuple_New(2);
            PyTuple_SET_ITEM(tup, 0, PyUnicode_FromStringAndSize(expr.data(), expr.size()));
            PyObject* inList = PyList_New(inputs.size());
            for (size_t i = 0; i < inputs.size(); ++i)
                PyList_SET_ITEM(inList, i, PyLong_FromLong(inputs[i]));
            PyTuple_SET_ITEM(tup, 1, inList);
            return tup;
        }
        // Downcast a generic OpenMM Force* to GluedForce*.
        // Needed after XmlSerializer.deserialize(), which returns the base Force type.
        static GluedPlugin::GluedForce* cast(OpenMM::Force* force) {
            return dynamic_cast<GluedPlugin::GluedForce*>(force);
        }
    }
};

}
