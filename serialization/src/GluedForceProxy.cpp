#include "GluedForceProxy.h"
#include "GluedForce.h"
#include "openmm/serialization/SerializationNode.h"
#include "openmm/OpenMMException.h"
#include <string>
#include <vector>

using namespace OpenMM;
using namespace GluedPlugin;
using namespace std;

GluedForceProxy::GluedForceProxy()
    : SerializationProxy("GluedForce") {}

void GluedForceProxy::serialize(const void* object,
                                    SerializationNode& node) const {
    node.setIntProperty("version", 1);
    const auto& force = *reinterpret_cast<const GluedForce*>(object);

    node.setIntProperty("forceGroup", force.getForceGroup());
    node.setDoubleProperty("temperature", force.getTemperature());
    node.setBoolProperty("usesPBC", force.usesPeriodicBoundaryConditions());

    // --- Collective variables ---
    SerializationNode& cvsNode = node.createChildNode("CollectiveVariables");
    int numSpecs = force.getNumCollectiveVariableSpecs();
    for (int i = 0; i < numSpecs; ++i) {
        int type;
        vector<int> atoms;
        vector<double> params;
        force.getCollectiveVariableInfo(i, type, atoms, params);

        SerializationNode& cvNode = cvsNode.createChildNode("CV");
        cvNode.setIntProperty("type", type);

        cvNode.setIntProperty("numAtoms", (int)atoms.size());
        for (int j = 0; j < (int)atoms.size(); ++j)
            cvNode.setIntProperty("atom" + to_string(j), atoms[j]);

        cvNode.setIntProperty("numParams", (int)params.size());
        for (int j = 0; j < (int)params.size(); ++j)
            cvNode.setDoubleProperty("param" + to_string(j), params[j]);

        if (type == GluedForce::CV_EXPRESSION) {
            string expr;
            vector<int> inputCVs;
            force.getExpressionCVInfo(i, expr, inputCVs);
            cvNode.setStringProperty("expression", expr);
            // inputCVIndices are already in atoms — nothing extra needed
        } else if (type == GluedForce::CV_PYTORCH) {
            cvNode.setStringProperty("modelPath", force.getPyTorchCVModelPath(i));
        }
    }

    // --- Biases ---
    SerializationNode& biasesNode = node.createChildNode("Biases");
    int numBiases = force.getNumBiases();
    for (int i = 0; i < numBiases; ++i) {
        int type;
        vector<int> cvIndices, intParams;
        vector<double> params;
        force.getBiasInfo(i, type, cvIndices, params, intParams);

        SerializationNode& biasNode = biasesNode.createChildNode("Bias");
        biasNode.setIntProperty("type", type);

        biasNode.setIntProperty("numCVs", (int)cvIndices.size());
        for (int j = 0; j < (int)cvIndices.size(); ++j)
            biasNode.setIntProperty("cv" + to_string(j), cvIndices[j]);

        biasNode.setIntProperty("numParams", (int)params.size());
        for (int j = 0; j < (int)params.size(); ++j)
            biasNode.setDoubleProperty("param" + to_string(j), params[j]);

        biasNode.setIntProperty("numIntParams", (int)intParams.size());
        for (int j = 0; j < (int)intParams.size(); ++j)
            biasNode.setIntProperty("intParam" + to_string(j), intParams[j]);
    }
}

void* GluedForceProxy::deserialize(const SerializationNode& node) const {
    if (node.getIntProperty("version") != 1)
        throw OpenMMException("GluedForceProxy: unsupported format version");

    auto* force = new GluedForce();
    force->setForceGroup(node.getIntProperty("forceGroup", 0));
    force->setTemperature(node.getDoubleProperty("temperature", -1.0));
    force->setUsesPeriodicBoundaryConditions(node.getBoolProperty("usesPBC", false));

    // --- Collective variables ---
    const SerializationNode& cvsNode = node.getChildNode("CollectiveVariables");
    for (const SerializationNode& cvNode : cvsNode.getChildren()) {
        int type = cvNode.getIntProperty("type");
        int numAtoms = cvNode.getIntProperty("numAtoms");
        vector<int> atoms(numAtoms);
        for (int j = 0; j < numAtoms; ++j)
            atoms[j] = cvNode.getIntProperty("atom" + to_string(j));

        int numParams = cvNode.getIntProperty("numParams");
        vector<double> params(numParams);
        for (int j = 0; j < numParams; ++j)
            params[j] = cvNode.getDoubleProperty("param" + to_string(j));

        if (type == GluedForce::CV_EXPRESSION) {
            string expr = cvNode.getStringProperty("expression");
            force->addExpressionCV(expr, atoms);
        } else if (type == GluedForce::CV_PYTORCH) {
            string modelPath = cvNode.getStringProperty("modelPath");
            force->addPyTorchCV(modelPath, atoms, params);
        } else {
            force->addCollectiveVariable(type, atoms, params);
        }
    }

    // --- Biases ---
    const SerializationNode& biasesNode = node.getChildNode("Biases");
    for (const SerializationNode& biasNode : biasesNode.getChildren()) {
        int type = biasNode.getIntProperty("type");
        int numCVs = biasNode.getIntProperty("numCVs");
        vector<int> cvIndices(numCVs);
        for (int j = 0; j < numCVs; ++j)
            cvIndices[j] = biasNode.getIntProperty("cv" + to_string(j));

        int numParams = biasNode.getIntProperty("numParams");
        vector<double> params(numParams);
        for (int j = 0; j < numParams; ++j)
            params[j] = biasNode.getDoubleProperty("param" + to_string(j));

        int numIntParams = biasNode.getIntProperty("numIntParams");
        vector<int> intParams(numIntParams);
        for (int j = 0; j < numIntParams; ++j)
            intParams[j] = biasNode.getIntProperty("intParam" + to_string(j));

        force->addBias(type, cvIndices, params, intParams);
    }

    return force;
}
