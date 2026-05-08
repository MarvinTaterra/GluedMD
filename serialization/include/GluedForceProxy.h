#ifndef OPENMM_GLUED_FORCE_PROXY_H_
#define OPENMM_GLUED_FORCE_PROXY_H_

#include "internal/windowsExportGlued.h"
#include "openmm/serialization/SerializationProxy.h"

namespace OpenMM {

class OPENMM_EXPORT_GLUED GluedForceProxy : public SerializationProxy {
public:
    GluedForceProxy();
    void serialize(const void* object, SerializationNode& node) const;
    void* deserialize(const SerializationNode& node) const;
};

} // namespace OpenMM

#endif // OPENMM_GLUED_FORCE_PROXY_H_
