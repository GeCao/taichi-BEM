from enum import Enum


class CellFluxType(Enum):
    NEUMANN_KNOWN = 0
    TOBESOLVED=1
    MIX=2

    def __int__(self):
        return self.value


class VertAttachType(Enum):
    DIRICHLET_KNOWN = 0
    TOBESOLVED=1

    def __int__(self):
        return self.value


class KernelType(Enum):
    LAPLACE = 0
    HELMHOLTZ = 1
    HELMHOLTZ_TRANSMISSION = 2
    
    def __int__(self):
        return self.value


class PanelsRelation(Enum):
    SEPARATE = 0
    COINCIDE = 1
    COMMON_VERTEX = 2
    COMMON_EDGE = 3
    
    def __int__(self):
        return self.value

class AssembleType(Enum):
    NOTHING = 0
    ADD_HALF_IDENTITY = 1
    ADD_M = 2
    ADD_P_PLUS = 3
    ADD_P_MINUS = 4
    
    def __int__(self):
        return self.value