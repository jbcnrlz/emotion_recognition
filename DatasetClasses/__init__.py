# DatasetClasses package

try:
    from .AffectNet import AffectNet
except ImportError:
    AffectNet = None

try:
    from .AffWild2 import AffWild2
except ImportError:
    AffWild2 = None

try:
    from .AFEWVA import AFEWVA
except ImportError:
    AFEWVA = None

try:
    from .CKPlus import CKPlus
except ImportError:
    CKPlus = None

try:
    from .DFEW import DFEW
except ImportError:
    DFEW = None

try:
    from .HedonicDataset import HedonicDataset
except ImportError:
    HedonicDataset = None

try:
    from .JoinedDataset import JoinedDataset
except ImportError:
    JoinedDataset = None

try:
    from .MIntPainDataset import MIntPainDataset
except ImportError:
    MIntPainDataset = None

try:
    from .OMG import OMG
except ImportError:
    OMG = None

try:
    from .SAM import SAM
except ImportError:
    SAM = None

try:
    from .TeachingDataset import TeachingDataset
except ImportError:
    TeachingDataset = None

try:
    from .WFD import WFD
except ImportError:
    WFD = None

__all__ = [
    'AffectNet',
    'AffWild2',
    'AFEWVA',
    'CKPlus',
    'DFEW',
    'HedonicDataset',
    'JoinedDataset',
    'MIntPainDataset',
    'OMG',
    'SAM',
    'TeachingDataset',
    'WFD'
]

