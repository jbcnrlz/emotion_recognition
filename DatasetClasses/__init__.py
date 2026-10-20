# DatasetClasses package (PEP 562 lazy loading)

_dataset_modules = {
    'AffectNet': '.AffectNet',
    'AffWild2': '.AffWild2',
    'AFEWVA': '.AFEWVA',
    'CKPlus': '.CKPlus',
    'DFEW': '.DFEW',
    'HedonicDataset': '.HedonicDataset',
    'JoinedDataset': '.JoinedDataset',
    'MIntPainDataset': '.MIntPainDataset',
    'OMG': '.OMG',
    'SAM': '.SAM',
    'TeachingDataset': '.TeachingDataset',
    'WFD': '.WFD'
}

def __getattr__(name):
    if name in _dataset_modules:
        import importlib
        mod = importlib.import_module(_dataset_modules[name], package=__name__)
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

def __dir__():
    return sorted(list(globals().keys()) + list(_dataset_modules.keys()))

__all__ = list(_dataset_modules.keys())
