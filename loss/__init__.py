# loss package

try:
    from .FocalLoss import FocalLoss
except ImportError:
    FocalLoss = None

try:
    from .FocalConsistencyLoss import (
        FocalConsistencyLoss,
        RegularizedLearnedConsistencyLoss,
        FocalConsistencyLoss14Emotions,
        RegularizedLearnedConsistencyLoss14Emotions
    )
except ImportError:
    FocalConsistencyLoss = None
    RegularizedLearnedConsistencyLoss = None
    FocalConsistencyLoss14Emotions = None
    RegularizedLearnedConsistencyLoss14Emotions = None

try:
    from .KnowledgeGuidedConsistencyLoss import (
        KnowledgeGuidedConsistencyLoss,
        KnowledgeGuidedConsistencyLoss8Classes,
        KnowledgeGuidedConsistencyLoss14Classes,
        GenericConsistencyLoss,
        EnhancedKnowledgeConsistencyLoss,
        create_enhanced_consistency_loss
    )
except ImportError:
    KnowledgeGuidedConsistencyLoss = None
    KnowledgeGuidedConsistencyLoss8Classes = None
    KnowledgeGuidedConsistencyLoss14Classes = None
    GenericConsistencyLoss = None
    EnhancedKnowledgeConsistencyLoss = None
    create_enhanced_consistency_loss = None

try:
    from .CenterLoss import CenterLoss
except ImportError:
    CenterLoss = None

try:
    from .ArcFace import ArcFace, CosFace, CombinedMarginLoss
except ImportError:
    ArcFace, CosFace, CombinedMarginLoss = None, None, None

try:
    from .EuclideanLoss import EuclideanLoss
except ImportError:
    EuclideanLoss = None

try:
    from .VAloss import CCCLoss, regularized_gmm_loss, elbo_loss
except ImportError:
    CCCLoss, regularized_gmm_loss, elbo_loss = None, None, None

__all__ = [
    'FocalLoss',
    'FocalConsistencyLoss',
    'RegularizedLearnedConsistencyLoss',
    'FocalConsistencyLoss14Emotions',
    'RegularizedLearnedConsistencyLoss14Emotions',
    'KnowledgeGuidedConsistencyLoss',
    'KnowledgeGuidedConsistencyLoss8Classes',
    'KnowledgeGuidedConsistencyLoss14Classes',
    'GenericConsistencyLoss',
    'EnhancedKnowledgeConsistencyLoss',
    'create_enhanced_consistency_loss',
    'CenterLoss',
    'ArcFace',
    'CosFace',
    'CombinedMarginLoss',
    'EuclideanLoss',
    'CCCLoss',
    'regularized_gmm_loss',
    'elbo_loss'
]

