# src.losses module

from loss import (
    FocalLoss,
    FocalConsistencyLoss,
    RegularizedLearnedConsistencyLoss,
    FocalConsistencyLoss14Emotions,
    RegularizedLearnedConsistencyLoss14Emotions,
    KnowledgeGuidedConsistencyLoss,
    KnowledgeGuidedConsistencyLoss8Classes,
    KnowledgeGuidedConsistencyLoss14Classes,
    GenericConsistencyLoss,
    EnhancedKnowledgeConsistencyLoss,
    create_enhanced_consistency_loss,
    CenterLoss,
    ArcFace,
    CosFace,
    CombinedMarginLoss,
    EuclideanLoss,
    CCCLoss,
    regularized_gmm_loss,
    elbo_loss
)

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

