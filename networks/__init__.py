# networks package
from .EmotionResnetVA import (
    ResnetWithBayesianGMMHead,
    ResNet50WithAttentionLikelihood,
    ResNet50WithAttentionGMM,
    ResNet50WithAttentionLikelihoodNoVA,
    ResNet50WithCrossAttention,
    Glint360kResNetWithCrossAttention,
    Glint360kResNetWithCrossAttentionVA,
    PainIndexCrossAttention,
    SpatialSelfAttention,
    BayesianLinearVI,
    BayesianNetworkVI,
    ResnetWithBayesianHead
)

try:
    from .PosterV2 import pyramid_trans_expr2
except ImportError:
    pyramid_trans_expr2 = None

try:
    from .DDAM import DDAMNet
except ImportError:
    DDAMNet = None

try:
    from .MixedFeatureNet import MixedFeatureNet
except ImportError:
    MixedFeatureNet = None

try:
    from .mobilefacenet import MobileFaceNet
except ImportError:
    MobileFaceNet = None

try:
    from .ir50 import Backbone as IR50
except ImportError:
    IR50 = None

try:
    from .iresnet import IResNet
except ImportError:
    IResNet = None

try:
    from .VAEForEmotion import VAEForEmotion, VAEOurEmotion, VAEOurEmotionAttention
except ImportError:
    VAEForEmotion, VAEOurEmotion, VAEOurEmotionAttention = None, None, None

try:
    from .ViTBasedModel import ViTEmotionHeadClassifier
except ImportError:
    ViTEmotionHeadClassifier = None

try:
    from .ResnetEmotionHead import ResnetEmotionHead
except ImportError:
    ResnetEmotionHead = None

__all__ = [
    'ResnetWithBayesianGMMHead',
    'ResNet50WithAttentionLikelihood',
    'ResNet50WithAttentionGMM',
    'ResNet50WithAttentionLikelihoodNoVA',
    'ResNet50WithCrossAttention',
    'Glint360kResNetWithCrossAttention',
    'Glint360kResNetWithCrossAttentionVA',
    'PainIndexCrossAttention',
    'SpatialSelfAttention',
    'BayesianLinearVI',
    'BayesianNetworkVI',
    'ResnetWithBayesianHead',
    'pyramid_trans_expr2',
    'DDAMNet',
    'MixedFeatureNet',
    'MobileFaceNet',
    'IR50',
    'IResNet',
    'VAEForEmotion',
    'VAEOurEmotion',
    'VAEOurEmotionAttention',
    'ViTEmotionHeadClassifier',
    'ResnetEmotionHead'
]

