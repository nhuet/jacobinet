# set loss as a keras layer
from jacobinet.layers import BackwardMergeNonLinearLayer
from .loss import Loss_Layer

# MeanSquaredError
# MeanAbsoluteError
# MeanAbsoluteError
# MeanSquaredLogarithmicError
# CosineSimilarity
# LogCosh
# Hinge
# SquaredHinge
# CategoricalHinge
# KLDivergence
# BinaryCrossentropy
# CategoricalCrossentropy categorical_crossentropy

class BackwardLoss(BackwardMergeNonLinearLayer):

    loss: Loss_Layer
