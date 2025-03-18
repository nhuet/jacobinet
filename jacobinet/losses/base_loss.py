# set loss as a keras layer
import keras
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

@keras.saving.register_keras_serializable()
class BackwardLoss(BackwardMergeNonLinearLayer):

    loss: Loss_Layer
