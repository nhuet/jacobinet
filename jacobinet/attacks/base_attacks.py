import keras
from keras.layers import Input, Layer
from keras.losses import Loss
from keras.models import Model, Sequential
import keras.ops as K

from .utils import FGSM
from jacobinet import clone_to_backward
from jacobinet.utils import to_list
from jacobinet.losses import deserialize, get_loss_as_layer
from jacobinet.models import BackwardModel
import numpy as np

from typing import Union, List, Any
from keras import KerasTensor as Tensor

def get_model_with_loss(model:Union[Model, Sequential], loss:Union[str, Loss, Layer], **kwargs):
    # duplicate inputs of the model

    inputs = [Input(input_i.shape[1:]) for input_i in to_list(model.inputs)]

    # groundtruth target: same shape as model.output if gt_shape undefined in kwargs
    if 'gt_shape' in kwargs:
        gt_shape = kwargs['gt_shape']
    else:
        gt_shape = to_list(model.outputs)[0].shape[1:]

    gt_input = Input(gt_shape)

    loss_layer:Layer
    if type(loss)=='str':
        loss:Loss = deserialize(loss)
        # convert loss which is a Loss object into a keras Layer
        loss_layer = get_loss_as_layer(loss)
    elif isinstance(loss, Loss):
        # convert loss which is a Loss object into a keras Layer
        loss_layer = get_loss_as_layer(loss)
    elif isinstance(loss, Layer):
        loss_layer = loss
    else:
        raise TypeError('unknown type for loss {}'.format(loss.__class__))
    
    # build a model: loss_layer takes as input y_true, y_pred
    output_pred = model(inputs)
    output_loss = loss_layer([gt_input]+to_list(output_pred)) #(None, 1)
    output_loss_dim_wo_batch = output_loss.shape[1:]
    assert len(output_loss_dim_wo_batch)==1 and output_loss_dim_wo_batch[0]==1, 'Wrong output shape for model that predicts a loss, Expected [1] got {}'.format(output_loss_dim_wo_batch)

    model_with_loss:Model = keras.models.Model(inputs+[gt_input], output_loss)
    
    return model_with_loss, [gt_input]



def get_adv_model_base(model, 
                  loss:Union[str, Layer]='categorical_crossentropy', 
                  attack:str=FGSM,
                  mapping_keras2backward_classes={},
                  mapping_keras2backward_losses={},
                  **kwargs)->Model: # we do not compute gradient on extra_inputs, loss should return (None, 1)
    
    if len(model.outputs)>1:
        raise NotImplementedError('actually not working wih multiple loss. Raise a dedicated PR if needed')
    if len(model.inputs)>1:
        raise NotImplementedError('actually not working wih multiple inputs. Raise a dedicated PR if needed')
    
    if loss=='logits':
        # simple backward
        backward_model_base_attack = clone_to_backward(
                                model=model, 
                                mapping_keras2backward_classes=mapping_keras2backward_classes,
                                )
    else:

        model_with_loss:Model
        label_tensors:List[Tensor]
        model_with_loss, label_tensors = get_model_with_loss(model, loss, **kwargs) # to define, same for every atacks

        input_mask = [label_tensor_i.name for label_tensor_i in label_tensors]

        if mapping_keras2backward_classes is None:
            mapping_keras2backward_classes = mapping_keras2backward_losses
        elif not(mapping_keras2backward_losses is None):
            mapping_keras2backward_classes.update(mapping_keras2backward_losses)   

        backward_model_base_attack = clone_to_backward(
                                    model=model_with_loss, 
                                    mapping_keras2backward_classes=mapping_keras2backward_classes,
                                    gradient=keras.Variable(np.ones((1, 1))),
                                    input_mask=input_mask
                                    )
    # convert it into an AdvModel
    return backward_model_base_attack

class AdvLayer(Layer):

    def __init__(
        self,
        epsilon: float = 0.0,
        lower:float = -np.inf,
        upper:float = np.inf,
        p:float = -1,
        radius:float = np.inf,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.epsilon = keras.Variable(epsilon, trainable=False)
        self.lower=lower
        self.upper = upper
        self.p = p
        self.radius = radius

    # @keras.ops.custom_gradient
    def call(self, inputs, training=None, mask=None):
        raise NotImplementedError()

    # saving
    def get_config(self):
        config = super().get_config()
        eps_config = keras.saving.serialize_keras_object(self.epsilon)
        config["epsilon"] = eps_config
        config["lower"] = self.lower
        config["upper"] = self.upper
        config["radius"] = self.radius
        return config
    
    def set_upper(self, upper):
        self.upper = upper

    def set_lower(self, lower):
        self.lower = lower

    def set_p(self, p):
        self.p = p

    def set_radius(self, radius):
        self.radius = radius

    def get_upper(self):
        return self.upper

    def get_lower(self):
        return self.lower

    def get_p(self):
        return self.p
    
    def get_radius(self):
        return self.radius
    
    def project_lp_ball(self, x):

        if self.p==-1:
            # no projection, return identity
            return x
        
        if self.p == np.inf:
            # project on l_inf norm: equivalent to clipping
            return K.clip(x, -self.radius, self.radius)
        
        axis=np.arange(len(x.shape)-1)+1
        if self.p==2:
            # compute l2 norm and normalize with it
            if self.radius < np.inf:
                norm_2 = K.sum(K.sqrt(x**2), axis=axis, keepdims=True)
                return self.radius*x/norm_2
            return x
        elif self.p==1:
            # compute l1 norm and normalize with it
            if self.radius < np.inf:
                norm_1 = K.sum(K.abs(x), axis=axis, keepdims=True)
                return x/norm_1
            return x
        else:
            raise ValueError('unknown lp norm p={}'.format(self.p))
        

    

class AdvModel(keras.Model):
    def __init__(
        self,
        layer_adv: AdvLayer,
        backward_model: BackwardModel,
        method='fgsm', # replace by Enum
        *args,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.backward_model = backward_model
        self.method = method
        self.layer_adv = layer_adv

    def get_config(self):
        config = super().get_config()
        backward_config = keras.saving.serialize_keras_object(
            self.backward_model
        )
        layer_config = keras.saving.serialize_keras_object(
            self.layer_adv
        )
        config["backward_model"] = backward_config
        config["method"] = self.method
        config["layer_adv"] = layer_config
        return config
    
    def set_upper(self, upper):
        self.layer_adv.set_upper(upper)

    def set_lower(self, lower):
        self.layer_adv.set_lower(lower)

    def set_p(self, p):
        self.layer_adv.set_p(p)

    def get_upper(self):
        return self.layer_adv.upper

    def get_lower(self):
        return self.layer_adv.get_lower()

    def get_p(self, p):
        return self.layer_adv.get_p()
    
