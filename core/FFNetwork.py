from argo.core.network.ArgoStochasticNetworkWithDefaults import ArgoStochasticNetworkWithDefaults
from argo.core.network.GeneralSonnetNetwork import GeneralSonnetNetwork
from argo.core.utils import argo_utils as utils


class FFNetwork(ArgoStochasticNetworkWithDefaults):
    """
    Network class for managing a FF network
    """
    
    default_params = {
        **ArgoStochasticNetworkWithDefaults.default_params,
        "network_architecture" : [
                ("Conv2D", {"output_channels" : 10, "kernel_shape" : (3,3)}, 1),
    	        ("BatchFlatten", {}, 0),
                ("Linear", {"output_size" : 200}, 1),
                ("Linear", {"output_size" : 100}, 1)]
                # ("conv2d", {"filters" : 10, "kernel_size" : (3,3)}),
    		    # ("flatten", {}),
                # ("dense", {"units" : 200}),
                # ("dense", {"units" : 100})]
    }
    
    def create_id(self):
        
        layers_ids = [utils.get_method_id(layer_tuple)
                        for layer_tuple in self._opts["network_architecture"]]
        
        _id = '-n' + "_".join(filter(None, layers_ids))
        
        super_id = super().create_id()
        
        _id += super_id
        return _id

    def __init__(self, opts, name, seed=None): # , remove_node_from_logits = False
        super().__init__(opts, name, seed)
        self._network_architecture = self._opts["network_architecture"]

        #self.remove_node_from_logits = remove_node_from_logits
         
    def _build(self, x, is_training=False, network_str=""): #drop_one_logit = False):
        
        '''
        # check if I need a n-1 logit layer, where the n_th is consider fixed to 1
        if self.remove_node_from_logits:
            # I need all this tricky steps, since the dictionary is shared between datastructures
            network_architecture = self._network_architecture.copy()
            last_module = self._network_architecture[-1]
            new_last_module = []
            new_last_module.append(last_module[0])
            new_last_module.append(last_module[1].copy())
            new_last_module[1]["output_size"] -= 1
            new_last_module.append(last_module[2])
            
            network_architecture[-1] = new_last_module
            
        else:
            network_architecture = self._network_architecture
        '''

        network_architecture = self._network_architecture
        #if drop_one_logit:
        #     network_architecture[-1][1]["output_size"] -= 1 

             
        self.module = GeneralSonnetNetwork(self._network_defaults["activation"],
                                           self._network_defaults["weights_init"],
                                           self._network_defaults["bias_init"],
                                           self._network_defaults["weights_reg"],
                                           self._network_defaults["bias_reg"],
                                           network_architecture,
                                           is_training = is_training,
                                           network_str=network_str,
                                           name="network")
        
        return self.module(x)
        # return self._create_deterministic_layers(x, self._network_architecture)
