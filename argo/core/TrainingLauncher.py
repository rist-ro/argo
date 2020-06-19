from .ArgoLauncher import ArgoLauncher

import pdb

class TrainingLauncher(ArgoLauncher):
            
    def execute(self, model, dataset, opts, config):
        model.init(dataset)

        model.create_session(opts, config)

        # import numpy as np
        # sess = model.get_raw_session()
        # raw_x_np, x_np = sess.run([model.raw_x, model.x], feed_dict={model.ds_handle: model.datasets_handles["train_loop"], model.is_training:True})
        
        model.train()
