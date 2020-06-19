
from .argo.core.ArgoLauncher import ArgoLauncher
from datasets.Dataset import TRAIN, VALIDATION, TEST

class AdversarialLauncher(ArgoLauncher):

    def __init__(self):
        super().__init__()

    def execute(self, model, dataset, opts, config):
        model.init(dataset)

        if not model._only_plot_statistics:
            model.attack(TEST)
            model.compute_accuracy(TEST, resize = 0)
            model.plot_accuracies()

            if(model._plot_reconstructions['enable']):
                #for value_kwargs in self._transform_kwargs_accuracy:
                initial_images, rec_images = model.generate_reconstructions(TEST)
                model.plot_reconstructions(initial_images, rec_images)

        if model._do_plot_statistics:
            model.compute_statistics(TEST)
            model.plot_mu_statistics(TEST)

        if model._resize['enable']:
            model.compute_accuracy(TEST, resize = 1)
