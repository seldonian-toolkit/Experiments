""" Module containing base class for running examples """
import os
from .generate_plots import SupervisedPlotGenerator
from .base_example import BaseExample


class HeadlessExample(BaseExample):
    def __init__(self, spec):
        """Class for running headless experiments"""
        super().__init__(spec=spec)
        assert self.regime == "supervised_learning","Headless examples are only supported for supervised learning"

    def run(
        self,
        full_pretraining_model,
        headless_pretraining_model,
        head_layer_names,
        latent_feature_shape,
        loss_func_pretraining,
        learning_rate_pretraining,
        pretraining_device,
        batch_epoch_dict_pretraining,
        safety_batch_size_pretraining,
        n_trials,
        data_fracs,
        results_dir,
        perf_eval_fn,
        perf_eval_kwargs,
        constraint_eval_kwargs,
        n_workers=1,
        batch_epoch_dict={},
        datagen_method="resample",
        verbose=False,
        baselines=[],
        performance_label="performance",
        performance_yscale="linear",
        plot_savename=None,
        plot_fontsize=12,
        legend_fontsize=8,
        model_label_dict={},
    ):
        """Run the experiment for this example.
        Runs any baseline models included in baselines
        parameter first. Then produces the three plots.
        """
        # assert baselines == [], "No baselines supported for headless examples yet"
        os.makedirs(results_dir, exist_ok=True)
        
        plot_generator = SupervisedPlotGenerator(
            spec=self.spec,
            n_trials=n_trials,
            data_fracs=data_fracs,
            n_workers=n_workers,
            datagen_method=datagen_method,
            perf_eval_fn=perf_eval_fn,
            constraint_eval_fns=[],
            results_dir=results_dir,
            perf_eval_kwargs=perf_eval_kwargs,
            constraint_eval_kwargs=constraint_eval_kwargs,
            batch_epoch_dict=batch_epoch_dict,
        )

        # Baselines first
        for baseline_model in baselines:
            plot_generator.run_baseline_experiment(
                baseline_model=baseline_model, verbose=verbose
            )
        
        # Run Seldonian headless experiment
        # A special thing we need to do for headless experiments is get the 
        # initial weights of the model and freeze them so we can 
        # re-initialize the same weights in each train when we pretrain
        initial_state_dict = full_pretraining_model.state_dict()

        plot_generator.run_headless_seldonian_experiment(
            full_pretraining_model=full_pretraining_model, 
            initial_state_dict=initial_state_dict,
            headless_pretraining_model=headless_pretraining_model, 
            head_layer_names=head_layer_names,
            latent_feature_shape=latent_feature_shape,
            loss_func_pretraining=loss_func_pretraining,
            learning_rate_pretraining=learning_rate_pretraining,
            pretraining_device=pretraining_device,
            batch_epoch_dict_pretraining=batch_epoch_dict_pretraining, 
            safety_batch_size_pretraining=1000,
            verbose=verbose)

        plot_generator.make_plots(
            fontsize=plot_fontsize,
            legend_fontsize=legend_fontsize,
            performance_label=performance_label,
            performance_yscale=performance_yscale,
            model_label_dict=model_label_dict,
            savename=plot_savename,
        )
