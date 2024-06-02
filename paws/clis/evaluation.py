import os

import click

from .main import cli, DelimitedStr
from paws.settings import DEFAULT_DATADIR, DEFAULT_OUTDIR, BASE_SEED, MASS_RANGE, MASS_INTERVAL, ModelType

__all__ = ["compute_semi_weakly_landscape", "gather_model_results"]
    
kCommonKeys = ["high_level", "decay_modes", "variables", "noise_dimension",
               "index_path", "split_index", "seed", "batchsize", "cache_dataset",
               "datadir", "outdir", "cache", "multi_gpu", "verbosity"]

def get_model_trainer(model_type:str, **kwargs):
    from paws.components import ModelTrainer
    init_kwargs = {key: kwargs.pop(key) for key in kCommonKeys}
    init_kwargs['model_options'] = kwargs
    feature_level = "high_level" if init_kwargs.pop("high_level") else "low_level"
    model_trainer = ModelTrainer(model_type, **init_kwargs)
    return model_trainer

@cli.command(name='compute_semi_weakly_landscape')
@click.option('-m', '--mass-point', required=True,
              help='Signal mass point (in the form m1:m2) to use for creating the dataset.')
@click.option('--min_mass', default=0, type=float, show_default=True,
              help='Minimum value of the mass (in GeV) in the 2D mass grid to scan for.')
@click.option('--max_mass', default=MASS_RANGE[1], type=float, show_default=True,
              help='Maximum value of the mass (in GeV) in the 2D mass grid to scan for.')
@click.option('--mass_interval', default=MASS_INTERVAL, type=float, show_default=True,
              help='Mass interval (in GeV) between grid points in the 2D mass grid to scan for.')
@click.option('--mu', required=True, type=float,
              help='Signal fraction used in the dataset.')
@click.option('--mu_list', default=None, type=float,
              help='The list of signal fractions to scan over (separated by commas). '
              'If None, the truth value of the dataset will be used.')
@click.option('--alpha', default=0.5, type=float, show_default=True,
              help='Signal branching fraction in the dataset. Ignored '
             'when only one signal decay mode is considered.')
@click.option('--alpha_list', default=None, type=float,
              help='The list of branching fractions to scan over (separated by commas). '
              'Only used when evaluating mixed two-prong + three-prong '
              'model.  If None, the truth value of the dataset will be used.')
@click.option('--high-level/--low-level', default=True, show_default=True,
              help='Whether to do training with low-evel or high-level features.')
@click.option('--decay-modes', default='qq', type=click.Choice(['qq', 'qqq', 'qq,qqq'], case_sensitive=False), show_default=True,
              help='Which decay mode should the signal undergo (qq or qqq).'
              'Use "qq,qqq" to include both decay modes.')
@click.option('--variables', default=None, show_default=True,
              help='Select certain high-level jet features to include in the training'
              'by the indices they appear in the feature vector. For example,'
              '"3,5,6" means select the 4th, 6th and 7th feature from the jet'
              'feature vector to be used in the training.')
@click.option('--noise', 'noise_dimension', default=0, type=int, show_default=True,
              help='Number of noise dimension to add to the train features.')
@click.option('--dataset-index-path', 'index_path', default=None, show_default=True,
              help='Path to the dataset split configuration file. It determines the'
              'shard indices for the train, validation, and test datasets in each'
              'random realization of data. If None, a default configuration '
              'will be created.')
@click.option('-i', '--split-index', default=0, type=int, show_default=True,
              help='Index for dataset split.')
@click.option('--fs-version', 'fs_version', default="v1", show_default=True,
              help='Version of the supervised model to use.')
@click.option('--fs-version-2', 'fs_version_2', default=None, show_default=True,
              help='When signals of mixed decay modes are considered, it corresponds to '
             'the version of the three-prone supervised model. If None, the '
             'same version as `fs_version` will be used.')
@click.option('--seed', default=BASE_SEED, type=int, show_default=True,
              help='The default seed used for all random processes.')
@click.option('--batchsize', default=None, type=int, show_default=True,
              help='Batch size for the dataset.')
@click.option('--cache-dataset/--no-cache-dataset', default=None, show_default=True,
              help='Whether to cache the dataset during training.')
@click.option('-d', '--datadir', default=DEFAULT_DATADIR, show_default=True,
              help='input directory where the tfrecord datasets are stored')
@click.option('-o', '--outdir', default=DEFAULT_OUTDIR, show_default=True,
              help='base output directory')
@click.option('--cache/--no-cache', default=True, show_default=True,
              help='Whether to cache the results.')
@click.option('--multi-gpu/--single-gpu', default=True, show_default=True,
              help='Whether to enable multi-GPU computation.')
@click.option('-v', '--verbosity',  default="INFO", show_default=True,
              help='Verbosity level ("DEBUG", "INFO", "WARNING" or "ERROR").')
def compute_semi_weakly_landscape(**kwargs):
    """
    Compute metric landscapes for a semi-weakly model
    """
    import os
    import json
    import numpy as np
    from quickstats.maths.numerics import cartesian_product
    from quickstats.utils.common_utils import NpEncoder
    from quickstats.utils.string_utils import split_str
    from paws.components import MetricLandscape
    
    mu_list = kwargs.pop("mu_list")
    if mu_list is None:
        mu_list = [kwargs['mu']]
        
    alpha_list = kwargs.pop("alpha_list")
    if alpha_list is None:
        alpha_list = [kwargs['alpha']]

    min_mass, max_mass, mass_interval = (kwargs.pop(key) for key in ("min_mass", "max_mass", "mass_interval"))

    model_trainer = get_model_trainer("semi_weakly", **kwargs)
    datasets = model_trainer.get_datasets()
    model = model_trainer.get_model()
    
    landscape = MetricLandscape()
    unique_masses = np.arange(min_mass, max_mass + mass_interval, mass_interval)
    mass_points = cartesian_product(unique_masses, unique_masses)

    result = landscape.predict_semiweakly(model, datasets['train'], mass_points=mass_points,
                                          mu_points=mu_list, alpha_points=alpha_list)
    parameters = model_trainer.model_loader._get_param_repr()
    model_options = model_trainer.model_options
    parameters['mass_point'] = model_options["mass_point"]
    parameters['mu'] = model_options["mu"]
    parameters['alpha'] = model_options["alpha"]
    parameters = model_trainer.path_manager.process_parameters(**parameters)
    outname = model_trainer.path_manager.get_file("semi_weakly_landscape", **parameters,
                                                  ds_type="train")
    os.makedirs(os.path.dirname(outname), exist_ok=True)
    with open(outname, 'w') as file:
        json.dump(result, file, cls=NpEncoder)
    model_trainer.stdout.info(f"Saved semi-weakly model landscape output to {outname}")

@cli.command(name='gather_model_results')
@click.option('-t', '--model-type', required=True,
              help='Type of model for which the results are gathered.')
@click.option('--high-level/--low-level', default=True, show_default=True,
              help='Whether to do training with low-evel or high-level features.')
@click.option('--decay-modes', default='qq,qqq', type=click.Choice(['qq', 'qqq', 'qq,qqq'], case_sensitive=False),
              show_default=True,
              help='Which decay mode should the signal undergo (qq or qqq).'
              'Use "qq,qqq" to include both decay modes.')
@click.option('--variables', default=None, show_default=True,
              help='Select certain high-level jet features to include in the training'
              'by the indices they appear in the feature vector. For example,'
              '"3,5,6" means select the 4th, 6th and 7th feature from the jet'
              'feature vector to be used in the training.')
@click.option('-m', '--mass-points', default="*:*", show_default=True, cls=DelimitedStr,
              help='Filter results by the list of signal mass points in the form m1:m2 '
              '(separated by commas, wildcard is accepted).')
@click.option('--split-indices', default="*", show_default=True, cls=DelimitedStr,
              help='Filter results by the list of dataset split indices '
              '(separated by commas, wildcard is accepted).')
@click.option('--mu-list', default="*", show_default=True, cls=DelimitedStr,
              help='Filter results by the list of signal fractions. '
              '(separated by commas, wildcard is accepted).')
@click.option('--alpha-list', default="*", show_default=True, cls=DelimitedStr,
              help='Filter results by the list of branching fractions. '
              '(separated by commas, wildcard is accepted).')
@click.option('--trial-list', default="*", show_default=True, cls=DelimitedStr,
              help='Filter results by the list of trial numbers. '
              '(separated by commas, wildcard is accepted).')
@click.option('--noise-list', default="*", show_default=True, cls=DelimitedStr,
              help='Filter results by the noise dimensions. '
              '(separated by commas, wildcard is accepted).')
@click.option('--version', default="v1", show_default=True,
              help='Version of the models.')
@click.option('--topk', default=5, show_default=True,
              help='(Weakly models only) Take the top-k trials with lowest loss.')
@click.option('--score-reduce-method', default='mean', type=click.Choice(["mean", "median"]), show_default=True,
              help='(Weakly models only) How to reduce the score over the trials.')
@click.option('--weight-reduce-method', default='median', type=click.Choice(["mean", "median"]), show_default=True,
              help='(Semi-weakly model only) How to reduce the model weights '
              '(predicted parameters) over the trials.')
@click.option('--metrics', default="auc,log_loss,sic_1e3", show_default=True,
              cls=DelimitedStr, type=click.Choice(["auc", "accuracy", "log_loss", "sic_1e3", "sic_1e4", "sic_1e5"]), 
              help='List of metrics to be included in the evaluation (separated by commas). '
              'Here sic_* refers to the Significance Improvement Characteristic at a 1 / FPR value of *.')
@click.option('--detailed/--simplified', default=False, show_default=True,
              help='Whether to save also the truth and predicted y values of the model results.')
@click.option('-o', '--outdir', default=DEFAULT_OUTDIR, show_default=True,
              help='Base output directory from which model results are extracted.')
@click.option('-f', '--filename', default="{model_type}_{feature_level}_{decay_mode}.parquet", required=True, show_default=True,
              help='Output filename where the gathered results are saved (on top of outdir). Keywords like '
              'model_type, feature_level and decay_mode will be automatically formatted.')
@click.option('-v', '--verbosity',  default="INFO", show_default=True,
              help='Verbosity level ("DEBUG", "INFO", "WARNING" or "ERROR").')
def gather_model_results(**kwargs):
    """
    Gather model results.
    """
    from quickstats import stdout
    from quickstats.utils.string_utils import split_str
    from paws.components import ResultLoader
    init_kwargs = {}
    for key in ["decay_modes", "variables", "outdir", "verbosity"]:
        init_kwargs[key] = kwargs.pop(key)
    merge_kwargs = {}
    for key in ['topk', 'score_reduce_method', 'weight_reduce_method']:
        merge_kwargs[key] = kwargs.pop(key)        
    feature_level = "high_level" if kwargs.pop("high_level") else "low_level"
    metrics, detailed, filename = [kwargs.pop(key) for key in ["metrics", "detailed", "filename"]]
    result_loader = ResultLoader(feature_level=feature_level, **init_kwargs)
    kwargs["mass_points"] = [split_str(mass_point, sep=":") for mass_point in kwargs["mass_points"]]
    result_loader.load(**kwargs)
    model_type = ModelType.parse(kwargs["model_type"])
    if model_type.key not in result_loader.dfs:
        stdout.warning("No results to save. Skipping.")
        return
    if model_type in [ModelType.SEMI_WEAKLY, ModelType.IDEAL_WEAKLY]:
        result_loader.merge_trials(**merge_kwargs)
    def get_metric(name:str):
        if name.startswith('sic_'):
            threshold = 1 / float(name.replace("sic_", ""))
            return (name, 'threshold_significance', {"fpr_thres": threshold})
        return name
    metrics = [get_metric(metric) for metric in metrics]
    result_loader.decorate_results(metrics)
    format_keys = result_loader._get_param_repr()
    format_keys['model_type'] = model_type.key
    result_loader.path_manager.makedirs(["combined_result"])
    save_outdir = result_loader.path_manager.get_directory("combined_result")
    outname = os.path.join(save_outdir, filename.format(**format_keys))
    result_loader.save_parquet(outname, detailed=detailed)