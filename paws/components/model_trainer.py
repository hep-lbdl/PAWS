from typing import Union, Optional, Dict, List
import os
import glob
import json
import time
import shutil

import numpy as np

from quickstats import AbstractObject
from quickstats.utils.common_utils import combine_dict, NpEncoder
from quickstats.utils.string_utils import split_str

from paws.components import DataLoader, ModelLoader
from paws.settings import (
    ModelType, DEDICATED_SUPERVISED, PARAM_SUPERVISED, IDEAL_WEAKLY, SEMI_WEAKLY,
    NUM_JETS, BASE_SEED, WEIGHT_CLIPPING, RETRAIN, NUM_TRIALS,
    DEFAULT_FEATURE_LEVEL, DEFAULT_DECAY_MODE, DEFAULT_DATADIR, DEFAULT_OUTDIR,
    INIT_MU, INIT_ALPHA, MASS_RANGE, MASS_INTERVAL, MASS_SCALE, PRIOR_RATIO,
    SAMPLING_METHOD, INIT_KAPPA
)

MODEL_OPTIONS = {
    DEDICATED_SUPERVISED: {
        'required': ['mass_point'],
        'optional': []
    },
    PARAM_SUPERVISED: {
        'required': [],
        'optional': ['include_masses', 'exclude_masses']
    },
    IDEAL_WEAKLY: {
        'required': ['mass_point', 'mu', 'alpha'],
        'optional': ['num_trials']
    },
    SEMI_WEAKLY: {
        'required': ['mass_point', 'mu', 'alpha'],
        'optional': ['num_trials', 'weight_clipping', 'retrain', 'kappa', 'fs_version', 'fs_version_2']
    },
    PRIOR_RATIO: {
        'required': [],
        'optional': ['sampling_method', 'param_expr', 'cache_dataset']
    }
}

class ModelTrainer(AbstractObject):
    """
    Class for training models of various machine learning methods.
    """

    def __init__(self, model_type: Union[str, ModelType],
                 model_options: Optional[Dict] = None,
                 feature_level: str = DEFAULT_FEATURE_LEVEL,
                 decay_modes: str = DEFAULT_DECAY_MODE,
                 cache: bool = True,
                 variables: Optional[str] = None,
                 noise_dimension: Optional[int] = None,
                 seed: int = BASE_SEED,
                 loss: str = 'bce',
                 split_index: int = 0,
                 batchsize: Optional[int] = None,
                 cache_dataset: Optional[bool] = None,
                 cache_test: bool = False,
                 version: str = 'v1',
                 multi_gpu: bool = True,
                 interrupt_freq: int = 0,
                 datadir: str = DEFAULT_DATADIR,
                 outdir: str = DEFAULT_OUTDIR,
                 index_path: Optional[str] = None,
                 verbosity: str = 'INFO'):
        """
        Initialize the ModelTrainer class.
        
        Parameters
        ----------------------------------------------------
        model_type : str or ModelType
            The type of the model to train.
        model_options : Dict, optional
            Options specific to the model type.
        feature_level : str or FeatureLevel, default "high_level"
            Features to use for the training. It can be either
            high-level ("high_level") or low-level ("low_level").
        decay_modes : str, list of str or list of DecayMode, default "qq,qqq"
            Decay modes of the signal to include in the training. Candidates are
            two-prong decay ("qq") or three-prong decay ("qqq"). If it is a
            string, it will be a comma delimited list of the decay modes.
        cache : bool, default True
            Whether to cache the results.
        variables : str, optional
            Select certain high-level jet features to include in the training
            by the indices they appear in the feature vector. For example,
            "3,5,6" means select the 4th, 6th and 7th feature from the jet
            feature vector to be used in the training.
        noise_dimension : int, optional
            Number of noise dimension per jet to include in the training.
        loss : str, default "bce"
            Name of the loss function. Choose between "bce" (binary
            cross entropy) and "nll" (negative log-likelihood). Note
            that nll loss is only allowed for semi-weakly models.
        seed : int, optional, default 2023
            The default seed used for all random processes.
        split_index : int
            Index for dataset split.
        batchsize : int
            Batch size for training.
        cache_dataset : bool
            Whether to cache the train and validation datasets.
        cache_test : bool
            Whether to cache the test dataset.
        version : str
            Version of the model.
        multi_gpu : bool, default True
            Whether to enable multi-GPU training.
        interrupt_freq : int, default 0
            Frequency of training interruption for early stopping.
        datadir : str, default "datasets"
            Directory for datasets.
        outdir : str, default "outputs"
            Directory for outputs.
        index_path : str, optional
            Path to the dataset split configuration file. It determines the
            shard indices for the train, validation, and test datasets in each
            random realization of data.
        verbosity : str, default "INFO"
            Verbosity level ("DEBUG", "INFO", "WARNING" or "ERROR").
        """
        super().__init__(verbosity=verbosity)

        self.model_type = ModelType.parse(model_type)
        self.base_config = {
            'feature_level': feature_level,
            'decay_modes': decay_modes,
            'variables': variables,
            'noise_dimension': noise_dimension,
            'seed': seed
        }
        self.cache = cache
        self.loss = loss
        self.batchsize = batchsize
        self.cache_dataset = cache_dataset
        self.cache_test = cache_test
        self.version = version
        self.split_index = split_index
        self.interrupt_freq = interrupt_freq
        self.datadir = datadir
        self.outdir = outdir
        self.index_path = index_path
        self.distribute_strategy = self.get_default_distribute_strategy() if multi_gpu else None

        self.initial_check()
        self.init_data_loader()
        self.init_model_loader()
        self.mixed_signal = len(self.model_loader.decay_modes) > 1
        self.set_model_options(model_options)

    @property
    def path_manager(self):
        return self.model_loader.path_manager
        
    def initial_check(self) -> None:
        import tensorflow as tf
        import aliad as ad

        # check software version
        self.stdout.info(f'     aliad version : {ad.__version__}')
        self.stdout.info(f'tensorflow version : {tf.__version__}')
        
        # check GPU setups
        if shutil.which("nvidia-smi") is not None:
            os.system("nvidia-smi")
        if shutil.which("nvcc") is not None:
            os.system("nvcc --version")
        
        physical_devices = tf.config.list_physical_devices('GPU')
        try:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
        except:
            # Invalid device or cannot modify virtual devices once initialized.
            pass

    def get_default_distribute_strategy(self, **kwargs) -> "tf.distribute.Strategy":
        import tensorflow as tf
        strategy = tf.distribute.MirroredStrategy(**kwargs)
        self.stdout.info("Created MirroredStrategy for distributed training")
        self.stdout.info(f"Number of devices : {strategy.num_replicas_in_sync}")
        return strategy

    def init_data_loader(self) -> None:
        self.data_loader = DataLoader(datadir=self.datadir, index_path=self.index_path,
                                      verbosity=self.stdout.verbosity, **self.base_config)

    def init_model_loader(self) -> None:
        self.model_loader = ModelLoader(distribute_strategy=self.distribute_strategy, datadir=self.datadir,
                                        outdir=self.outdir, verbosity=self.stdout.verbosity, loss=self.loss,
                                        **self.base_config)

    def set_model_options(self, options: Optional[Dict] = None) -> None:
        """
        Set the options used in model training.

        Parameters
        ----------------------------------------------------
        options : dict, optional
            Options specific to the model type.
        """
        options = combine_dict(options)
        keys = list(options)
        required_keys = MODEL_OPTIONS[self.model_type]['required']
        optional_keys = MODEL_OPTIONS[self.model_type]['optional']

        for key in required_keys:
            if key not in keys:
                raise ValueError(f'Missing required model option: {key}')

        for key in keys:
            if key not in required_keys and key not in optional_keys:
                raise ValueError(f'Unknown model option: {key}')

        if ('mass_point' in options) and isinstance(options['mass_point'], str):
            options['mass_point'] = split_str(options['mass_point'], sep=':', cast=int)

        def parse_masses_expr(expr: str):
            return [split_str(mass_expr, sep=":", cast=int) for mass_expr in split_str(expr, sep=",")]

        for key in ['include_masses', 'exclude_masses']:
            if (key in options) and isinstance(options[key], str):
                options[key] = parse_masses_expr(options[key])

        if self.model_type in [IDEAL_WEAKLY, SEMI_WEAKLY]:
            options.setdefault('num_trials', NUM_TRIALS)
            if self.mixed_signal and (options['alpha'] is None):
                raise ValueError('Branching ratio cannot be None when multiple decay modes are involved in weakly supervised model training.')
            if not self.mixed_signal:
                options['alpha'] = None
                
        if self.model_type == SEMI_WEAKLY:
            parameters = {
                'model_type': PARAM_SUPERVISED,
                'mass_point': options['mass_point'],
                'split_index': self.split_index
            }
            basename = self.model_loader.path_manager.get_basename('model_full_train')
            options["fs_version"] = options.get("fs_version", None) or self.version
            decay_modes = self.model_loader.decay_modes
            fs_checkpoint_dir = self.model_loader.get_checkpoint_dir(version=options['fs_version'], **parameters,
                                                                     decay_modes=[decay_modes[0]])
            options['fs_model_path'] = os.path.join(fs_checkpoint_dir, basename)
            if self.mixed_signal:
                options["fs_version_2"] = options.get("fs_version_2", None) or options['fs_version']
                fs_checkpoint_dir_2 = self.model_loader.get_checkpoint_dir(version=options['fs_version_2'], **parameters,
                                                                           decay_modes=[decay_modes[1]])
                options['fs_model_path_2'] = os.path.join(fs_checkpoint_dir_2, basename)
            else:
                options['fs_model_path_2'] = None
            options.setdefault('weight_clipping', WEIGHT_CLIPPING)
            options.setdefault('retrain', RETRAIN)
            options.setdefault('kappa', INIT_KAPPA)

        if self.model_type == PRIOR_RATIO:
            options.setdefault('sampling_method', SAMPLING_METHOD)

        self.model_options = options

    def get_datasets(self) -> "tf.data.Dataset":
        """
        Get the datasets for training, validation, and testing.

        The datasets corresponding to the correct model type and configuration will be generated.

        Returns
        ----------------------------------------------------
        tf.data.Dataset
            The datasets.
        """        
        kwargs = {
            'split_index': self.split_index,
            'batchsize': self.batchsize,
            'cache_dataset': self.cache_dataset,
            'cache_test': self.cache_test
        }
        
        if self.model_type in [DEDICATED_SUPERVISED, IDEAL_WEAKLY, SEMI_WEAKLY]:
            kwargs['mass_point'] = self.model_options['mass_point']
            
        if self.model_type == PARAM_SUPERVISED:
            for key in ['include_masses', 'exclude_masses']:
                kwargs[key] = self.model_options.get(key, None)
                
        if self.model_type in [IDEAL_WEAKLY, SEMI_WEAKLY]:
            kwargs['mu'] = self.model_options['mu']
            kwargs['alpha'] = self.model_options['alpha']

        if self.model_type == PRIOR_RATIO:
            kwargs['mass_point'] = [MASS_RANGE[0], MASS_RANGE[0]]
            kwargs['samples'] = ['qcd', 'extra_qcd']
            
        return self.data_loader.get_datasets(**kwargs)

    def get_feature_metadata(self) -> Dict:
        feature_level = self.base_config['feature_level']
        dirname = self.path_manager.get_directory('dedicated_dataset', feature_level=feature_level, sample='*', m1='*', m2='*')
        metadata_filenames = glob.glob(os.path.join(dirname, "*_metadata.json"))
        with open(metadata_filenames[0], 'r') as file:
            metadata = json.load(file)
        return metadata['features']

    def get_model(self) -> "tf.keras.Model":
        """
        Get the model to train based on the model type.

        Returns
        ----------------------------------------------------
        tf.keras.Model
            The model.
        """        
        feature_metadata = self.get_feature_metadata()
        if self.model_type in [DEDICATED_SUPERVISED, IDEAL_WEAKLY]:
            return self.model_loader.get_supervised_model(feature_metadata, parametric=False)
        if self.model_type == PARAM_SUPERVISED:
            return self.model_loader.get_supervised_model(feature_metadata, parametric=True)
        if self.model_type == SEMI_WEAKLY:
            return self.model_loader.get_semi_weakly_model(feature_metadata,
                                                           fs_model_path=self.model_options['fs_model_path'],
                                                           fs_model_path_2=self.model_options['fs_model_path_2'],
                                                           kappa=self.model_options['kappa'])
        if self.model_type == PRIOR_RATIO:
            return self.model_loader.get_prior_ratio_model(feature_metadata)
        raise RuntimeError(f'Unknown model type: {self.model_type}')

    def get_checkpoint_dir(self, trial: Optional[int] = None) -> str:
        """
        Get the directory to save model checkpoints.

        Parameters
        ----------------------------------------------------
        trial : int, optional
            Trial index for the model training (weakly and semi-weakly models only).

        Returns
        ----------------------------------------------------
        str
            The directory for saving model checkpoints.
        """
        parameters = {
            'split_index': self.split_index,
            'trial': trial,
            'version': self.version
        }
        if self.model_type in [DEDICATED_SUPERVISED, IDEAL_WEAKLY, SEMI_WEAKLY]:
            parameters['mass_point'] = self.model_options['mass_point']

        if self.model_type in [IDEAL_WEAKLY, SEMI_WEAKLY]:
            parameters['mu'] = self.model_options['mu']
            parameters['alpha'] = self.model_options['alpha']

        if self.model_type == PRIOR_RATIO:
            return self.model_loader.get_checkpoint_dir(PARAM_SUPERVISED, **parameters)
            
        return self.model_loader.get_checkpoint_dir(self.model_type, **parameters)

    def get_model_path(self, trial: Optional[int] = None) -> str:
        checkpoint_dir = self.get_checkpoint_dir(trial=trial)
        if self.model_type == PRIOR_RATIO:
            sampling_method = self.model_options['sampling_method']
            basename = self.model_loader.path_manager.get_file("model_prior_ratio", basename_only=True,
                                                               sampling_method=sampling_method)
            return os.path.join(checkpoint_dir, basename)
        return self.model_loader.get_checkpoint_path('model_full_train', checkpoint_dir)

    def load_trained_model(self, trial: Optional[int] = None) -> str:
        model_path = self.get_model_path(trial=trial)
        model = self.model_loader.load_model(model_path)
        self.stdout.info(f'Loaded model from {model_path}')
        return model

    def get_prior_ratio_dataset(self, sampling_method:str='sampled',
                                param_expr: Optional[str]=None,
                                cache_dataset: bool = True):
        # inferred, sampled
        from paws.components import MetricLandscape
        import pandas as pd
        sampling_method = sampling_method.lower()
        checkpoint_dir = self.get_checkpoint_dir()
        basename = self.model_loader.path_manager.get_file("prior_ratio_dataset",
                                                           basename_only=True,
                                                           sampling_method=sampling_method)
        prior_ratio_dataset_path = os.path.join(checkpoint_dir, basename)
        if cache_dataset and os.path.exists(prior_ratio_dataset_path):
            data = json.load(open(prior_ratio_dataset_path))
            self.stdout.info(f'Cached prior ratio dataset from "{prior_ratio_dataset_path}"')
            return np.array(data['x']), np.array(data['y'])
        if sampling_method == 'sampled':
            basename = self.model_loader.path_manager.get_file("model_full_train",
                                                               basename_only=True)
            supervised_model_path = os.path.join(checkpoint_dir, basename)
            if not os.path.exists(supervised_model_path):
                raise FileNotFoundError(f'parametersed model file does not exist: {supervised_model_path}')
            supervised_model = self.model_loader.load_model(supervised_model_path)
            datasets = self.get_datasets()
            landscape = MetricLandscape()
            if param_expr is None:
                param_expr = (f"m1={MASS_RANGE[0]}_{MASS_RANGE[1]}_{MASS_INTERVAL},"
                              f"m2={MASS_RANGE[0]}_{MASS_RANGE[1]}_{MASS_INTERVAL}")
            output = landscape.eval_supervised(supervised_model, datasets['train'],
                                               param_expr=param_expr,
                                               metrics=['prior_ratio'])['predictions']
            result = {
                'x': np.column_stack((output['m1'], output['m2'])),
                'y': output['prior_ratio']
            }
        elif sampling_method == 'inferred':
            from quickstats.maths.numerics import cartesian_product
            unique_masses =np.arange(MASS_RANGE[0], MASS_RANGE[1] + MASS_INTERVAL, MASS_INTERVAL)
            unique_masses = unique_masses.astype('float64')
            mass_points = cartesian_product(unique_masses, unique_masses)
            result = {
                'x': [],
                'y': []
            }
            for mass_point in mass_points:
                result['x'].append(mass_point)
                df = self.data_loader.get_dataset_specs(mass_point)
                counts = df.groupby('sample')['size'].sum().to_dict()
                sig_count = np.sum([count for key, count in counts.items() if 'qcd' not in key.lower()])
                bkg_count = np.sum([count for key, count in counts.items() if 'qcd' in key.lower()])
                ratio = bkg_count / sig_count
                result['y'].append(ratio)
        else:
            raise ValueError(f'invalid sampling method: {sampling_method}')
        from quickstats.utils.common_utils import NpEncoder
        with open(prior_ratio_dataset_path, 'w') as outfile:
            json.dump(result, outfile, cls=NpEncoder)
        self.stdout.info(f'Saved prior ratio dataset as "{prior_ratio_dataset_path}"')
        return np.array(result['x']), np.array(result['y'])

    def print_train_summary(self) -> None:
        """
        Print a summary of the training configuration.
        """
        model_str = self.model_type.key.replace("_", " ").capitalize()
        param_components = []
        if self.model_type in [DEDICATED_SUPERVISED, IDEAL_WEAKLY, SEMI_WEAKLY]:
            m1, m2 = self.model_options['mass_point']
            param_components.extend([("m1", str(m1)), ("m2", str(m2))])
        if self.model_type in [IDEAL_WEAKLY, SEMI_WEAKLY]:
            mu = self.model_options['mu']
            param_components.append(("mu", f'{mu:.3g}'))
            alpha = self.model_options['alpha']
            if alpha is not None:
                param_components.append(("alpha", f'{alpha:.3g}'))
        if param_components:
            param_str = ("with " + "(" + ", ".join([c[0] for c in param_components]) + 
                         ") = (" + ", ".join([c[1] for c in param_components]) + ")")
        else:
            param_str = ""
        param_repr = self.model_loader._get_param_repr()
        feature_level_str = param_repr['feature_level'].replace("_", " ")
        decay_mode_str = param_repr['decay_mode'].replace("_", ", ")
        noise_dim_str = param_repr['noise_dim']
        var_str = param_repr['variables'].replace("_", ", ")

        if self.model_type == SEMI_WEAKLY:
            if self.mixed_signal:
                fs_model_paths = [self.model_options['fs_model_path'], self.model_options['fs_model_path_2']]
                decay_modes = [self.model_loader.decay_modes[0], self.model_loader.decay_modes[1]]
            else:
                fs_model_paths = [self.model_options['fs_model_path']]
                decay_modes = [self.model_loader.decay_modes[0]]
            extra_str = "\n".join([f'{decay_mode.name.replace("_", " ").capitalize()} supervised model path: {path}' \
                                  for decay_mode, path in zip(decay_modes, fs_model_paths)]) + "\n"
        else:
            extra_str = ""

        summary_str = ("\n##############################################################\n"
                      f"{model_str} training {param_str}                              \n"
                      f"Decay mode(s): {decay_mode_str}                               \n"
                      f"Feature level: {feature_level_str}                            \n"
                      f"Jet feature indices: {var_str}                                \n"
                      f"Noise dimension: {noise_dim_str}                              \n"
                      f"Dataset directory: {self.datadir}                             \n"
                      f"Output directory: {self.outdir}                               \n"
                      f"{extra_str}"
                       "##############################################################\n")
        self.stdout.info(summary_str, bare=True)

    def train(self) -> None:
        t0 = time.time()
        
        # random dataset initialization determined by the split index
        if self.model_type != PRIOR_RATIO:
            datasets = self.get_datasets()
            dataset_summary = self.data_loader.dataset_summary
        else:
            datasets = None
            dataset_summary = None
        
        if self.model_type in [DEDICATED_SUPERVISED, PARAM_SUPERVISED, PRIOR_RATIO]:
            trials = np.arange(1)
        else:
            trials = np.arange(self.model_options['num_trials'])
            
        self.print_train_summary()

        if self.model_type == SEMI_WEAKLY:
            seed = 1000 * int(1 / self.model_options['mu']) + self.split_index
            size = (self.model_options['num_trials'], NUM_JETS)
            low, high = MASS_RANGE[0] * MASS_SCALE, MASS_RANGE[1] * MASS_SCALE
            np.random.seed(seed)
            random_masses = np.random.uniform(low=low, high=high, size=size).astype('float32')
        else:
            random_masses = None
            
        y_true = None
        # run over random model parameter initialization (same dataset used)
        for trial in trials:
            self.stdout.info(f"Trial ({trial + 1} / {len(trials)})")
            if self.model_type == PRIOR_RATIO:
                model, normalizer = self.get_model()
            else:
                model = self.get_model()
            checkpoint_dir = self.get_checkpoint_dir(trial=trial)
            os.makedirs(checkpoint_dir, exist_ok=True)
            weight_clipping = self.model_options['weight_clipping'] if self.model_type == SEMI_WEAKLY else False
            train_config = self.model_loader.get_train_config(checkpoint_dir, model_type=self.model_type, weight_clipping=weight_clipping)
            if self.interrupt_freq:
                train_config['callbacks']['early_stopping']['interrupt_freq'] = self.interrupt_freq
                
            config_savepath = self.model_loader.get_checkpoint_path('train_config', checkpoint_dir)
            with open(config_savepath, 'w') as file:
                json.dump(train_config, file, indent=2, default=lambda o: o.name)

            if dataset_summary is not None:
                ds_summary_path = self.model_loader.get_checkpoint_path('dataset_summary', checkpoint_dir)
                with open(ds_summary_path, 'w') as file:
                    json.dump(dataset_summary, file, indent=2)

            callbacks_map = self.model_loader.get_callbacks(self.model_type, config=train_config)
            early_stopping = callbacks_map.get('early_stopping', None)
            callbacks = list(callbacks_map.values())

            self.model_loader.compile_model(model, train_config)
            
            if self.model_type == SEMI_WEAKLY:
                init_m1, init_m2, init_mu = random_masses[trial][0], random_masses[trial][1], INIT_MU
                init_alpha = INIT_ALPHA if self.mixed_signal else None
                self.model_loader.set_semi_weakly_model_weights(model, m1=init_m1, m2=init_m2, mu=init_mu, alpha=init_alpha)
                weight_str = f"m1 = {init_m1}, m2 = {init_m2}, mu = {init_mu}"
                if init_alpha is not None:
                    weight_str += f", alpha = {init_alpha}"
                self.stdout.info(f'--------------------------------------------------------------', bare=True)
                self.stdout.info(f'Initial Weights: {weight_str}')

            if self.interrupt_freq and (early_stopping is not None):
                self.model_loader.restore_model(early_stopping, model, checkpoint_dir)

            t1 = time.time()

            # run model training
            model_savepath = self.get_model_path(trial=trial)
            if os.path.exists(model_savepath) and self.cache:
                self.stdout.info(f'Cached model from "{model_savepath}"')
                model = self.model_loader.load_model(model_savepath)
                new_path = os.path.splitext(model_savepath)[0] + "_new.keras"
                model.save(new_path)
            else:
                if self.model_type != PRIOR_RATIO:
                    model.fit(datasets['train'],
                              validation_data=datasets['val'],
                              epochs=train_config['epochs'],
                              initial_epoch=early_stopping.initial_epoch,
                              callbacks=callbacks)
                else:
                    x, y = self.get_prior_ratio_dataset(**self.model_options)
                    normalizer.adapt(x)
                    model.fit(x, y, batch_size=32,
                              epochs=train_config['epochs'],
                              callbacks=callbacks,
                              validation_split=0.2)
                
                if (early_stopping is not None) and early_stopping.interrupted:
                    self.stdout.info('Training interrupted!')
                    return None

                if self.model_type == SEMI_WEAKLY:
                    self._semi_weakly_mass_symmetry_check(model, datasets, train_config=train_config, callbacks_map=callbacks_map)
                    predicted_params = self.model_loader.get_semi_weakly_model_weights(model)
                    param_savepath = self.model_loader.get_checkpoint_path('predicted_parameters', checkpoint_dir)
                    with open(param_savepath, 'w') as file:
                        json.dump(predicted_params, file, indent=2, cls=NpEncoder)
               
                model.save(model_savepath)
                self.stdout.info(f'Saved model as "{model_savepath}"')

            t2 = time.time()
            self.stdout.info(f'Finished training! Training time = {t2 - t1:.3f}s.')

            # release memory
            if (trial == trials[-1]) and (datasets is not None):
                for ds_type in datasets:
                    if ds_type != 'test':
                        datasets[ds_type] = None

            if self.model_type != PRIOR_RATIO:
                test_result_savepath = self.model_loader.get_checkpoint_path('test_result', checkpoint_dir)
                if os.path.exists(test_result_savepath) and self.cache:
                    self.stdout.info(f'Cached test results from "{test_result_savepath}".')
                else:
                    test_results = self.evaluate(model, datasets['test'], y_true=y_true)
                    y_true = test_results['y_true'] if y_true is None else y_true
                    with open(test_result_savepath, 'w') as file:
                        json.dump(test_results, file, cls=NpEncoder)
                t3 = time.time()
                self.stdout.info(f'Finished prediction! Test time = {t3 - t2:.3f}s.')

        total_time = time.time() - t0
        self.stdout.info(f"Task finished! Total time taken = {total_time:.3f}s.")

    def _semi_weakly_mass_symmetry_check(self, model: "tf.keras.Model",
                                         datasets: Dict[str, "tf.data.Dataset"],
                                         train_config: Dict,
                                         callbacks_map: Dict) -> None:
        predicted_params = self.model_loader.get_semi_weakly_model_weights(model)
        pred_m1, pred_m2 = predicted_params['m1'], predicted_params['m2']
        current_loss = model.evaluate(datasets['val'])[0]
        self.model_loader.set_semi_weakly_model_weights(model, m1=pred_m2, m2=pred_m1)
        flipped_loss = model.evaluate(datasets['val'])[0]
        if flipped_loss >= current_loss:
            self.model_loader.set_semi_weakly_model_weights(model, m1=pred_m1, m2=pred_m2)
            return
        self.stdout.info('Improved loss upon mass flipping: '
                         f'(m1: {pred_m1} -> {pred_m2}, m2: {pred_m2} -> {pred_m1}, loss: {current_loss} -> {flipped_loss}).')
        if self.model_options['retrain']:
            early_stopping = callbacks_map['early_stopping']
            early_stopping.reset()
            callbacks = list(callbacks_map.values())
            self.stdout.info('Retrain with flipped mass '
                             f'(m1: {pred_m1} -> {pred_m2}, m2: {pred_m2} -> {pred_m1}, loss: {current_loss} -> {flipped_loss}).')
            initial_epoch = early_stopping.final_epoch + 1
            epochs = initial_epoch + train_config['epochs']
            model.fit(datasets['train'], validation_data=datasets['val'], epochs=epochs,
                      initial_epoch=initial_epoch, callbacks=callbacks)

    def evaluate(self, model: "tf.keras.Model",
                 dataset: "tf.data.Dataset",
                 y_true: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        predicted_proba = np.concatenate(model.predict(dataset)).flatten()
        if y_true is None:
            # NB: Assuming no event weight
            y_true = np.concatenate([y for _, y in dataset]).flatten()
        
        results = {
            'predicted_proba': predicted_proba,
            'y_true': y_true
        }
        
        if self.model_type == PARAM_SUPERVISED:
            masses = np.concatenate([x[-1] for x, _ in dataset])
            masses = masses.reshape([-1, 2])
            results['m1'] = masses[:, 0]
            results['m2'] = masses[:, 1]

        return results