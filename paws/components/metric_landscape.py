from typing import Optional, Dict, List
from itertools import repeat
import numpy as np

from aliad.components import ModelOutput
from quickstats import AbstractObject, semistaticmethod
from quickstats.utils.common_utils import execute_multi_tasks

from paws.components.model_loader import ModelLoader
from paws.settings import MASS_SCALE

class MetricLandscape(AbstractObject):
    """
    Tool for evaluating metric landscapes of supervised/semi-weakly models.

    Metrics can be loss, auc or accuracy.
    """

    def __init__(self, verbosity: Optional[str] = 'INFO'):
        super().__init__(verbosity=verbosity)

    @semistaticmethod
    def _get_y_true(self, dataset):
        data = list(dataset.take(1))
        if len(data[0]) == 2:  # X, Y
            y_index = -1
        elif len(data[0]) == 3:  # X, Y, weight
            y_index = -2
        else:
            raise RuntimeError('Invalid dataset format')
        y_true = np.concatenate([data[y_index] for data in dataset]).flatten()
        return y_true

    def predict_semiweakly(self, model, dataset,
                           mass_points: List[List[float]],
                           mu_points: Optional[List[float]]=None,
                           alpha_points: Optional[List[float]] = None) -> Dict:
        """
        Get prediction from a semi-weakly model for the given dataset.

        Parameters
        ----------
        model : keras.Model
            The semi-weakly model from which the predictions are made.
        dataset : tf.data.Dataset
            The dataset used for prediction.
        mass_points : list of list of floats
            The list of mass points (m1, m2) for which the landscapes are profiled over.
        mu_points : list of floats, optional
            The list of signal fractions to scan over. If None, the initial value of
            the model will be used.
        alpha_points : list of floats, optional
            The list of branching fractions to scan over. Only used when evaluating
            mixed two-prong + three-prong model.  If None, the initial value of
            the model will be used.

        Returns
        -------
        result : dict
            A dictionary of the predicted outputs and the truth labels.
        """
        init_weights = ModelLoader.get_semi_weakly_model_weights(model)
        if ('m1' not in init_weights) or ('m2' not in init_weights):
            raise ValueError('Mass parameters not found in model weights. Please '
                             'double check your model.')
        if ('alpha' not in init_weights) and (alpha_points is not None):
            raise ValueError('Cannot scan over alpha points: semi-weakly model '
                             'does not contain alpha as trainable weights.')
        if ('mu' in init_weights) and (mu_points is None):
            mu_points = [init_weights['mu']]
        if ('alpha' in init_weights) and (alpha_points is None):
            alpha_points = [init_weights['alpha']]
        if mu_points is None:
            mu_points = [None]
        if alpha_points is None:
            alpha_points = [None]

        outputs = {
            'predictions': {
                'm1': [],
                'm2': [],
                'mu': [],
                'alpha': [],
                'y_pred': []
            }
        }
        outputs['y_true'] = self._get_y_true(dataset)

        for mu in mu_points:
            for alpha in alpha_points:
                for mass_point in mass_points:
                    m1, m2 = mass_point
                    self.stdout.info(f"Running model prediction with (m1, m2, mu) = ({m1}, {m2}, {mu})")
                    weights = {
                        'm1': m1 * MASS_SCALE,
                        'm2': m2 * MASS_SCALE,
                        'mu': np.log(mu),
                        'alpha': alpha
                    }
                    ModelLoader.set_model_weights(model, weights)
                    y_pred = model.predict(dataset).flatten()
                    outputs['predictions']['m1'].append(m1)
                    outputs['predictions']['m2'].append(m2)
                    outputs['predictions']['mu'].append(mu)
                    outputs['predictions']['alpha'].append(alpha)
                    outputs['predictions']['y_pred'].append(y_pred)
        if None in alpha_points:
            outputs['predictions'].pop('alpha')
        if None in mu_points:
            outputs['predictions'].pop('mu')
        for key in outputs['predictions']:
            outputs['predictions'][key] = np.array(outputs['predictions'][key])

        return outputs

    def predict_supervised(self, model, dataset,
                           mass_points: List[List[float]]) -> Dict:
        """
        Get prediction from a supervised model for the given dataset.

        Parameters
        ----------
        model : keras.Model
            The supervised model from which the predictions are made.
        dataset : tf.data.Dataset
            The dataset used for prediction.
        mass_points : list of list of floats
            The list of mass points (m1, m2) for which the landscapes are profiled over.

        Returns
        -------
        result : dict
            A dictionary of the predicted outputs and the truth labels.
        """
        import tensorflow as tf
        from tensorflow.keras import Input
        weights = ModelLoader.get_semi_weakly_weights(m1=0., m2=0.)
        features = list(dataset.take(1))[0][0]
        train_inputs = [Input(shape=feature.shape[1:], dtype=feature.dtype) for feature in features]
        dummy_input = tf.ones_like(tf.keras.layers.Flatten()(train_inputs[0]))[:, 0]
        m1_out = weights['m1'](dummy_input)
        m2_out = weights['m2'](dummy_input)
        mass_params = tf.keras.layers.concatenate([m1_out, m2_out])
        model_inputs = [inpt for inpt in train_inputs]
        model_inputs.append(mass_params)
        ModelLoader.freeze_all_layers(model)
        model_out = model(model_inputs)
        param_model = tf.keras.Model(inputs=train_inputs, outputs=model_out)

        outputs = {
            'predictions': {
                'm1': [],
                'm2': [],
                'y_pred': []
            }
        }
        outputs['y_true'] = self._get_y_true(dataset)

        for mass_point in mass_points:
            m1, m2 = mass_point
            self.stdout.info(f"Running model prediction with (m1, m2) = ({m1}, {m2})")
            weights = {
                'm1': m1,
                'm2': m2
            }
            ModelLoader.set_model_weights(param_model, weights)
            y_pred = param_model.predict(dataset)
            outputs['predictions']['m1'].append(m1)
            outputs['predictions']['m2'].append(m2)
            outputs['predictions']['y_pred'].append(y_pred)

        for key in outputs['predictions']:
            outputs['predictions'][key] = np.array(outputs['predictions'][key])

        return outputs

    @staticmethod
    def _evaluate(metric: str, y_pred: np.ndarray, y_true: np.ndarray, label: Optional[int] = None):
        if label is not None:
            mask = y_true == label
            y_true = y_true[mask]
            y_pred = y_pred[mask]
        output = ModelOutput(y_true=y_true, y_score=y_pred)
        return output.get(metric)

    @semistaticmethod
    def _get_metric_landscape(self, data, metric: str,
                              label: Optional[int] = None,
                              parallel: int = -1):
        y_true = data['y_true']
        predictions = data['predictions']
        # check if awkward array
        if hasattr(predictions, 'fields'):
            parameters = [key for key in predictions.fields if key != 'y_pred']
        else:
            parameters = [key for key in predictions.keys() if key != 'y_pred']
        num_points = len(predictions[parameters[0]])
        landscape = []
        metric_values = execute_multi_tasks(self._evaluate, repeat(metric),
                                            predictions['y_pred'], repeat(y_true),
                                            repeat(label), parallel=parallel)
        for i in range(num_points):
            point = {param: predictions[param][i] for param in parameters}
            point[metric] = metric_values[i]
            landscape.append(point)
        return landscape

    @semistaticmethod
    def get_loss_landscape(self, data, label: Optional[int] = None, parallel: int = -1):
        return self._get_metric_landscape(data, 'log_loss', label=label, parallel=parallel)

    @semistaticmethod
    def get_auc_landscape(self, data, label: Optional[int] = None, parallel: int = -1):
        return self._get_metric_landscape(data, 'auc', label=label, parallel=parallel)

    @semistaticmethod
    def get_accuracy_landscape(self, data, label: Optional[int] = None, parallel: int = -1):
        return self._get_metric_landscape(data, 'accuracy', label=label, parallel=parallel)