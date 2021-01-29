from zookeeper import registry, HParams
import larq as lq
import tensorflow as tf
from bnn_optimization import optimizers


@registry.register_model
def BinaryConnect(hparams, input_shape, num_classes):
    kwhparams = dict(
        input_quantizer="ste_sign",
        kernel_quantizer=hparams.kernel_quantizer,
        kernel_constraint=hparams.kernel_constraint,
        use_bias=False,
    )
    return tf.keras.models.Sequential(
        [
            # don't quantize inputs in first layer
            tf.keras.layers.Dropout(hparams.dropout_rate),
            lq.layers.QuantDense(hparams.dense_units, **kwhparams),
            tf.keras.layers.ReLU(),
            tf.keras.layers.BatchNormalization(scale=False),
            tf.keras.layers.Dropout(hparams.dropout_rate),
            lq.layers.QuantDense(hparams.dense_units, **kwhparams),
            tf.keras.layers.ReLU(),
            tf.keras.layers.BatchNormalization(scale=False),
            tf.keras.layers.Dropout(hparams.dropout_rate),
            lq.layers.QuantDense(hparams.dense_units, **kwhparams),
            tf.keras.layers.ReLU(),
            tf.keras.layers.BatchNormalization(scale=False),
            tf.keras.layers.Dropout(hparams.dropout_rate),
            lq.layers.QuantDense(hparams.dense_units, **kwhparams),
            tf.keras.layers.BatchNormalization(scale=False),
            tf.keras.layers.Activation("softmax"),
        ]
    )


@registry.register_hparams(BinaryConnect)
class default(HParams):
    epochs = 100
    dense_units = 2048
    kernel_size = 3
    batch_size = 256
    optimizer = tf.keras.optimizers.Adam(5e-3)
    kernel_quantizer = "ste_sign"
    kernel_constraint = "weight_clip"


@registry.register_hparams(BinaryConnect)
class bop_mnist(default):
    batch_size = 100
    epochs = 100
    kernel_quantizer = None
    kernel_constraint = None
    threshold = 1e-8
    gamma = 1e-5
    gamma_decay = 10 ** (-3 / 500)
    decay_step = int((54000 / 100) * 1)
    dropout_rate = 0.2

    @property
    def optimizer(self):
        return optimizers.Bop(
            fp_optimizer=tf.keras.optimizers.Adam(0.01),
            threshold=self.threshold,
            gamma=tf.keras.optimizers.schedules.ExponentialDecay(
                self.gamma, self.decay_step, self.gamma_decay, staircase=True
            ),
        )
