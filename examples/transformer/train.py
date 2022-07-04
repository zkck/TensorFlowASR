import os
import random
from glob import glob
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class CustomSchedule(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(
        self,
        init_lr=0.00001,
        lr_after_warmup=0.001,
        final_lr=0.00001,
        warmup_epochs=15,
        decay_epochs=85,
        steps_per_epoch=203,
    ):
        super().__init__()
        self.init_lr = init_lr
        self.lr_after_warmup = lr_after_warmup
        self.final_lr = final_lr
        self.warmup_epochs = warmup_epochs
        self.decay_epochs = decay_epochs
        self.steps_per_epoch = steps_per_epoch

    def calculate_lr(self, epoch):
        """linear warm up - linear decay"""
        warmup_lr = (
            self.init_lr
            + ((self.lr_after_warmup - self.init_lr) / (self.warmup_epochs - 1)) * epoch
        )
        decay_lr = tf.math.maximum(
            self.final_lr,
            self.lr_after_warmup
            - (epoch - self.warmup_epochs)
            * (self.lr_after_warmup - self.final_lr)
            / (self.decay_epochs),
        )
        return tf.math.minimum(warmup_lr, decay_lr)

    def __call__(self, step):
        epoch = step // self.steps_per_epoch
        return self.calculate_lr(epoch)


"""
## Create & train the end-to-end model
"""

# Copyright 2020 Huy Le Nguyen (@usimarit)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import fire
import math
from tensorflow_asr.utils import env_util

logger = env_util.setup_environment()
import tensorflow as tf

from tensorflow_asr.configs.config import Config
from tensorflow_asr.helpers import featurizer_helpers, dataset_helpers

from tensorflow_asr.models.transformer import Transformer

DEFAULT_YAML = os.path.join(os.path.abspath(os.path.dirname(__file__)), "config.yml")


def main(
    config: str = DEFAULT_YAML,
    tfrecords: bool = False,
    sentence_piece: bool = False,
    subwords: bool = True,
    bs: int = None,
    spx: int = 1,
    metadata: str = None,
    static_length: bool = False,
    devices: list = [0],
    mxp: bool = False,
    pretrained: str = None,
):
    tf.keras.backend.clear_session()
    tf.config.optimizer.set_experimental_options({"auto_mixed_precision": mxp})
    strategy = env_util.setup_strategy(devices)

    config = Config(config)

    speech_featurizer, text_featurizer = featurizer_helpers.prepare_featurizers(
        config=config,
        subwords=subwords,
        sentence_piece=sentence_piece,
    )

    train_dataset, eval_dataset = dataset_helpers.prepare_training_datasets(
        config=config,
        speech_featurizer=speech_featurizer,
        text_featurizer=text_featurizer,
        tfrecords=tfrecords,
        metadata=metadata,
    )

    if not static_length:
        speech_featurizer.reset_length()
        text_featurizer.reset_length()

    train_data_loader, eval_data_loader, global_batch_size = dataset_helpers.prepare_training_data_loaders(
        config=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        strategy=strategy,
        batch_size=bs,
    )

    with strategy.scope():
        model = Transformer(
            num_hid=200,
            num_head=2,
            num_feed_forward=400,
            target_maxlen=200,
            num_layers_enc=4,
            num_layers_dec=1,
            num_classes=34,
        )
        loss_fn = tf.keras.losses.CategoricalCrossentropy(
            from_logits=True,
            label_smoothing=0.1,
        )

        learning_rate = CustomSchedule(
            init_lr=0.00001,
            lr_after_warmup=0.001,
            final_lr=0.00001,
            warmup_epochs=15,
            decay_epochs=85,
            steps_per_epoch=len(train_data_loader),
        )
        optimizer = keras.optimizers.Adam(learning_rate)
        model.compile(optimizer=optimizer, loss=loss_fn)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(**config.learning_config.running_config.checkpoint),
        tf.keras.callbacks.experimental.BackupAndRestore(config.learning_config.running_config.states_dir),
        tf.keras.callbacks.TensorBoard(**config.learning_config.running_config.tensorboard),
    ]

    model.fit(
        train_data_loader,
        epochs=config.learning_config.running_config.num_epochs,
        validation_data=eval_data_loader,
        callbacks=callbacks,
        steps_per_epoch=train_dataset.total_steps,
        validation_steps=eval_dataset.total_steps if eval_data_loader else None,
    )


if __name__ == "__main__":
    fire.Fire(main)
