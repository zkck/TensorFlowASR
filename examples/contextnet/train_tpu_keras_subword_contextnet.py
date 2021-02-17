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
import math
import argparse
from tensorflow_asr.utils import setup_environment, setup_tpu

setup_environment()
import tensorflow as tf

DEFAULT_YAML = os.path.join(os.path.abspath(os.path.dirname(__file__)), "config.yml")

tf.keras.backend.clear_session()

parser = argparse.ArgumentParser(prog="Conformer Training")

parser.add_argument("--config", type=str, default=DEFAULT_YAML, help="The file path of model configuration file")

parser.add_argument("--sentence_piece", default=False, action="store_true", help="Whether to use `SentencePiece` model")

parser.add_argument("--bs", type=int, default=None, help="Batch size per replica")

parser.add_argument("--tpu_address", type=str, default=None, help="TPU address. Leave None on Colab")

parser.add_argument("--max_lengths_prefix", type=str, default=None, help="Path to file containing max lengths")

parser.add_argument("--compute_lengths", default=False, action="store_true", help="Whether to compute lengths")

parser.add_argument("--mxp", default=False, action="store_true", help="Enable mixed precision")

parser.add_argument("--subwords", type=str, default=None, help="Path to file that stores generated subwords")

parser.add_argument("--subwords_corpus", nargs="*", type=str, default=[], help="Transcript files for generating subwords")

args = parser.parse_args()

tf.config.optimizer.set_experimental_options({"auto_mixed_precision": args.mxp})

strategy = setup_tpu(args.tpu_address)

from tensorflow_asr.configs.config import Config
from tensorflow_asr.datasets.keras import ASRTFRecordDatasetKeras
from tensorflow_asr.featurizers.speech_featurizers import TFSpeechFeaturizer
from tensorflow_asr.featurizers.text_featurizers import SubwordFeaturizer, SentencePieceFeaturizer
from tensorflow_asr.models.keras.contextnet import ContextNet
from tensorflow_asr.optimizers.schedules import TransformerSchedule

config = Config(args.config)
speech_featurizer = TFSpeechFeaturizer(config.speech_config)

if args.sentence_piece:
    print("Loading SentencePiece model ...")
    text_featurizer = SentencePieceFeaturizer.load_from_file(config.decoder_config, args.subwords)
elif args.subwords and os.path.exists(args.subwords):
    print("Loading subwords ...")
    text_featurizer = SubwordFeaturizer.load_from_file(config.decoder_config, args.subwords)
else:
    print("Generating subwords ...")
    text_featurizer = SubwordFeaturizer.build_from_corpus(
        config.decoder_config,
        corpus_files=args.subwords_corpus
    )
    text_featurizer.save_to_file(args.subwords)

train_dataset = ASRTFRecordDatasetKeras(
    speech_featurizer=speech_featurizer, text_featurizer=text_featurizer,
    **vars(config.learning_config.train_dataset_config)
)
eval_dataset = ASRTFRecordDatasetKeras(
    speech_featurizer=speech_featurizer, text_featurizer=text_featurizer,
    **vars(config.learning_config.eval_dataset_config)
)

if args.compute_lengths:
    train_dataset.update_lengths(args.max_lengths_prefix)
    eval_dataset.update_lengths(args.max_lengths_prefix)

# Update max lengths calculated from both train and eval datasets
train_dataset.load_max_lengths(args.max_lengths_prefix)
eval_dataset.load_max_lengths(args.max_lengths_prefix)

with strategy.scope():
    batch_size = args.bs if args.bs is not None else config.learning_config.running_config.batch_size
    global_batch_size = batch_size
    global_batch_size *= strategy.num_replicas_in_sync
    # build model
    contextnet = ContextNet(**config.model_config, vocabulary_size=text_featurizer.num_classes)
    contextnet._build(speech_featurizer.shape, prediction_shape=text_featurizer.prepand_shape, batch_size=global_batch_size)
    contextnet.summary(line_length=120)

    optimizer = tf.keras.optimizers.Adam(
        TransformerSchedule(
            d_model=contextnet.dmodel,
            warmup_steps=config.learning_config.optimizer_config["warmup_steps"],
            max_lr=(0.05 / math.sqrt(contextnet.dmodel))
        ),
        beta_1=config.learning_config.optimizer_config["beta1"],
        beta_2=config.learning_config.optimizer_config["beta2"],
        epsilon=config.learning_config.optimizer_config["epsilon"]
    )

    contextnet.compile(optimizer=optimizer, global_batch_size=global_batch_size, blank=text_featurizer.blank)

    train_data_loader = train_dataset.create(global_batch_size).take(10)
    eval_data_loader = eval_dataset.create(global_batch_size)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(**config.learning_config.running_config.checkpoint),
        tf.keras.callbacks.experimental.BackupAndRestore(config.learning_config.running_config.states_dir),
        tf.keras.callbacks.TensorBoard(**config.learning_config.running_config.tensorboard)
    ]

    contextnet.fit(
        train_data_loader, epochs=config.learning_config.running_config.num_epochs,
        validation_data=eval_data_loader, callbacks=callbacks,
    )