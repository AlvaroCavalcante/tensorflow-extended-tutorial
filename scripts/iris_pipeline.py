from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import os 
from typing import Text

from tfx.components import CsvExampleGen

from tfx.components import StatisticsGen
from tfx.components import SchemaGen 
from tfx.components import ExampleValidator

from tfx.components import Transform

from tfx.components import Trainer
from tfx.proto import trainer_pb2
import tensorflow_model_analysis as tfma

from tfx.components.base import executor_spec
from tfx.components.trainer.executor import GenericExecutor
from tfx.orchestration import metadata
from tfx.orchestration import pipeline
from tfx.orchestration.airflow.airflow_dag_runner import AirflowDagRunner
from tfx.orchestration.airflow.airflow_dag_runner import AirflowPipelineConfig
from tfx.types import Channel
from tfx.types.standard_artifacts import Model
from tfx.types.standard_artifacts import ModelBlessing
from tfx.utils.dsl_utils import external_input

_pipeline_name = 'iris'

_project_root = os.path.join(os.environ['HOME'], 'airflow')
_data_root = os.path.join(_project_root, 'data', 'iris_data')

_module_file = os.path.join(_project_root, 'dags', 'iris_utils.py')

_serving_model_dir = os.path.join(_project_root, 'serving_model', _pipeline_name)

_tfx_root = os.path.join(_project_root, 'tfx') # root dos arquivos do TFX (metadata e pipe)
_pipeline_root = os.path.join(_tfx_root, 'pipelines', _pipeline_name)
_metadata_path = os.path.join(_tfx_root, 'metadata', _pipeline_name, 'metadata.db')

_airflow_config = {
    'schedule_interval': None,
    'start_date': datetime.datetime(2019, 1, 1),
}


def _create_pipeline(pipeline_name, pipeline_root, data_root,
                     module_file, serving_model_dir,
                     metadata_path, direct_num_workers):

    examples = external_input(data_root)

    example_gen = CsvExampleGen(input=examples)

    statistics_gen = StatisticsGen(examples=example_gen.outputs['examples'])

    infer_schema = SchemaGen(
        statistics=statistics_gen.outputs['statistics'])

    validate_stats = ExampleValidator(
        statistics=statistics_gen.outputs['statistics'],
        schema=infer_schema.outputs['schema'])

    transform = Transform(
        examples=example_gen.outputs['examples'],
        schema=infer_schema.outputs['schema'],
        module_file=module_file)

    trainer = Trainer(
        module_file=module_file,
        custom_executor_spec=executor_spec.ExecutorClassSpec(GenericExecutor),
        examples=transform.outputs['transformed_examples'],
        transform_graph=transform.outputs['transform_graph'],
        schema=infer_schema.outputs['schema'],
        train_args=trainer_pb2.TrainArgs(num_steps=1000),
        eval_args=trainer_pb2.EvalArgs(num_steps=500))

    return pipeline.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        components = [
            example_gen,
            statistics_gen,
            infer_schema,
            validate_stats,
            transform,
            trainer
        ],
        enable_cache = True,
        metadata_connection_config=metadata.sqlite_metadata_connection_config(
            metadata_path
        ),
        beam_pipeline_args=['--direcit_num_workers=%d' %direct_num_workers])

DAG = AirflowDagRunner(AirflowPipelineConfig(_airflow_config)).run(
    _create_pipeline(
        pipeline_name = _pipeline_name,
        pipeline_root = _pipeline_root,
        data_root = _data_root,
        module_file = _module_file,
        serving_model_dir = _serving_model_dir,
        metadata_path = _metadata_path,
        direct_num_workers = 0 ))
