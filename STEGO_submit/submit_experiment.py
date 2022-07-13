"""Submit a run on Azure ML to train an unsupervised semantic segmentation model."""

import argparse
import logging.config
import azureml.core
from azureml.core import Dataset, Environment, Experiment, Workspace
from azureml.core.compute import AmlCompute, ComputeTarget
from azureml.data.output_dataset_config import OutputFileDatasetConfig
from azureml.core.compute_target import ComputeTargetException
from azureml.core.runconfig import (
    DockerConfiguration,
    RunConfiguration,
)
from azureml.pipeline.core import Pipeline
from azureml.pipeline.steps import PythonScriptStep

NUM_GPUS = {
    "STANDARD_NC6S_V2": 1,
    "STANDARD_NC12S_V2": 2,
    "STANDARD_NC24S_V2": 4,
    "STANDARD_NC24RS_V2": 4,
    "STANDARD_NC6S_V3": 1,
    "STANDARD_NC12S_V3": 2,
    "STANDARD_NC24S_V3": 4,
}

SHM_SIZE = {
    "STANDARD_NC6S_V2": "2g",
    "STANDARD_NC12S_V2": "100g",
    "STANDARD_NC24S_V2": "200g",
    "STANDARD_NC24RS_V2": "200g",
    "STANDARD_NC6S_V3": "2g",
    "STANDARD_NC12S_V3": "100g",
    "STANDARD_NC24S_V3": "200g",
}


class TrainingExperiment:
    """Train a model using Azure ML."""

    def __init__(self, workspace, cluster_name, cluster_size) -> None:
        """Create a training experiment."""
        self._logger = logging.getLogger(__name__)
        self._workspace = workspace
        self._cluster_name = cluster_name
        self._cluster_size = cluster_size

    def create_compute_target(self, cluster_name, cluster_size):
        """Create a compute target if it does not exist."""
        try:
            # Verify that cluster exists
            training_cluster = ComputeTarget(  # pylint: disable=abstract-class-instantiated
                workspace=self._workspace, name=cluster_name
            )
            self._logger.info("Found existing cluster, use it.")
        except ComputeTargetException:
            # If not, create it
            compute_config = AmlCompute.provisioning_configuration(
                vm_size=cluster_size, max_nodes=5, vm_priority="dedicated"
            )
            training_cluster = ComputeTarget.create(self._workspace, cluster_name, compute_config)

        training_cluster.wait_for_completion(show_output=False)
    
    def create_environment(self):
        """Create the training environment with conda and pip."""
        conda_env = Environment("pytorch-1.9-gpu")
        conda_env.docker.enabled = True
        conda_env.python.user_managed_dependencies = True
        conda_env.docker.base_image = None
        conda_env.docker.base_dockerfile = "docker/Dockerfile"
        return conda_env
    
    def submit(self, registered_dataset):
        """Submit the experiment."""
        self.create_compute_target(cluster_name=self._cluster_name, cluster_size=self._cluster_size)
        environment = self.create_environment()

        gpu_run_config = RunConfiguration(script="train.py")
        gpu_run_config.environment = environment
        gpu_run_config.docker = DockerConfiguration(use_docker=True, shm_size=SHM_SIZE[self._cluster_size.upper()])

        input_dataset = Dataset.get_by_name(workspace=self._workspace, name=registered_dataset)
        output_dataset = OutputFileDatasetConfig(name="precomputed_knns", source="/mnt/data/output_dataset/")

        precompute_knns_step = PythonScriptStep(
            name="Precompute KNN",
            script_name="precompute_knns.py",
            source_directory=".",
            inputs=[input_dataset.as_mount(path_on_compute="/mnt/data/input_dataset")],
            outputs=[output_dataset.as_mount()],
            compute_target=self._cluster_name,
            runconfig=gpu_run_config,
        )


        train_step = PythonScriptStep(
            name="Train",
            script_name="train_segmentation.py",
            source_directory=".",
            inputs=[input_dataset.as_mount(path_on_compute="/mnt/data/input_dataset"), output_dataset.as_input(name="precomputed_knns").as_mount(path_on_compute="/mnt/data/output_dataset")],
            compute_target=self._cluster_name,
            runconfig=gpu_run_config,
        )

        pipeline = Pipeline(workspace=self._workspace, steps=[precompute_knns_step, train_step])
        pipeline.validate()
        return pipeline


def main():
    """Submit a run on Azure ML to train an object detection model."""
    parser = argparse.ArgumentParser(
        description="Submit a run on Azure ML to train an object detection model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--cluster-name",
        default="NC6V3-cluster",
        help="Name of the Compute cluster in Azure ML (optional)",
    )
    parser.add_argument(
        "--cluster-size",
        default="STANDARD_NC6S_V3",
        help="Azure Compute Virtual Machine size for GPU training node(optional)",
    )
    parser.add_argument(
        "--experiment-name",
        default="STEGO-experinment_2",
        help="Name of the experiment in Azure ML.",
    )
    parser.add_argument(
        "--registered-dataset",
        default="high_resolution_small",
        help="Name of the dataset registered in Azure Machine Learning (required).",
    )
    args = parser.parse_args()

    logging_config = dict(
        version=1,
        formatters={"f": {"format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"}},
        handlers={
            "h": {
                "class": "logging.StreamHandler",
                "formatter": "f",
                "level": logging.DEBUG,
                "stream": "ext://sys.stdout",
            }
        },
        root={"handlers": ["h"], "level": logging.INFO},
    )
    logging.config.dictConfig(logging_config)
    logging.info("Start")

    workspace = Workspace.from_config()

    logging.info("Ready to use Azure ML %s to work with %s", azureml.core.VERSION, workspace.name)

    training_experiment = TrainingExperiment(
        workspace=workspace,
        cluster_name=args.cluster_name,
        cluster_size=args.cluster_size,
    )
    logging.info("Finished here1")
    pipeline = training_experiment.submit(registered_dataset=args.registered_dataset)

    logging.info("Finished here")

    experiment = Experiment(workspace=workspace, name=args.experiment_name)
    experiment.submit(pipeline)

    logging.info("Done")


if __name__ == "__main__":
    main()
