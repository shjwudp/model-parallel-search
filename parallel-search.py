import csv
from collections import namedtuple
from itertools import product

import hydra
from omegaconf import DictConfig

ParallelConfig = namedtuple("ParallelConfig", ["ngpus", "tp", "pp", "ep", "cp", "dp"])
MemoryEstimation = namedtuple(
    "MemoryEstimation",
    [
        "name",
        "ngpus",
        "TP",
        "PP",
        "EP",
        "CP",
        "DP",
        "sub_seq_length",
        "micro_batch_size",
        "num_of_micro_batches",
        "pipeline_parallelism_buble_rate",
        "data_parallel_sharding_strategy",
        "total_memory_gb",
        "model_and_optimizer_states_memory_gb",
        "activations_memory_gb",
        "cross_entropy_loss_temp_memory_gb",
        "practice_activations_memory_gb",
        "expert_parameters_m",
        "non_expert_parameters_m",
        "expert_layers_activations_gb",
        "practice_expert_layers_activations_gb",
        "non_expert_layers_activations_gb",
        "mlp_activation_mb",
        "moe_activation_mb",
        "token_imbalance_hypothesis",
    ],
)


def The_coefficient_of_the_model_state_size_with_respect_to_the_parameter_quantity(
    trainer_config, data_parallel_size
):
    shard_strategy = trainer_config.data_parallel_sharding_strategy
    param_dtype = trainer_config.param_dtype
    grad_dtype = trainer_config.grad_dtype
    d = data_parallel_size
    psi_table = {
        "NO_OP": {
            ("float16", "float16"): 20.0,
            ("float16", "float32"): 18.0,
            ("float32", "float32"): 16.0,
        },
        "OPTIMIZER_STATES": {
            ("float16", "float16"): 4 + 16.0 / d,
            ("float16", "float32"): 6 + 12.0 / d,
            ("float32", "float32"): 8 + 8.0 / d,
        },
        "OPTIMIZER_STATES_AND_GRADS": {
            ("float16", "float16"): 2 + 18.0 / d,
            ("float16", "float32"): 2 + 16.0 / d,
            ("float32", "float32"): 4 + 12.0 / d,
        },
        "FULLY_SHARD": {
            ("float16", "float16"): 20.0 / d,
            ("float16", "float32"): 16.0 / d,
            ("float32", "float32"): 16.0 / d,
        },
    }
    return psi_table[shard_strategy][(param_dtype, grad_dtype)]


def Memory_estimation(
    model_config,
    trainer_config,
    parallel_config: ParallelConfig,
    token_imbalance_hypothesis=1.0,
) -> MemoryEstimation:
    """Estimate memory usage for a given model and parallel configuration."""
    # Parallel configuration
    ngpus = parallel_config.ngpus
    tp = parallel_config.tp
    pp = parallel_config.pp
    ep = parallel_config.ep
    cp = parallel_config.cp
    dp = parallel_config.dp
    assert (
        tp * pp * cp * dp == ngpus
    ), "Parallelism configuration does not match the number of GPUs."
    assert dp % ep == 0, "Data parallelism must be divisible by expert parallelism."
    data_module_expert_parallelism = dp // ep

    # Model parameters
    v = model_config.vocab_size
    h = model_config.hidden_size
    h_ffn = model_config.ffn_hidden_size
    nlayers = model_config.num_layers
    assert (
        nlayers % pp == 0
    ), "Number of layers must be divisible by pipeline parallelism."
    assert (
        model_config.seq_length % cp == 0
    ), "Sequence length must be divisible by context parallelism world size."
    s = model_config.seq_length // cp
    b = model_config.micro_batch_size
    m = model_config.global_batch_size / dp / b
    activation = model_config.activation

    # Training parameters
    assert trainer_config.param_dtype == "float16", "Only float16 is supported for now."
    assert trainer_config.grad_dtype == "float32", "Only float32 is supported for now."

    # MoE configuration
    f_expert = model_config.moe.expert_frequency
    k = model_config.moe.k
    assert (
        model_config.moe.num_experts % ep == 0
    ), "Number of experts must be divisible by number of expert parallelism."
    n_experts = model_config.moe.num_experts
    token_imbalance_factor = 1.0 if ep == 1 else token_imbalance_hypothesis
    c = token_imbalance_factor  # expert capacity factor

    # Memory usage for model & optimizer states
    embedding_parameters = v * h // tp * (2 if pp == 1 else 1)
    attention_layer_parameters = h * h * 4 // tp
    if activation == "swiglu":
        mlp_layer_parameters = h * h_ffn * 3 // tp
    else:
        mlp_layer_parameters = h * h_ffn * 2 // tp
    non_expert_parameters = (
        embedding_parameters
        + (attention_layer_parameters + mlp_layer_parameters * (1.0 - f_expert))
        * nlayers
        // pp
    )
    expert_parameters = (
        (mlp_layer_parameters * f_expert * n_experts // ep) * nlayers // pp
    )
    non_expert_psi = The_coefficient_of_the_model_state_size_with_respect_to_the_parameter_quantity(
        trainer_config, dp
    )
    expert_psi = The_coefficient_of_the_model_state_size_with_respect_to_the_parameter_quantity(
        trainer_config, data_module_expert_parallelism
    )
    model_and_optimizer_states_memory = (
        non_expert_parameters * non_expert_psi + expert_parameters * expert_psi
    )

    # Memory usage for activations
    cross_entropy_activation = 4 * s * b * v // tp
    if activation == "swiglu":
        # SwiGLU MLP
        mlp_activation = 2 * s * b * (h + 4 * h_ffn) / tp
    else:
        # Vanilla MLP
        mlp_activation = 2 * s * b * (h + 2 * h_ffn) / tp
    moe_activation = 2 * (s * b * h + c * k * s * b * h) // tp + c * k * mlp_activation
    practice_moe_activation = (
        s * b * h // tp + 4 * c * k * s * b * h + c * k * mlp_activation
    )
    expert_layers_activations = nlayers * f_expert * moe_activation
    practice_expert_layers_activations = nlayers * f_expert * practice_moe_activation
    non_expert_layers_activations = (
        nlayers * ((1.0 - f_expert) * mlp_activation + 2 * 7 * s * b * h / tp)
        + cross_entropy_activation
    )
    activations_memory = expert_layers_activations + non_expert_layers_activations
    practice_activations_memory = (
        practice_expert_layers_activations + non_expert_layers_activations
    )

    # Cross entropy temporary memory
    cross_entropy_loss_temp_memory = 2 * s * b * v // tp if tp > 1 and pp == 1 else 0.0

    # Pipeline parallelism bubble ratio
    pipeline_parallelism_buble_rate = (pp - 1) / m

    return MemoryEstimation(
        name=model_config.name,
        ngpus=ngpus,
        TP=tp,
        PP=pp,
        EP=ep,
        CP=cp,
        DP=dp,
        sub_seq_length=model_config.seq_length // cp,
        micro_batch_size=model_config.micro_batch_size,
        num_of_micro_batches=m,
        pipeline_parallelism_buble_rate=pipeline_parallelism_buble_rate,
        data_parallel_sharding_strategy=trainer_config.data_parallel_sharding_strategy,
        total_memory_gb=(
            model_and_optimizer_states_memory
            + activations_memory
            + cross_entropy_loss_temp_memory
        )
        / 1024 ** 3,
        model_and_optimizer_states_memory_gb=model_and_optimizer_states_memory
        / 1024 ** 3,
        activations_memory_gb=activations_memory / 1024 ** 3,
        cross_entropy_loss_temp_memory_gb=cross_entropy_loss_temp_memory / 1024 ** 3,
        practice_activations_memory_gb=practice_activations_memory / 1024 ** 3,
        expert_parameters_m=expert_parameters / 1024 ** 2,
        non_expert_parameters_m=non_expert_parameters / 1024 ** 2,
        expert_layers_activations_gb=expert_layers_activations / 1024 ** 3,
        practice_expert_layers_activations_gb=practice_expert_layers_activations
        / 1024 ** 3,
        non_expert_layers_activations_gb=non_expert_layers_activations / 1024 ** 3,
        mlp_activation_mb=mlp_activation / 1024 ** 2,
        moe_activation_mb=moe_activation / 1024 ** 2,
        token_imbalance_hypothesis=token_imbalance_hypothesis,
    )


def print_memory_estimation(memory_estimation: MemoryEstimation):
    print(f"Model: {memory_estimation.name}")
    print(f"Number of GPUs: {memory_estimation.ngpus}")
    print(f"Token Parallelism: {memory_estimation.TP}")
    print(f"Pipeline Parallelism: {memory_estimation.PP}")
    print(f"Expert Parallelism: {memory_estimation.EP}")
    print(f"Context Parallelism: {memory_estimation.CP}")
    print(f"Data Parallelism: {memory_estimation.DP}")
    print(f"Sub-sequence length: {memory_estimation.sub_seq_length}")
    print(f"Micro-batch size: {memory_estimation.micro_batch_size}")
    print(f"Number of micro-batches: {memory_estimation.num_of_micro_batches}")
    print(
        f"Pipeline Parallelism Bubble Rate: {memory_estimation.pipeline_parallelism_buble_rate}"
    )
    print(
        f"Data Parallel Sharding Strategy: {memory_estimation.data_parallel_sharding_strategy}"
    )
    print(f"Total Memory (GB): {memory_estimation.total_memory_gb}")
    print(
        f"Model and Optimizer States Memory (GB): {memory_estimation.model_and_optimizer_states_memory_gb}"
    )
    print(f"Activations Memory (GB): {memory_estimation.activations_memory_gb}")
    print(
        f"Practice Activations Memory (GB): {memory_estimation.practice_activations_memory_gb}"
    )
    print(f"Expert Parameters (M): {memory_estimation.expert_parameters_m}")
    print(f"Non-Expert Parameters (M): {memory_estimation.non_expert_parameters_m}")
    print(
        f"Expert Layers Activations (GB): {memory_estimation.expert_layers_activations_gb}"
    )
    print(
        f"Practice Expert Layers Activations (GB): {memory_estimation.practice_expert_layers_activations_gb}"
    )
    print(
        f"Non-Expert Layers Activations (GB): {memory_estimation.non_expert_layers_activations_gb}"
    )
    print(f"MLP Activation (MB): {memory_estimation.mlp_activation_mb}")
    print(f"MoE Activation (MB): {memory_estimation.moe_activation_mb}")
    print(f"Token Imbalance Hypothesis: {memory_estimation.token_imbalance_hypothesis}")


@hydra.main(
    version_base="1.1", config_path="./", config_name="Mixtral_8x7b.yaml",
)
def main(cfg: DictConfig):
    csv_file = open("memory_estimation.csv", "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(MemoryEstimation._fields)

    # Parallelism configuration
    num_experts = cfg.model.moe.num_experts
    mbs_range = [1, 2]
    cp_range = [1, 2, 4, 8]
    ep_range = [2 ** i for i in range(int(num_experts).bit_length())]
    tp_range = [1, 2, 4, 8]
    pp_range = [1, 2, 4, 8]
    ngpus_range = [64, 128, 512, 1024, 10240, 16384]
    shard_strategys = ["OPTIMIZER_STATES", "OPTIMIZER_STATES_AND_GRADS", "FULLY_SHARD"]

    if "ngpus_range" in cfg:
        ngpus_range = cfg.ngpus_range

    for ngpus, tp, pp, ep, cp, dp_sharding_strategy, mbs in product(
        ngpus_range, tp_range, pp_range, ep_range, cp_range, shard_strategys, mbs_range
    ):
        mp = tp * pp * cp
        if ngpus % mp != 0:
            # Parallelism configuration does not match the number of GPUs.
            continue
        dp = ngpus // mp
        if dp % ep != 0:
            # Data parallelism must be divisible by expert parallelism.
            continue
        if cfg.model.global_batch_size % (dp * mbs) != 0:
            # Global batch size must be divisible by data parallelism and micro batch size.
            continue

        # Update model configuration
        cfg.model.micro_batch_size = mbs

        # Update parallelism configuration
        cfg.trainer.data_parallel_sharding_strategy = dp_sharding_strategy

        parallel_config = ParallelConfig(ngpus=ngpus, tp=tp, pp=pp, ep=ep, cp=cp, dp=dp)
        memory_estimation = Memory_estimation(
            cfg.model,
            cfg.trainer,
            parallel_config,
            token_imbalance_hypothesis=cfg.model.moe.token_imbalance_hypothesis,
        )
        csv_writer.writerow(memory_estimation)


if __name__ == "__main__":
    main()
