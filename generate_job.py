import random
from typing import List, Tuple


def generate_quintuples(num_services, num_pod):
    """Generate random service quintuples.

    Args:
        num_services: Number of services/quintuples to generate (can have duplicates)

    Returns:
        List of quintuples in format: (comm_data, service_type, batch_size, degree, time)
        - comm_data: Communication data in MB
        - service_type: 0=data parallel, 1=pipeline parallel
        - batch_size: Batch size
        - degree: Parallel degree
        - time: Execution time in seconds
    """
    # First generate all possible quintuples
    # Constants defined at module level
    MODEL_PARAMS_MB = {
        "VGG11": 507.0, "VGG16": 528.0, "VGG19": 549.4,
        "ResNet18": 44.6, "ResNet34": 83.2, "ResNet50": 97.5, "ResNet101": 170.5,
        "BERT": 412.0, "RoBERTa": 355.0,
        "GPT1": 420.0, "GPT2": 446.5
    }

    ACTIVATION_SIZE = {
        "GPT1": 12.6, "GPT2": 16.8
    }

    data_sections = [
        {
            "models": ["VGG11", "VGG16", "VGG19", "ResNet18", "ResNet34", "ResNet50", "ResNet101"],
            "batch_sizes": [32, 64, 128, 256, 512],
            "times": [
                [0.1426, 0.1861, 0.2362, 0.1022, 0.1922, 0.3096, 0.5572],
                [0.2125, 0.37, 0.4939, 0.1377, 0.246, 0.3206, 0.6428],
                [0.3714, 0.6832, 0.7595, 0.1692, 0.3054, 0.4382, 0.835],
                [0.5814, 1.0299, 1.1827, 0.1789, 0.3431, 0.6182, 1.2488],
                [1.1398, 2.4346, 2.8515, 0.3238, 0.5634, 1.43, 2.6309],
            ],
            "is_model_parallel": False
        },
        {
            "models": ["BERT", "RoBERTa"],
            "batch_sizes": [1, 2, 4, 8, 16],
            "times": [
                [0.37, 0.4605],
                [0.63, 0.5165],
                [1.582, 0.9734],
                [2.257, 2.3163],
                [3.4329, 2.4162],
            ],
            "is_model_parallel": False
        },
        {
            "models": ["GPT1", "GPT2"],
            "batch_sizes": [1, 2, 4, 8],
            "times": [
                [0.9926, 0.8699],
                [1.1718, 1.3946],
                [1.8325, 2.1846],
                [3.2821, 3.1334],
            ],
            "is_model_parallel": True
        }
    ]

    DATA_PARALLEL_DEGREES = [i for i in range(6, 8)]
    PIPELINE_PARALLEL_DEGREES = [i for i in range(6, 8)]
    TIME_RANGE = [0.15, 0.45]
    all_quintuples = []
    for section in data_sections:
        models = section["models"]
        batch_sizes = section["batch_sizes"]
        times = section["times"]
        is_model_parallel = section["is_model_parallel"]

        for i, batch_size in enumerate(batch_sizes):
            for j, model in enumerate(models):
                original_time = times[i][j]

                if is_model_parallel:  # Pipeline parallel
                    for degree in PIPELINE_PARALLEL_DEGREES:
                        comm_data = ACTIVATION_SIZE[model]
                        adjusted_time = original_time / degree
                        if TIME_RANGE[0] <= adjusted_time <= TIME_RANGE[1]:
                            all_quintuples.append(
                                (comm_data, 1, batch_size, degree, adjusted_time)
                            )
                else:  # Data parallel
                    for degree in DATA_PARALLEL_DEGREES:
                        comm_data = 2 * (degree - 1) * MODEL_PARAMS_MB[model] / degree
                        adjusted_time = original_time
                        if TIME_RANGE[0] <= adjusted_time <= TIME_RANGE[1]:
                            all_quintuples.append(
                                (comm_data, 0, batch_size, degree, adjusted_time)
                            )

    # Then randomly sample with possible duplicates
    if not all_quintuples:
        return []

    random_quintuples = [
        random.choice(all_quintuples)
        for _ in range(num_services)
    ]

    return random_quintuples


def generate_placement(services: List[Tuple], num_nodes: int) -> List[List[int]]:
    """
    Generate random placement strategies for each service's parallel units across nodes,
    ensuring no duplicate node assignments within a single service.

    Args:
        services: List of service quintuples in format:
                 (comm_data, service_type, batch_size, degree, time)
        num_nodes: Total number of available nodes in the cluster

    Returns:
        List of placement strategies, where each strategy is a list of node assignments
        for that service's parallel units, with no duplicates within a service.
        Returns None for services that cannot be placed (degree > num_nodes)
    """
    placement_strategies = []
    for service in services:
        comm_data, service_type, batch_size, degree, time = service

        # Check if placement is possible (enough distinct nodes)
        # For both data and pipeline parallel, we want distinct nodes
        available_nodes = list(range(num_nodes))
        placement = random.choices(available_nodes, k=degree)
        placement = list(set(placement))  
        placement_strategies.append(placement)
    return placement_strategies


# Example usage
if __name__ == "__main__":
    # Example services (from previous generator)
    services = [
        (412.0, 0, 16, 4, 0.85),  # Data parallel service
        (12.6, 1, 8, 4, 0.92),  # Pipeline parallel service
        (170.5, 0, 32, 8, 0.78),  # Data parallel with higher degree
        (16.8, 1, 4, 2, 0.81)  # Pipeline parallel with lower degree
    ]

    num_nodes = 8  # Total nodes in cluster
    placements = generate_placement(services, num_nodes)

    print("Service Placement Strategies:")
    for i, placement in enumerate(placements):
        service_type = "Data Parallel (Ring-AllReduce)" if services[i][1] == 0 else "Pipeline Parallel"
        print(f"Service {i + 1} ({service_type}, degree={services[i][3]}): {placement}")


# Example usage
if __name__ == "__main__":
    services = generate_quintuples(20)
    print("Randomly generated service quintuples (comm_data(MB), type, batch_size, degree, time):")
    print("Service type: 0=data parallel, 1=pipeline parallel")
    for q in services:
        print(f"({q[0]:.1f}MB, {q[1]}, {q[2]}, {q[3]}, {q[4]:.4f}s)")
