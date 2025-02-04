from collections import defaultdict, deque

def find_task_order(dependencies):
    # Build the graph and in-degree dictionary
    graph = defaultdict(list)
    in_degree = defaultdict(int)
    all_tasks = set()

    # Populate the graph and count in-degrees
    for prereq, task in dependencies:
        graph[prereq].append(task)
        in_degree[task] += 1
        # Ensure all tasks are recorded
        all_tasks.add(prereq)
        all_tasks.add(task)

    # Initialize queue with tasks having no prerequisites
    queue = deque([t for t in all_tasks if in_degree[t] == 0])

    order = []
    while queue:
        current = queue.popleft()
        order.append(current)

        # Reduce in-degree of neighbors
        for neighbor in graph[current]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    # If the order includes all tasks, we have a valid topological sort
    if len(order) == len(all_tasks):
        return order
    else:
        # A cycle must exist, so no valid ordering
        return None


# Example usage:
dependencies_1 = [
    ("inspect_machinery", "run_optimization_models"),
    ("inspect_machinery", "refine_crude_oil"),
    ("run_optimization_models", "refine_crude_oil"),
    ("refine_crude_oil", "quality_check"),
    ("quality_check", "package_products"),
    ("package_products", "ship_gasoline")
]

order_1 = find_task_order(dependencies_1)
print(order_1)  # Possible output: ['inspect_machinery', 'run_optimization_models', 'refine_crude_oil', 'quality_check', 'package_products', 'ship_gasoline']

dependencies_2 = [
    ("prepare_chemicals", "mix_solutions"),
    ("clean_containers", "mix_solutions"),
    ("mix_solutions", "perform_reaction"),
    ("perform_reaction", "distillation")
]

order_2 = find_task_order(dependencies_2)
print(order_2)  # Possible valid output: ['clean_containers', 'prepare_chemicals', 'mix_solutions', 'perform_reaction', 'distillation']
