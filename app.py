from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Simulated system state
processes = []
resources = ["R1", "R2", "R3"]
max_matrix = []  # Max resource needs (for Banker's)
alloc_matrix = []  # Currently allocated resources
avail_resources = [5, 5, 5]  # Available resources


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/add_process', methods=['POST'])
def add_process():
    global processes, max_matrix, alloc_matrix
    data = request.json

    # Extract inputs
    max_needs = [int(data['max_r1']), int(data['max_r2']), int(data['max_r3'])]
    alloc_needs = [int(data['alloc_r1']), int(data['alloc_r2']), int(data['alloc_r3'])]

    # Validation: Check if allocation exceeds max need
    for i in range(3):
        if alloc_needs[i] > max_needs[i]:
            return jsonify({
                "status": "error",
                "message": f"Allocated R{i+1} ({alloc_needs[i]}) exceeds maximum need R{i+1} ({max_needs[i]})"
            }), 400  # Bad Request

    # Proceed with adding process
    process_id = len(processes)
    processes.append(f"P{process_id}")
    max_matrix.append(max_needs)
    alloc_matrix.append(alloc_needs)

    for i in range(3):
        avail_resources[i] -= alloc_matrix[-1][i]

    process_data = [{"id": p, "alloc": alloc} for p, alloc in zip(processes, alloc_matrix)]
    return jsonify({"status": "Process added", "processes": process_data})


@app.route('/check_deadlock', methods=['POST'])
def check_deadlock():
    algorithm = request.json.get('algorithm', 'bankers')
    print(f"Checking deadlock with algorithm: {algorithm}")
    if algorithm == 'bankers':
        result = bankers_algorithm()
        print(f"Banker's result: {result.get_json()}")
        return result
    elif algorithm == 'wfg':
        return wfg_algorithm()
    return jsonify({"error": "Invalid algorithm"})


def bankers_algorithm():
    work = avail_resources.copy()
    finish = [False] * len(processes)
    safe_sequence = []

    print(f"Initial state - Work: {work}, Finish: {finish}, Processes: {processes}")

    while False in finish:
        found = False
        process_needs = [(i, sum(max_matrix[i][j] - alloc_matrix[i][j] for j in range(3)))
                         for i in range(len(processes)) if not finish[i]]
        process_needs.sort(key=lambda x: x[1])

        for i, _ in process_needs:
            need = [max_matrix[i][j] - alloc_matrix[i][j] for j in range(3)]
            print(f"Checking {processes[i]} - Need: {need}, Work: {work}")
            if not finish[i] and all(need[j] <= work[j] for j in range(3)):
                for j in range(3):
                    work[j] += alloc_matrix[i][j]
                finish[i] = True
                safe_sequence.append(processes[i])
                found = True
                print(f"{processes[i]} can finish. New Work: {work}, Safe Sequence: {safe_sequence}")
                break
        if not found:
            print("Deadlock detected. No process can proceed.")
            return jsonify({"deadlock": True, "safe_sequence": [], "suggestion": suggest_resolution()})

    print(f"Safe sequence determined: {safe_sequence}")
    return jsonify({"deadlock": False, "safe_sequence": safe_sequence, "suggestion": None})


def wfg_algorithm():
    wfg = build_wfg()
    if detect_cycle(wfg):
        suggestion = suggest_resolution_wfg(wfg)
        return jsonify({"deadlock": True, "safe_sequence": [], "suggestion": suggestion})
    return jsonify({"deadlock": False, "safe_sequence": [], "suggestion": None})


def build_wfg():
    wfg = {p: [] for p in processes}
    avail = avail_resources.copy()

    for i in range(len(processes)):
        need = [max_matrix[i][j] - alloc_matrix[i][j] for j in range(3)]
        for j in range(3):
            if need[j] > avail[j]:
                for k in range(len(processes)):
                    if k != i and alloc_matrix[k][j] > 0:
                        wfg[processes[i]].append(processes[k])
    return wfg


def detect_cycle(graph):
    visited = set()
    rec_stack = set()
    for node in graph:
        if node not in visited:
            if dfs_cycle(graph, node, visited, rec_stack):
                return True
    return False


def dfs_cycle(graph, node, visited, rec_stack):
    visited.add(node)
    rec_stack.add(node)
    for neighbor in graph[node]:
        if neighbor not in visited:
            if dfs_cycle(graph, neighbor, visited, rec_stack):
                return True
        elif neighbor in rec_stack:
            return True
    rec_stack.remove(node)
    return False


def suggest_resolution():
    max_alloc = -1
    preempt_pid = 0
    for i, alloc in enumerate(alloc_matrix):
        total = sum(alloc)
        if total > max_alloc:
            max_alloc = total
            preempt_pid = i
    return f"Preempt {processes[preempt_pid]} and release its resources."


def suggest_resolution_wfg(wfg):
    max_edges = -1
    preempt_pid = 0
    for i, p in enumerate(processes):
        edges = len(wfg[p])
        if edges > max_edges:
            max_edges = edges
            preempt_pid = i
    return f"Preempt {processes[preempt_pid]} and release its resources."


@app.route('/resolve_deadlock', methods=['POST'])
def resolve_deadlock():
    global processes, alloc_matrix, avail_resources
    algorithm = request.json.get('algorithm', 'bankers')

    initial_graph = get_graph_data_internal()

    if algorithm == 'bankers':
        max_alloc = -1
        preempt_pid = 0
        for i, alloc in enumerate(alloc_matrix):
            total = sum(alloc)
            if total > max_alloc:
                max_alloc = total
                preempt_pid = i
    elif algorithm == 'wfg':
        wfg = build_wfg()
        max_edges = -1
        preempt_pid = 0
        for i, p in enumerate(processes):
            edges = len(wfg[p])
            if edges > max_edges:
                max_edges = edges
                preempt_pid = i

    released_resources = alloc_matrix[preempt_pid]
    for i in range(3):
        avail_resources[i] += released_resources[i]

    preempted_process = processes.pop(preempt_pid)
    alloc_matrix.pop(preempt_pid)
    max_matrix.pop(preempt_pid)

    final_graph = get_graph_data_internal()

    return jsonify({
        "initial_graph": initial_graph,
        "final_graph": final_graph,
        "message": f"Preempted {preempted_process} to resolve deadlock."
    })


@app.route('/reset', methods=['POST'])
def reset():
    global processes, max_matrix, alloc_matrix, avail_resources
    processes = []
    max_matrix = []
    alloc_matrix = []
    avail_resources = [5, 5, 5]
    return jsonify({"status": "System reset"})


def get_graph_data_internal():
    nodes = [{"id": p, "label": p, "group": "process"} for p in processes] + \
            [{"id": r, "label": r, "group": "resource"} for r in resources]
    edges = []
    for i, alloc in enumerate(alloc_matrix):
        for j, count in enumerate(alloc):
            if count > 0:
                edges.append({"from": processes[i], "to": resources[j], "label": str(count)})
    return {"nodes": nodes, "edges": edges}


@app.route('/get_graph_data', methods=['GET'])
def get_graph_data():
    return jsonify(get_graph_data_internal())


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)