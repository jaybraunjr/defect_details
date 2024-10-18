import numpy as np
import MDAnalysis as mda
import os
import json
import numpy as np
from collections import defaultdict
from utils import defect_loader
from tqdm import tqdm
from scipy import stats

def _dfs(graph, start):

    visited, stack = set(), [start]
    while stack:
        vertex = stack.pop()
        if vertex not in visited:
            visited.add(vertex)
            stack.extend(graph[vertex] - visited)
    return visited

def _make_graph(matrix):

    graph = {}
    xis, yis = matrix.shape
    for (xi, yi), value in np.ndenumerate(matrix):
        if value == 0:
            continue  
        n = xi * yis + yi 
        nlist = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue  
                x = divmod(xi + dx, xis)[1] 
                y = divmod(yi + dy, yis)[1]  
                if matrix[x, y] == 1:
                    ndn = x * yis + y  
                    nlist.append(ndn)
        graph[n] = set(nlist)  
    return graph


def construct_matrix(u, selection):
    xarray, yarray = np.arange(int(u.dimensions[0])), np.arange(int(u.dimensions[1]))
    xx, yy = np.meshgrid(xarray, yarray)
    matrix = np.zeros_like(xx)
    xind = np.mod(selection.positions[:, 0].astype(np.int64), u.dimensions[0].astype(np.int64))
    yind = np.mod(selection.positions[:, 1].astype(np.int64), u.dimensions[1].astype(np.int64))
    matrix[xind, yind] = 1
    return matrix

def find_defects(matrix, min_size):
    graph, visited, defects = _make_graph(matrix), set(), []
    for n in graph:
        if n not in visited:
            defect_loc = _dfs(graph, n)
            visited.update(defect_loc)
            if len(defect_loc) >= min_size:
                defects.append(defect_loc)
    return defects


def check_overlap(defect, prev_defects, yis):
    return [prev_id for prev_id, prev_defect in prev_defects.items() if any(node in prev_defect or is_adjacent(node, prev_defect, yis) for node in defect)]

def is_adjacent(node1, prev_defect, yis):
    x1, y1 = divmod(node1, yis)
    return any(abs(x1 - x2) <= 1 and abs(y1 - y2) <= 1 for x2, y2 in (divmod(node2, yis) for node2 in prev_defect))

def track_defect_lifetimes(defects_per_frame, yis, defect_type):
    defect_details, defect_mapping, current_defect_id = {}, {}, 0
    for frame_idx, frame_defects in enumerate(defects_per_frame):
        previous_defects, defect_mapping, assigned_defects = defect_mapping.copy(), {}, set()
        for defect in frame_defects:
            overlapping_prev_ids = check_overlap(defect, previous_defects, yis)
            if overlapping_prev_ids:
                largest_prev_id = max(overlapping_prev_ids, key=lambda id: len(previous_defects[id]))
                if largest_prev_id not in defect_details:
                    defect_details[largest_prev_id] = {'lifetime': 1, 'size': len(defect), 'frames': [frame_idx], 'type': defect_type}
                defect_details[largest_prev_id]['lifetime'] += 1
                defect_details[largest_prev_id]['frames'].append(frame_idx)
                defect_mapping[largest_prev_id] = defect
                assigned_defects.add(largest_prev_id)
                for prev_id in overlapping_prev_ids:
                    if prev_id != largest_prev_id and prev_id in defect_details:
                        defect_details.pop(prev_id, None)
                        defect_mapping.pop(prev_id, None)
            else:
                defect_details[current_defect_id] = {'lifetime': 1, 'size': len(defect), 'frames': [frame_idx], 'type': defect_type}
                defect_mapping[current_defect_id] = defect
                current_defect_id += 1
    return defect_details



def process_universes_and_save(base_name, universes, min_size):
    all_defects_details = process_universes(universes, min_size)
    save_defect_details(base_name, all_defects_details, min_size)  # Use base_name directly
    return all_defects_details


def process_universes(universes, min_size):
    all_defects_details = []
    for key, universe in universes.items():
        # Extract the type from the second element of the key and remove '.gro'
        defect_type = key[1].replace('.gro', '')
        defects_per_frame = calculate_defects_frame(universe, min_size)
        yis = int(universe.dimensions[1])
        defect_details = track_defect_lifetimes(defects_per_frame, yis, defect_type)
        all_defects_details.append(defect_details)
    return all_defects_details


def save_defect_details(base_name, all_defects_details, min_size, output_dir='output'):
    unique_name = create_unique_name(base_name)
    file_path = os.path.join(output_dir, f'defect_details_{unique_name}_minsize{min_size}_v2.json')
    os.makedirs(output_dir, exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(all_defects_details, f, indent=4)


def load_defect_details(base_name, min_size, output_dir='output'):
    unique_name = create_unique_name(base_name)
    file_path = os.path.join(output_dir, f'defect_details_{unique_name}_minsize{min_size}_v2.json')
    with open(file_path, 'r') as f:
        all_defects_details = json.load(f)
        print(f"Loaded {len(all_defects_details)} defect details for {unique_name} with min_size {min_size}")
    return all_defects_details

# 9/4
def calculate_average_lifetime_per_type_with_error(all_defects_details):
    lifetimes_per_type = defaultdict(list)

    for defects_details in all_defects_details:
        for defect_id, details in defects_details.items():
            defect_type = details['type']
            lifetime = details['lifetime']
            lifetimes_per_type[defect_type].append(lifetime)

    avg_lifetime_with_error = {}
    for defect_type, lifetimes in lifetimes_per_type.items():
        if lifetimes:
            mean_lifetime = np.mean(lifetimes)
            std_lifetime = np.std(lifetimes)  # Standard deviation
            sem_lifetime = std_lifetime / np.sqrt(len(lifetimes))  # Standard error of the mean (SEM)
            avg_lifetime_with_error[defect_type] = {
                'mean': mean_lifetime,
                'std': std_lifetime,
                'sem': sem_lifetime
            }
        else:
            avg_lifetime_with_error[defect_type] = {
                'mean': 0,
                'std': 0,
                'sem': 0
            }

    return avg_lifetime_with_error

def create_unique_name(key_tuple):
    base_name = key_tuple[0]
    type_name = key_tuple[1].replace('.gro', '').replace('.xtc', '')
    unique_name = f"{base_name}_{type_name}"
    return unique_name


def calculate_and_save_averages_with_error(base_name, min_size, output_dir='output'):
    all_defects_details = load_defect_details(base_name, min_size, output_dir)
    avg_lifetime_with_error = calculate_average_lifetime_per_type_with_error(all_defects_details)
    unique_name = create_unique_name(base_name)
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f'average_lifetime_{unique_name}_{min_size}.json')
    with open(file_path, 'w') as f:
        json.dump(avg_lifetime_with_error, f, indent=4)
    print(f"Saved average lifetimes and errors for {unique_name}")

def collect_all_defect_details(base_names, min_size, output_dir='output'):
    all_defects_details = []
    for base_name in base_names:
        # List all files in the output directory
        files = os.listdir(output_dir)
        # Filter files that match the pattern
        for file in files:
            if file.startswith(f'defect_details_{base_name}_') and f'_minsize{min_size}_v2.json' in file:
                file_path = os.path.join(output_dir, file)
                with open(file_path, 'r') as f:
                    defect_details = json.load(f)
                    all_defects_details.extend(defect_details)
    return all_defects_details

def calculate_and_save_averages_with_error_for_all(base_names, min_size, output_dir='output'):
    all_defects_details = collect_all_defect_details(base_names, min_size, output_dir)
    avg_lifetime_with_error = calculate_average_lifetime_per_type_with_error(all_defects_details)
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f'average_lifetime_all_{min_size}.json')
    with open(file_path, 'w') as f:
        json.dump(avg_lifetime_with_error, f, indent=4)
    print(f"Saved average lifetimes and errors for all types in {file_path}")



## Block averaging

def calc_prob(a, bin_centers, probabilities):
    total_prob = 0.0
    for i, j in zip(bin_centers, probabilities):
        if i > a:
            total_prob += j
    return total_prob

def calc_probs_for_chunks(a, bin_centers_chunks, probabilities_chunks):
    probs = []
    for bin_centers, probabilities in zip(bin_centers_chunks, probabilities_chunks):
        probs.append(calc_prob(a, bin_centers, probabilities))
    return np.mean(probs), np.std(probs)

def block_average(data, n_chunks):
    chunk_size = len(data) // n_chunks
    chunks = [data[i * chunk_size:(i + 1) * chunk_size] for i in range(n_chunks)]
    print(f"Data split into {len(chunks)} chunks of size {chunk_size}.")
    return chunks

def plot_histogram(defects, nbins=200, bin_max=200):
    bins = np.linspace(0, bin_max, nbins)
    hist, bin_edges = np.histogram(defects, bins)
    hist = hist.astype(np.float64)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    if np.sum(hist) > 0:
        hist /= np.sum(hist)  # Normalize to get probabilities
    return bin_centers, hist

def plot_all_histograms(*defect_datasets, n_chunks=100):
    histograms = []
    for defects in defect_datasets:
        chunks = block_average(defects, n_chunks)
        all_bin_centers = []
        all_histograms = []
        for chunk in chunks:
            bin_centers, probabilities = plot_histogram(chunk)
            all_bin_centers.append(bin_centers)
            all_histograms.append(probabilities)
        histograms.append((all_bin_centers, all_histograms))
    return histograms



def calculate_average_lifetime_per_type(all_defects_details):
    lifetimes_per_type = defaultdict(list)
    for defects_details in all_defects_details:
        for defect_id, details in defects_details.items():
            defect_type = details['type']
            lifetime = details['lifetime']
            lifetimes_per_type[defect_type].append(lifetime)
    
    average_lifetime_per_type = {defect_type: (sum(lifetimes) / len(lifetimes) if lifetimes else 0) for defect_type, lifetimes in lifetimes_per_type.items()}
    
    return average_lifetime_per_type



### Defect stuff 10-17-24


def calculate_defects_for_universes(universes):
    """
    Calculate defects for each universe.
    """
    defects = {}
    for key, universe in universes.items():
        print(universe.trajectory.n_frames)
        defects[key] = calculate_defects(universe)
    return defects

def get_specific_defects(defects, base_name, topo, traj):
    """
    Access defects for specific cases.
    """
    return defects[(base_name, topo, traj)]


def calculate_defects(universe):
    """
    Calculate defects in a molecular dynamics universe.
    """
    defects = []
    for ts in universe.trajectory:
        ag = universe.select_atoms('prop z > 0')
        hz = np.average(ag.positions[:, 2])
        agup = universe.select_atoms('prop z > %f' % hz)
        agdw = universe.select_atoms('prop z < %f' % hz)

        xarray = np.arange(0, universe.dimensions[0], 1)
        yarray = np.arange(0, universe.dimensions[1], 1)
        xx, yy = np.meshgrid(xarray, yarray)
        Mup = np.zeros_like(xx)
        Mdw = np.zeros_like(xx)

        # UP
        xind = agup.positions[:, 0].astype(np.int64)
        yind = agup.positions[:, 1].astype(np.int64)
        Mup[xind, yind] = 1

        graph = _make_graph(Mup)
        visited = set()
        for n in graph:
            if n not in visited:
                defect_loc = _dfs(graph, n)
                visited = visited.union(defect_loc)
                defects.append(len(defect_loc))

        # DW
        xind = agdw.positions[:, 0].astype(np.int64)
        yind = agdw.positions[:, 1].astype(np.int64)
        Mdw[xind, yind] = 1

        graph = _make_graph(Mdw)
        visited = set()
        for n in graph:
            if n not in visited:
                defect_loc = _dfs(graph, n)
                visited = visited.union(defect_loc)
                defects.append(len(defect_loc))
                
    return defects

def calculate_defects_frame(u, min_size=1, max_frames=100):
    defects_per_frame = []
    for frame_num, ts in enumerate(tqdm(u.trajectory[:max_frames], desc=f"Processing frames up to {max_frames}")):
        hz = np.average(u.select_atoms('prop z > 0').positions[:, 2])
        agup, agdw = u.select_atoms(f'prop z > {hz}'), u.select_atoms(f'prop z < {hz}')
        
        Mup, Mdw = construct_matrix(u, agup), construct_matrix(u, agdw)
        
        # Find defects for both "up" and "down" regions
        defects = find_defects(Mup, min_size) + find_defects(Mdw, min_size)
        defect_count = len(defects)
        
        # Print frame number and defect count
        print(f"Frame {frame_num}: {defect_count} defects larger than {min_size}")
        
        # Store the defect count per frame
        defects_per_frame.append({
            'frame': frame_num,
            'defect_count': defect_count
        })
    
    return defects_per_frame
