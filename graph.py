import numpy as np
import torch
import scipy.signal as signal
import scipy.stats as stats
import scipy.linalg as linalg
from torch_geometric.data import Data
from scipy.io import loadmat

def load_asce_shm_data(filename="DATAfile.mat"):
    """Load the ASCE-SHM dataset from a MATLAB .mat file (non-HDF5)"""
    raw_data = loadmat(filename)
    data = {}
    
    for key in raw_data:
        if key.startswith("__"):
            continue  # Skip MATLAB metadata
        value = raw_data[key]
        if hasattr(value, 'shape'):
            data[key] = np.array(value)
        else:
            data[key] = value  # Could be string, list, etc.

    # Transpose acc to match expected format (16, time_steps)
    if 'acc' in data and data['acc'].shape[1] == 16:
        data['acc'] = data['acc'].T
        print(f"Transposed acc to shape {data['acc'].shape}")

    return data

def map_sensors_to_structure():
    """Map 16 sensors to their positions on the structure"""
    # From papers: acc(:,1)- floor 1 of column 2 in x-direction, etc.
    sensor_mapping = [
        (1, 2, 0), (1, 6, 1), (1, 8, 0), (1, 4, 1),  # Floor 1
        (2, 2, 0), (2, 6, 1), (2, 8, 0), (2, 4, 1),  # Floor 2
        (3, 2, 0), (3, 6, 1), (3, 8, 0), (3, 4, 1),  # Floor 3
        (4, 2, 0), (4, 6, 1), (4, 8, 0), (4, 4, 1)   # Floor 4
    ]
    
    # Column positions based on paper
    column_coords = {
        2: (0, 0),      # Southwest corner
        4: (0, 2.5),    # Southeast corner
        6: (2.5, 0),    # Northwest corner
        8: (2.5, 2.5)   # Northeast corner
    }
    
    # Floor heights (m)
    floor_heights = {1: 0.9, 2: 1.8, 3: 2.7, 4: 3.6}
    
    return sensor_mapping, column_coords, floor_heights

def create_transformation_matrix(sensor_mapping, column_coords):
    """Create transformation matrix from 12 DOFs to 16 sensor locations"""
    T = np.zeros((16, 12))
    
    for sensor_idx, (floor, column, direction) in enumerate(sensor_mapping):
        # Map to DOFs (floor indices are 0-based in the DOF numbering)
        dof_idx = (floor-1)*3
        
        # Get column coordinates
        x, y = column_coords[column]
        
        # Center coordinates (center of mass at (1.25, 1.25))
        x_rel = x - 1.25
        y_rel = y - 1.25
        
        if direction == 0:  # x-direction sensor
            T[sensor_idx, dof_idx] = 1.0  # x-component
            T[sensor_idx, dof_idx+2] = -y_rel  # θ-component affects x based on y-position
        else:  # y-direction sensor
            T[sensor_idx, dof_idx+1] = 1.0  # y-component
            T[sensor_idx, dof_idx+2] = x_rel  # θ-component affects y based on x-position
            
    return T

def compute_mode_shapes(K, M, num_modes=4):
    """Compute mode shapes from K and M matrices"""
    # Solve generalized eigenvalue problem: Kφ = λMφ
    # eigvals, eigvecs = np.linalg.eigh(K, M, subset_by_index=[0, num_modes-1])
    eigvals, eigvecs = linalg.eigh(K, M, subset_by_index=(0, num_modes-1))

    # Natural frequencies (Hz)
    nat_freqs = np.sqrt(eigvals) / (2 * np.pi)
    
    return nat_freqs, eigvecs

def transform_modes_to_sensors(mode_shapes, T):
    """Project mode shapes from DOF space to sensor space"""
    sensor_mode_shapes = T @ mode_shapes
    return sensor_mode_shapes

def calculate_effective_stiffness_mass(K, M, T):
    """Calculate effective stiffness and mass at sensor locations"""
    # For each sensor, calculate effective properties
    K_eff = np.zeros(T.shape[0])
    M_eff = np.zeros(T.shape[0])
    
    for i in range(T.shape[0]):
        # Map from DOFs to sensor i
        sensor_dof_map = T[i, :]
        
        # Effective stiffness as weighted sum of K
        K_eff[i] = np.sum(sensor_dof_map * (K @ sensor_dof_map))
        
        # Effective mass as weighted sum of M
        M_eff[i] = np.sum(sensor_dof_map * (M @ sensor_dof_map))
    
    return K_eff, M_eff

def extract_time_domain_features(acc):
    """Extract time domain features from acceleration data"""
    features = []
    
    # Ensure we're dealing with the correct shape
    # acc should be shape (16, time_steps)
    num_sensors = acc.shape[0]
    
    for i in range(num_sensors):
        print(f"Extracting time domain features for sensor {i+1}")
        signal = acc[i, :]  # Get time series for sensor i
        
        # Calculate features
        rms = np.sqrt(np.mean(signal**2))
        peak = np.max(np.abs(signal))
        kurtosis = stats.kurtosis(signal)
        skewness = stats.skew(signal)
        mad = np.mean(np.abs(signal - np.mean(signal)))
        zero_crossings = np.sum(np.diff(np.signbit(signal)))
        crest_factor = peak / rms if rms > 0 else 0
        p2p = np.max(signal) - np.min(signal)
        
        features.append([rms, peak, kurtosis, skewness, mad, zero_crossings, crest_factor, p2p])
    
    return np.array(features)

def extract_frequency_domain_features(acc, fs=1000):
    """Extract frequency domain features from acceleration data"""
    features = []
    
    # acc should be shape (16, time_steps)
    num_sensors = acc.shape[0]
    
    for i in range(num_sensors):
        signal = acc[i, :]  # Get time series for sensor i
        
        # Compute FFT
        fft = np.abs(np.fft.rfft(signal))
        freq = np.fft.rfftfreq(len(signal), d=1/fs)
        
        # Calculate features
        # Power in different bands (0-10Hz, 10-20Hz, 20-30Hz)
        band1 = np.sum(fft[(freq >= 0) & (freq < 10)]**2)
        band2 = np.sum(fft[(freq >= 10) & (freq < 20)]**2)
        band3 = np.sum(fft[(freq >= 20) & (freq < 30)]**2)
        
        # Dominant frequency
        dom_freq_idx = np.argmax(fft)
        dom_freq = freq[dom_freq_idx]
        dom_amp = fft[dom_freq_idx]
        
        # Spectral centroid
        centroid = np.sum(freq * fft) / np.sum(fft) if np.sum(fft) > 0 else 0
        
        features.append([band1, band2, band3, dom_freq, dom_amp, centroid])
    
    return np.array(features)

def calculate_correlation_matrix(acc):
    """Calculate correlation matrix between sensor signals"""
    # acc should be shape (16, time_steps)
    return np.corrcoef(acc)  # Correlation across rows (sensors)

def calculate_coherence_and_tf(acc, fs=1000, nperseg=1024, nat_freqs=None):
    """Calculate coherence and transfer function between sensors"""
    # acc should be shape (16, time_steps)
    num_sensors = acc.shape[0]
    
    # Default frequency bands if natural frequencies not provided
    if nat_freqs is None:
        freq_bands = [(0, 10), (10, 20), (20, 30)]
        num_bands = len(freq_bands)
    else:
        num_bands = len(nat_freqs)
    
    coherence_matrix = np.zeros((num_sensors, num_sensors, num_bands))
    tf_magnitude = np.zeros((num_sensors, num_sensors))
    
    for i in range(num_sensors):
        for j in range(i+1, num_sensors):
            # Calculate coherence
            f, Cxy = signal.coherence(acc[i], acc[j], fs=fs, nperseg=nperseg)
            
            if nat_freqs is None:
                # Calculate coherence in frequency bands
                for b, (f_min, f_max) in enumerate(freq_bands):
                    mask = (f >= f_min) & (f <= f_max)
                    avg_coherence = np.mean(Cxy[mask]) if np.any(mask) else 0
                    coherence_matrix[i, j, b] = coherence_matrix[j, i, b] = avg_coherence
            else:
                # Calculate coherence at natural frequencies
                for m, freq in enumerate(nat_freqs[:num_bands]):  # Limit to num_bands
                    idx = np.abs(f - freq).argmin()
                    coherence_matrix[i, j, m] = coherence_matrix[j, i, m] = Cxy[idx]
            
            # Calculate transfer function
            f, Txy = signal.csd(acc[i], acc[j], fs=fs, nperseg=nperseg)
            f, Pxx = signal.welch(acc[i], fs=fs, nperseg=nperseg)
            
            # Average TF magnitude in 0-30Hz band
            mask = (f >= 0) & (f <= 30)
            if np.any(mask) and np.all(Pxx[mask] > 0):
                Hxy = np.abs(Txy[mask] / Pxx[mask])
                avg_tf = np.mean(Hxy)
            else:
                avg_tf = 0
            
            tf_magnitude[i, j] = tf_magnitude[j, i] = avg_tf
    
    return coherence_matrix, tf_magnitude

def create_graph_structure(data):
    """Create graph structure for ASCE-SHM dataset"""
    # Extract data
    acc = data['acc']
    K = data['K']
    M = data['M']
    node = data['node']
    elem = data['elem']
    
    # Calculate dt from time array if available
    dt = 0.01  
    if 'dt' in data:
        dt = float(data['dt'].item()) if isinstance(data['dt'], np.ndarray) else float(data['dt'])
    elif 'time' in data and data['time'].size > 1:
        dt = data['time'][0, 1] - data['time'][0, 0]
    
    # Ensure acc is in format (16, time_steps)
    if acc.shape[1] == 16 and acc.shape[0] != 16:
        acc = acc.T

    # Get sensor mapping and coordinates
    sensor_mapping, column_coords, floor_heights = map_sensors_to_structure()
    
    # Create transformation matrix
    T = create_transformation_matrix(sensor_mapping, column_coords)
    
    # Compute mode shapes and frequencies
    nat_freqs, mode_shapes = compute_mode_shapes(K, M, num_modes=4)
    sensor_mode_shapes = transform_modes_to_sensors(mode_shapes, T)
    
    # Calculate effective stiffness and mass at sensor locations
    K_eff, M_eff = calculate_effective_stiffness_mass(K, M, T)
    
    # Extract time domain features
    time_features = extract_time_domain_features(acc)
    
    # Extract frequency domain features
    fs = 1.0 / dt
    freq_features = extract_frequency_domain_features(acc, fs=fs)
    
    # Calculate correlation matrix
    corr_matrix = calculate_correlation_matrix(acc)
    
    # Calculate coherence and transfer function
    coherence_matrix, tf_matrix = calculate_coherence_and_tf(acc, fs=fs, nat_freqs=nat_freqs)
    
    # Create node features
    num_sensors = acc.shape[0]  # Should be 16
    node_features = []
    
    for i in range(num_sensors):
        floor, column, direction = sensor_mapping[i]
        x, y = column_coords[column]
        z = floor_heights[floor]
        
        # 1. Spatial Features (5D)
        spatial_features = [
            x/2.5,             # Normalized x coordinate
            y/2.5,             # Normalized y coordinate
            z/3.6,             # Normalized z coordinate
            (floor-1)/3,       # Normalized floor index
            direction          # Measurement direction
        ]
        
        # 2. Response Features (14D)
        response_features = np.concatenate([
            time_features[i],  # Time domain features (8D)
            freq_features[i]   # Frequency domain features (6D)
        ])
        
        # 3. Modal Features (10D)
        modal_features = np.concatenate([
            sensor_mode_shapes[i, :4],  # Mode shape components for first 4 modes
            [nat_freqs[0]/nat_freqs[0], nat_freqs[1]/nat_freqs[0], 
             nat_freqs[2]/nat_freqs[0], nat_freqs[3]/nat_freqs[0]],  # Normalized frequencies
            [sensor_mode_shapes[i, 0]/np.max(np.abs(sensor_mode_shapes[:, 0])), 
             sensor_mode_shapes[i, 1]/np.max(np.abs(sensor_mode_shapes[:, 1]))]  # Normalized modal amplitudes
        ])
        
        # 4. Structural Features (5D)
        structural_features = [
            K_eff[i]/np.max(K_eff),  # Normalized effective stiffness
            M_eff[i]/np.max(M_eff),  # Normalized effective mass
            K_eff[i]/M_eff[i]/np.max(K_eff/M_eff) if M_eff[i] > 0 else 0,  # K/M ratio (related to frequency)
            sum(1 for f, c, _ in sensor_mapping if f == floor)/4,  # Normalized connections on same floor
            1 if floor == 1 else 0    # Is boundary node (on first floor)
        ]
        
        # Combine all features
        features = np.concatenate([
            spatial_features,
            response_features,
            modal_features,
            structural_features
        ])
        
        node_features.append(features)
    
    # Create edge index and edge features
    edge_index = []
    edge_attr = []
    
    for i in range(num_sensors):
        for j in range(i+1, num_sensors):
            # Add bidirectional edges
            edge_index.append([i, j])
            edge_index.append([j, i])
            
            # Get sensor properties
            floor_i, column_i, direction_i = sensor_mapping[i]
            floor_j, column_j, direction_j = sensor_mapping[j]
            x_i, y_i = column_coords[column_i]
            x_j, y_j = column_coords[column_j]
            z_i = floor_heights[floor_i]
            z_j = floor_heights[floor_j]
            
            # 1. Physical Connectivity (5D)
            same_floor = 1 if floor_i == floor_j else 0
            adjacent_floor = 1 if abs(floor_i - floor_j) == 1 else 0
            
            # Direct connection determined by structure
            direct_connection = 0
            if same_floor and (column_i == column_j or 
                              abs(x_i - x_j) <= 2.5 and abs(y_i - y_j) <= 2.5):
                direct_connection = 1  # Connected by beam on same floor
            elif adjacent_floor and column_i == column_j:
                direct_connection = 1  # Connected by column
            elif adjacent_floor and abs(x_i - x_j) <= 2.5 and abs(y_i - y_j) <= 2.5:
                direct_connection = 1  # Potentially connected by brace
            
            # Determine connection type
            if direct_connection == 0:
                connection_type = 0  # No direct connection
            elif same_floor:
                connection_type = 2  # Beam connection
            elif adjacent_floor and column_i == column_j:
                connection_type = 1  # Column connection
            else:
                connection_type = 3  # Brace connection
            
            # Connection stiffness (placeholder - would be better with actual values from K)
            connection_stiffness = connection_type / 3 if connection_type > 0 else 0
            
            # 2. Dynamic Relationship (8D)
            correlation = corr_matrix[i, j]
            coherence_values = coherence_matrix[i, j, :3]  # First 3 modes
            tf_value = tf_matrix[i, j]
            
            # Phase consistency (approximate from correlation)
            phase_consistency = np.abs(correlation) / 2
            
            # Impulse response similarity (simplified approximation)
            impulse_response = tf_value * correlation
            
            # 3. Structural Relationship (6D)
            # Distance between sensors
            euclidean_dist = np.sqrt((x_i-x_j)**2 + (y_i-y_j)**2 + (z_i-z_j)**2) / 5.0
            
            # Directional indicators
            same_vertical = 1 if np.isclose(x_i, x_j) and np.isclose(y_i, y_j) else 0
            same_direction = 1 if direction_i == direction_j else 0
            orthogonal_dirs = 1 if direction_i != direction_j else 0
            
            # Mass-normalized stiffness ratio
            stiffness_ratio = min(K_eff[i]/M_eff[i], K_eff[j]/M_eff[j]) / max(K_eff[i]/M_eff[i], K_eff[j]/M_eff[j]) if M_eff[i] > 0 and M_eff[j] > 0 else 0
            
            # Combine edge features
            edge_features = np.concatenate([
                [direct_connection, same_floor, adjacent_floor, connection_type, connection_stiffness],
                [correlation], coherence_values, [tf_value, phase_consistency, impulse_response],
                [euclidean_dist, same_vertical, same_direction, orthogonal_dirs, stiffness_ratio]
            ])
            
            # Add same edge features in both directions (symmetric)
            edge_attr.append(edge_features)
            edge_attr.append(edge_features)
    
    # Convert to PyTorch tensors
    node_features = torch.tensor(np.array(node_features), dtype=torch.float)
    edge_index = torch.tensor(np.array(edge_index).T, dtype=torch.long)
    edge_attr = torch.tensor(np.array(edge_attr), dtype=torch.float)
    
    # Create graph data object
    graph_data = Data(
        x=node_features,
        edge_index=edge_index,
        edge_attr=edge_attr
    )
    
    return graph_data

def create_and_save_graph(filename="DATAfile.mat", output_file="asce_shm_graph.pt"):
    """Create and save graph structure from ASCE-SHM dataset"""
    # Load data
    data = load_asce_shm_data(filename)
    
    # Create graph structure
    graph = create_graph_structure(data)
    graph.acceleration_data = torch.tensor(data['acc'], dtype=torch.float)

    # Save graph
    torch.save(graph, output_file)
    
    print(f"Graph created with {graph.num_nodes} nodes and {graph.num_edges} edges")
    print(f"Node feature dimension: {graph.num_node_features}")
    print(f"Edge feature dimension: {graph.num_edge_features}")
    print(f"Acceleration data shape: {graph.acceleration_data.shape}")
    
    return graph

if __name__ == "__main__":
    create_and_save_graph()
