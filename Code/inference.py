from idtxl.multivariate_te import MultivariateTE
from idtxl.visualise_graph import plot_network
from idtxl.estimators_jidt import JidtKraskovMI
from idtxl.estimators_jidt import JidtDiscreteCMI
from idtxl.estimators_jidt import JidtKraskovCMI
from idtxl.active_information_storage import ActiveInformationStorage
import contextlib
import matplotlib.pyplot as plt
import geopandas as gpd
import networkx as nx
import numpy as np
from matplotlib.patches import FancyArrowPatch
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from PIL import Image
from pyproj import Transformer
from rasterio import features
from rasterio.enums import Resampling
from rasterio.mask import mask, raster_geometry_mask
from rasterio.plot import show
from shapely.geometry import Polygon, MultiPolygon, mapping
from skimage.measure import block_reduce
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize
from matplotlib.colors import LogNorm
from bridson import poisson_disc_samples
import matplotlib.patches as patches
import scienceplots 
import os 
import Tesselation as tess
import ipykernel
from matplotlib.colors import Normalize
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import leidenalg as la
import igraph as ig
import cpnet


from idtxl.data import Data
import numpy as np

def compute_and_save_lagged_correlations(file_path, output_dir):
    # Read the file
    gdf = gpd.read_file(file_path)
    
    # Select only numeric columns
    numeric_gdf = gdf.select_dtypes(include=[float, int])
    
    # Drop the population column
    if 'population' in numeric_gdf.columns:
        numeric_gdf = numeric_gdf.drop(columns=['population'])
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each time series independently
    for column in numeric_gdf.columns:
        # Extract the time series
        time_series = numeric_gdf[[column]]
        
        for lag in range(-4, 5):  # Including negative lags
            if lag == 0:
                relative_diff_ts = time_series.pct_change(periods=1)
            else:
                # Calculate the differences between time steps
                diff_ts = time_series.diff(periods=abs(lag))
                # Compute relative differences
                relative_diff_ts = time_series.pct_change(periods=abs(lag))

            transposed_ts = relative_diff_ts.T

            # Calculate the correlation matrix of the transposed DataFrame
            row_correlation_matrix = transposed_ts.corr()

            # Set correlation values below abs(0.85) to 0
            adjusted_corr_m = row_correlation_matrix.applymap(lambda x: 0 if abs(x) < 0.85 else x)

            # Delete first row and column
            adjusted_corr_m = adjusted_corr_m.iloc[1:, 1:]

            # Save the adjusted correlation matrix
            lag_label = f'lag_{lag}' if lag >= 0 else f'neg_lag_{abs(lag)}'
            output_file = os.path.join(output_dir, f'adjusted_corr_matrix_{column}_{lag_label}.csv')
            adjusted_corr_m.to_csv(output_file)

def compute_MI(data_array):
    """
    Compute mutual information matrix for the given data array.

    Parameters:
    data_array (numpy.ndarray): A 2D array where each row represents a variable and each column a sample.

    Returns:
    numpy.ndarray: A mutual information matrix.
    """
    # Initialize the mutual information estimator
    mi_estimator = JidtKraskovMI()

    # Compute mutual information between all pairs of variables
    n_processes = data_array.shape[0]
    mi_matrix = np.zeros((n_processes, n_processes))

    for i in range(n_processes):
        for j in range(i + 1, n_processes):
            mi_value = mi_estimator.estimate(data_array[i, :], data_array[j, :])
            mi_matrix[i, j] = mi_value
            mi_matrix[j, i] = mi_value
    
    return mi_matrix

def compute_TE(data_array, settings=None, num_permutations=50):
    """
    Compute transfer entropy matrix for the given data array using JidtDiscreteCMI and test statistical significance.

    Parameters:
    data_array (numpy.ndarray): A 2D array where each row represents a variable and each column a sample.
    settings (dict): Optional settings for the JidtDiscreteCMI estimator.
    num_permutations (int): Number of permutations for significance testing.

    Returns:
    (numpy.ndarray, numpy.ndarray): A tuple containing the transfer entropy matrix and the p-value matrix.
    """
    if settings is None:
        settings = {
            'discretise_method': 'max_ent', # Discretise continuous data into equal-sized bins
            'n_discrete_bins': 3,  # Number of discrete bins/levels
            'alph1': 3,
            'alph2': 3,
            'alphc': 3
        }

    # Initialize the conditional mutual information estimator
    te_estimator = JidtDiscreteCMI(settings=settings)
    
    # Compute transfer entropy for all pairs of variables
    n_processes = data_array.shape[0]
    te_matrix = np.zeros((n_processes, n_processes))
    p_value_matrix = np.ones((n_processes, n_processes))
    
    for target in range(n_processes):
        for source in range(n_processes):
            if source != target:
                # Define the time-shifted variables for transfer entropy calculation
                target_past = data_array[target, :-1]  # Target at time t-1
                source_past = data_array[source, :-1]  # Source at time t-1
                target_present = data_array[target, 1:]  # Target at time t

                # Calculate transfer entropy using CMI (source past -> target present | target past)
                te_value = te_estimator.estimate(source_past, target_present, target_past)
                te_matrix[source, target] = te_value
                
                # Perform permutation testing
                null_distribution = np.zeros(num_permutations)
                for i in range(num_permutations):
                    np.random.shuffle(source_past)  # Permute the source data
                    null_te_value = te_estimator.estimate(source_past, target_present, target_past)
                    null_distribution[i] = null_te_value
                
                # Calculate p-value as the proportion of null distribution values greater than or equal to the observed value
                p_value = (np.sum(null_distribution >= te_value) + 1) / (num_permutations + 1)
                p_value_matrix[source, target] = p_value

    return te_matrix, p_value_matrix

# def compute_TE(data_array, settings=None):
#     """
#     Compute transfer entropy matrix for the given data array using JidtDiscreteCMI.

#     Parameters:
#     data_array (numpy.ndarray): A 2D array where each row represents a variable and each column a sample.
#     settings (dict): Optional settings for the JidtDiscreteCMI estimator.

#     Returns:
#     numpy.ndarray: A transfer entropy matrix.
#     """
#     if settings is None:
#         settings = {
#             'discretise_method': 'max_ent', # Discretise continuous data into equal-sized bins
#             'n_discrete_bins': 3,  # Number of discrete bins/levels
#             'alph1': 3,
#             'alph2': 3,
#             'alphc': 3
#         }

#     # Initialize the conditional mutual information estimator
#     te_estimator = JidtDiscreteCMI(settings=settings)
    
#     # Compute transfer entropy for all pairs of variables
#     n_processes = data_array.shape[0]
#     te_matrix = np.zeros((n_processes, n_processes))
    
#     for target in range(n_processes):
#         for source in range(n_processes):
#             if source != target:
#                 # Define the time-shifted variables for transfer entropy calculation
#                 target_past = data_array[target, :-1]  # Target at time t-1
#                 source_past = data_array[source, :-1]  # Source at time t-1
#                 target_present = data_array[target, 1:]  # Target at time t

#                 # Calculate transfer entropy using CMI (source past -> target present | target past)
#                 te_value = te_estimator.estimate(source_past, target_present, target_past)
#                 te_matrix[source, target] = te_value

#     return te_matrix

def compute_AIS(data_array, processes):
    """
    Compute the Active Information Storage (AIS) for the given data array and processes.

    Parameters:
    data_array (numpy.ndarray): A 3D array of shape (n_processes, n_samples, n_replications).
    processes (list): A list of process indices to compute AIS for.

    Returns:
    dict: A dictionary mapping process indices to their AIS values.
    """
    data_array = data_array.reshape((data_array.shape[1], data_array.shape[0], 1))
    # Initialize the data object
    data = Data(data_array, dim_order='psr')

    # Define the settings for the analysis
    settings = {
        'cmi_estimator': 'JidtKraskovCMI',
        'n_perm_max_stat': 20,
        'n_perm_min_stat': 20,
        'alpha_max_stat': 0.1,
        'alpha_min_stat': 0.1,
        'alpha_mi': 0.1,
        'max_lag': 3,
        'tau': 1
    }

    # Initialize the Active Information Storage analysis
    network_analysis = ActiveInformationStorage()

    # Perform the network analysis
    with open(os.devnull, 'w') as fnull:
        with contextlib.redirect_stdout(fnull):
            results = network_analysis.analyse_network(settings, data, processes=processes)

    # Initialize a dictionary to store AIS values
    ais_values = {}
    ais_p_value = {}

    # Iterate over the processes analysed
    for process in results.processes_analysed:
        process_result = results.get_single_process(process,fdr=False)

        # Extract the AIS value
        ais_value = process_result['ais']
        p_value = process_result['ais_pval']

        # # Replace nan with 0 if necessary
        # if np.isnan(ais_value):
        #     ais_value = 0

        # Store the AIS value in the dictionary
        ais_values[process] = ais_value
        ais_p_value[process] = p_value


    return ais_values, ais_p_value


# def normalize_transfer_entropy(te_matrix):
#     """
#     Normalize the transfer entropy matrix over the number of inputs to a given node.
    
#     Args:
#         te_matrix (np.ndarray): Transfer entropy matrix.
    
#     Returns:
#         np.ndarray: Normalized transfer entropy matrix.
#     """
#     # Initialize normalized matrix
#     normalized_te_matrix = np.zeros_like(te_matrix)
    
#     # Normalize each column by the sum of its values
#     for i in range(te_matrix.shape[1]):
#         column_sum = np.sum(te_matrix[:, i])
#         if column_sum > 0:
#             normalized_te_matrix[:, i] = te_matrix[:, i] / column_sum
    
#     return normalized_te_matrix

# def update_transfer_entropy_with_ais(te_matrix, ais_values):
#     """
#     Update the transfer entropy matrix with AIS values.
    
#     Args:
#         te_matrix (np.ndarray): Normalized transfer entropy matrix.
#         ais_values (dict): AIS values for each node.
    
#     Returns:
#         np.ndarray: Updated transfer entropy matrix.
#     """
#     updated_te_matrix = np.zeros_like(te_matrix)
    
#     for target_node in range(te_matrix.shape[1]):
#         ais_value = ais_values.get(target_node, 0)
#         updated_te_matrix[:, target_node] = te_matrix[:, target_node] * ais_value
    
#     return updated_te_matrix



def geojson_to_timeseries(geojson_path):
    # Load GeoJSON file and ignore geometry
    gdf = gpd.read_file(geojson_path, ignore_geometry=True)

    # Determine the number of processes and samples
    n_processes = len(gdf)
    numerical_columns = gdf.select_dtypes(include=[np.number]).columns.tolist()
    n_samples = len(numerical_columns)
    
    if n_processes == 0 or n_samples == 0:
        raise ValueError("The GeoDataFrame does not contain enough data to create a time series.")

    # Create an empty array to hold the data
    data = np.zeros((n_samples, n_processes))
    
    # Fill the array with data from the DataFrame
    for i, row in enumerate(gdf.itertuples(index=False)):
        data[:, i] = np.array(row[:n_samples])
    
    # Convert to DataFrame
    df = pd.DataFrame(data, columns=[f'Process {i+1}' for i in range(n_processes)])
    #remove the first row as it is the population column
    df = df.iloc[1:]
    
    
    return df



def visualize_voronoi_with_network( voronoi_gdf, boundary, centroids_gdf, correlation_matrix, percentile=90, title = 'Economic Network', figsiz=(10,10), ax=None):
    """
    Visualizes the Voronoi GeoDataFrame alongside the boundary with an overlaid network
    based on centroid points and a correlation matrix, filtering edges by top percentile.

    Parameters:
    voronoi_gdf (geopandas.GeoDataFrame): GeoDataFrame of the Voronoi polygons.
    boundary (shapely.geometry): Boundary geometry used for clipping.
    centroids_gdf (geopandas.GeoDataFrame): GeoDataFrame containing centroid points.
    correlation_matrix (numpy.array): Square matrix of correlation values between centroids.
    percentile (float): Percentile for filtering edges based on their weight (default 90).
    """

    plt.ioff()

    if ax is None:
        fig, ax = plt.subplots(figsize=figsiz)
    else:
        fig = ax.figure

    
    plt.style.use(['science', 'ieee'])
    boundary_gdf = gpd.GeoDataFrame(geometry=[boundary], crs="EPSG:4326")
    boundary_gdf.boundary.plot(ax=ax, color="black", linewidth=2)
    voronoi_gdf.plot(ax=ax, alpha=0.5, edgecolor='black', cmap='viridis')

    # Check if the correlation matrix is symmetric
    is_symmetric = np.allclose(correlation_matrix, correlation_matrix.T, atol=1e-8)
    G = nx.DiGraph() if not is_symmetric else nx.Graph()

    # Create and populate graph
    positions = {idx: (point.x, point.y) for idx, point in enumerate(centroids_gdf.geometry)}
    for idx, pos in positions.items():
        G.add_node(idx, pos=pos)
    
    edges_weights = []
    for i in range(len(correlation_matrix)):
        for j in range(i + 1, len(correlation_matrix)):
            weight = abs(correlation_matrix[i, j])
            if weight > 0:  # Filter out non-connections
                edges_weights.append((i, j, weight))
                if not is_symmetric:
                    weight_reverse = abs(correlation_matrix[j, i])
                    if weight_reverse > 0:
                        edges_weights.append((j, i, weight_reverse))

    # Filter edges to include only top percentile
    edge_threshold = np.percentile([weight for _, _, weight in edges_weights], percentile)
    for i, j, weight in edges_weights:
        if weight >= edge_threshold:
            G.add_edge(i, j, weight=weight)

    # Draw the network with filtered edges
    pos = nx.get_node_attributes(G, 'pos')
    edges = G.edges()
    weights = np.array([G[u][v]['weight'] for u, v in edges])
    widths = (weights - weights.min()) / (weights.max() - weights.min()) * 3 + 0.5  # Scale edge width
    edge_colors = ['blue' if G[u][v]['weight'] > 0 else 'red' for u, v in G.edges()]  # Color by sign
    #make title
    ax.set_title(title)
    nx.draw_networkx_nodes(G, pos, node_color='red', node_size=50, ax=ax)
    # Draw bezier edges
    draw_network(G, pos, ax)
 
    plt.tight_layout()
    #plt.show()
    return ax



# def draw_network(G, pos, ax):
#     # Get edge weights and calculate scaled widths
#     weights = np.array([d['weight'] for (u, v, d) in G.edges(data=True)])
#     widths = (weights - weights.min()) / (weights.max() - weights.min()) * 3 + 0.5

#     # Iterate over edges to draw them with bezier curves
#     for ((u, v, d), width) in zip(G.edges(data=True), widths):
#         node_u_pos, node_v_pos = pos[u], pos[v]
#         # Set color based on the sign of the weight
#         edge_color = 'blue' if d['weight'] > 0 else 'red'
#         if isinstance(G, nx.DiGraph):
#             # Create a bezier curved arrow for the directed edge
#             arrow = FancyArrowPatch(node_u_pos, node_v_pos, arrowstyle='-|>',
#                                     color=edge_color, alpha=0.7,
#                                     mutation_scale=10.0,
#                                     linewidth=width, connectionstyle="arc3,rad=0.2")
#         else:
#             # Create a bezier curved line for the undirected edge
#             arrow = FancyArrowPatch(node_u_pos, node_v_pos, arrowstyle='-',
#                                     color=edge_color, alpha=0.7,
#                                     mutation_scale=10.0,
#                                     linewidth=width, connectionstyle="arc3,rad=0.2")
#         ax.add_patch(arrow)

def draw_network(G, pos, ax, edge_colors=None):
    # Get edge weights and calculate scaled widths
    weights = np.array([d['weight'] for (u, v, d) in G.edges(data=True)])
    widths = (weights - weights.min()) / (weights.max() - weights.min()) * 3 + 0.5

    # If no edge colors are provided, default to blue for positive and red for negative weights
    if edge_colors is None:
        edge_colors = ['blue' if d['weight'] > 0 else 'red' for (u, v, d) in G.edges(data=True)]
    
    # Iterate over edges to draw them with bezier curves
    for ((u, v, d), width, edge_color) in zip(G.edges(data=True), widths, edge_colors):
        node_u_pos, node_v_pos = pos[u], pos[v]
        if isinstance(G, nx.DiGraph):
            # Create a bezier curved arrow for the directed edge
            arrow = FancyArrowPatch(node_u_pos, node_v_pos, arrowstyle='-|>',
                                    color=edge_color, alpha=0.7,
                                    mutation_scale=10.0,
                                    linewidth=width, connectionstyle="arc3,rad=0.2")
        else:
            # Create a bezier curved line for the undirected edge
            arrow = FancyArrowPatch(node_u_pos, node_v_pos, arrowstyle='-',
                                    color=edge_color, alpha=0.7,
                                    mutation_scale=10.0,
                                    linewidth=width, connectionstyle="arc3,rad=0.2")
        ax.add_patch(arrow)



def visualize_voronoi_with_analysis(voronoi_gdf, boundary, centroids_gdf, correlation_matrix, clusters, centrality, percentile=90, title='Economic Network', figsiz=(10,10), ax=None):
    """
    Visualizes the Voronoi GeoDataFrame alongside the boundary with an overlaid network
    based on centroid points and a correlation matrix, filtering edges by top percentile.

    Parameters:
    voronoi_gdf (geopandas.GeoDataFrame): GeoDataFrame of the Voronoi polygons.
    boundary (shapely.geometry): Boundary geometry used for clipping.
    centroids_gdf (geopandas.GeoDataFrame): GeoDataFrame containing centroid points.
    correlation_matrix (numpy.array): Square matrix of correlation values between centroids.
    clusters (dict): Dictionary of node clusters.
    centrality (dict): Dictionary of node centrality values.
    percentile (float): Percentile for filtering edges based on their weight (default 90).
    """

    plt.ioff()

    if ax is None:
        fig, ax = plt.subplots(figsize=figsiz)
    else:
        fig = ax.figure

    plt.style.use(['science', 'ieee'])
    boundary_gdf = gpd.GeoDataFrame(geometry=[boundary], crs="EPSG:4326")
    boundary_gdf.boundary.plot(ax=ax, color="black", linewidth=2)
    
    # Check if the correlation matrix is symmetric
    is_symmetric = np.allclose(correlation_matrix, correlation_matrix.T, atol=1e-8)
    G = nx.DiGraph() if not is_symmetric else nx.Graph()

    # Create and populate graph
    positions = {idx: (point.x, point.y) for idx, point in enumerate(centroids_gdf.geometry)}
    for idx, pos in positions.items():
        G.add_node(idx, pos=pos)
    
    edges_weights = []
    for i in range(len(correlation_matrix)):
        for j in range(i + 1, len(correlation_matrix)):
            weight = abs(correlation_matrix[i, j])
            if weight > 0:  # Filter out non-connections
                edges_weights.append((i, j, weight))
                if not is_symmetric:
                    weight_reverse = abs(correlation_matrix[j, i])
                    if weight_reverse > 0:
                        edges_weights.append((j, i, weight_reverse))

    # Filter edges to include only top percentile
    edge_threshold = np.percentile([weight for _, _, weight in edges_weights], percentile)
    for i, j, weight in edges_weights:
        if weight >= edge_threshold:
            G.add_edge(i, j, weight=weight)

    # Assign colors to clusters
    unique_clusters = list(set(clusters.values()))
    cluster_colors = {cluster: idx for idx, cluster in enumerate(unique_clusters)}
    node_colors = [cluster_colors[clusters[node]] for node in G.nodes()]

    # Assign colors to clusters
    unique_clusters = list(set(clusters.values()))
    cmap = cm.get_cmap('tab20', len(unique_clusters))  # Using 'tab20' colormap for distinct colors
    cluster_colors = {cluster: mcolors.rgb2hex(cmap(idx)) for idx, cluster in enumerate(unique_clusters)}
    node_colors = [cluster_colors[clusters[node]] for node in G.nodes()]


    plt.ioff()  

    # Normalize centrality values for colormap
    centrality_values = np.array(list(centrality.values()))
    norm = Normalize(vmin=min(centrality_values), vmax=max(centrality_values))
    cmap = cm.get_cmap('Greys')

    # Color Voronoi cells by centrality
    voronoi_gdf['centrality'] = [centrality[idx] for idx in voronoi_gdf.index]
    voronoi_gdf.plot(ax=ax, column='centrality', cmap=cmap, norm=norm, alpha=0.5, edgecolor='black')


    plt.ioff()

    # Draw the network
    pos = nx.get_node_attributes(G, 'pos')
    edges = G.edges()
    weights = np.array([G[u][v]['weight'] for u, v in edges])
    widths = (weights - weights.min()) / (weights.max() - weights.min()) * 3 + 0.5  # Scale edge width
    edge_colors = [cluster_colors[clusters[u]] for u, v in G.edges()]

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=50, cmap='viridis', ax=ax)
    draw_network(G, pos, ax, edge_colors=edge_colors)
    
    ax.set_title(title)
    plt.tight_layout()

    return ax



def normalize_incoming_te(te_matrix, ais_values, factor_std=1):
    """
    Normalize the transfer entropy (TE) matrix.

    Parameters:
    te_matrix_path (str): Path to the CSV file containing the TE matrix.
    ais_values (dict): Dictionary containing the AIS values for each node.

    Returns:
    np.ndarray: Normalized TE matrix with values below the threshold set to 0.
    """
    # Set diagonal to 0
    np.fill_diagonal(te_matrix, 0)
    
    # Compute incoming TE sum for each node
    incoming_te_sum = np.sum(te_matrix, axis=1)
    
    # Get AIS values as np array
    ais_array = np.array(list(ais_values.values()))
    
    # Add AIS to values in incoming_te_sum
    incoming_te_sum = incoming_te_sum + ais_array
    
    # Divide every column in te_matrix by corresponding incoming_te_sum element
    normalized_te_matrix = te_matrix / incoming_te_sum[:, None]
    
    # Compute mean and standard deviation of normalized_te_matrix elements
    mean_te = np.mean(normalized_te_matrix)
    std_te = np.std(normalized_te_matrix)
    threshold = mean_te + std_te * factor_std
    
    # Set values in normalized_te_matrix below threshold to 0
    normalized_te_matrix[normalized_te_matrix < threshold] = 0
    
    return normalized_te_matrix


import numpy as np

# def AIS_to_te_matrix(te_matrix, ais_values, factor = 1):
#     """
#     Process the transfer entropy (TE) matrix by normalizing and thresholding.

#     Parameters:
#     te_matrix_path (str): Path to the CSV file containing the TE matrix.
#     ais_values (dict): Dictionary containing the AIS values for each node.

#     Returns:
#     np.ndarray: Processed TE matrix with thresholded values set to 0.
#     """

#     # Set diagonal to 0
#     np.fill_diagonal(te_matrix, 0)
    
#     # # Compute incoming TE sum for each node
#     # incoming_te_sum = np.sum(te_matrix, axis=1)
    
#     # Get AIS values as np array
#     ais_array = np.array(list(ais_values.values()))
    
#     # # Normalize the TE matrix by the incoming TE sum
#     # normalized_te_matrix = te_matrix / incoming_te_sum[:, None]
    
#     # Set all column entries to 0 if less than corresponding AIS value
#     threshold_matrix =  factor * ais_array[:, None]
#     te_matrix[te_matrix < threshold_matrix] = 0
    
#     return te_matrix

def AIS_TE_ratio(te_matrix, ais_values, ais_pvalues, factor=1):
    """
    Process the transfer entropy (TE) matrix by normalizing and thresholding based on AIS values and p-values.

    Parameters:
    te_matrix (np.ndarray): The TE matrix.
    ais_values (dict): Dictionary containing the AIS values for each node.
    ais_pvalues (dict): Dictionary containing the AIS p-values for each node.
    factor (float): Factor to multiply AIS values for thresholding.

    Returns:
    np.ndarray: The ratio of AIS to incoming TE sum for each node.
    """

    # Set diagonal to 0
    np.fill_diagonal(te_matrix, 0)
    
    # Compute incoming TE sum for each node
    incoming_te_sum = np.sum(te_matrix, axis=1)
    #count the number of incoming edges for each node
    incoming_edges = np.count_nonzero(te_matrix, axis=1)
    #compute mean of incoming TEs for each node
    mean_incoming_te = incoming_te_sum / incoming_edges

    
    # Get AIS values as np array, setting to NaN if p-value < 0.05
    ais_array = np.array([ais_values[node] if ais_pvalues.get(node, np.nan) >= 0.05 else np.nan for node in sorted(ais_values.keys())])
    
    # Compute the ratio of AIS to incoming TE sum
    ratio = ais_array / mean_incoming_te
    
    return ratio


def BH_correction(p_value_matrix, alpha=0.05):
    """
    Apply the Benjamini-Hochberg correction to a matrix of p-values.

    Parameters:
    p_value_matrix (numpy.ndarray): 2D array of p-values.
    alpha (float): Desired significance level.

    Returns:
    numpy.ndarray: 2D array indicating significant edges after correction (1 for significant, 0 for not significant).
    """
    # Flatten the p-value matrix and get sorted indices
    p_values = p_value_matrix.flatten()
    m = len(p_values)
    sorted_indices = np.argsort(p_values)
    sorted_p_values = p_values[sorted_indices]
    
    # Calculate thresholds
    thresholds = alpha * np.arange(1, m + 1) / m
    
    # Identify significant p-values
    significant = sorted_p_values <= thresholds
    max_significant_index = np.where(significant)[0].max() if significant.any() else -1

    # Create a matrix to hold the corrected significance levels
    corrected_significance_matrix = np.zeros_like(p_value_matrix, dtype=int)
    if max_significant_index >= 0:
        significant_threshold = sorted_p_values[max_significant_index]
        corrected_significance_matrix = p_value_matrix <= significant_threshold

    return corrected_significance_matrix


def BY_correction(p_value_matrix, alpha=0.05):
    """
    Apply the Benjamini-Yekutieli correction to a matrix of p-values.

    Parameters:
    p_value_matrix (numpy.ndarray): 2D array of p-values.
    alpha (float): Desired significance level.

    Returns:
    numpy.ndarray: 2D array indicating significant edges after correction (1 for significant, 0 for not significant).
    """
    # Flatten the p-value matrix and get sorted indices
    p_values = p_value_matrix.flatten()
    m = len(p_values)
    sorted_indices = np.argsort(p_values)
    sorted_p_values = p_values[sorted_indices]

    # Calculate the harmonic number
    harmonic_number = np.sum(1.0 / np.arange(1, m + 1))
    
    # Calculate thresholds
    thresholds = alpha * np.arange(1, m + 1) / (m * harmonic_number)
    
    # Identify significant p-values
    significant = sorted_p_values <= thresholds
    max_significant_index = np.where(significant)[0].max() if significant.any() else -1

    # Create a matrix to hold the corrected significance levels
    corrected_significance_matrix = np.zeros_like(p_value_matrix, dtype=int)
    if max_significant_index >= 0:
        significant_threshold = sorted_p_values[max_significant_index]
        corrected_significance_matrix = p_value_matrix <= significant_threshold

    return corrected_significance_matrix

def classify_core_periphery(te_matrix):
    """
    Classifies nodes in a graph as core or periphery based on the adjacency matrix.

    Parameters:
    te_matrix (np.ndarray): Adjacency matrix of the graph.

    Returns:
    dict: Dictionary of nodes with core/periphery classification.
    """
    # Initialize the algorithm
    algorithm = cpnet.KM_config()
    
    # Create a graph from the adjacency matrix
    G = nx.from_numpy_array(te_matrix)
    
    # Detect core-periphery structure
    algorithm.detect(G)
    
    # Get the coreness classification
    coreness = algorithm.get_coreness()
    
    # Create the attributes dictionary
    attributes = {}
    for node in G.nodes():
        attributes[node] = {
            'core_periphery_classification': coreness[node]
        }
    
    return attributes


def create_dataframe_from_voronoi(voronoi_gdf_path):

    dataframe = geojson_to_timeseries(voronoi_gdf_path)

    #compute percentage change for each process
    dataframe = dataframe.pct_change()
    #drop the first row
    dataframe = dataframe.iloc[1:]
    #reshape to np arraw with columns as entries
    data_array = dataframe.to_numpy().T

    return data_array

def compute_TE_significance(data_array,):
    #sturges rule for binning
    num_bins = int(1 + np.log2(data_array.shape[1]))
    #estimate perm needed for desired lowest p-value
    num_perm = int(1/0.05)-1
    #compute transfer entropy matrix
    settings = {
        'discretise_method': 'max_ent', # Discretise continuous data into equal-sized bins
        'n_discrete_bins': num_bins,  # Number of discrete bins/levels
        'alph1': 3,
        'alph2': 3,
        'alphc': 3
    }
    te_matrix, p_value_matrix = compute_TE(data_array,settings=settings , num_permutations=num_perm)
    
    return te_matrix, p_value_matrix

def correct_TE_matrix(te_matrix, p_value_matrix, save = False):
    "Correct for relevant subset of Te matrix by correcting for significance and tie strength"
    te_matrix[p_value_matrix > 0.051] = 0
    matrix_values = te_matrix.flatten()
    matrix_nozero = matrix_values[matrix_values != 0]
    #only consider values above 1 std from mean and enough entries exist
    if len(matrix_nozero) > 0.01*len(matrix_values):
        te_matrix[te_matrix < np.mean(matrix_values) + np.std(matrix_values) ] = 0
    
    if save:
        #save matrix as csv
        np.savetxt('te_matrix.csv', te_matrix, delimiter=',')

    return te_matrix

def partition_network(adj_matrix, directed=False):
    """
    Partitions a network into clusters using the Louvain method.

    Parameters:
    - adj_matrix (numpy.ndarray): The adjacency matrix of the network.
    - directed (bool): If True, treats the network as directed. Otherwise, treats it as undirected.

    Returns:
    - dict: A dictionary where keys are node indices and values are cluster indices.
    """
    # Create an igraph graph from the adjacency matrix
    if directed:
        g = ig.Graph.Weighted_Adjacency(adj_matrix.tolist(), mode=ig.ADJ_DIRECTED)
    else:
        g = ig.Graph.Weighted_Adjacency(adj_matrix.tolist(), mode=ig.ADJ_UNDIRECTED)
    
    # Use the Leiden method for community detection
    partition = la.find_partition(g, la.CPMVertexPartition, resolution_parameter=0.01)

    # Convert partition to a dictionary
    partition_dict = {i: membership for i, membership in enumerate(partition.membership)}

    return partition_dict

def measure_attributes_of_graph(te_matrix):
    G = nx.from_numpy_array(te_matrix)
    
    # Compute centrality for each node
    centrality = nx.eigenvector_centrality(G)
    partition = partition_network(te_matrix, directed=True)
    
    # Compute additional metrics
    degree_centrality = nx.degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)
    closeness_centrality = nx.closeness_centrality(G)
    pagerank = nx.pagerank(G)
    clustering_coefficient = nx.clustering(G)
    
    # Compute the number of different partitions a node is connected to
    partition_neighbors_count = {}
    for node in G.nodes():
        neighbor_partitions = {partition[neighbor] for neighbor in G.neighbors(node)}
        partition_neighbors_count[node] = len(neighbor_partitions)
    
    # Combine all metrics into a dictionary
    attributes = {}
    for node in G.nodes():
        attributes[node] = {
            'degree_centrality': degree_centrality[node],
            'betweenness_centrality': betweenness_centrality[node],
            'closeness_centrality': closeness_centrality[node],
            'eigenvector_centrality': centrality[node],
            'pagerank': pagerank[node],
            'clustering_coefficient': clustering_coefficient[node],
            'partition': partition[node],
            'partition_neighbors_count': partition_neighbors_count[node]
        }

    return attributes

def assign_attributes_to_geojson(attributes, geojson_data):
    # Iterate through the GeoJSON features and add the corresponding attributes
    for i, feature in enumerate(geojson_data['features']):
        # Use the feature 'id' if it exists, otherwise use the index 'i'
        node_id = feature.get('id', i)
        if node_id in attributes:
            for attr_name, attr_value in attributes[node_id].items():
                feature['properties'][attr_name] = attr_value
        else:
            raise ValueError(f"Feature ID {node_id} not found in attribute dictionary.")
    
    return geojson_data
