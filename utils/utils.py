import pandas as pd
import numpy as np
import random
import torch
from torch_geometric.data import Data
import scipy.stats as stats
from collections import Counter

column_name_mapping = {
    # Basic Player Information
    "AGE": "Age",
    "player_height": "Player Height",
    "player_weight": "Player Weight",
    "country": "Country",
    # Traditional Statistics - Playing Time
    "MP": "Minutes Played",
    "GP": "Games Played",
    "MPG": "Minutes Per Game",
    # Traditional Statistics - Scoring
    "FG": "Field Goals Made",
    "FGA": "Field Goal Attempts",
    "FG%": "Field Goal Percentage",
    "3P": "3-Point Field Goals Made",
    "3PA": "3-Point Field Goal Attempts",
    "3P%": "3-Point Field Goal Percentage",
    "2P": "2-Point Field Goals Made",
    "2PA": "2-Point Field Goal Attempts",
    "2P%": "2-Point Field Goal Percentage",
    "eFG%": "Effective Field Goal Percentage",
    "FT": "Free Throws Made",
    "FTA": "Free Throw Attempts",
    "FT%": "Free Throw Percentage",
    "POINTS": "Points Scored",
    # Traditional Statistics - Rebounding
    "ORB": "Offensive Rebounds",
    "DRB": "Defensive Rebounds",
    "TRB": "Total Rebounds",
    # Traditional Statistics - Playmaking and Defense
    "AST": "Assists",
    "STL": "Steals",
    "BLK": "Blocks",
    "TOV": "Turnovers",
    "PF_x": "Personal Fouls",  # Note: '_x' may result from merging datasets
    "PF_y": "Power Forward Position",  # If 'PF' refers to position; '_y' from merging
    # Advanced Metrics
    "ORPM": "Offensive Real Plus-Minus",
    "DRPM": "Defensive Real Plus-Minus",
    "RPM": "Real Plus-Minus",
    "WINS_RPM": "Wins Above Replacement Player (RPM)",
    "PIE": "Player Impact Estimate",
    "PACE": "Team Pace",
    "W": "Team Wins",
    # Player Positions
    "C": "Center Position",
    "PF-C": "Power Forward-Center Position",
    "PG": "Point Guard Position",
    "SF": "Small Forward Position",
    "SG": "Shooting Guard Position",
    ##### TEAM NAMES
    #     "ATL": "Atlanta Hawks",
    #     "ATL/CLE": "Atlanta Hawks / Cleveland Cavaliers",
    #     "ATL/LAL": "Atlanta Hawks / Los Angeles Lakers",
    #     "BKN": "Brooklyn Nets",
    #     "BKN/WSH": "Brooklyn Nets / Washington Wizards",
    #     "BOS": "Boston Celtics",
    #     "CHA": "Charlotte Hornets",
    #     "CHI": "Chicago Bulls",
    #     "CHI/OKC": "Chicago Bulls / Oklahoma City Thunder",
    #     "CLE": "Cleveland Cavaliers",
    #     "CLE/DAL": "Cleveland Cavaliers / Dallas Mavericks",
    #     "CLE/MIA": "Cleveland Cavaliers / Miami Heat",
    #     "DAL": "Dallas Mavericks",
    #     "DAL/BKN": "Dallas Mavericks / Brooklyn Nets",
    #     "DAL/PHI": "Dallas Mavericks / Philadelphia 76ers",
    #     "DEN": "Denver Nuggets",
    #     "DEN/CHA": "Denver Nuggets / Charlotte Hornets",
    #     "DEN/POR": "Denver Nuggets / Portland Trail Blazers",
    #     "DET": "Detroit Pistons",
    #     "GS": "Golden State Warriors",
    #     "GS/CHA": "Golden State Warriors / Charlotte Hornets",
    #     "GS/SAC": "Golden State Warriors / Sacramento Kings",
    #     "HOU": "Houston Rockets",
    #     "HOU/LAL": "Houston Rockets / Los Angeles Lakers",
    #     "HOU/MEM": "Houston Rockets / Memphis Grizzlies",
    #     "IND": "Indiana Pacers",
    #     "LAC": "Los Angeles Clippers",
    #     "LAL": "Los Angeles Lakers",
    #     "MEM": "Memphis Grizzlies",
    #     "MIA": "Miami Heat",
    #     "MIL": "Milwaukee Bucks",
    #     "MIL/CHA": "Milwaukee Bucks / Charlotte Hornets",
    #     "MIN": "Minnesota Timberwolves",
    #     "NO": "New Orleans Pelicans",
    #     "NO/DAL": "New Orleans Pelicans / Dallas Mavericks",
    #     "NO/MEM": "New Orleans Pelicans / Memphis Grizzlies",
    #     "NO/MIL": "New Orleans Pelicans / Milwaukee Bucks",
    #     "NO/MIN/SAC": "New Orleans Pelicans / Minnesota Timberwolves / Sacramento Kings",
    #     "NO/ORL": "New Orleans Pelicans / Orlando Magic",
    #     "NO/SAC": "New Orleans Pelicans / Sacramento Kings",
    #     "NY": "New York Knicks",
    #     "NY/PHI": "New York Knicks / Philadelphia 76ers",
    #     "OKC": "Oklahoma City Thunder",
    #     "ORL": "Orlando Magic",
    #     "ORL/TOR": "Orlando Magic / Toronto Raptors",
    #     "PHI": "Philadelphia 76ers",
    #     "PHI/OKC": "Philadelphia 76ers / Oklahoma City Thunder",
    #     "PHX": "Phoenix Suns",
    #     "POR": "Portland Trail Blazers",
    #     "SA": "San Antonio Spurs",
    #     "SAC": "Sacramento Kings",
    #     "TOR": "Toronto Raptors",
    #     "UTAH": "Utah Jazz",
    #     "WSH": "Washington Wizards",
}

columns_to_drop = {
    # Team Affiliations
    "ATL": "Atlanta Hawks",
    "ATL/CLE": "Atlanta Hawks / Cleveland Cavaliers",
    "ATL/LAL": "Atlanta Hawks / Los Angeles Lakers",
    "BKN": "Brooklyn Nets",
    "BKN/WSH": "Brooklyn Nets / Washington Wizards",
    "BOS": "Boston Celtics",
    "CHA": "Charlotte Hornets",
    "CHI": "Chicago Bulls",
    "CHI/OKC": "Chicago Bulls / Oklahoma City Thunder",
    "CLE": "Cleveland Cavaliers",
    "CLE/DAL": "Cleveland Cavaliers / Dallas Mavericks",
    "CLE/MIA": "Cleveland Cavaliers / Miami Heat",
    "DAL": "Dallas Mavericks",
    "DAL/BKN": "Dallas Mavericks / Brooklyn Nets",
    "DAL/PHI": "Dallas Mavericks / Philadelphia 76ers",
    "DEN": "Denver Nuggets",
    "DEN/CHA": "Denver Nuggets / Charlotte Hornets",
    "DEN/POR": "Denver Nuggets / Portland Trail Blazers",
    "DET": "Detroit Pistons",
    "GS": "Golden State Warriors",
    "GS/CHA": "Golden State Warriors / Charlotte Hornets",
    "GS/SAC": "Golden State Warriors / Sacramento Kings",
    "HOU": "Houston Rockets",
    "HOU/LAL": "Houston Rockets / Los Angeles Lakers",
    "HOU/MEM": "Houston Rockets / Memphis Grizzlies",
    "IND": "Indiana Pacers",
    "LAC": "Los Angeles Clippers",
    "LAL": "Los Angeles Lakers",
    "MEM": "Memphis Grizzlies",
    "MIA": "Miami Heat",
    "MIL": "Milwaukee Bucks",
    "MIL/CHA": "Milwaukee Bucks / Charlotte Hornets",
    "MIN": "Minnesota Timberwolves",
    "NO": "New Orleans Pelicans",
    "NO/DAL": "New Orleans Pelicans / Dallas Mavericks",
    "NO/MEM": "New Orleans Pelicans / Memphis Grizzlies",
    "NO/MIL": "New Orleans Pelicans / Milwaukee Bucks",
    "NO/MIN/SAC": "New Orleans Pelicans / Minnesota Timberwolves / Sacramento Kings",
    "NO/ORL": "New Orleans Pelicans / Orlando Magic",
    "NO/SAC": "New Orleans Pelicans / Sacramento Kings",
    "NY": "New York Knicks",
    "NY/PHI": "New York Knicks / Philadelphia 76ers",
    "OKC": "Oklahoma City Thunder",
    "ORL": "Orlando Magic",
    "ORL/TOR": "Orlando Magic / Toronto Raptors",
    "PHI": "Philadelphia 76ers",
    "PHI/OKC": "Philadelphia 76ers / Oklahoma City Thunder",
    "PHX": "Phoenix Suns",
    "POR": "Portland Trail Blazers",
    "SA": "San Antonio Spurs",
    "SAC": "Sacramento Kings",
    "TOR": "Toronto Raptors",
    "UTAH": "Utah Jazz",
    "WSH": "Washington Wizards",
}


columns_for_percentile = [
    "Age",
    "Minutes Played",
    "Field Goals Made",
    "Field Goal Attempts",
    "Field Goal Percentage",
    "3-Point Field Goals Made",
    "3-Point Field Goal Attempts",
    "3-Point Field Goal Percentage",
    "2-Point Field Goals Made",
    "2-Point Field Goal Attempts",
    "2-Point Field Goal Percentage",
    "Effective Field Goal Percentage",
    "Free Throws Made",
    "Free Throw Attempts",
    "Free Throw Percentage",
    "Offensive Rebounds",
    "Defensive Rebounds",
    "Total Rebounds",
    "Assists",
    "Steals",
    "Blocks",
    "Turnovers",
    "Personal Fouls",
    "Points Scored",
    "Games Played",
    "Minutes Per Game",
    "Offensive Real Plus-Minus",
    "Defensive Real Plus-Minus",
    "Real Plus-Minus",
    "Wins Above Replacement Player (RPM)",
    "Player Impact Estimate",
    "Team Pace",
    "Team Wins",
    "Player Height",
    "Player Weight",
]


def feature_norm(features):

    min_values = features.min(axis=0)[0]
    max_values = features.max(axis=0)[0]

    return 2 * (features - min_values).div(max_values - min_values) - 1


def prepare_graph_data(csv_path, edge_path, seed=42):
    # columns_to_drop_list = []
    # Load user features and edges
    user_features = pd.read_csv(csv_path)
    user_ids = user_features["user_id"].values

    targets = torch.tensor(user_features["SALARY"].values, dtype=torch.long)
    targets = torch.where(
        targets <= 0,
        torch.tensor(0, dtype=torch.long),
        torch.tensor(1, dtype=torch.long),
    )  # Merging the -1 and 0 class to 0

    columns_to_drop_list = list(columns_to_drop.keys())
    columns_to_drop_list.append("SALARY")

    user_features.drop(columns=columns_to_drop_list, axis=1, inplace=True)
    user_features.rename(columns=column_name_mapping, inplace=True)
    features = torch.tensor(
        user_features.drop("user_id", axis=1).values, dtype=torch.float
    )
    header = list(user_features.drop("user_id", axis=1).columns)

    edges = pd.read_csv(edge_path, sep="\t", header=None, names=["source", "target"])
    source_nodes = edges["source"].values
    target_nodes = edges["target"].values

    # Filter out isolated nodes
    node_set = set(source_nodes).union(set(target_nodes))
    user_features_filtered = user_features[user_features["user_id"].isin(node_set)]
    filtered_indices = [i for i, user_id in enumerate(user_ids) if user_id in node_set]
    features = features[filtered_indices]
    features = feature_norm(features)
    targets = targets[filtered_indices]
    user_ids = user_features_filtered["user_id"].values

    # Create node ID mapping
    user_id_map = {user_id: i for i, user_id in enumerate(user_ids)}
    mapped_sources = [user_id_map[src] for src in source_nodes if src in user_id_map]
    mapped_targets = [user_id_map[dst] for dst in target_nodes if dst in user_id_map]

    # Make the graph undirected
    all_sources = mapped_sources + mapped_targets
    all_targets = mapped_targets + mapped_sources

    # Create edge_index and remove duplicate edges
    edge_index = torch.tensor([all_sources, all_targets], dtype=torch.long)
    edge_index = torch.unique(edge_index, dim=1)

    # Create train, val, and test masks with random shuffling
    num_nodes = features.shape[0]
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    indices = torch.randperm(num_nodes)

    # Define sizes for train, validation, and test splits
    train_size = int(num_nodes * 0.6)
    val_size = int(num_nodes * 0.2)
    test_size = num_nodes - train_size - val_size

    # Split indices into train, val, and test
    train_indices = indices[:train_size]
    val_indices = indices[train_size : train_size + val_size]
    test_indices = indices[train_size + val_size :]

    # Initialize masks
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    # Set masks based on the indices
    train_mask[train_indices] = True
    val_mask[val_indices] = True
    test_mask[test_indices] = True

    # Create the PyTorch Geometric Data object
    data = Data(
        x=features,
        edge_index=edge_index,
        y=targets,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )

    return data, header, user_id_map, user_features_filtered


def binary_accuracy(preds, labels):
    preds = (preds > 0.5).float()
    correct = (preds == labels).float()
    return correct.sum() / len(correct)


def set_seed(seed=42):
    """Set seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = (
        True  # Ensure deterministic behavior in CuDNN (if using GPU)
    )
    torch.backends.cudnn.benchmark = (
        False  # Disable CuDNN benchmark to ensure reproducibility
    )


def get_subnodes_feat_importance(
    edges_df, data_x, data_edge_index, explainer, header, topk=5
):
    """
    Generates explanations for each node_index in edges_df['Source'] using explainer_gnnx.

    Parameters:
    - edges_df: pandas DataFrame containing at least a 'Source' column with node indices.
    - data: A data object containing 'x' and 'edge_index' attributes (e.g., PyTorch Geometric data object).
    - explainer_gnnx: A function or method that generates explanations, callable as explainer_gnnx(data.x, data.edge_index, index=node_index).

    Returns:
    - explanations: A dictionary with node_index as keys and their explanations as values.
    """
    explanations = {}
    for node_index in edges_df["Source"].unique():
        # Ensure node_index is an integer
        node_index = int(node_index)
        # Generate the explanation for the current node_index

        explanation_gnnx = explainer(data_x, data_edge_index, index=node_index)

        _, df_score_fi = explanation_gnnx.visualize_feature_importance(
            feat_labels=header, top_k=topk
        )

        explanations[node_index] = df_score_fi

    return explanations


def dataframe_to_tuples(df):
    """
    Convert a pandas DataFrame into a list of tuples, preserving original data types, and rounding float values
    to 3 decimal places.

    Parameters:
    df (pd.DataFrame): The DataFrame to convert

    Returns:
    list: A list of tuples containing only the column values, with original types (ints preserved, floats rounded)
    """

    def round_floats(value):
        # Round only float values to 2 decimal places, leave others unchanged
        if isinstance(value, float):
            return round(value, 2)
        return value

    # Create a list of tuples from DataFrame values, rounding floats but preserving other types
    return [
        tuple(round_floats(val) for val in row)
        for row in df.itertuples(index=False, name=None)
    ]


def dataframe_to_dict(df, first_col_key=False):
    """
    Convert a pandas DataFrame into a dictionary where:
    - If there are 2 columns, the key is the value of the first column and the value is the second column.
    - If there are more than 2 columns, the key is the column name and the value is a set of (feature value, percentile).

    Parameters:
    df (pd.DataFrame): The DataFrame to convert
    first_col_key (bool): If True, use the first column as the key in the dictionary.

    Returns:
    dict: A dictionary representation of the DataFrame
    """
    # Case where DataFrame has more than 2 columns
    # if df.shape[1] > 2:
    #     result = {}
    #     for col in df.columns:
    #         # If first column key is True, ignore first column in keys
    #         if first_col_key and col == df.columns[0]:
    #             continue
    #         # Create dictionary where each key is a column name and values are sets of (feature value, percentile)
    #         result[col] = set(df[col])
    #     return result

    # # Default case when DataFrame has 2 or fewer columns
    # else:
    if first_col_key:
        first_col = df.columns[0]
        result = df.set_index(first_col).to_dict(orient="index")
    else:
        result = df.to_dict(orient="index")

    return result


def add_percentile_to_node_features(
    user_features, columns_for_percentile, target_user_id
):
    """
    Calculate percentiles for the specified user and append them to the user_features DataFrame.
    The index will reflect 0 for feature values and 1 for percentile values.

    Parameters:
    user_features (pd.DataFrame): The entire user features DataFrame.
    columns_for_percentile (list): List of columns for which percentile is to be calculated.
    target_user_id (int): The ID of the user for which percentiles will be calculated.

    Returns:
    pd.DataFrame: A DataFrame with the original features and their corresponding percentiles.
    """
    # Select the user's feature values
    node_features = user_features[user_features["user_id"] == target_user_id]

    # Initialize a new row with 'Percentile' in the 'user_id' column
    percentile_row = node_features.copy()
    percentile_row["user_id"] = "Percentile"

    # Calculate percentile rank for specified columns
    for col in node_features.columns:
        if col in columns_for_percentile:
            # Calculate the percentile rank for specified columns
            user_value = node_features[col].values[0]
            percentile_rank = stats.percentileofscore(user_features[col], user_value)
            percentile_row[col] = round(percentile_rank, 2)  # Round to 2 decimal points
        elif col != "user_id":  # Exclude 'user_id' from this process
            # Insert "Binary Column" or other informative placeholder for non-percentile columns
            percentile_row[col] = "Binary Column"

    # Concatenate the original row and the percentile row
    node_features = pd.concat([node_features, percentile_row], ignore_index=True)
    node_features.drop("user_id", axis=1, inplace=True)

    # Rename the index to reflect 0 -> feature value, 1 -> percentile
    node_features.index = ["Feature Values", "Percentile"]

    # Round all numeric columns in node_features to 3 decimal places
    node_features = node_features.map(
        lambda x: round(x, 2) if isinstance(x, (int, float)) else x
    )

    return node_features


def calculate_node_edges_and_label_ratios(
    data, node_id, label_0="Low Salary", label_1="High Salary"
):
    """
    Calculate the number of edges and the label ratios of connected nodes for a specified node in a PyTorch Geometric graph.

    Parameters:
    data (torch_geometric.data.Data): The PyG Data object containing the graph and node labels.
    node_id (int): The target node for which to calculate the degree and label ratios.
    label_0 (str): Custom name for the label corresponding to class 0.
    label_1 (str): Custom name for the label corresponding to class 1.

    Returns:
    dict: A dictionary containing the number of edges ('Number of Edges') and a dictionary of label ratios ('Label Ratios').
    """
    # 1. Calculate the number of edges (degree) for the specified node
    edge_index = data.edge_index
    is_connected = (edge_index[0] == node_id) | (edge_index[1] == node_id)
    connected_edges = edge_index[:, is_connected]
    num_edges = connected_edges.size(1)

    # 2. Get the labels of the nodes connected to the specified node
    connected_nodes = torch.where(
        edge_index[0] == node_id, edge_index[1], edge_index[0]
    )[is_connected]
    connected_labels = data.y[connected_nodes].tolist()

    # 3. Calculate the ratio of each label
    label_counts = Counter(connected_labels)
    total_connections = sum(label_counts.values())

    # Map labels to custom names based on label_0 and label_1
    label_mapping = {0: label_0, 1: label_1}

    label_ratios = {
        label_mapping[label]: round(count / total_connections, 2)
        for label, count in label_counts.items()
    }

    # Return the number of edges and the label ratios
    return {"Number of Edges": num_edges, "Label Ratios": label_ratios}


def add_percentile_for_multiple_users_as_df_and_convert(
    user_features, columns_for_percentile, target_node_ids, user_id_map
):
    """
    Calculate percentiles for a list of users, return a list of dictionaries, and replace the user_id with the corresponding
    node_id from the user_id_map. The target_node_ids represent the node IDs (values in user_id_map), which are mapped
    back to the corresponding user_id.

    Parameters:
    user_features (pd.DataFrame): The entire user features DataFrame.
    columns_for_percentile (list): List of columns for which percentile is to be calculated.
    target_node_ids (list): A list of node IDs (values in user_id_map) for which percentiles will be calculated.
    user_id_map (dict): A dictionary mapping user IDs to node IDs.

    Returns:
    list: A list of dictionaries, where each dictionary represents the feature values and percentiles for each user.
    """
    # Reverse the user_id_map so we can map node_ids back to user_ids
    node_to_user_id_map = {v: k for k, v in user_id_map.items()}

    all_user_percentiles = []

    for target_node_id in target_node_ids:
        # Map node_id back to user_id
        target_user_id = node_to_user_id_map.get(target_node_id)

        if target_user_id is None:
            continue  # If no mapping found, skip this node

        # Select the user's feature values
        node_features = user_features[user_features["user_id"] == target_user_id]

        # Initialize a new row with 'Percentile' in the 'user_id' column
        percentile_row = node_features.copy()
        percentile_row["user_id"] = "Percentile"

        # Calculate percentile rank for specified columns
        for col in node_features.columns:
            if col in columns_for_percentile:
                # Calculate the percentile rank for specified columns
                user_value = node_features[col].values[0]
                percentile_rank = stats.percentileofscore(
                    user_features[col], user_value
                )
                percentile_row[col] = round(
                    percentile_rank, 2
                )  # Round to 2 decimal points
            elif col != "user_id":  # Exclude 'user_id' from this process
                # Insert "Binary Column" or other informative placeholder for non-percentile columns
                percentile_row[col] = "Binary Column"

        # Concatenate the original row and the percentile row
        node_features = pd.concat([node_features, percentile_row], ignore_index=True)

        # Add user_id column to identify user
        node_features["user_id"] = [target_user_id, "Percentile"]

        # Convert the DataFrame (two rows) to a dictionary
        user_data_df = node_features.iloc[
            0:2
        ].copy()  # Get two rows (feature values + percentiles)
        user_dict = dataframe_to_dict(user_data_df)  # Convert them into a dictionary

        # Now, use the node_id as the key and remove "Percentile"
        user_dict_final = {
            target_node_id: {k: v for k, v in user_dict.items() if k != "user_id"}
        }

        # Append the resulting dictionary
        all_user_percentiles.append(user_dict_final)

    return all_user_percentiles


def predict_node_class_and_prob(gcn, data, node_index, device):
    """
    Predict the class and probability for a specific node in a GCN model.

    Parameters:
    gcn (torch.nn.Module): The GCN model.
    data (torch_geometric.data.Data): The PyG Data object containing the graph and features.
    node_index (int): The index of the node for which to make the prediction.
    device (torch.device): The device to move the data and model to (e.g., 'cpu' or 'cuda').

    Returns:
    dict: A dictionary containing the predicted probability and the binary class prediction.
    """
    # Ensure the model is in evaluation mode
    gcn.eval()

    # Move the data to the appropriate device (e.g., GPU or CPU)
    data = data.to(device)

    # Get the model's output (probabilities) for all nodes, then select the specific node
    with torch.no_grad():
        # Forward pass to get probabilities for all nodes
        out = gcn(
            data.x, data.edge_index
        ).squeeze()  # out will be in the range [0, 1] due to sigmoid

        # Extract the probability for the specified node
        node_prob = out[node_index].item()

        # Convert the probability to a binary prediction (0 or 1)
        node_prediction = 1 if node_prob > 0.5 else 0

    if node_prediction == 1:
        node_prediction = "High Salary"
    else:
        node_prediction = "Low Salary"

    # Return the probability and predicted class for the node
    return {
        "Target Node Index": node_index,
        "Predicted Class": node_prediction,
    }


def predict_reg(model, data, node_index, device):
    """
    Predict the regression score for a specific node in a GCN model.

    Parameters:
    model (torch.nn.Module): The GNN model.
    data (torch_geometric.data.Data): The PyG Data object containing the graph and features.
    node_index (int): The index of the node for which to make the prediction.
    device (torch.device): The device to move the data and model to (e.g., 'cpu' or 'cuda').

    Returns:
    dict: Predicted regression score for the node.
    """
    # Ensure the model is in evaluation mode
    model.eval()

    # Move the data and model to the specified device
    data = data.to(device)
    model = model.to(device)

    # Perform inference to get the regression score for the node
    with torch.no_grad():
        # Forward pass to get output for all nodes
        out = model(data.x, data.edge_index)

        # Extract the prediction for the specific node
        node_prediction = out[node_index].item()  # Convert to a scalar value

    # Return the regression score in a dictionary
    return {"node_index": node_index, "predicted_score": round(node_prediction, 2)}


def get_source_node_labels_as_dict_with_titles(edges_df, data, dataset, dataframe):
    """
    Given an edges DataFrame, a PyG Data object, and a categorical DataFrame, return the labels of nodes listed
    in the 'Source Node' column as a dictionary. For the 'NBA' dataset, it returns the salary category
    ('Low Salary' for 0 and 'High Salary' for 1). For the 'IMDB' dataset, it returns the IMDB rating score,
    using movie titles from the categorical DataFrame as keys.

    Parameters:
    edges_df (pd.DataFrame): The DataFrame containing the 'Source Node' column with node IDs.
    data (torch_geometric.data.Data): The PyG Data object containing node labels in 'data.y'.
    dataset (str): The name of the dataset ('NBA' or 'IMDB').
    categorical_df (pd.DataFrame): The DataFrame containing the 'Series_Title' column with movie titles.

    Returns:
    dict: A dictionary where keys are movie titles and values are dictionaries with either 'Label' or 'Rating'.
    """
    # Convert the 'Source Node' column to a list of node IDs
    source_nodes = edges_df["Source Node"].tolist()

    # Get labels for each node in 'Source' using the 'data.y' attribute
    source_labels = data.y[source_nodes]

    # Get corresponding movie titles from the categorical DataFrame
    movie_titles = dataframe.loc[source_nodes, "Series_Title"].tolist()

    if dataset == "NBA":
        # Convert label values to salary categories ('Low Salary' for 0 and 'High Salary' for 1)
        salary_mapping = {0: "Low Salary", 1: "High Salary"}

        # Create a dictionary with movie titles as keys and 'Label' as value
        source_node_labels_dict = {
            title: {"Label": salary_mapping[label.item()]}
            for title, label in zip(movie_titles, source_labels)
        }

        return source_node_labels_dict

    elif dataset == "IMDB":
        # Assuming the labels are floating-point numbers representing ratings
        source_node_labels_dict = {
            title: {"Rating": round(label.item(), 2)}
            for title, label in zip(movie_titles, source_labels)
        }

        return source_node_labels_dict

    else:
        raise ValueError(f"Unsupported dataset: {dataset}")


def get_movie_stats(df, movie_idx):
    """
    Create a DataFrame with numerical values and their percentiles for a specific movie.

    Args:
        df: Original DataFrame with movie data
        movie_idx: Index of the movie to analyze

    Returns:
        DataFrame with values and percentiles for selected numerical features
    """
    # Selected columns
    columns = ["Duration", "Meta_score", "No_of_Votes", "Gross"]

    # Get values for the specific movie
    values = df.loc[movie_idx, columns]

    # Calculate percentiles for each value
    percentiles = {}
    for col in columns:
        value = df.loc[movie_idx, col]
        percentile = (df[col] <= value).mean() * 100
        percentiles[col] = percentile

    # Create DataFrame with both values and percentiles
    stats_df = pd.DataFrame({"Value": values, "Percentile": percentiles})

    # Format values based on column type
    stats_df["Value"] = stats_df["Value"].apply(
        lambda x: f"{x:,.0f}" if x >= 1000 else f"{x:.2f}"
    )
    stats_df["Percentile"] = stats_df["Percentile"].apply(lambda x: f"{x:.1f}%")

    categorical_cols = df.columns.difference(columns).tolist()
    categorical_cols.remove("IMDB_Rating")
    categorical_values = df.loc[movie_idx, categorical_cols]

    categorical_df = pd.DataFrame(
        {
            "Categories": categorical_values,
        }
    )

    return stats_df, categorical_df


df_column_descriptions = {
    "Series_Title": ("The official title of the movie or TV series as listed on IMDb."),
    "Released_Year": ("The year the movie or TV series was released to the public."),
    "Certificate": (
        "The film classification indicating the appropriate audience based on content. "
        "Examples include PG-13 (Parents Strongly Cautioned), R (Restricted), etc."
    ),
    "Duration": (
        "The duration of the movie or TV series in minutes. "
        "For movies, it's the total Duration; for TV series, it might refer to the average episode length."
    ),
    "Genre": (
        "The categories or genres that describe the movie's content. "
        "Multiple genres are typically separated by commas, e.g., Action, Drama."
    ),
    "IMDB_Rating": (
        "The average IMDb rating of the movie on a scale from 0 to 10, reflecting user reviews and ratings."
    ),
    "Overview": (
        "A brief summary or synopsis of the movie's plot, providing an overview of the storyline."
    ),
    "Meta_score": (
        "The Metascore of the movie, which aggregates critic reviews and ratings to provide an overall score."
    ),
    "Director": (
        "The name(s) of the director(s) responsible for the movie's direction. "
        "Multiple directors are typically separated by commas."
    ),
    "Star1": ("The name of the first lead actor or actress in the movie."),
    "Star2": ("The name of the second lead actor or actress in the movie."),
    "Star3": ("The name of the third lead actor or actress in the movie."),
    "Star4": ("The name of the fourth lead actor or actress in the movie."),
    "Star1_appearances": ("Count of movies first actor appears in (normalized)."),
    "Star2_appearances": ("Count of movies second actor appears in (normalized)."),
    "Star3_appearances": ("Count of movies third actor appears in (normalized)."),
    "Star4_appearances": ("Count of movies fourth actor appears in (normalized)."),
    "No_of_Votes": (
        "The total number of user votes or ratings the movie has received on IMDb. "
        "A higher number indicates more widespread viewer engagement."
    ),
    "Gross": (
        "The total box office revenue generated by the movie, typically expressed in USD. "
    ),
}
