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
    "PF_x": "Personal Fouls",
    "PF_y": "Power Forward Position", 
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

    user_features = pd.read_csv(csv_path)
    user_ids = user_features["user_id"].values

    targets = torch.tensor(user_features["SALARY"].values, dtype=torch.long)
    targets = torch.where(
        targets <= 0,
        torch.tensor(0, dtype=torch.long),
        torch.tensor(1, dtype=torch.long),
    )  # Merging the -1 and 0 class to 0 as a 'Low salary'

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

    node_set = set(source_nodes).union(set(target_nodes))
    user_features_filtered = user_features[user_features["user_id"].isin(node_set)]
    filtered_indices = [i for i, user_id in enumerate(user_ids) if user_id in node_set]
    features = features[filtered_indices]
    features = feature_norm(features)
    targets = targets[filtered_indices]
    user_ids = user_features_filtered["user_id"].values

    user_id_map = {user_id: i for i, user_id in enumerate(user_ids)}
    mapped_sources = [user_id_map[src] for src in source_nodes if src in user_id_map]
    mapped_targets = [user_id_map[dst] for dst in target_nodes if dst in user_id_map]

    all_sources = mapped_sources + mapped_targets
    all_targets = mapped_targets + mapped_sources

    edge_index = torch.tensor([all_sources, all_targets], dtype=torch.long)
    edge_index = torch.unique(edge_index, dim=1)

    num_nodes = features.shape[0]
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    indices = torch.randperm(num_nodes)

    train_size = int(num_nodes * 0.6)
    val_size = int(num_nodes * 0.2)
    test_size = num_nodes - train_size - val_size

    train_indices = indices[:train_size]
    val_indices = indices[train_size : train_size + val_size]
    test_indices = indices[train_size + val_size :]

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[train_indices] = True
    val_mask[val_indices] = True
    test_mask[test_indices] = True

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
        True
    )
    torch.backends.cudnn.benchmark = (
        False
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
        node_index = int(node_index)
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
        if isinstance(value, float):
            return round(value, 2)
        return value
    
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
    if first_col_key:
        first_col = df.columns[0]
        result = df.set_index(first_col).to_dict(orient="index")
    else:
        result = df.to_dict(orient="index")

    return result


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
    edge_index = data.edge_index
    is_connected = (edge_index[0] == node_id) | (edge_index[1] == node_id)
    connected_edges = edge_index[:, is_connected]
    num_edges = connected_edges.size(1)

    connected_nodes = torch.where(
        edge_index[0] == node_id, edge_index[1], edge_index[0]
    )[is_connected]
    connected_labels = data.y[connected_nodes].tolist()
    label_counts = Counter(connected_labels)
    total_connections = sum(label_counts.values())
    label_mapping = {0: label_0, 1: label_1}

    label_ratios = {
        label_mapping[label]: round(count / total_connections, 2)
        for label, count in label_counts.items()
    }

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
    node_to_user_id_map = {v: k for k, v in user_id_map.items()}

    all_user_percentiles = []

    for target_node_id in target_node_ids:
        target_user_id = node_to_user_id_map.get(target_node_id)

        if target_user_id is None:
            continue

        node_features = user_features[user_features["user_id"] == target_user_id]
        percentile_row = node_features.copy()
        percentile_row["user_id"] = "Percentile"

        for col in node_features.columns:
            if col in columns_for_percentile:
                user_value = node_features[col].values[0]
                percentile_rank = stats.percentileofscore(
                    user_features[col], user_value
                )
                percentile_row[col] = round(
                    percentile_rank, 2
                )
            elif col != "user_id":
                percentile_row[col] = "Binary Column"

        node_features = pd.concat([node_features, percentile_row], ignore_index=True)

        node_features["user_id"] = [target_user_id, "Percentile"]
        user_data_df = node_features.iloc[
            0:2
        ].copy()
        user_dict = dataframe_to_dict(user_data_df)
        user_dict_final = {
            target_node_id: {k: v for k, v in user_dict.items() if k != "user_id"}
        }

        all_user_percentiles.append(user_dict_final)

    return all_user_percentiles


def get_source_node_labels_as_dict_with_salary(edges_df, data):
    """
    Given an edges DataFrame and a PyG Data object, return the labels of nodes listed in the 'Source' column
    as a dictionary where the node ID is the key and the value is a dictionary with the node's salary category
    ('Low salary' for label 0 and 'High salary' for label 1).

    Parameters:
    edges_df (pd.DataFrame): The DataFrame containing the 'Source' column with node IDs.
    data (torch_geometric.data.Data): The PyG Data object containing node labels in 'data.y'.

    Returns:
    dict: A dictionary where keys are node IDs and values are dictionaries with the salary category.
    """
    source_nodes = edges_df["Source Node"].tolist()
    source_labels = data.y[source_nodes]
    salary_mapping = {0: "Low Salary", 1: "High Salary"}

    source_node_labels_dict = {
        node_id: {"Label": salary_mapping[label.item()]}
        for node_id, label in zip(source_nodes, source_labels)
    }

    return source_node_labels_dict


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
    gcn.eval()
    data = data.to(device)

    with torch.no_grad():
        out = gcn(
            data.x, data.edge_index
        ).squeeze()

        node_prob = out[node_index].item()
        node_prediction = 1 if node_prob > 0.5 else 0

    if node_prediction == 1:
        node_prediction = "High Salary"
    else:
        node_prediction = "Low Salary"

    return {
        "Target Node Index": node_index,
        "Predicted Class": node_prediction,
    }

