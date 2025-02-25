"""
pyg_modifications.py

This file consolidates all necessary modifications for PyTorch Geometric:

1) torch_geometric/explain/explanation.py
   - Explanation.visualize_graph
   - HeteroExplanation.visualize_feature_importance
   - _visualize_score

2) torch_geometric/visualization/graph.py
   - _visualize_graph_via_graphviz

INSTRUCTIONS:
-------------
1. Open the PyTorch Geometric repo or installation directory.
2. Locate:
   - torch_geometric/explain/explanation.py
   - torch_geometric/visualization/graph.py
3. Copy the functions below into the respective files, replacing the existing
   versions or inserting them if they do not exist.
4. Reinstall (or rebuild) PyTorch Geometric from the edited source.
5. Verify that your environment points to the updated code.
"""

# =============================================================================
# 1) torch_geometric/explain/explanation.py
# =============================================================================


# --- (A) Explanation class -> visualize_graph --------------------------------
def explanation_visualize_graph():
    """
    Place this code inside the 'Explanation' class in
    torch_geometric/explain/explanation.py,
    replacing the existing 'visualize_graph' method.
    """
    code = r'''
def visualize_graph(
    self,
    path: Optional[str] = None,
    backend: Optional[str] = None,
    node_labels: Optional[List[str]] = None,
) -> None:
    r"""Visualizes the explanation graph with edge opacity corresponding to
    edge importance.

    Args:
        path (str, optional): The path to where the plot is saved.
            If set to :obj:`None`, will visualize the plot on-the-fly.
            (default: :obj:`None`)
        backend (str, optional): The graph drawing backend to use for
            visualization (:obj:`"graphviz"`, :obj:`"networkx"`).
            If set to :obj:`None`, will use the most appropriate
            visualization backend based on available system packages.
            (default: :obj:`None`)
        node_labels (list[str], optional): The labels/IDs of nodes.
            (default: :obj:`None`)
    """
    edge_mask = self.get('edge_mask')
    if edge_mask is None:
        raise ValueError(
            f"The attribute 'edge_mask' is not available in "
            f"'{self.__class__.__name__}' (got {self.available_explanations})"
        )
    g, edges_df = visualize_graph(
        self.edge_index, edge_mask, path, backend, node_labels
    )
    return g, edges_df
'''
    return code


# --- (B) HeteroExplanation class -> visualize_feature_importance -------------
def hetero_explanation_visualize_feature_importance():
    """
    Place this code inside the 'HeteroExplanation' class in
    torch_geometric/explain/explanation.py,
    replacing the existing 'visualize_feature_importance' method.
    """
    code = r'''
def visualize_feature_importance(
    self,
    path: Optional[str] = None,
    feat_labels: Optional[Dict[NodeType, List[str]]] = None,
    top_k: Optional[int] = None,
):
    r"""Creates a bar plot of the node feature importances by summing up
    node masks across all nodes for each node type.

    Args:
        path (str, optional): The path to where the plot is saved.
            If set to :obj:`None`, will visualize the plot on-the-fly.
            (default: :obj:`None`)
        feat_labels (Dict[NodeType, List[str]], optional): The labels of
            features for each node type. (default: :obj:`None`)
        top_k (int, optional): Top k features to plot. If :obj:`None`
            plots all features. (default: :obj:`None`)
    """
    node_mask_dict = self.node_mask_dict
    for node_mask in node_mask_dict.values():
        if node_mask.dim() != 2:
            raise ValueError(
                f"Cannot compute feature importance for object-level 'node_mask' "
                f"(got shape {node_mask.size()})"
            )

    if feat_labels is None:
        feat_labels = {}
        for node_type, node_mask in node_mask_dict.items():
            feat_labels[node_type] = range(node_mask.size(1))

    score = torch.cat(
        [node_mask.sum(dim=0) for node_mask in node_mask_dict.values()],
        dim=0
    )

    all_feat_labels = []
    for node_type in node_mask_dict.keys():
        all_feat_labels += [
            f'{node_type}#{label}' for label in feat_labels[node_type]
        ]

    plot, df = _visualize_score(score, all_feat_labels, path, top_k)
    return plot, df
'''
    return code


# --- (C) _visualize_score function -------------------------------------------
def explanation_visualize_score():
    """
    Place this standalone function inside torch_geometric/explain/explanation.py,
    replacing or adding to the existing code. Make sure it matches any internal
    imports/structure.
    """
    code = r"""
def _visualize_score(
    score: torch.Tensor,
    labels: List[str],
    path: Optional[str] = None,
    top_k: Optional[int] = None,
):
    import matplotlib.pyplot as plt
    import pandas as pd

    if len(labels) != score.numel():
        raise ValueError(
            f"The number of labels (got {len(labels)}) must match the number "
            f"of scores (got {score.numel()})"
        )

    score = score.cpu().numpy()

    df = pd.DataFrame({'feature_importance': score}, index=labels)
    df = df.sort_values('feature_importance', ascending=False)

    # Optional: Limit to top_k features
    if top_k is not None:
        df = df.head(top_k)
        title = f"Feature importance for top {len(df)} features"
    else:
        title = f"Feature importance for {len(df)} features"

    df = df.round(decimals=3)

    # Plotting the DataFrame with labels set as index
    ax = df.plot(
        kind='barh',
        figsize=(10, 7),
        title=title,
        xlabel='Feature Importance',
        xlim=[0, df['feature_importance'].max() + 0.075],
        legend=False,
    )
    ax.set_title(title, fontsize=17)

    # Invert y-axis to have the most important feature at the top
    plt.gca().invert_yaxis()

    # Increase the font size of the y-axis labels
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=13)

    ax.tick_params(axis='x', labelsize=12)
    ax.set_xlabel('Feature Importance', fontsize=14)

    # Add labels to the bars
    ax.bar_label(
        ax.containers[0],
        labels=[f"{v:.3f}" for v in df['feature_importance']],
        label_type='edge',
        fontsize=11,
        padding=2
    )

    # Adjust layout to make labels more readable
    plt.subplots_adjust(left=0.3, right=0.95)

    # Save or show the plot
    if path is not None:
        plt.savefig(path)
    else:
        plt.show()

    plt.close()

    df.reset_index(inplace=True)
    df.columns = ['feature_name', 'feature_importance']

    return ax, df
"""
    return code


# =============================================================================
# 2) torch_geometric/visualization/graph.py
# =============================================================================


# --- _visualize_graph_via_graphviz function ----------------------------------
def visualization_graphviz():
    """
    Place this standalone function in torch_geometric/visualization/graph.py,
    replacing or adding to the existing code for '_visualize_graph_via_graphviz'.
    """
    code = r"""
def _visualize_graph_via_graphviz(
    edge_index: Tensor,
    edge_weight: Tensor,
    path: Optional[str] = None,
    node_labels: Optional[List[str]] = None,
) -> Any:
    import graphviz

    src_list = []
    dst_list = []
    w_list = []

    suffix = path.split('.')[-1] if path is not None else None
    g = graphviz.Digraph('graph', format=suffix)
    g.attr('node', shape='circle', fontsize='11pt')

    for node in edge_index.view(-1).unique().tolist():
        g.node(str(node) if node_labels is None else node_labels[node])

    sub_edges = list(zip(edge_index.t().tolist(), edge_weight.tolist()))
    sub_edges.sort(key=lambda x: x[1], reverse=True)

    # Now iterate over the sorted edges
    for (src, dst), w in sub_edges:
        hex_color = hex(255 - round(255 * w))[2:]
        hex_color = f'{hex_color}0' if len(hex_color) == 1 else hex_color
        if node_labels is not None:
            src = node_labels[src]
            dst = node_labels[dst]
        g.edge(str(src), str(dst), color=f'#{hex_color}{hex_color}{hex_color}')

        src_list.append(src)
        dst_list.append(dst)
        w_list.append(w)

    data = {'Source Node': src_list, 'Destination Node': dst_list, 'Importance': w_list}
    edges_df = pd.DataFrame(data)
    edges_df = edges_df.sort_values('Importance', ascending=False)
    edges_df = edges_df.round(decimals=3)
    edges_df.reset_index(drop=True, inplace=True)

    if path is not None:
        path = '.'.join(path.split('.')[:-1])
        g.render(path, cleanup=True)
    else:
        g.view()

    return g, edges_df
"""
    return code


# =============================================================================
# Helper function to print all code for easy copying
# =============================================================================


def print_all_modifications():
    """Prints all the modified code blocks in a user-friendly format."""
    print("\n\n===== Explanation.visualize_graph =====")
    print(explanation_visualize_graph())

    print("\n\n===== HeteroExplanation.visualize_feature_importance =====")
    print(hetero_explanation_visualize_feature_importance())

    print("\n\n===== _visualize_score function =====")
    print(explanation_visualize_score())

    print("\n\n===== _visualize_graph_via_graphviz function =====")
    print(visualization_graphviz())


if __name__ == "__main__":
    # To print all modifications.
    print_all_modifications()
