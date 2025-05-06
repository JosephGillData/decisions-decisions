import matplotlib.pyplot as plt
import networkx as nx

class DecisionTreeGraph:

    def __init__(self):

        pass

    def _create_node_label(self, node):

        # Create a label for the current node
        if node.is_leaf == False:

            if (self.X_dtypes.values[node.feature_idx] == int) | (self.X_dtypes.values[node.feature_idx] == float):
                label = f"""{self.X_dtypes.keys()[node.feature_idx]} < {node.threshold}"""
            elif self.X_dtypes.values[node.feature_idx] == object:
                label = f"""{self.X_dtypes.keys()[node.feature_idx]} = {node.threshold}"""
            else:
                print(f"New column data type: {self.X_dtypes.keys()[node.feature_idx]}")
        else:
            label = node.leaf_node_value.round(2)

        return label

    def _add_decision_node(self, node, parent_id=None):

        # Exit the loop once leaf node is reached
        if node is None:
            return
        
        self.node_count += 1
        current_id = self.node_count

        label = self._create_node_label(node)
        self.G.add_node(current_id, label=label, is_leaf=node.is_leaf)
        
        # If there is a parent, add an edge from the parent to the current node
        if parent_id is not None:
            self.G.add_edge(parent_id, current_id)

        # Recursively add child nodes
        self._add_decision_node(node.left_child_node, parent_id=current_id)
        self._add_decision_node(node.right_child_node, parent_id=current_id)

    def _create_label_positions(self):

        pos = nx.nx_pydot.graphviz_layout(self.G, prog='dot')

        for node, (x, y) in pos.items():
            if self.G.nodes[node]['is_leaf']:
                pos[node] = (x, y - 8)
            else:
                pos[node] = (x, y + 17)

        return pos

    def visualise(self, tree, X_dtypes):

        self.tree = tree
        self.X_dtypes = X_dtypes
        self.G = nx.DiGraph()
        self.node_count = 0

        # Add nodes and edges
        self._add_decision_node(self.tree)

        # Extract labels for nodes
        labels = nx.get_node_attributes(self.G, 'label')
        
        plt.figure(figsize=(13, 8))

        # Create a new dictionary for label positions with an offset.
        # Adjust offset as needed (here, we subtract 20 units from the y-coordinate).
        pos = nx.nx_pydot.graphviz_layout(self.G, prog='dot')

        # Extract leaf node information for color differentiation and label positions
        node_colors = [
            "lightgreen" if self.G.nodes[node]["is_leaf"] else "lightblue"
            for node in self.G.nodes
        ]
        label_pos = self._create_label_positions()

        nx.draw(self.G, pos, with_labels=True, node_size=500, node_color=node_colors, font_size=10)
        nx.draw_networkx_labels(self.G, label_pos, labels, font_size=11, verticalalignment='top')
        plt.show()
