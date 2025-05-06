

class Node:

    def __init__(
            self,
            data = None,
            feature_idx: int = None,
            threshold: float = None,
            left_child_node: 'Node' = None,
            right_child_node: 'Node' = None,
            leaf_node_value: float = None,
            is_leaf: bool = False,
            ):
        """
        The node of a tree is an object.

        Args:
            feature_idx (int, optional): The index of the feature that is split. Defaults to None.
            threshold (float, optional): The value of the split. Defaults to None.
            left_child_node (Node, optional): The left child node. Defaults to None.
            right_child_node (Node, optional): The right child node. Defaults to None.
            value (float, optional): If the node is leaf, store the predicted value. Defaults to None.
        """
        
        # for decision node
        self.data = data
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left_child_node = left_child_node
        self.right_child_node = right_child_node
        
        # for leaf node
        self.leaf_node_value = leaf_node_value
        self.is_leaf = is_leaf