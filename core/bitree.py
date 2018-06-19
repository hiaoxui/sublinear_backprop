

class Node(object):
    """
    The node of binary tree.
    If the node is a leaf, it contains an integer (index) as its value.
    If the node is an internal node, it contains a series of contiguous integers.
    """
    def __init__(self, is_leaf, parent, *, scope=None, active=None, value=None):
        """
        :param bool is_leaf: Whether this is a leaf or internal node.
        :param Node/None parent: The predecessor of this node.
        :param list[int] scope: In the form of [beg_point, end_point).
        :param set[int] active: A subset of scope. Active indexes.
        :param int value: If it is leaf, value should be specified.
        """
        self.is_leaf = is_leaf
        if self.is_leaf:
            self.value = value
            self.scope = range(value, value+1)
            self.active = set()
            self.active.add(value)
        else:
            self.active = active
            if isinstance(scope, range):
                self.scope = scope
            else:
                self.scope = range(*scope)
            self.left_child = None
            self.right_child = None
        self.parent = parent

    def right_left_most(self, value=True):
        """
        The leftmost leaf of its right child.
        :param bool value: If True, this method returns its value only. (faster)
        If False, this method returns the node itself.
        :rtype: int or Node
        """
        assert not self.is_leaf
        if value:
            return self.right_child.scope.start
        curr_node = self.right_child
        while not curr_node.is_leaf:
            curr_node = curr_node.left_child
        return curr_node


class BiTree(object):
    """
    A binary tree with useful methods for RNNs memory management.
    """
    def __init__(self, total, first_state, stepper):
        """
        :param int total: The number of leaves.
        :param first_state: The value of the first leaf.
        :param func stepper: v_{i+1} = stepper(v_i).
        In another word, we could use stepper to generate values of all the leaves with the value of
        the first node.
        """
        self.total = total
        self.root = self._build_tree()
        # We may use other steppers to accelerate.
        self.default_stepper = stepper
        # Storage is used to store the data, and the values in leaves are just the indexes (keys) of
        # this dict.
        self.storage = dict()
        self._activate(0, first_state)

    def _build_tree(self):
        """
        The tree built should be a complete and optimal binary tree.
        """
        def recursive(scope, parent):
            if scope.stop - scope.start == 1:
                return Node(True, parent, value=scope.start)
            else:
                left_scope = range(scope.start, (scope.start+scope.stop)//2)
                right_scope = range((scope.start+scope.stop)//2, scope.stop)
                new_node = Node(False, parent, scope=scope, active=set())
                new_node.left_child = recursive(left_scope, new_node)
                new_node.right_child = recursive(right_scope, new_node)
                return new_node

        return recursive(range(0, self.total), None)

    def _deactivate(self, idx):
        """
        Free the content indexed idx.
        :param int idx: Index.
        """
        assert idx in self.root.active
        assert idx in self.storage
        if idx == 0:
            return

        # Free it from the memory.
        del self.storage[idx]

        # Delete it from the tree.
        node = self.root
        while not node.is_leaf:
            assert idx in node.active
            node.active.discard(idx)
            if idx in node.left_child.scope:
                node = node.left_child
            else:
                node = node.right_child

    def _activate(self, idx, stuff):
        """
        Add new content, and index it as idx.
        :param int idx: Index.
        :param stuff: Content.
        """
        assert idx not in self.root.active
        assert idx not in self.storage

        # Allocate new memory.
        self.storage[idx] = stuff

        # Add it to the tree.
        node = self.root
        while not node.is_leaf:
            assert idx not in node.active
            node.active.add(idx)
            if idx in node.left_child.scope:
                node = node.left_child
            else:
                node = node.right_child

    def _previous(self, idx):
        """
        The index of previous active node of idx.
        Example: 0(active), 1, 2(active), 3, 4, ...
        self._previous(4) => 2
        :param index idx: Index.
        """
        return max(filter(lambda x_: x_ < idx, self.root.active))

    def forward_generator(self, node=None):
        """
        Iterate all the leaves of specified node (root as default), yielding them one by one.
        At the same time, cache the content of necessary leaves (in a portion of \log n) for later
        use.
        For more details of this algorithm:
        https://timvieira.github.io/blog/post/2016/10/01/reversing-a-sequence-with-sublinear-space/
        :param Node node: Default is root.
        """
        node = node or self.root
        if node.is_leaf:
            return
        target_idx = node.right_left_most()
        prev_node_idx = self._previous(target_idx)
        curr_state = self.storage[prev_node_idx]
        while prev_node_idx < target_idx:
            # Send specified stepper or send nothing to use the default stepper.
            # Note that other than side effect, both the specified stepper and the default stepper
            # should have the same output.
            stepper = yield curr_state
            stepper = stepper or self.default_stepper
            curr_state = stepper(curr_state, prev_node_idx)
            prev_node_idx += 1
        self._activate(target_idx, curr_state)
        yield from self.forward_generator(node.right_child)

    def forward(self):
        """
        Iterate all the leaves, but yield or return nothing.
        Cache the content of necessary nodes for later use.
        """
        gen = self.forward_generator()
        for _ in gen:
            pass

    def backward_generator(self):
        """
        Call self.forward_generator or self.forward firstly.
        Iterate all the leaves reversely. Note that it is not as trivial as forward, since there
        is no method like v_i = reverse_stepper(v_{i+1}), so we need the use the cached content to
        generate every new v_i, which might be costly but inevitable.
        It will free the old cache and cache new content during the iteration.
        """
        last_node = self.root
        while not last_node.is_leaf:
            last_node = last_node.right_child
        curr_node = last_node
        assert curr_node.value == self.total - 1
        assert last_node.value in self.root.active
        yield self.storage[last_node.value]
        while curr_node.value != 0:
            next_node = curr_node
            target_idx = curr_node.value - 1
            while target_idx not in next_node.scope:
                next_node = next_node.parent
            self._clear_storage(next_node.right_child)
            for _ in self.forward_generator(next_node.left_child):
                pass
            while not next_node.is_leaf:
                if target_idx in next_node.left_child.scope:
                    next_node = next_node.left_child
                else:
                    next_node = next_node.right_child
            curr_node = next_node
            yield self.storage[target_idx]

    def _clear_storage(self, node):
        """
        Free all the cache for the given node.
        :param Node node: If it is a leaf, we will free the content indexed by its value.
        If it is a internal node, we will free all its leaves.
        """
        to_clear = node.active.copy()
        for idx in to_clear:
            self._deactivate(idx)


if __name__ == '__main__':
    bt = BiTree(64, 0, lambda x_, _: x_ + 1)
    bt.forward()
    x = 1
