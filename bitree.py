
class Node(object):
    def __init__(self, is_leaf, parent, *, scope=None, active=None, value=None):
        """

        :param bool is_leaf:
        :param Node/None parent:
        :param list[int] scope:
        :param set[int] active:
        :param int value:
        """
        self.is_leaf = is_leaf
        if self.is_leaf:
            self.value = value
            self.scope = [value, value+1]
        else:
            self.active = active
            self.scope = scope
            self.left_child = None
            self.right_child = None
        self.parent = parent

    def right_left_most(self, value=True):
        assert not self.is_leaf
        if value:
            return self.right_child.scope[0]
        curr_node = self.right_child
        while not curr_node.is_leaf:
            curr_node = curr_node.left_child
        return curr_node


class BiTree(object):
    def __init__(self, total, first_state, stepper):
        """

        :param int total:
        """
        self.total = total
        self.root = self._build_tree()
        self.default_stepper = stepper
        self._activate(0)
        self.storage = dict()
        self.storage[0] = first_state

    def _build_tree(self):

        def recursive(scope, parent):
            if scope[1] - scope[0] == 1:
                return Node(True, parent, value=scope[0])
            else:
                left_scope = [scope[0], (scope[0]+scope[1])//2]
                right_scope = [(scope[0]+scope[1])//2, scope[1]]
                new_node = Node(False, parent, scope=scope, active=set())
                new_node.left_child = recursive(left_scope, new_node)
                new_node.right_child = recursive(right_scope, new_node)
                return new_node

        return recursive([0, self.total], None)

    def _deactivate(self, idx):
        if idx not in self.root.active:
            return

        node = self.root
        while not node.is_leaf:
            assert idx in node.active
            node.active.discard(idx)
            if idx in node.left_child.scope:
                node = node.left_child
            else:
                node = node.right_child

    def _activate(self, idx):
        if idx in self.root.active:
            return

        node = self.root
        while not node.is_leaf:
            assert idx not in node.active
            node.active.add(idx)
            if idx in node.left_child.scope:
                node = node.left_child
            else:
                node = node.right_child

    def _previous(self, idx):
        return max(filter(lambda x_: x_ < idx, self.root.active))

    def forward_generator(self, node=None):
        node = node or self.root
        if node.is_leaf:
            return
        next_stop_idx = node.right_left_most()
        prev_node_idx = self._previous(next_stop_idx)
        curr_state = self.storage[prev_node_idx]
        while prev_node_idx < next_stop_idx:
            stepper = yield curr_state
            stepper = stepper or self.default_stepper
            curr_state = stepper(curr_state)
            prev_node_idx += 1
        self.storage[next_stop_idx] = curr_state
        self._activate(next_stop_idx)
        yield from self.forward_generator(node.right_child)

    def forward(self):
        gen = self.forward_generator()
        for _ in gen:
            pass

    def backward_generator(self):
        last_node = self.root
        while not last_node.is_leaf:
            last_node = last_node.right_child
        curr_node = last_node
        assert curr_node.value == self.total - 1
        while curr_node.value != 0:
            next_node = curr_node
            next_value = curr_node.value - 1
            while next_node.value not in next_node.scope:
                next_node = next_node.parent
            self._clear_storage(next_node)
            for _ in self.forward_generator(next_node.left_child):
                pass
            while not next_node.is_leaf:
                if next_value in next_node.left_child.scope:
                    next_node = next_node.left_child
                else:
                    next_node = next_node.right_child
            yield self.storage[next_value]

    def _clear_storage(self, node):
        to_clear = node.active.copy()
        for idx in to_clear:
            del self.storage[idx]
            self._deactivate(idx)


if __name__ == '__main__':
    bt = BiTree(10, None, None)
    x = 1
