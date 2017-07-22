def memory_plan(self, feed_shapes):
    """Allocates ndarray.NDArray for every node except feed_dict nodes.

    Implementation note:
    Option 1: Alloc a ndarray.NDArray per node that persists across run()
    Option 2: Implement a memory pool to reuse memory for nodes of same
            shapes. More details see Lecture 7.

    For both options, self.node_to_arr_map stores node->NDArray mapping to
    allow mapping to persist across multiple executor.run().

    Hint: use ndarray.empty(shape, ctx=self.ctx) to allocate NDArray.

    Parameters
    ----------
    feed_shapes: node->shapes mapping for feed_dict nodes.
    """
    shape_to_arr_pool = {}
    for node, shape in self.node_to_shape_map.items():
        if node not in feed_shapes:
            shape_to_arr_pool[shape] = []

    node_to_out_degree = {}
    for node in self.topo_order:
        node_to_out_degree[node] = 0
    for node in self.topo_order:
        for i in node.inputs:
            node_to_out_degree[i] += 1

    counter = [0, 0]
    from operator import mul
    from functools import reduce

    def real_alloc(shape):
        counter[0] += 1
        counter[1] += reduce(mul, shape)
        return ndarray.empty(shape, ctx=self.ctx)

    def alloc(shape):
        if len(shape_to_arr_pool[shape]) == 0:
            return real_alloc(shape)
        else:
            return shape_to_arr_pool[shape].pop()

    def free(array):
        if array is None:
            return
        shape_to_arr_pool[array.shape].append(array)

    total_count = 0
    total_memory = 0
    for node, shape in self.node_to_shape_map.items():
        if node not in feed_shapes:
            total_count += 1
            total_memory += reduce(mul, shape)

    self.node_to_arr_map = {}
    eval_result_set = set()

    # allocate unique memory for all eval nodes
    for node in self.eval_node_list:
        shape = self.node_to_shape_map[node]
        self.node_to_arr_map[node] = real_alloc(shape)
        eval_result_set.add(self.node_to_arr_map[node])

    for node in self.topo_order:
        # this is a feed variable, which needs no allocation
        # also do not iterate inputs since there is none
        if node in feed_shapes:
            continue
        if node not in self.eval_node_list:
            # this node should not be allocated
            assert node not in self.node_to_arr_map
            self.node_to_arr_map[node] = alloc(self.node_to_shape_map[node])
        for i in node.inputs:
            node_to_out_degree[i] -= 1
            # release unused memory
            if node_to_out_degree[i] == 0:
                mem = self.node_to_arr_map.get(i, None)
                # put memory back to pool only when
                # this is not the node to do evaluation
                if mem not in eval_result_set:
                    free(mem)

    self.graph()
    print 'Memory Plan Stat: %.2f%%(%d/%d)' % (float(counter[1]) * 100. / float(total_memory), counter[1], total_memory)
