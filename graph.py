import sys
from collections import deque
import heapq
from typing import Any

def swap(list,a,b):
    temp = list[a]
    list[a] = list[b]
    list[b] = temp

parent = lambda k : (k - 1)//2
left_child =  lambda k : 2*k + 1
right_child =  lambda k : 2*k + 2

class betterPriorityQueue():
    def __init__(this, comparable_iterable=None):
        this.size : int = 0
        this.heap : list[list[Any]] = [] # minHeap
        this.refs : int = [] # listIndexes
        if comparable_iterable != None: 
            for e in comparable_iterable: this.add(e) # heapify method

    def add(this, el : Any) -> None:
        this.heap.append(el)
        this.refs.append(this.size)
        this.size += 1
        this.heapifyUp(this.size - 1)

    
    def pop(this) -> tuple[Any,Any]:
        if this.size == 0: raise Exception("No items to pop in heap object.")
        min_ = this.heap[0]
        this.heap[0] = this.heap[this.size - 1]
        this.size -= 1
        this.__heapifyDown()
        return min_

    def heapifyUp(this, index) -> None:
        p = parent(index)
        while index > 0 and this.heap[p][1] > this.heap[index][1]:
            this.refs[this.heap[p][0]] = index
            this.refs[this.heap[index][0]] = p
            swap(this.heap,index,p)
            index = p
            p = parent(index)
            print(p)
    
    def __heapifyDown(this) -> None:
        index = 0
        while left_child(index) < this.size:
            left = left_child(index)
            right = right_child(index)
            smallestChildIndex = left
            if right < this.size and this.heap[right][1] < this.heap[left][1]:
                smallestChildIndex = right
            if this.heap[index] < this.heap[smallestChildIndex]:
                break
            this.refs[this.heap[smallestChildIndex][0]] = index
            this.refs[this.heap[index][0]] = smallestChildIndex
            swap(this.heap,index,smallestChildIndex)
            index = smallestChildIndex



class Graph():
    def __init__(this, edges : list[tuple[int,int,float]], representations : str | list[str] ) -> None:
        this.edges : list[tuple[int,int,float]] = edges
        this.representations : list[str] = []
        this.n : int = this.__getVertexSetCardinality(edges)
        if isinstance(representations, str): representations = [representations]
        for r in representations:
            match r:
                case 'adj_list':
                    this.adj_list : list[list[tuple[int,float]]] = this.__constructAdjList()
                case 'adj_matrix':
                    this.adj_matrix : list[list[tuple[bool,float] | bool]] = this.__constructAdjMatrix()
                case 'incidence_matrix':
                    this.inc_matrix : list[list[int]] = this.__constructIncidenceMatrix()

    def __constructAdjList(this) -> list[list[tuple[int,float]]]: # complexity Omega(V) and O(V + E) = O(V²) 
        adj_list : list[list[tuple[int,float]]] = [ [] for v in range(this.n)]
        for e in this.edges: adj_list[e[0]].append((e[1],e[2]))
        if 'adj_list' not in this.representations: this.representations.append('adj_list')
        return adj_list

    def __constructAdjMatrix(this) -> list[list[tuple[bool,float] | bool]]: # complexity Theta(V² + E) = Theta(V²)
        adj_matrix : list[list[tuple[bool,float]]] = [[ False for v in range(this.n)] for v in range(this.n)]
        for e in this.edges: adj_matrix[e[0]][e[1]] = (True,e[2])
        if 'adj_matrix' not in this.representations: this.representations.append('adj_matrix')
        return adj_matrix

    def __constructIncidenceMatrix(this) -> list[list[int]]: # complexity Theta(V*E) and O(V³)
        inc_matrix : list[list[int]] = [[ 0 for e in range(len(this.edges))] for v in range(this.n)]
        for e in range(len(this.edges)):
            inc_matrix[this.edges[e][0]][e] = -1
            inc_matrix[this.edges[e][1]][e] = 1
        if 'incidence_matrix' not in this.representations: this.representations.append('incidence_matrix')
        return inc_matrix

    def __getVertexSetCardinality(this, edges : list[tuple[int,int]]) -> int:
        n : int = 0
        for e in edges:
            n = max(n,e[0],e[1])
        if edges != []: n += 1
        return n
    
    # def adjSearch(u : int, v : int) -> bool:
    #     return v in this.adj_list[u] if 'adj_list' in this.representations else this.adj_matrix[u][v]
        
    def topologicalSort(this) -> list[int]:  # needs adj_list or adj_matrix constructed

        adjacencies = this.adj_list if 'adj_list' in this.representations else ( ( v for v in range(len(adj)) if adj[v] == True) for adj in g.adj_matrix )
        visited = [False for _ in range(this.n)]
        top_order = [] # the topological ordering
        for v in range(this.n):
            if not visited[v]:

                working_stack = [v]
                while working_stack:
                    v2 = working_stack.pop()
                    visited[v2] = True
                    
                    for next_neighbor,_ in adjacencies[v2]:
                        if not visited[next_neighbor]:
                            working_stack.append(v2)
                            working_stack.append(next_neighbor)
                            break
                    else: # else neighbourhood fully visited
                        top_order.append(v2)
        return top_order

    def __initializeSingleSource(this, s : int) -> tuple[list[int],list[int]]:
        shortest_paths_tree = [float('inf') for _ in range(this.n)]
        shortest_paths_tree[s] = -1 # s is the root
        distances = [float('inf') for _ in range(this.n)]
        distances[s] = 0
        return shortest_paths_tree, distances
    
    def __getWeight(this, u : int, v : int) -> float:
        if 'adj_matrix' in this.representations:
            return this.adj_matrix[u][v][1] if this.adj_matrix[u][v] != False else float('inf')
        else:
            for x in this.adj_list[u]:
                if x[0] == v:
                    return x[1]
        return float('inf')

    def brute_force_SP(this, s : int, t : int) -> tuple[list[int|float],int|float]:
        adjacencies = this.adj_list if 'adj_list' in this.representations else ( ( v for v in range(len(adj)) if adj[v] != False) for adj in g.adj_matrix )
        sp_w = float('inf')
        sp = []
        stack : tuple[int,float,list[int]] = [(s,0,[s],[False for _ in range(this.n)])]
        while len(stack) > 0:
            u, p_w, p, visited = stack.pop()
            for v,w in adjacencies[u]:
                if not visited[v]: # if p + [v] is not a cycle then 
                    if v == t:
                        if sp_w > p_w + w:
                            sp = p + [t]
                            sp_w = p_w + w
                    else:
                        visited_rec = [e for e in visited]
                        visited_rec[u] = True
                        stack.append((v, p_w + w, p + [v], visited_rec))
        return sp,sp_w
    
    def DAG_shortest_paths(this, s : int) -> tuple[list[int|float],list[int|float],list[int|float]]:
        top_sort = this.topologicalSort()
        shortest_paths_tree,distances = this.__initializeSingleSource(s)
        adjacencies = this.adj_list if 'adj_list' in this.representations else ( ( v for v in range(len(adj)) if adj[v] != False) for adj in g.adj_matrix )

        for u in reversed(top_sort):
            for v,w in adjacencies[u]:
                if distances[v] > distances[u] + w: # edge relaxation
                    distances[v] = distances[u] + w
                    shortest_paths_tree[v] = u
        return shortest_paths_tree,distances,top_sort

    def Bellman_Ford(this, s : int) -> tuple[list[int|float],list[int|float]] | bool:
        shortest_paths_tree,distances = this.__initializeSingleSource(s)
        for _ in range(this.n - 1):
            for u,v,w in this.edges:
                if distances[v] > distances[u] + w: # edge relaxation
                    distances[v] = distances[u] + w
                    shortest_paths_tree[v] = u
        for u,v,w in this.edges:
            if distances[v] > distances[u] + w: return False
        return shortest_paths_tree,distances
    
    def Floyd_Warshall(this) -> tuple[list[list[int|float]],list[list[int|float]]]:
        shortest_paths_trees = [[-1 if i == j or this.__getWeight(i,j) == float('inf') else i for j in range(this.n)] for i in range(this.n)]
        distances = [[0 if i == j else this.__getWeight(i,j) for j in range(this.n)] for i in range(this.n) ]
        for k in range(this.n):
            for i in range(this.n):
                for k in range(this.n):
                    if distances[i][k] > distances[i][k] + distances[k][k]: # edge relaxation
                        distances[i][k] = distances[i][k] + distances[k][k]
                        shortest_paths_trees[i][k] = shortest_paths_trees[k][k]
        return shortest_paths_trees,distances
    
    def dijkstra(this, s : int) -> tuple[list[int|float],list[int|float]]:

        adjacencies = this.adj_list if 'adj_list' in this.representations else ( ( v for v in range(len(adj)) if adj[v] == True) for adj in g.adj_matrix )
        shortest_paths_tree, distances = this.__initializeSingleSource(s)
        heap = betterPriorityQueue([[v,distances[v]] for v in range(this.n)])

        while heap.size > 0:
            u = heap.pop()[0]
            for v,w in adjacencies[u]: # Takes O(V + ElogV)
                if distances[v] > distances[u] + w:
                    distances[v] = distances[u] + w
                    shortest_paths_tree[v] = u
                    heap.heap[heap.refs[v]][1] = distances[u] + w
                    heap.heapifyUp(heap.refs[v]) # takes O(logV)
        return shortest_paths_tree, distances
    
    def findMinimumColoring(this) -> tuple[list[int],int]: # the result is not guaranteed to be optimal and has a bound of delta(G) <= min_coloring <= delta(G) + 1
        coloring = [float('inf') for _ in range(this.n)] # no initial coloring
        adjacencies = this.adj_list if 'adj_list' in this.representations else ( ( v for v in range(len(adj)) if adj[v] != False) for adj in g.adj_matrix )
        visited = [False for _ in range(this.n)]
        for u in range(this.n):
            if not visited[u]:
                visited[u] = True
                temp = [coloring[v] for v,_ in adjacencies[u] if coloring[v] < float('inf')]
                if len(temp) == 0:
                    coloring[u] = 0
                    continue
                temp.sort()
                item = temp[0]
                j = 0
                for i in range(this.n):
                    if i < item: # if not i < item then i == item, therefore we must increment.
                        coloring[u] = i
                        break
                    if j < len(temp) - 1 and temp[j+1] != temp[j]: 
                        j += 1
                        item = temp[j]
                    else:
                        coloring[u] = i + 1
                        break
        return coloring,max(coloring) + 1

    
    def universalSinks_BF(this) -> list[int]: # "brute force" version or universalSinks
        adjacencies = this.adj_list if 'adj_list' in this.representations else ( ( v for v in range(len(adj)) if adj[v] != False) for adj in g.adj_matrix )
        sinks_matrix = []
        for s in range(this.n):
            stack = [s]
            sinks = []
            visited = [False for _ in range(this.n)]
            while len(stack) > 0:
                u = stack.pop()
                if not visited[u]:
                    sinks.append(u)
                    visited[u] = True
                    for v in adjacencies[u]:
                        stack.append(v)
            sinks_matrix.append(sinks)
        sinks = set([v for v in range(this.n)])
        for s in [set(sinks) for sinks in sinks_matrix]:
            sinks = sinks.intersection(s)
        return list(sinks)
    
    def universalSinks(this) -> list[int]: # tem que usar o union find
        adjacencies = this.adj_list if 'adj_list' in this.representations else ( ( v for v in range(len(adj)) if adj[v] != False) for adj in g.adj_matrix )
        visited = [False for v in range(this.n)]
        universal_sinks = set([v for v in range(this.n)])
        for s in range(this.n):
            if not visited[s]:
                stack = [(s,{s})]
                while len(stack) > 0:
                    u,sinks = stack.pop()
                    for v in adjacencies[u]:
                        if not visited[v]: pass
                    else: pass
                    
                        


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        # class constructor tests
        g = Graph([(0,1,1),(1,0,1),(1,2,1),(2,1,1),(2,3,1),(3,4,1)], ['adj_list','adj_matrix','incidence_matrix'])
        assert(g.n == 5)
        assert(g.edges == [(0,1,1),(1,0,1),(1,2,1),(2,1,1),(2,3,1),(3,4,1)])
        assert(g.adj_list == [[(1,1)],[(0,1),(2,1)],[(1,1),(3,1)],[(4,1)],[]])
        assert(g.adj_matrix == [[0,(1,1),0,0,0],[(1,1),0,(1,1),0,0],[0,(1,1),0,(1,1),0],[0,0,0,0,(1,1)],[0,0,0,0,0]])
        assert(g.inc_matrix == [[-1,1,0,0,0,0],[1,-1,-1,1,0,0],[0,0,1,-1,-1,0],[0,0,0,0,1,-1],[0,0,0,0,0,1]])
        
        g2 = Graph([(1,0,1),(1,2,1),(1,3,1),(3,4,1),(3,5,1),(3,6,1)], ['adj_list','adj_matrix','incidence_matrix'])
        g3 = Graph([(0,1,1),(0,2,2),(1,3,2),(1,4,2),(2,3,1),(2,4,1),(3,5,1),(4,5,2)], ['adj_list','adj_matrix','incidence_matrix'])

        # Bruteforce Shortest Path test:
        assert(g3.brute_force_SP(0,5) == ([0, 2, 3, 5], 4))

        # Dijkstra Test:
        assert(g3.dijkstra(0) == ([-1, 0, 0, 1, 1, 3], [0.0, 1.0, 2.0, 3.0, 3.0, 4.0]))

        assert(g3.DAG_shortest_paths(0) == ([-1, 0, 0, 2, 2, 3], [0.0, 1.0, 2.0, 3.0, 3.0, 4.0], [5, 3, 4, 1, 2, 0]))

        # Top sort and DAG_Shortest_paths tests
        assert(g2.topologicalSort() == [0, 2, 4, 5, 6, 3, 1])


        # Bellman Ford tests

        assert(g3.Bellman_Ford(1) == ([float('inf'), -1, float('inf'), 1, 1, 3], [float('inf'), 0, float('inf'), 2, 2, 3]))
        g4 = Graph([(0,1,1),(1,2,3),(2,3,2),(3,1,-8),(0,4,3),(3,5,2),(4,5,2)], ['adj_list','adj_matrix','incidence_matrix'])
        assert(g4.Bellman_Ford(0) == False)
        assert(g4.Bellman_Ford(4) == ([float('inf'), float('inf'), float('inf'), float('inf'), -1, 4], [float('inf'), float('inf'), float('inf'), float('inf'), 0, 2]))
        
        # Vertex coloring tests
        edges_ = []
        for i in range(4):
            for j in range(4):
                if i != j: edges_.append((i,j,1))
        K4 = Graph(edges_, ['adj_list','adj_matrix','incidence_matrix'])
        assert(K4.findMinimumColoring() == ([0, 1, 2, 3], 4))
        g5 = Graph([(0,1,1),(1,0,1),(0,2,1),(2,0,1),(1,2,1),(2,1,1),(0,3,1),(3,0,1),(1,5,1),(5,1,1),(2,4,1),(4,2,1),(3,4,1),(4,3,1),(4,5,1),(5,4,1),(3,5,1),(5,3,1)], ['adj_list','adj_matrix','incidence_matrix'])
        assert(g5.findMinimumColoring() == ([0, 1, 2, 1, 0, 2], 3))

        print('tests completed')
    