from pathlib import Path
from typing import List, Tuple, Optional
import networkx as nx
import json
import random
import numpy as np

class SPEA2GraphOptimizer:
    def __init__(
        self,
        graph_edges_path: Path,
        graph_nodes_path: Path,
        origin: int,
        destination: int,
        vehicle_allowed_in_lez: bool,
        population_size: int = 50,
        generations: int = 50,
        max_archive_size: int = 25,
        penalty_signal: float = 0.005,
        penalty_lanes_factor: float = 1.2,
        seed: Optional[int] = None
    ):
        self.origin = origin
        self.destination = destination
        self.vehicle_allowed_in_lez = vehicle_allowed_in_lez
        self.population_size = population_size
        self.generations = generations
        self.max_archive_size = max_archive_size
        self.penalty_signal = penalty_signal
        self.penalty_lanes_factor = penalty_lanes_factor
        # reproducible random
        self.random = random.Random(seed)

        # History and output
        self.archive: List[Tuple[List[Tuple[int,int,int]], Tuple[float,float,float]]] = []
        self.history_hv: List[float] = []
        self.history_spread: List[float] = []
        self.n_evaluations = 0

        # Load graph
        self.graph = self._load_graph(graph_edges_path, graph_nodes_path)
        # normalization constants
        self.norm_distance = max(nx.get_edge_attributes(self.graph, 'length').values())
        self.norm_time = max(nx.get_edge_attributes(self.graph, 'travel_time').values())
        self.norm_fuel = max(nx.get_edge_attributes(self.graph, 'fuel_consumption').values())

    def _load_graph(self, edges_path: Path, nodes_path: Path) -> nx.MultiDiGraph:
        with open(edges_path) as fe, open(nodes_path) as fn:
            edges = json.load(fe)
            nodes = json.load(fn)
        G = nx.MultiDiGraph()
        for node in nodes:
            G.add_node(int(node['osmid']), **node)
        for edge in edges:
            data = dict(edge)
            key = data.pop('key')
            G.add_edge(edge['u'], edge['v'], key=key, **data)
        return G

    def _evaluate_route(self, route: List[Tuple[int,int,int]]) -> Tuple[float,float,float]:
        d = t = f = 0.0
        for u,v,k in route:
            if not self.graph.has_edge(u,v,key=k):
                return float('inf'),float('inf'),float('inf')
            e = self.graph[u][v][k]
            # oneway/LEZ
            if e.get('oneway',False) and (u,v)!=(e['u'],e['v']):
                return float('inf'),float('inf'),float('inf')
            if e.get('lez',False) and not self.vehicle_allowed_in_lez:
                return float('inf'),float('inf'),float('inf')
            # costs
            d += e['length']
            t += e['travel_time']
            fuel = e['fuel_consumption']
            # signal
            if self.graph.nodes[u].get('highway') in ['traffic_signals','stop'] or \
               self.graph.nodes[v].get('highway') in ['traffic_signals','stop']:
                fuel += self.penalty_signal
            # lanes
            try:
                lanes = max(int(x) for x in e.get('lanes',[])) if isinstance(e.get('lanes'),list) else int(e.get('lanes',1))
            except:
                lanes = 1
            if lanes==1:
                fuel *= self.penalty_lanes_factor
            f += fuel
        return (
            round(d/self.norm_distance,6),
            round(t/self.norm_time,6),
            round(f/self.norm_fuel,6)
        )

    def _construct_solution(self) -> List[Tuple[int,int,int]]:
        current = self.origin
        route = []
        visited = set()
        tries = 0
        while current!=self.destination and tries<250:
            nbrs = [(u,v,k) for u,v,k in self.graph.out_edges(current,keys=True) if (u,v,k) not in visited]
            if not nbrs:
                break
            step = self.random.choice(nbrs)
            route.append(step)
            visited.add(step)
            current = step[1]
            tries+=1
        return route if current==self.destination else []

    def _crossover(self, parents: List[List[Tuple[int,int,int]]]) -> List[List[Tuple[int,int,int]]]:
        children = []
        for i in range(0,len(parents),2):
            p1 = parents[i]
            p2 = parents[i+1] if i+1<len(parents) else p1
            com = set(p1)&set(p2)
            if not com:
                children.extend([p1,p2])
            else:
                cut = self.random.choice(list(com))
                c1 = p1[:p1.index(cut)+1] + p2[p2.index(cut)+1:]
                c2 = p2[:p2.index(cut)+1] + p1[p1.index(cut)+1:]
                children.append(c1 if self._is_valid_route(c1) else p1)
                children.append(c2 if self._is_valid_route(c2) else p2)
        return children

    def _mutate(self, population: List[List[Tuple[int,int,int]]]) -> List[List[Tuple[int,int,int]]]:
        out = []
        for route in population:
            if len(route)<2:
                out.append(route)
            else:
                idx = self.random.randrange(len(route))
                suf = self._construct_solution()
                out.append(route[:idx]+suf if suf else route)
        return out

    def _is_valid_route(self, r: List[Tuple[int,int,int]]) -> bool:
        return bool(r and r[0][0]==self.origin and r[-1][1]==self.destination and all(r[i][1]==r[i+1][0] for i in range(len(r)-1)))

    def run(self):
        # initial pop
        pop = [self._construct_solution() for _ in range(self.population_size)]
        fit = [self._evaluate_route(sol) for sol in pop]
        self.n_evaluations = len(pop)
        archive = []
        for gen in range(self.generations):
            # union
            union = [(pop[i],fit[i]) for i in range(len(pop))] + archive
            # strength
            S = [sum(1 for _,f2 in union if self._dominates(f1,f2)) for _,f1 in union]
            # raw
            R = [sum(S[j] for j,(_,f2) in enumerate(union) if self._dominates(f2,f1)) for _,f1 in union]
            # density
            arr = np.array([f for _,f in union])
            dmat = np.linalg.norm(arr - arr[:, None], axis=2)
            k = max(1, int(np.sqrt(len(union))))
            D = [1.0/(np.partition(dmat[i],k)[k]+2.0) for i in range(len(union))]
            # fitness
            F = [R[i]+D[i] for i in range(len(union))]
            # environmental selection
            idx = np.argsort(F)
            sel = [union[i] for i in idx[:self.max_archive_size]]
            archive = sel
            # record metrics
            front = [f for _,f in archive]
            self.history_hv.append(self._compute_hypervolume(front))
            self.history_spread.append(self._compute_spread(front))
            # selection + variation
            mating = []
            for _ in range(self.population_size//2):
                a,b = self.random.sample(range(len(archive)),2)
                mating.append(archive[a][0])
                mating.append(archive[b][0])
            children = self._crossover(mating)
            pop = self._mutate(children)[:self.population_size]
            fit = [self._evaluate_route(sol) for sol in pop]
            self.n_evaluations += len(pop)
        # final archive
        self.archive = archive
        return self.archive

    def _dominates(self, f1, f2):
        return all(a<=b for a,b in zip(f1,f2)) and any(a<b for a,b in zip(f1,f2))

    def _compute_hypervolume(self, front: List[Tuple[float,float,float]], ref=(1.1,1.1,1.1)) -> float:
        pts = np.array([f for f in front if all(f[i]<=ref[i] for i in range(3))])
        if pts.size==0: return 0.0
        pts = pts[np.lexsort(np.rot90(pts))]
        return float(sum(np.prod(np.maximum(0,np.array(ref)-p)) for p in pts))

    def _compute_spread(self, front: List[Tuple[float,float,float]]) -> float:
        pts = np.array(front)
        if len(pts)<2: return 0.0
        d = [np.linalg.norm(pts[i]-pts[i+1]) for i in range(len(pts)-1)]
        return float(np.std(d))
