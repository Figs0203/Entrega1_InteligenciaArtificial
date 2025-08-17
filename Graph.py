import networkx as nx
import matplotlib.pyplot as plt
import os

class Graph:
    def __init__(self):
        self.G = nx.Graph()

    def add_node(self, node):
        self.G.add_node(node)

    def add_edge(self, u, v, weight):
        self.G.add_edge(u, v, weight=weight)

    def create_from_dict(self, graph_dict):
        for node, neighbors in graph_dict.items():
            self.add_node(node)
            for neighbor in neighbors:
                self.add_edge(node, neighbor[0], weight=neighbor[1])

    def draw(self,  result_path, pos=None,):
        if pos is None:
            pos = nx.spring_layout(self.G)

        plt.figure(figsize=(8, 6))
        nx.draw(self.G, pos, with_labels=True, node_color="lightblue", font_size=10, node_size=100)
        plt.title("Graph Visualization")
        plt.savefig(result_path)
        plt.close()

    def draw_with_path(self, path, pos=None,  title="Graph Visualization with Path", result_path=None):
        if pos is None:
            pos = nx.spring_layout(self.G)

        fig, ax = plt.subplots(figsize=(8, 6))

        nx.draw(
            self.G, pos, with_labels=True, ax=ax,
            node_size=2000, node_color="lightblue", font_size=10
        )

        if path:
            path_edges = list(zip(path, path[1:]))
            nx.draw_networkx_edges(self.G, pos, edgelist=path_edges, edge_color="red", width=2)

        edge_labels = nx.get_edge_attributes(self.G, 'weight')
        nx.draw_networkx_edge_labels(self.G, pos, edge_labels=edge_labels)

        ax.set_title(title)
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)


        if result_path:
            if not os.path.exists(os.path.dirname(result_path)):
                os.makedirs(os.path.dirname(result_path))
            plt.savefig(result_path)
        else:
            plt.show()
