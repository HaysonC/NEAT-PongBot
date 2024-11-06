import time
import json
from typing import Any, Sequence, Optional, override
import neat
from graphviz import Digraph
from matplotlib import pyplot as plt

# use this analytics for other types of training as well

# TODO: IMPLEMENT THE ANALYTICS CLASS
class Analytics(object):
    def __init__(self, pop: int,
                 gen: int = 0,
                 winRate: float = 0.0,
                 startTime: float = time.time(),
                 prevTime: float = time.time()
                 ) -> None:
        """
        Initialize the analytics object

        :param pop: The population of the generation
        :param gen: The generation number
        :param winRate: The win rate of the generation against a certain agent
        :param startTime: The previous time
        :param prevTime: The previous time
        """
        
        self._startTime = startTime
        self._prevTime = prevTime
        self._pop: int = pop
        self._gen: int = gen
        self._time : list[float] = self._entry(gen=gen, lastEntry=startTime)
        self.winCount: float = 0
        self._winRate: list[float] = self._entry(gen=gen, lastEntry=winRate)
        self._timesForEachGen: list[float] = self._entry(gen=gen, lastEntry=0)

    class _entry(list):
        def __init__(self, lastEntry: Any = None, gen: int = 0) -> None:
            super().__init__()
            self.gen = gen
            super().extend([None] * gen)
            super().append(lastEntry)
        
        def load(self, data: list):
            self.pop(0)
            self.extend(data)

        @override
        def __str__(self) -> str:
            return str(self.gen) + "\n" + super().__str__()


    def save_to_file(self, path: str) -> None:
        """
        Save the analytics to a file

        :param path: The path to the file
        """
        data = {}
        for attr_item in self._get_attr_list():
            data[attr_item] = getattr(self, f"_{attr_item}")
        with open(path, 'w') as f:
            json.dump(data, f)

    def load_from_file(self, path: str) -> 'Analytics':
        """
        Load the analytics from a file

        :param path: The path to the file
        """
        with open(path, 'r') as f:
            data = json.load(f)
        for attr_item in data.keys():
            entry = self._entry()
            entry.load(data[attr_item])
            setattr(self, f"_{attr_item}", entry)
        return self

    @staticmethod
    def _load_data(path: str):
        """
        Load the data from a json object
        """
        with open(path, 'r') as f:
            data = json.load(f)
        return data

    def append(self,
               data: dict[str, Any]) -> None:
        """
        Append the analytics of the new generation

        :param data: The data of the new generation
        """
        self._gen += 1
        self._prevTime = time.time()
        for attr_item in self._get_attr_list():
            try:
                getattr(self, f"{attr_item}").append(data[attr_item if not attr_item.startswith("_") else attr_item[1:]])
            except KeyError:
                if attr_item == "_time":
                    getattr(self, f"{attr_item}").append(time.time())
                elif attr_item == "_timesForEachGen":
                    getattr(self, f"{attr_item}").append(time.time() - self._prevTime)
                else:
                    getattr(self, f"{attr_item}").append(None)


    def __add__(self, other: "Analytics") -> "Analytics":
        """
        Add two analytics objects together
        """
        newAnalytics = Analytics(self._pop, self._gen, self._winRates[-1], self._timesForEachGen[-1])
        for attr_item in self._get_attr_list():
            newAnalytics[f"_{attr_item}"] = getattr(self, f"_{attr_item}") + getattr(other, f"_{attr_item}")
        return newAnalytics

    def __delitem__(self, key: int) -> None:
        """
        Delete an item from the analytics of a certain generation
        """
        for attr_item in self._get_attr_list():
            try:
                getattr(self, f"_{attr_item}").pop(key)
            except IndexError:
                print(f"Could not delete the value of {attr_item} at index {key}")

    def __getitem__(self, item: int) -> dict:
        """
        Get the analytics of a certain generation
        """
        attr_items = self._get_attr_list()
        return {attr_item: getattr(self, f"_{attr_item}")[item] for attr_item in attr_items}


    def get_latest(self, attr: str) -> Any:
        """
        Get the latest value of an attribute

        :param attr: The attribute to get the latest value of
        """
        try:
            return getattr(self, f"_{attr}")[-1]
        except AttributeError:
            print(f"Could not find the attribute {attr}")
            return None
        except IndexError:
            return None

    def __setitem__(self, key, value : dict) -> None:
        """
        Set the analytics of a certain generation
        """
        attrs = self._get_attr_list()
        for attr in attrs:
            try:
                getattr(self, f"_{attr}")[key] = value[attr]
            except IndexError:
                print(f"Could not set the value of {attr} at index {key}")
            except KeyError:
                print(f"Could not find the key {attr} in the dictionary")

    def __len__(self) -> int:
        """
        Get the length of the analytics
        """
        return self._gen


    def plot(self, attr_itemsToPlot: Sequence[str] | str):
        """
        Plot the analytics

        Example usage:
        analytics.plot(["winRate", "time"])
        # or
        analytics.plot("winRate")
        analytics.plot("time")

        :param attr_itemsToPlot: The attr_items to plot (variable names)
        """
        if isinstance(attr_itemsToPlot, str):
            attr_itemsToPlot = [attr_itemsToPlot]
        for attr_item in attr_itemsToPlot:
            # new plt
            if not attr_item.startswith("_"):
                attr_item = f"_{attr_item}"
            try:
                entry: Analytics._entry = getattr(self, f"_{attr_item}")
                generations = range(self._gen)
                plt.plot(generations, entry)
                plt.xlabel("Generation")
                plt.ylabel(attr_item)
                plt.title(f"{attr_item} vs Generation")
                plt.show()
            except AttributeError:
                print(f"Could not find the attribute {attr_item}")
            pass
        
    def _get_attr_list(self) -> list[str]:
        """
        Get the list of attributes in the analytics
        """
        return [i for i in self.__dict__.keys() if isinstance(getattr(self, i), Analytics._entry)]

    def get_totTime(self) -> float:
        """
        Get the time elapsed from the start of the training
        """
        return time.time() - self._startTime

    def add_win(self, n: int = 1) -> None:
        """
        Add a win to the win count
        """
        self.winCount  += n

    def get_winRate(self, matchesCount: int, popCount: int) -> float:
        """
        Get the win rate of the generation
        Will reset the win count

        :param matchesCount: The number of matches played
        :param popCount: The number of agents in the population
        :return: The win rate of the generation
        """
        ret = self.winCount / (matchesCount * popCount)
        print(f"Win rate against set agent: {ret}")
        self.winCount = 0
        return ret


# TODO: Implement the Neat_Analytics class specifically for NEAT
class Neat_Analytics(Analytics):
    def __init__(self, pop: int,
                 gen: int = 0,
                 winRate: float = 0.0,
                 bestGenomes: list[Optional[neat.DefaultGenome]] = None,
                 bestGenomesFitness: list[float] = None,
                 bestGenomesSpecies: list[int] = None,
                 avgFitness: list[float] = None,
                 stagnation: list[int] = None,
                 startTime: float = time.time()) -> None:
        """
        Initialize the analytics object

        :param pop: The population of the generation
        :param gen: The generation number
        :param winRate: The win rate of the generation against a certain agent
        :param startTime: The previous time
        """
        super().__init__(pop, gen, winRate, startTime)
        self._bestGenome: list[Optional[neat.DefaultGenome]] = super()._entry(gen=gen, lastEntry=bestGenomes)
        self._bestGenomesFitness: list[float] = super()._entry(gen=gen, lastEntry=bestGenomesFitness)
        self._bestGenomesSpecies: list[int] = super()._entry(gen=gen, lastEntry=bestGenomesSpecies)
        self._avgFitness: list[float] = super()._entry(gen=gen, lastEntry=avgFitness)
        self._stagnation: list[int] = super()._entry(gen=gen, lastEntry=stagnation)
        # implement the rest of the functions, change the code to also take in neat specific info

    # TODO: FIX THIS FUNCTION
    @staticmethod
    def visualize_genome(genome: neat.DefaultGenome) -> Digraph:
        """
        Function to visualize a neat genome using graphviz.

        ONLY FOR NEAT

        :param genome: The genome object.
        :return: The Digraph object representing the neural network.
        """

        dot = Digraph(comment='Neural Network', format='png')
        dot.attr(rankdir='LR')  # Left to right layout

        # Define node names for input and output
        node_names = {
            -1: 'ball.x', -2: 'ball.y', -3: 'paddle.x', -4: 'paddle.y',  # Input nodes
            0: 'Output_U', 1: 'Output_D', 2: 'Output_Stay'  # Output nodes
        }

        # Identify all nodes
        input_nodes = [-1, -2, -3, -4]
        output_nodes = [0, 1, 2]
        hidden_nodes = [node for node in genome.nodes if node not in input_nodes and node not in output_nodes]

        # Add input nodes
        for node in input_nodes:
            if node in node_names:
                dot.node(str(node), label=node_names[node], style='filled', fillcolor='lightblue')

        # Add hidden nodes
        for node in hidden_nodes:
            dot.node(str(node), label=f'Hidden {node}', style='filled', fillcolor='lightgrey')

        # Add output nodes
        for node in output_nodes:
            if node in node_names:
                dot.node(str(node), label=node_names[node], style='filled', fillcolor='lightgreen')

        # Add edges with weights
        for conn in genome.connections.values():
            if conn.enabled:
                input_node = conn.key[0]
                output_node = conn.key[1]
                weight = conn.weight
                # Adjust the edge color based on weight direction
                if weight > 0:
                    color = 'green'
                elif weight < 0:
                    color = 'red'
                else:
                    color = 'grey'
                dot.edge(str(input_node), str(output_node), label=f"{weight:.2f}", color=color)

        # Render the graph and view it
        dot.render('neural_network', view=True, cleanup=True)

        print("Visualization generated: neural_network.png")
        return dot


class Q_Analytics(Analytics): ...