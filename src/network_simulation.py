"""
Network Simulation Module
Simulates phishing URL spread through social networks
"""

import networkx as nx
import numpy as np
import random
from typing import List, Dict, Set


class PhishingSpreadSimulator:
    """Simulate how phishing URLs spread through social networks"""
    
    def __init__(self, num_nodes: int, network_type: str, infection_rate: float):
        """
        Initialize the phishing spread simulator
        
        Args:
            num_nodes: Number of nodes (users) in the network
            network_type: Type of network ('random', 'barabasi', 'watts_strogatz')
            infection_rate: Probability of infection spread (0.0 to 1.0)
        """
        if num_nodes < 10:
            raise ValueError("Number of nodes must be at least 10")
        if not 0.0 <= infection_rate <= 1.0:
            raise ValueError("Infection rate must be between 0.0 and 1.0")
        
        self.num_nodes = num_nodes
        self.network_type = network_type
        self.infection_rate = infection_rate
        self.graph = None
        self.infected_nodes = set()
        self.susceptible_nodes = set()
        self.infection_timeline = []
        
        # Set random seed for reproducibility
        random.seed(42)
        np.random.seed(42)
    
    def generate_network(self) -> nx.Graph:
        """
        Generate a social network graph based on specified type
        
        Returns:
            NetworkX graph object
        """
        print(f"Generating {self.network_type} network with {self.num_nodes} nodes...")
        
        if self.network_type == 'facebook' or self.network_type == 'real':
            # Load real Facebook network
            try:
                from facebook_network_loader import load_facebook_network
                self.graph = load_facebook_network('data/facebook_combined.txt')
                # Update num_nodes to match actual network
                self.num_nodes = self.graph.number_of_nodes()
                print(f"Using REAL Facebook social network!")
            except Exception as e:
                print(f"Could not load Facebook network: {e}")
                print("Falling back to Barabási-Albert network...")
                m = min(3, self.num_nodes - 1)
                self.graph = nx.barabasi_albert_graph(self.num_nodes, m, seed=42)
        
        elif self.network_type == 'random' or self.network_type == 'erdos_renyi':
            # Erdős-Rényi random graph
            # Probability chosen to ensure connected graph
            p = 2 * np.log(self.num_nodes) / self.num_nodes
            self.graph = nx.erdos_renyi_graph(self.num_nodes, p, seed=42)
            
        elif self.network_type == 'barabasi' or self.network_type == 'scale_free':
            # Barabási-Albert scale-free network
            # Each new node attaches to m existing nodes
            m = min(3, self.num_nodes - 1)
            self.graph = nx.barabasi_albert_graph(self.num_nodes, m, seed=42)
            
        elif self.network_type == 'watts_strogatz' or self.network_type == 'small_world':
            # Watts-Strogatz small-world network
            k = min(6, self.num_nodes - 1)  # Each node connected to k nearest neighbors
            p = 0.1  # Rewiring probability
            self.graph = nx.watts_strogatz_graph(self.num_nodes, k, p, seed=42)
            
        else:
            raise ValueError(f"Unknown network type: {self.network_type}")
        
        # Initialize all nodes as susceptible
        self.susceptible_nodes = set(self.graph.nodes())
        
        print(f"Network generated: {self.graph.number_of_nodes()} nodes, "
              f"{self.graph.number_of_edges()} edges")
        
        return self.graph
    
    def initialize_infection(self, seed_nodes: List[int] = None, num_seeds: int = 1):
        """
        Start infection from seed nodes
        
        Args:
            seed_nodes: List of node IDs to start infection (optional)
            num_seeds: Number of random seed nodes if seed_nodes not provided
        """
        if self.graph is None:
            self.generate_network()
        
        if seed_nodes is None:
            # Select random seed nodes
            seed_nodes = random.sample(list(self.graph.nodes()), num_seeds)
        
        # Infect seed nodes
        for node in seed_nodes:
            if node in self.susceptible_nodes:
                self.infected_nodes.add(node)
                self.susceptible_nodes.remove(node)
        
        # Record initial state
        self.infection_timeline = [len(self.infected_nodes)]
        
        print(f"Infection initialized with {len(self.infected_nodes)} seed node(s): {seed_nodes}")
    
    def simulate_step(self) -> int:
        """
        Run one simulation step
        
        Returns:
            Number of newly infected nodes in this step
        """
        newly_infected = set()
        
        # For each infected node, try to infect neighbors
        for infected_node in list(self.infected_nodes):
            # Get susceptible neighbors
            neighbors = set(self.graph.neighbors(infected_node))
            susceptible_neighbors = neighbors.intersection(self.susceptible_nodes)
            
            # Try to infect each susceptible neighbor
            for neighbor in susceptible_neighbors:
                # Infection occurs with probability = infection_rate
                if random.random() < self.infection_rate:
                    newly_infected.add(neighbor)
        
        # Update infected and susceptible sets
        self.infected_nodes.update(newly_infected)
        self.susceptible_nodes.difference_update(newly_infected)
        
        # Record timeline
        self.infection_timeline.append(len(self.infected_nodes))
        
        return len(newly_infected)
    
    def run_simulation(self, max_steps: int = 100) -> Dict:
        """
        Run the complete simulation
        
        Args:
            max_steps: Maximum number of simulation steps
            
        Returns:
            Dictionary containing simulation results and metrics
        """
        if self.graph is None:
            self.generate_network()
        
        if len(self.infected_nodes) == 0:
            self.initialize_infection()
        
        print(f"\nRunning simulation (max {max_steps} steps)...")
        
        step = 0
        while step < max_steps:
            newly_infected = self.simulate_step()
            step += 1
            
            # Check for convergence (no new infections)
            if newly_infected == 0:
                print(f"Simulation converged at step {step}")
                break
            
            if step % 10 == 0:
                print(f"Step {step}: {len(self.infected_nodes)} infected "
                      f"({len(self.infected_nodes)/self.num_nodes*100:.1f}%)")
        
        # Calculate final metrics
        results = {
            'network_type': self.network_type,
            'num_nodes': self.num_nodes,
            'num_edges': self.graph.number_of_edges(),
            'infection_rate': self.infection_rate,
            'final_infected_count': len(self.infected_nodes),
            'infection_percentage': len(self.infected_nodes) / self.num_nodes * 100,
            'steps_to_stabilize': step,
            'timeline': self.infection_timeline,
            'infected_nodes': list(self.infected_nodes),
            'susceptible_nodes': list(self.susceptible_nodes)
        }
        
        print(f"\nSimulation completed:")
        print(f"  Final infected: {results['final_infected_count']} "
              f"({results['infection_percentage']:.1f}%)")
        print(f"  Steps: {results['steps_to_stabilize']}")
        
        return results
    
    def get_infection_timeline(self) -> List[int]:
        """
        Return infection count at each time step
        
        Returns:
            List of infection counts
        """
        return self.infection_timeline
    
    def identify_critical_nodes(self, top_n: int = 10) -> List[tuple]:
        """
        Identify high-centrality nodes that could accelerate spread
        
        Args:
            top_n: Number of top nodes to return
            
        Returns:
            List of (node_id, centrality_score) tuples
        """
        if self.graph is None:
            self.generate_network()
        
        # Calculate degree centrality
        centrality = nx.degree_centrality(self.graph)
        
        # Sort by centrality and get top N
        sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
        critical_nodes = sorted_nodes[:top_n]
        
        return critical_nodes
    
    def reset_simulation(self):
        """Reset the simulation to initial state"""
        self.infected_nodes = set()
        self.susceptible_nodes = set(self.graph.nodes()) if self.graph else set()
        self.infection_timeline = []


if __name__ == "__main__":
    # Test the network simulation
    print("Testing Phishing Spread Simulator\n")
    print("="*60)
    
    # Test 1: Barabási-Albert network
    print("\nTest 1: Barabási-Albert Scale-Free Network")
    print("-"*60)
    simulator1 = PhishingSpreadSimulator(
        num_nodes=150,
        network_type='barabasi',
        infection_rate=0.3
    )
    results1 = simulator1.run_simulation(max_steps=50)
    
    # Test 2: Watts-Strogatz network
    print("\n" + "="*60)
    print("\nTest 2: Watts-Strogatz Small-World Network")
    print("-"*60)
    simulator2 = PhishingSpreadSimulator(
        num_nodes=150,
        network_type='watts_strogatz',
        infection_rate=0.5
    )
    results2 = simulator2.run_simulation(max_steps=50)
    
    # Test 3: Identify critical nodes
    print("\n" + "="*60)
    print("\nTest 3: Critical Node Identification")
    print("-"*60)
    critical_nodes = simulator1.identify_critical_nodes(top_n=5)
    print("Top 5 critical nodes (by degree centrality):")
    for node_id, centrality in critical_nodes:
        print(f"  Node {node_id}: {centrality:.4f}")
    
    print("\n" + "="*60)
    print("\nAll tests completed successfully!")



def run_parameter_comparison(num_nodes: int = 150, network_type: str = 'barabasi',
                            infection_rates: List[float] = None, max_steps: int = 50) -> Dict:
    """
    Run multiple simulations with different infection rates and compare results
    
    Args:
        num_nodes: Number of nodes in the network
        network_type: Type of network to generate
        infection_rates: List of infection rates to test (default: [0.1, 0.3, 0.5, 0.7, 0.9])
        max_steps: Maximum steps for each simulation
        
    Returns:
        Dictionary containing comparison results
    """
    if infection_rates is None:
        infection_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    print("\n" + "="*60)
    print("PARAMETER COMPARISON: Infection Rate Variation")
    print("="*60)
    print(f"Network: {network_type}, Nodes: {num_nodes}")
    print(f"Testing infection rates: {infection_rates}\n")
    
    results = []
    
    for rate in infection_rates:
        print(f"\n--- Testing infection rate: {rate} ---")
        simulator = PhishingSpreadSimulator(num_nodes, network_type, rate)
        sim_results = simulator.run_simulation(max_steps)
        
        results.append({
            'infection_rate': rate,
            'final_infected_count': sim_results['final_infected_count'],
            'infection_percentage': sim_results['infection_percentage'],
            'steps_to_stabilize': sim_results['steps_to_stabilize'],
            'timeline': sim_results['timeline']
        })
    
    # Print summary table
    print("\n" + "="*60)
    print("SUMMARY TABLE")
    print("="*60)
    print(f"{'Rate':<10} {'Infected':<12} {'Percentage':<12} {'Steps':<10}")
    print("-"*60)
    
    for result in results:
        print(f"{result['infection_rate']:<10.1f} "
              f"{result['final_infected_count']:<12} "
              f"{result['infection_percentage']:<12.1f} "
              f"{result['steps_to_stabilize']:<10}")
    
    print("="*60)
    
    return {
        'network_type': network_type,
        'num_nodes': num_nodes,
        'results': results
    }
