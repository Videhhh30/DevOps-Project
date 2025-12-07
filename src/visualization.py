"""
Visualization Module
Generate plots and visualizations for phishing detection and network simulation
"""

import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


class Visualizer:
    """Generate visualizations for ML models and network simulations"""
    
    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, save_path: str = 'outputs/confusion_matrix.png'):
        """
        Generate confusion matrix heatmap
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            save_path: Path to save the plot
        """
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Create figure
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Legitimate', 'Phishing'],
                   yticklabels=['Legitimate', 'Phishing'])
        plt.title('Confusion Matrix - Phishing URL Detection', fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        # Save plot
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
        plt.close()
    
    @staticmethod
    def plot_feature_importance(model, feature_names: list, 
                               save_path: str = 'outputs/feature_importance.png',
                               top_n: int = 10):
        """
        Plot top feature importances
        
        Args:
            model: Trained model with feature_importances_ attribute
            feature_names: List of feature names
            save_path: Path to save the plot
            top_n: Number of top features to display
        """
        if not hasattr(model, 'feature_importances_'):
            print("Model does not have feature_importances_ attribute")
            return
        
        # Get feature importances
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        
        # Create DataFrame for plotting
        top_features = [feature_names[i] for i in indices]
        top_importances = [importances[i] for i in indices]
        
        # Create figure
        plt.figure(figsize=(10, 6))
        bars = plt.barh(range(len(top_features)), top_importances, color='steelblue')
        plt.yticks(range(len(top_features)), top_features)
        plt.xlabel('Importance Score', fontsize=12)
        plt.title(f'Top {top_n} Feature Importances', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, top_importances)):
            plt.text(value, i, f' {value:.4f}', va='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Feature importance plot saved to {save_path}")
        plt.close()
    
    @staticmethod
    def plot_network_graph(graph: nx.Graph, infected_nodes: set, 
                          critical_nodes: list = None,
                          save_path: str = 'outputs/network_graph.png'):
        """
        Visualize network with infection status
        
        Args:
            graph: NetworkX graph object
            infected_nodes: Set of infected node IDs
            critical_nodes: List of (node_id, centrality) tuples for critical nodes
            save_path: Path to save the plot
        """
        plt.figure(figsize=(12, 10))
        
        # Use faster layout for large networks
        num_nodes = graph.number_of_nodes()
        if num_nodes > 500:
            print(f"  Using fast layout for large network ({num_nodes} nodes)...")
            # Use random layout for very large networks (much faster)
            pos = nx.random_layout(graph, seed=42)
        else:
            # Use spring layout for smaller networks (better visualization)
            pos = nx.spring_layout(graph, seed=42, k=0.5, iterations=50)
        
        # Separate nodes by status
        susceptible = [n for n in graph.nodes() if n not in infected_nodes]
        infected = list(infected_nodes)
        
        # Draw edges
        nx.draw_networkx_edges(graph, pos, alpha=0.2, width=0.5)
        
        # Draw susceptible nodes
        nx.draw_networkx_nodes(graph, pos, nodelist=susceptible,
                              node_color='lightblue', node_size=50,
                              label='Susceptible')
        
        # Draw infected nodes
        nx.draw_networkx_nodes(graph, pos, nodelist=infected,
                              node_color='red', node_size=100,
                              label='Infected')
        
        # Highlight critical nodes if provided
        if critical_nodes:
            critical_ids = [node_id for node_id, _ in critical_nodes]
            nx.draw_networkx_nodes(graph, pos, nodelist=critical_ids,
                                  node_color='gold', node_size=200,
                                  node_shape='*', label='Critical Nodes')
        
        plt.title('Social Network - Phishing Spread Visualization', 
                 fontsize=14, fontweight='bold')
        plt.legend(loc='upper right', fontsize=10)
        plt.axis('off')
        plt.tight_layout()
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Network graph saved to {save_path}")
        plt.close()
    
    @staticmethod
    def plot_spread_timeline(timeline: list, infection_rate: float = None,
                           save_path: str = 'outputs/spread_timeline.png'):
        """
        Plot infection count over time
        
        Args:
            timeline: List of infection counts at each step
            infection_rate: Infection rate parameter (for title)
            save_path: Path to save the plot
        """
        plt.figure(figsize=(10, 6))
        
        steps = range(len(timeline))
        plt.plot(steps, timeline, marker='o', linewidth=2, markersize=6, color='red')
        plt.fill_between(steps, timeline, alpha=0.3, color='red')
        
        plt.xlabel('Simulation Step', fontsize=12)
        plt.ylabel('Number of Infected Nodes', fontsize=12)
        
        title = 'Phishing URL Spread Over Time'
        if infection_rate is not None:
            title += f' (Infection Rate: {infection_rate})'
        plt.title(title, fontsize=14, fontweight='bold')
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Spread timeline plot saved to {save_path}")
        plt.close()
    
    @staticmethod
    def plot_parameter_comparison(results: dict, 
                                 save_path: str = 'outputs/parameter_comparison.png'):
        """
        Compare simulation results across different parameters
        
        Args:
            results: Dictionary with 'results' key containing list of simulation results
            save_path: Path to save the plot
        """
        # Extract data
        infection_rates = [r['infection_rate'] for r in results['results']]
        final_infected = [r['final_infected_count'] for r in results['results']]
        infection_pct = [r['infection_percentage'] for r in results['results']]
        steps = [r['steps_to_stabilize'] for r in results['results']]
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Infection percentage vs rate
        ax1.plot(infection_rates, infection_pct, marker='o', linewidth=2, 
                markersize=8, color='red')
        ax1.fill_between(infection_rates, infection_pct, alpha=0.3, color='red')
        ax1.set_xlabel('Infection Rate', fontsize=12)
        ax1.set_ylabel('Final Infection Percentage (%)', fontsize=12)
        ax1.set_title('Impact of Infection Rate on Spread', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 105)
        
        # Plot 2: Steps to stabilize vs rate
        ax2.plot(infection_rates, steps, marker='s', linewidth=2, 
                markersize=8, color='blue')
        ax2.fill_between(infection_rates, steps, alpha=0.3, color='blue')
        ax2.set_xlabel('Infection Rate', fontsize=12)
        ax2.set_ylabel('Steps to Stabilize', fontsize=12)
        ax2.set_title('Convergence Speed vs Infection Rate', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, 1)
        
        plt.suptitle(f'Parameter Comparison - {results["network_type"]} Network', 
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Parameter comparison plot saved to {save_path}")
        plt.close()
    
    @staticmethod
    def plot_multiple_timelines(timelines_dict: dict,
                               save_path: str = 'outputs/multiple_timelines.png'):
        """
        Plot multiple infection timelines for comparison
        
        Args:
            timelines_dict: Dictionary of {label: timeline} pairs
            save_path: Path to save the plot
        """
        plt.figure(figsize=(12, 6))
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(timelines_dict)))
        
        for (label, timeline), color in zip(timelines_dict.items(), colors):
            steps = range(len(timeline))
            plt.plot(steps, timeline, marker='o', linewidth=2, 
                    markersize=4, label=label, color=color)
        
        plt.xlabel('Simulation Step', fontsize=12)
        plt.ylabel('Number of Infected Nodes', fontsize=12)
        plt.title('Phishing Spread Comparison', fontsize=14, fontweight='bold')
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Multiple timelines plot saved to {save_path}")
        plt.close()


if __name__ == "__main__":
    # Test visualizations
    print("Testing Visualization Module\n")
    
    # Test 1: Confusion Matrix
    print("Test 1: Confusion Matrix")
    y_true = np.array([0, 0, 1, 1, 0, 1, 1, 0, 1, 0])
    y_pred = np.array([0, 0, 1, 1, 0, 1, 0, 0, 1, 0])
    Visualizer.plot_confusion_matrix(y_true, y_pred)
    
    # Test 2: Feature Importance (mock data)
    print("\nTest 2: Feature Importance")
    from sklearn.ensemble import RandomForestClassifier
    
    # Create mock model with feature importances
    X_mock = np.random.rand(100, 10)
    y_mock = np.random.randint(0, 2, 100)
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_mock, y_mock)
    
    feature_names = [f'feature_{i}' for i in range(10)]
    Visualizer.plot_feature_importance(model, feature_names)
    
    # Test 3: Network Graph
    print("\nTest 3: Network Graph")
    G = nx.barabasi_albert_graph(50, 2, seed=42)
    infected = set(list(G.nodes())[:10])
    critical = [(0, 0.5), (1, 0.4), (2, 0.3)]
    Visualizer.plot_network_graph(G, infected, critical)
    
    # Test 4: Spread Timeline
    print("\nTest 4: Spread Timeline")
    timeline = [1, 3, 7, 15, 28, 45, 60, 70, 75, 78, 80]
    Visualizer.plot_spread_timeline(timeline, infection_rate=0.3)
    
    # Test 5: Parameter Comparison
    print("\nTest 5: Parameter Comparison")
    mock_results = {
        'network_type': 'barabasi',
        'num_nodes': 100,
        'results': [
            {'infection_rate': 0.1, 'final_infected_count': 20, 
             'infection_percentage': 20.0, 'steps_to_stabilize': 15},
            {'infection_rate': 0.3, 'final_infected_count': 50, 
             'infection_percentage': 50.0, 'steps_to_stabilize': 12},
            {'infection_rate': 0.5, 'final_infected_count': 80, 
             'infection_percentage': 80.0, 'steps_to_stabilize': 10},
            {'infection_rate': 0.7, 'final_infected_count': 95, 
             'infection_percentage': 95.0, 'steps_to_stabilize': 8},
            {'infection_rate': 0.9, 'final_infected_count': 100, 
             'infection_percentage': 100.0, 'steps_to_stabilize': 6},
        ]
    }
    Visualizer.plot_parameter_comparison(mock_results)
    
    print("\n" + "="*60)
    print("All visualization tests completed successfully!")
    print("Check the 'outputs/' directory for generated plots.")
