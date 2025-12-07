"""
Main Application Entry Point
Phishing URL Detection and Social Network Spread Simulation System
"""

import sys
import os
import argparse
import pandas as pd

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from network_simulation import PhishingSpreadSimulator, run_parameter_comparison
from visualization import Visualizer


def train_models(dataset_path: str):
    """
    Train CNN model on the dataset
    
    Args:
        dataset_path: Path to CSV dataset
    """
    from simple_cnn import train_simple_cnn
    
    print("\n" + "="*60)
    print("TRAINING CNN MODEL FOR PHISHING DETECTION")
    print("="*60)
    
    # Train CNN model
    cnn, metrics = train_simple_cnn(dataset_path)
    
    print("\n" + "="*60)
    print("Training completed successfully!")
    print("="*60)
    
    return {'model': cnn, 'metrics': metrics}


def predict_url(url: str, model_path: str = 'models/simple_cnn_model.pkl'):
    """
    Predict whether a URL is phishing or legitimate using CNN
    
    Args:
        url: URL to classify
        model_path: Path to trained CNN model
        
    Returns:
        Prediction result dictionary
    """
    from simple_cnn import SimpleCNNClassifier
    
    # Load CNN model
    cnn = SimpleCNNClassifier()
    cnn.load_model(model_path)
    
    # Make prediction
    prob = cnn.predict([url])[0]
    pred = 1 if prob > 0.5 else 0
    
    # Format result
    result = {
        'url': url,
        'prediction': 'phishing' if pred == 1 else 'legitimate',
        'confidence': prob if pred == 1 else (1 - prob),
        'probability_phishing': prob,
        'probability_legitimate': 1 - prob
    }
    
    # Print result
    print("\n" + "="*60)
    print("PHISHING URL DETECTION RESULT (CNN)")
    print("="*60)
    print(f"URL: {result['url']}")
    print(f"\nPrediction: {result['prediction'].upper()}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"\nProbabilities:")
    print(f"  Legitimate: {result['probability_legitimate']:.2%}")
    print(f"  Phishing:   {result['probability_phishing']:.2%}")
    
    if result['prediction'] == 'phishing':
        print(f"\n⚠️  WARNING: This URL appears to be a PHISHING attempt!")
    else:
        print(f"\n✓ This URL appears to be LEGITIMATE")
    
    print("="*60 + "\n")
    
    return result


def run_simulation(infection_rate: float = 0.5, network_type: str = 'barabasi',
                  num_nodes: int = 150, max_steps: int = 50,
                  ml_confidence: float = None):
    """
    Run phishing spread simulation
    
    Args:
        infection_rate: Base infection rate (0.0 to 1.0)
        network_type: Type of network ('random', 'barabasi', 'watts_strogatz')
        num_nodes: Number of nodes in network
        max_steps: Maximum simulation steps
        ml_confidence: ML prediction confidence to adjust infection rate
        
    Returns:
        Simulation results dictionary
    """
    print("\n" + "="*60)
    print("PHISHING SPREAD SIMULATION")
    print("="*60)
    
    # Adjust infection rate based on ML confidence if provided
    if ml_confidence is not None:
        # Higher confidence in phishing = higher infection rate
        adjusted_rate = min(infection_rate * (1 + ml_confidence), 1.0)
        print(f"ML Confidence: {ml_confidence:.2%}")
        print(f"Base infection rate: {infection_rate}")
        print(f"Adjusted infection rate: {adjusted_rate:.2f}")
        infection_rate = adjusted_rate
    
    # Run simulation
    simulator = PhishingSpreadSimulator(num_nodes, network_type, infection_rate)
    results = simulator.run_simulation(max_steps)
    
    # Identify critical nodes
    critical_nodes = simulator.identify_critical_nodes(top_n=10)
    results['critical_nodes'] = critical_nodes
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    Visualizer.plot_network_graph(
        simulator.graph,
        simulator.infected_nodes,
        critical_nodes[:5],  # Show top 5 critical nodes
        'outputs/network_graph.png'
    )
    
    Visualizer.plot_spread_timeline(
        results['timeline'],
        infection_rate,
        'outputs/spread_timeline.png'
    )
    
    print("\n" + "="*60)
    print("Simulation completed successfully!")
    print("="*60)
    
    return results


def run_parameter_study(network_type: str = 'barabasi', num_nodes: int = 150):
    """
    Run simulations with varying infection rates
    
    Args:
        network_type: Type of network
        num_nodes: Number of nodes
    """
    print("\n" + "="*60)
    print("PARAMETER VARIATION STUDY")
    print("="*60)
    
    results = run_parameter_comparison(num_nodes, network_type)
    
    # Generate comparison visualization
    Visualizer.plot_parameter_comparison(results, 'outputs/parameter_comparison.png')
    
    return results


def integrated_workflow(url: str, dataset_path: str = None):
    """
    Integrated workflow: predict URL and simulate spread based on confidence
    
    Args:
        url: URL to analyze
        dataset_path: Path to dataset for training (if models don't exist)
    """
    print("\n" + "="*60)
    print("INTEGRATED PHISHING ANALYSIS WORKFLOW")
    print("="*60)
    
    # Check if models exist, train if not
    if not os.path.exists('models/best_model.pkl'):
        if dataset_path is None:
            print("Error: No trained model found and no dataset provided for training")
            return
        print("\nNo trained model found. Training models first...")
        train_models(dataset_path)
    
    # Predict URL
    print("\nStep 1: Analyzing URL with ML model...")
    result = predict_url(url)
    
    # Run simulation based on prediction
    if result['prediction'] == 'phishing':
        print("\nStep 2: Simulating phishing spread...")
        print(f"Using ML confidence ({result['confidence']:.2%}) to adjust infection rate")
        
        # Use confidence to determine infection rate
        # High confidence phishing = high infection rate
        base_rate = 0.3
        sim_results = run_simulation(
            infection_rate=base_rate,
            network_type='barabasi',
            num_nodes=150,
            ml_confidence=result['probability_phishing']
        )
        
        print(f"\nSimulation Results:")
        print(f"  Final infected: {sim_results['final_infected_count']} nodes "
              f"({sim_results['infection_percentage']:.1f}%)")
        print(f"  Convergence: {sim_results['steps_to_stabilize']} steps")
    else:
        print("\nURL classified as legitimate. Skipping spread simulation.")
    
    print("\n" + "="*60)
    print("Integrated analysis completed!")
    print("="*60)


def main():
    """Main entry point with command-line interface"""
    parser = argparse.ArgumentParser(
        description='Phishing URL Detection and Social Network Spread Simulation'
    )
    
    parser.add_argument('--mode', type=str, required=True,
                       choices=['train', 'predict', 'simulate', 'parameter-study', 'integrated'],
                       help='Operation mode')
    
    parser.add_argument('--dataset', type=str,
                       help='Path to dataset CSV file (for training)')
    
    parser.add_argument('--url', type=str,
                       help='URL to predict (for predict/integrated mode)')
    
    parser.add_argument('--infection-rate', type=float, default=0.5,
                       help='Infection rate for simulation (0.0 to 1.0)')
    
    parser.add_argument('--network-type', type=str, default='barabasi',
                       choices=['random', 'barabasi', 'watts_strogatz', 'facebook', 'real'],
                       help='Type of social network (use "facebook" for real Facebook network)')
    
    parser.add_argument('--num-nodes', type=int, default=150,
                       help='Number of nodes in network')
    
    parser.add_argument('--max-steps', type=int, default=50,
                       help='Maximum simulation steps')
    
    args = parser.parse_args()
    
    # Execute based on mode
    if args.mode == 'train':
        if not args.dataset:
            print("Error: --dataset required for training mode")
            return
        train_models(args.dataset)
    
    elif args.mode == 'predict':
        if not args.url:
            print("Error: --url required for predict mode")
            return
        predict_url(args.url)
    
    elif args.mode == 'simulate':
        run_simulation(
            infection_rate=args.infection_rate,
            network_type=args.network_type,
            num_nodes=args.num_nodes,
            max_steps=args.max_steps
        )
    
    elif args.mode == 'parameter-study':
        run_parameter_study(
            network_type=args.network_type,
            num_nodes=args.num_nodes
        )
    
    elif args.mode == 'integrated':
        if not args.url:
            print("Error: --url required for integrated mode")
            return
        integrated_workflow(args.url, args.dataset)


if __name__ == "__main__":
    # If no arguments provided, show help
    if len(sys.argv) == 1:
        print("\n" + "="*60)
        print("PHISHING URL DETECTION & NETWORK SIMULATION SYSTEM")
        print("="*60)
        print("\nUsage examples:")
        print("\n1. Train CNN model:")
        print("   python main.py --mode train --dataset data/dataset.csv")
        print("\n2. Predict a URL:")
        print("   python main.py --mode predict --url http://example.com")
        print("\n3. Run simulation:")
        print("   python main.py --mode simulate --infection-rate 0.5 --network-type barabasi")
        print("\n4. Parameter study:")
        print("   python main.py --mode parameter-study --network-type barabasi")
        print("\n5. Integrated analysis:")
        print("   python main.py --mode integrated --url http://phishing-site.tk --dataset data/dataset.csv")
        print("\nFor more options, use: python main.py --help")
        print("="*60 + "\n")
    else:
        main()
