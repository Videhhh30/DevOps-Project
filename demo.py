"""
Demo Script
Demonstrates all features of the Phishing URL Detection and Network Simulation System
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from feature_extraction import URLFeatureExtractor
from model_training import ModelTrainer
from prediction import PredictionService
from network_simulation import PhishingSpreadSimulator, run_parameter_comparison
from visualization import Visualizer
from dataset_handler import generate_synthetic_dataset


def print_section(title):
    """Print a formatted section header"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70 + "\n")


def demo_feature_extraction():
    """Demonstrate URL feature extraction"""
    print_section("DEMO 1: URL Feature Extraction")
    
    extractor = URLFeatureExtractor()
    
    test_urls = [
        "https://www.google.com",
        "http://paypal-secure-login.tk",
        "http://192.168.1.1/admin",
        "http://amaz0n-account@verify.com"
    ]
    
    print("Extracting features from sample URLs:\n")
    for url in test_urls:
        print(f"URL: {url}")
        features = extractor.extract_features(url)
        # Show key features
        print(f"  Length: {features['url_length']}, "
              f"Dots: {features['num_dots']}, "
              f"HTTPS: {features['is_https']}, "
              f"Suspicious TLD: {features['suspicious_tld']}, "
              f"Has IP: {features['has_ip']}")
        print()
    
    input("Press Enter to continue...")


def demo_model_training():
    """Demonstrate model training"""
    print_section("DEMO 2: Model Training and Evaluation")
    
    # Generate synthetic dataset
    print("Generating synthetic dataset for demonstration...")
    generate_synthetic_dataset('data/demo_dataset.csv', num_samples=500)
    
    # Train models
    print("\nTraining machine learning models...")
    extractor = URLFeatureExtractor()
    trainer = ModelTrainer(extractor)
    
    results = trainer.train_and_evaluate('data/demo_dataset.csv')
    
    # Save models
    trainer.save_model(results['best_model'], 'models/demo_model.pkl')
    trainer.save_scaler('models/demo_scaler.pkl')
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    Visualizer.plot_confusion_matrix(
        results['metrics']['y_test'],
        results['metrics']['y_pred'],
        'outputs/demo_confusion_matrix.png'
    )
    
    if hasattr(results['best_model'], 'feature_importances_'):
        Visualizer.plot_feature_importance(
            results['best_model'],
            results['feature_names'],
            'outputs/demo_feature_importance.png'
        )
    
    print("\n✓ Models trained and saved successfully!")
    print("✓ Visualizations saved to outputs/ directory")
    
    input("\nPress Enter to continue...")


def demo_prediction():
    """Demonstrate URL prediction"""
    print_section("DEMO 3: URL Prediction")
    
    predictor = PredictionService('models/demo_model.pkl', 'models/demo_scaler.pkl')
    
    test_urls = [
        "https://www.google.com",
        "http://paypal-secure-login.tk",
        "http://192.168.1.1/admin",
        "http://amaz0n-account@verify.com",
        "https://www.github.com",
        "http://bit.ly/suspicious123"
    ]
    
    print("Predicting URLs:\n")
    for url in test_urls:
        result = predictor.predict(url)
        status = "⚠️  PHISHING" if result['prediction'] == 'phishing' else "✓ LEGITIMATE"
        print(f"{status:15} | {result['confidence']:6.1%} | {url}")
    
    input("\nPress Enter to continue...")


def demo_network_simulation():
    """Demonstrate network simulation"""
    print_section("DEMO 4: Social Network Phishing Spread Simulation")
    
    print("Simulating phishing spread through a social network...\n")
    
    # Run simulation
    simulator = PhishingSpreadSimulator(
        num_nodes=150,
        network_type='barabasi',
        infection_rate=0.4
    )
    
    results = simulator.run_simulation(max_steps=50)
    
    # Identify critical nodes
    critical_nodes = simulator.identify_critical_nodes(top_n=5)
    
    print(f"\nCritical Nodes (high influence):")
    for node_id, centrality in critical_nodes:
        print(f"  Node {node_id}: centrality = {centrality:.4f}")
    
    # Generate visualizations
    print("\nGenerating network visualizations...")
    Visualizer.plot_network_graph(
        simulator.graph,
        simulator.infected_nodes,
        critical_nodes,
        'outputs/demo_network_graph.png'
    )
    
    Visualizer.plot_spread_timeline(
        results['timeline'],
        0.4,
        'outputs/demo_spread_timeline.png'
    )
    
    print("\n✓ Network simulation completed!")
    print("✓ Visualizations saved to outputs/ directory")
    
    input("\nPress Enter to continue...")


def demo_parameter_comparison():
    """Demonstrate parameter variation study"""
    print_section("DEMO 5: Parameter Variation Study")
    
    print("Running simulations with different infection rates...\n")
    
    # Run parameter comparison
    results = run_parameter_comparison(
        num_nodes=150,
        network_type='barabasi',
        infection_rates=[0.1, 0.3, 0.5, 0.7, 0.9],
        max_steps=50
    )
    
    # Generate comparison visualization
    Visualizer.plot_parameter_comparison(
        results,
        'outputs/demo_parameter_comparison.png'
    )
    
    print("\n✓ Parameter study completed!")
    print("✓ Comparison plot saved to outputs/ directory")
    
    input("\nPress Enter to continue...")


def demo_integrated_workflow():
    """Demonstrate integrated ML + simulation workflow"""
    print_section("DEMO 6: Integrated Workflow (ML + Simulation)")
    
    print("Analyzing a phishing URL and simulating its spread...\n")
    
    # Predict URL
    test_url = "http://paypal-secure-login.tk"
    print(f"Analyzing URL: {test_url}\n")
    
    predictor = PredictionService('models/demo_model.pkl', 'models/demo_scaler.pkl')
    result = predictor.predict(test_url)
    
    print(f"Prediction: {result['prediction'].upper()}")
    print(f"Confidence: {result['confidence']:.2%}")
    
    if result['prediction'] == 'phishing':
        print(f"\n⚠️  Phishing detected! Simulating spread through network...")
        print(f"Using ML confidence to adjust infection rate...\n")
        
        # Adjust infection rate based on confidence
        base_rate = 0.3
        adjusted_rate = min(base_rate * (1 + result['probability_phishing']), 1.0)
        
        print(f"Base infection rate: {base_rate}")
        print(f"ML-adjusted rate: {adjusted_rate:.2f}\n")
        
        # Run simulation
        simulator = PhishingSpreadSimulator(
            num_nodes=150,
            network_type='barabasi',
            infection_rate=adjusted_rate
        )
        
        sim_results = simulator.run_simulation(max_steps=50)
        
        print(f"\nSimulation Results:")
        print(f"  Final infected: {sim_results['final_infected_count']} nodes "
              f"({sim_results['infection_percentage']:.1f}%)")
        print(f"  Convergence: {sim_results['steps_to_stabilize']} steps")
        
        print("\n✓ Integrated analysis completed!")
    
    input("\nPress Enter to continue...")


def main():
    """Run the complete demo"""
    print("\n" + "="*70)
    print("  PHISHING URL DETECTION & NETWORK SIMULATION SYSTEM")
    print("  Complete Feature Demonstration")
    print("="*70)
    
    print("\nThis demo will showcase all features of the system:")
    print("  1. URL Feature Extraction")
    print("  2. Model Training and Evaluation")
    print("  3. URL Prediction")
    print("  4. Network Simulation")
    print("  5. Parameter Variation Study")
    print("  6. Integrated ML + Simulation Workflow")
    
    print("\nAll visualizations will be saved to the 'outputs/' directory.")
    
    input("\nPress Enter to start the demo...")
    
    try:
        # Run all demos
        demo_feature_extraction()
        demo_model_training()
        demo_prediction()
        demo_network_simulation()
        demo_parameter_comparison()
        demo_integrated_workflow()
        
        # Final summary
        print_section("DEMO COMPLETED!")
        
        print("All features demonstrated successfully!\n")
        print("Generated files:")
        print("  Models:")
        print("    - models/demo_model.pkl")
        print("    - models/demo_scaler.pkl")
        print("\n  Visualizations:")
        print("    - outputs/demo_confusion_matrix.png")
        print("    - outputs/demo_feature_importance.png")
        print("    - outputs/demo_network_graph.png")
        print("    - outputs/demo_spread_timeline.png")
        print("    - outputs/demo_parameter_comparison.png")
        print("\n  Dataset:")
        print("    - data/demo_dataset.csv")
        
        print("\n" + "="*70)
        print("Thank you for trying the Phishing Detection System!")
        print("="*70 + "\n")
        
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        print(f"\n\nError during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
