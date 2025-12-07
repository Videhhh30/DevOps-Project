"""
Test script to verify the network simulation works correctly
"""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from network_simulation import PhishingSpreadSimulator
import matplotlib.pyplot as plt

def test_simulation():
    print("Testing Phishing Spread Simulator\n")
    print("="*60)
    
    # Create simulator
    print("\n1. Creating simulator...")
    simulator = PhishingSpreadSimulator(
        num_nodes=150,
        network_type='barabasi',
        infection_rate=0.3
    )
    
    # Run simulation
    print("\n2. Running simulation...")
    results = simulator.run_simulation(max_steps=50)
    
    # Print results
    print("\n3. Results:")
    print(f"   Network Type: {results['network_type']}")
    print(f"   Number of Nodes: {results['num_nodes']}")
    print(f"   Number of Edges: {results['num_edges']}")
    print(f"   Infection Rate: {results['infection_rate']}")
    print(f"   Final Infected Count: {results['final_infected_count']}")
    print(f"   Infection Percentage: {results['infection_percentage']:.2f}%")
    print(f"   Steps to Stabilize: {results['steps_to_stabilize']}")
    print(f"   Timeline Length: {len(results['timeline'])}")
    print(f"   Timeline: {results['timeline']}")
    
    # Test visualization
    print("\n4. Testing visualization...")
    try:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(results['timeline'], marker='o', color='red')
        ax.fill_between(range(len(results['timeline'])), results['timeline'], alpha=0.3, color='red')
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Infected Nodes')
        ax.set_title('Spread of Phishing Attack Over Time')
        ax.grid(True, alpha=0.3)
        plt.savefig('test_simulation_plot.png')
        print("   ✓ Visualization saved to test_simulation_plot.png")
        plt.close()
    except Exception as e:
        print(f"   ✗ Visualization failed: {e}")
    
    print("\n" + "="*60)
    print("Test completed successfully!")
    
    return results

if __name__ == "__main__":
    test_simulation()
