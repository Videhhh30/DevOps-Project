"""
Test infection rate 0.06 multiple times to see variation
"""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from network_simulation import PhishingSpreadSimulator

def test_rate_multiple_times():
    print("Testing infection rate 0.06 multiple times")
    print("="*60)
    
    rate = 0.06
    results_list = []
    
    for i in range(5):
        print(f"\n--- Run {i+1} ---")
        simulator = PhishingSpreadSimulator(
            num_nodes=150,
            network_type='barabasi',
            infection_rate=rate
        )
        results = simulator.run_simulation(max_steps=50)
        
        results_list.append({
            'run': i+1,
            'infected': results['final_infected_count'],
            'percentage': results['infection_percentage'],
            'steps': results['steps_to_stabilize']
        })
        
        print(f"  Infected: {results['final_infected_count']} ({results['infection_percentage']:.1f}%)")
        print(f"  Steps: {results['steps_to_stabilize']}")
    
    print("\n" + "="*60)
    print("SUMMARY:")
    avg_infected = sum(r['infected'] for r in results_list) / len(results_list)
    avg_percentage = sum(r['percentage'] for r in results_list) / len(results_list)
    avg_steps = sum(r['steps'] for r in results_list) / len(results_list)
    
    print(f"Average Infected: {avg_infected:.1f} nodes ({avg_percentage:.1f}%)")
    print(f"Average Steps: {avg_steps:.1f}")

if __name__ == "__main__":
    test_rate_multiple_times()
