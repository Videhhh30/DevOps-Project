"""
Test different infection rates to verify realistic results
"""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from network_simulation import PhishingSpreadSimulator

def test_infection_rates():
    print("Testing Different Infection Rates")
    print("="*60)
    
    rates_to_test = [0.03, 0.05, 0.08, 0.10, 0.12, 0.15]
    
    for rate in rates_to_test:
        print(f"\n--- Testing infection rate: {rate} ({rate*100:.0f}%) ---")
        simulator = PhishingSpreadSimulator(
            num_nodes=150,
            network_type='barabasi',
            infection_rate=rate
        )
        results = simulator.run_simulation(max_steps=50)
        
        print(f"  Final Infected: {results['final_infected_count']} nodes")
        print(f"  Infection %: {results['infection_percentage']:.1f}%")
        print(f"  Steps: {results['steps_to_stabilize']}")
    
    print("\n" + "="*60)
    print("Recommendation: Use 0.05 (5%) for realistic 10-20% infection")

if __name__ == "__main__":
    test_infection_rates()
