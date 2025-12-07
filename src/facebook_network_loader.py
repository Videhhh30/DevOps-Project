"""
Facebook Network Loader
Loads real Facebook social network data for simulation
"""

import networkx as nx


def load_facebook_network(filepath: str = 'facebook_combined.txt') -> nx.Graph:
    """
    Load the Facebook social network dataset
    
    Args:
        filepath: Path to facebook_combined.txt file
        
    Returns:
        NetworkX graph with Facebook network structure
    """
    print(f"Loading Facebook social network from {filepath}...")
    
    # Load edge list
    G = nx.read_edgelist(filepath, nodetype=int)
    
    print(f"Facebook network loaded:")
    print(f"  Nodes (users): {G.number_of_nodes()}")
    print(f"  Edges (friendships): {G.number_of_edges()}")
    print(f"  Average degree: {sum(dict(G.degree()).values()) / G.number_of_nodes():.2f}")
    
    # Check if connected
    if nx.is_connected(G):
        print(f"  Network is connected")
    else:
        components = list(nx.connected_components(G))
        print(f"  Network has {len(components)} components")
        print(f"  Largest component: {len(max(components, key=len))} nodes")
    
    return G


def get_network_statistics(G: nx.Graph) -> dict:
    """
    Get detailed statistics about the network
    
    Args:
        G: NetworkX graph
        
    Returns:
        Dictionary with network statistics
    """
    stats = {
        'num_nodes': G.number_of_nodes(),
        'num_edges': G.number_of_edges(),
        'avg_degree': sum(dict(G.degree()).values()) / G.number_of_nodes(),
        'density': nx.density(G),
        'is_connected': nx.is_connected(G)
    }
    
    # Clustering coefficient (can be slow for large networks)
    try:
        stats['avg_clustering'] = nx.average_clustering(G)
    except:
        stats['avg_clustering'] = None
    
    return stats


if __name__ == "__main__":
    # Test loading the Facebook network
    print("Testing Facebook Network Loader\n")
    print("="*60)
    
    G = load_facebook_network('facebook_combined.txt')
    
    print("\n" + "="*60)
    print("Network Statistics:")
    print("="*60)
    
    stats = get_network_statistics(G)
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    # Find most connected users
    print("\n" + "="*60)
    print("Top 10 Most Connected Users:")
    print("="*60)
    
    degrees = dict(G.degree())
    top_users = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:10]
    
    for user_id, degree in top_users:
        print(f"  User {user_id}: {degree} connections")
    
    print("\n" + "="*60)
    print("Facebook network loaded successfully!")
