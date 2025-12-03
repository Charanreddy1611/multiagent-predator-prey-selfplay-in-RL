"""
Visualize training results for MAPPO or IPPO.
"""

import sys
import json
from analysis.metrics import MetricsTracker
from analysis.plots import plot_training_curves, plot_emergent_behaviors

def visualize_training(metrics_file: str, algorithm_name: str = ""):
    """
    Visualize training results from metrics file.
    
    Args:
        metrics_file: Path to metrics JSON file
        algorithm_name: Name to display (e.g., "MAPPO", "IPPO")
    """
    print(f"Loading metrics from {metrics_file}...")
    
    # Load metrics
    metrics = MetricsTracker()
    try:
        metrics.load(metrics_file)
    except FileNotFoundError:
        print(f"Error: Could not find {metrics_file}")
        return
    
    print(f"✓ Loaded {len(metrics.episode_data)} episodes")
    
    # Print summary
    metrics.print_summary()
    
    # Generate plots
    print(f"\nGenerating training curves for {algorithm_name}...")
    plot_training_curves(
        metrics.get_all_metrics(),
        save_path=f'{algorithm_name.lower()}_training_curves.png',
        window=100
    )
    
    print(f"\nGenerating behavior analysis for {algorithm_name}...")
    plot_emergent_behaviors(
        metrics.get_episode_data(),
        save_path=f'{algorithm_name.lower()}_emergent_behaviors.png'
    )
    
    print(f"\n✓ Plots saved!")
    print(f"  - {algorithm_name.lower()}_training_curves.png")
    print(f"  - {algorithm_name.lower()}_emergent_behaviors.png")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        metrics_file = sys.argv[1]
        algorithm_name = sys.argv[2] if len(sys.argv) > 2 else "Algorithm"
        visualize_training(metrics_file, algorithm_name)
    else:
        print("Usage: python visualize_training.py <metrics_file> [algorithm_name]")
        print("\nExamples:")
        print("  python visualize_training.py checkpoints/metrics_final.json MAPPO")
        print("  python visualize_training.py checkpoints/metrics_final.json IPPO")

