import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def visualize_trajectory(csv_file="trajectory.csv", save_path="trajectory_plot.png"):
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found!")
        return

    # Load the data
    df = pd.read_csv(csv_file)

    # Set up the plot
    plt.figure(figsize=(10, 8))

    # 1. Plot the continuous path (a blue line connecting the x, y points)
    plt.plot(df['x'], df['y'], color='blue', label='Path', alpha=0.5, linestyle='-')

    # 2. Plot the orientations (arrows showing the theta)
    u = np.cos(df['theta'])
    v = np.sin(df['theta'])

    # Quiver plots vectors (arrows)
    plt.quiver(df['x'], df['y'], u, v, color='red', angles='xy', scale_units='xy', scale=1.5, width=0.004, label='Heading (Theta)')

    # 3. Mark the Start and End points for clarity
    plt.scatter(df['x'].iloc[0], df['y'].iloc[0], color='green', marker='o', s=100, label='Start', zorder=5)
    plt.scatter(df['x'].iloc[-1], df['y'].iloc[-1], color='purple', marker='X', s=100, label='End', zorder=5)

    # Formatting
    plt.title("2D Optimized Robot Trajectory")
    plt.xlabel("X (meters)")
    plt.ylabel("Y (meters)")
    plt.axis("equal") 
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    # Save the plot BEFORE showing it
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot successfully saved to {save_path}")

    # Show the plot
    plt.show()


if __name__ == "__main__":
    visualize_trajectory("results/initial_traj.csv", "results/initial_traj.png")
    visualize_trajectory("results/optimized_traj.csv", "results/optimized_traj.png")
