import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

class PSOClusteringPure:
    """
    Pure Particle Swarm Optimization (PSO) for data clustering.
    Each particle represents a set of 'k' cluster centroids.
    """
    def __init__(self, n_clusters, n_particles, data, w=0.7, c1=2.0, c2=2.0, max_iter=100):
        self.n_clusters = n_clusters
        self.n_particles = n_particles
        self.data = data
        # FIX: The error occurs here. data.shape is a tuple (n_samples, n_features).
        # We need to take only the number of features, which is data.shape[1].
        self.n_features = data.shape[1]
        self.w = w  # Inertia weight
        self.c1 = c1  # Cognitive factor
        self.c2 = c2  # Social factor
        self.max_iter = max_iter

        # Initialize particles randomly across the data range
        min_vals = np.min(data, axis=0)
        max_vals = np.max(data, axis=0)
        self.particles = np.random.rand(n_particles, n_clusters, self.n_features) * (max_vals - min_vals) + min_vals
        self.velocities = np.zeros_like(self.particles)
        
        # Initialize personal best positions and scores
        self.pbest_positions = self.particles.copy()
        self.pbest_scores = np.full(n_particles, np.inf)

        # Initialize global best position and score
        self.gbest_position = None
        self.gbest_score = np.inf

    def fitness_function(self, centroids):
        """
        Calculates the fitness (Sum of Squared Errors) of a particle.
        Lower SSE means a better clustering.
        """
        # Assign each data point to the nearest centroid
        distances = np.sqrt(((self.data - centroids[:, np.newaxis])**2).sum(axis=2))
        cluster_assignments = np.argmin(distances, axis=0)

        # Calculate the SSE for the current centroids
        sse = 0
        for i in range(self.n_clusters):
            cluster_points = self.data[cluster_assignments == i]
            if len(cluster_points) > 0:
                sse += np.sum((cluster_points - centroids[i])**2)
        
        return sse

    def optimize(self):
        """
        The main optimization loop for pure PSO clustering.
        """
        # Store the initial position of the first particle for visualization
        self.initial_centroids = self.particles[0].copy()

        for i in range(self.max_iter):
            for j in range(self.n_particles):
                # Calculate fitness for the current particle
                current_fitness = self.fitness_function(self.particles[j])

                # Update personal best if current position is better
                if current_fitness < self.pbest_scores[j]:
                    self.pbest_scores[j] = current_fitness
                    self.pbest_positions[j] = self.particles[j].copy()

                # Update global best if personal best is better than current global best
                if current_fitness < self.gbest_score:
                    self.gbest_score = current_fitness
                    self.gbest_position = self.particles[j].copy()

            # Update velocity and position for each particle
            for j in range(self.n_particles):
                r1, r2 = np.random.rand(2)
                
                # Update velocity based on inertia, cognitive, and social components
                cognitive_velocity = self.c1 * r1 * (self.pbest_positions[j] - self.particles[j])
                social_velocity = self.c2 * r2 * (self.gbest_position - self.particles[j])
                self.velocities[j] = self.w * self.velocities[j] + cognitive_velocity + social_velocity

                # Update position based on new velocity
                self.particles[j] += self.velocities[j]
        
        return self.gbest_position

    def get_labels(self, centroids):
        """
        Assigns data points to clusters based on a given set of centroids.
        """
        distances = np.sqrt(((self.data - centroids[:, np.newaxis])**2).sum(axis=2))
        return np.argmin(distances, axis=0)

# --- Main execution block ---
if __name__ == '__main__':
    # 1. Generate synthetic data
    X, _ = make_blobs(n_samples=500, centers=4, cluster_std=0.60, random_state=42)
    n_clusters = 4

    # 2. Run pure PSO to find the clusters
    print("Running pure PSO clustering...")
    pso = PSOClusteringPure(n_clusters=n_clusters, n_particles=50, data=X, max_iter=200)
    final_centroids = pso.optimize()
    final_labels = pso.get_labels(final_centroids)
    
    # Get initial labels using the stored initial centroids
    initial_labels = pso.get_labels(pso.initial_centroids)
    print("PSO clustering complete.")
    
    # 3. Visualize the results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot initial state
    ax1.scatter(X[:, 0], X[:, 1], c=initial_labels, cmap='viridis', s=50, alpha=0.7)
    ax1.scatter(pso.initial_centroids[:, 0], pso.initial_centroids[:, 1], marker='X', s=200, c='red', edgecolor='black', linewidth=2, label='Initial Centroids')
    ax1.set_title('Initial Random State')
    ax1.set_xlabel('Feature 1')
    ax1.set_ylabel('Feature 2')
    ax1.legend()

    # Plot final state
    ax2.scatter(X[:, 0], X[:, 1], c=final_labels, cmap='viridis', s=50, alpha=0.7)
    ax2.scatter(final_centroids[:, 0], final_centroids[:, 1], marker='X', s=200, c='red', edgecolor='black', linewidth=2, label='Final Centroids')
    ax2.set_title('Final Optimized State (Pure PSO)')
    ax2.set_xlabel('Feature 1')
    ax2.set_ylabel('Feature 2')
    ax2.legend()
    
    plt.suptitle('Comparison of Initial vs. Final PSO Clustering', fontsize=16)
    plt.tight_layout()
    plt.show()
