"""
Adaptive Ensemble Evolutionary Algorithm (AEEA) for GNBG Benchmark
Author: [Tran Qui Don]
Last Edited: [16/05/2025]

This implementation combines multiple state-of-the-art evolutionary computation techniques
with adaptive mechanisms for robust performance across the GNBG benchmark suite.
"""

import numpy as np
from typing import Tuple, List, Optional, Dict
import logging
from dataclasses import dataclass
from scipy.stats import rankdata
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@dataclass
class AEEAParams:
    """Configuration parameters for the Adaptive Ensemble EA"""
    population_size: int = 100
    archive_size: int = 50
    max_generations: int = 5000
    crossover_rate: float = 0.9
    mutation_rate: float = 0.1
    f_lower: float = 0.1
    f_upper: float = 0.9
    cr_lower: float = 0.1
    cr_upper: float = 0.9
    memory_size: int = 5
    restart_threshold: float = 1e-6
    diversity_threshold: float = 0.1
    ensemble_size: int = 3

class AdaptiveEnsembleEA:
    def __init__(self, 
                 problem_dim: int,
                 bounds: Tuple[np.ndarray, np.ndarray],
                 params: Optional[AEEAParams] = None):
        """
        Initialize the Adaptive Ensemble EA
        
        Args:
            problem_dim: Problem dimension
            bounds: Tuple of (lower_bounds, upper_bounds)
            params: Optional configuration parameters
        """
        self.dim = problem_dim
        self.lb, self.ub = bounds
        self.params = params or AEEAParams()
        
        # Initialize memory for adaptive parameters
        self.f_memory = np.ones(self.params.memory_size) * 0.5
        self.cr_memory = np.ones(self.params.memory_size) * 0.5
        self.success_memory = np.zeros(self.params.memory_size)
        
        # Initialize population and archive
        self.population = self._initialize_population()
        self.archive = np.empty((self.params.archive_size, self.dim))
        self.archive_fitness = np.full(self.params.archive_size, np.inf)
        
        # Initialize fitness history
        self.fitness_history = []
        self.best_fitness = np.inf
        self.best_solution = None
        
        # Initialize ensemble weights
        self.ensemble_weights = np.ones(self.params.ensemble_size) / self.params.ensemble_size
        
    def _initialize_population(self) -> np.ndarray:
        """Initialize population using Latin Hypercube Sampling"""
        population = np.zeros((self.params.population_size, self.dim))
        for i in range(self.dim):
            population[:, i] = np.random.permutation(self.params.population_size)
        population = (population + 0.5) / self.params.population_size
        population = self.lb + population * (self.ub - self.lb)
        return population
    
    def _calculate_diversity(self, population: np.ndarray) -> float:
        """Calculate population diversity using average distance to centroid"""
        centroid = np.mean(population, axis=0)
        distances = np.sqrt(np.sum((population - centroid) ** 2, axis=1))
        return np.mean(distances)
    
    def _adaptive_parameters(self, gen: int) -> Tuple[float, float]:
        """Generate adaptive F and CR parameters"""
        if gen < self.params.memory_size:
            f = np.random.uniform(self.params.f_lower, self.params.f_upper)
            cr = np.random.uniform(self.params.cr_lower, self.params.cr_upper)
        else:
            # Use success-based adaptation
            success_rate = np.mean(self.success_memory)
            if success_rate > 0.5:
                f = np.mean(self.f_memory) + np.random.normal(0, 0.1)
                cr = np.mean(self.cr_memory) + np.random.normal(0, 0.1)
            else:
                f = np.random.uniform(self.params.f_lower, self.params.f_upper)
                cr = np.random.uniform(self.params.cr_lower, self.params.cr_upper)
            
            f = np.clip(f, self.params.f_lower, self.params.f_upper)
            cr = np.clip(cr, self.params.cr_lower, self.params.cr_upper)
        
        return f, cr
    
    def _ensemble_variation(self, 
                          target_idx: int, 
                          f: float, 
                          cr: float) -> np.ndarray:
        """Generate offspring using ensemble of variation operators"""
        target = self.population[target_idx]
        
        # Select parents
        candidates = np.arange(self.params.population_size)
        candidates = candidates[candidates != target_idx]
        parents = np.random.choice(candidates, size=3, replace=False)
        
        # Generate trial vectors using different operators
        trials = []
        
        # DE/rand/1
        trial1 = target + f * (self.population[parents[0]] - self.population[parents[1]])
        trials.append(trial1)
        
        # DE/best/1
        best_idx = np.argmin(self.fitness_history[-self.params.population_size:])
        trial2 = self.population[best_idx] + f * (self.population[parents[0]] - self.population[parents[1]])
        trials.append(trial2)
        
        # DE/current-to-best/1
        trial3 = target + f * (self.population[best_idx] - target) + f * (self.population[parents[0]] - self.population[parents[1]])
        trials.append(trial3)
        
        # Combine trials using ensemble weights
        trial = np.zeros_like(target)
        for i, t in enumerate(trials):
            trial += self.ensemble_weights[i] * t
        
        # Apply binomial crossover
        mask = np.random.random(self.dim) < cr
        if not np.any(mask):
            mask[np.random.randint(self.dim)] = True
        trial = np.where(mask, trial, target)
        
        # Ensure bounds
        trial = np.clip(trial, self.lb, self.ub)
        
        return trial
    
    def _update_archive(self, solution: np.ndarray, fitness: float):
        """Update archive with new solution"""
        if fitness < np.max(self.archive_fitness):
            worst_idx = np.argmax(self.archive_fitness)
            self.archive[worst_idx] = solution
            self.archive_fitness[worst_idx] = fitness
    
    def _update_ensemble_weights(self, 
                               trials: List[np.ndarray], 
                               trial_fitness: List[float]):
        """Update ensemble weights based on performance"""
        if len(trials) == 0:
            return
        
        # Calculate improvement ratios
        improvements = []
        for i, fitness in enumerate(trial_fitness):
            if fitness < self.best_fitness:
                improvements.append((self.best_fitness - fitness) / self.best_fitness)
            else:
                improvements.append(0)
        
        # Update weights using softmax
        if np.sum(improvements) > 0:
            exp_improvements = np.exp(improvements)
            self.ensemble_weights = exp_improvements / np.sum(exp_improvements)
    
    def optimize(self, 
                fitness_func, 
                max_evals: int,
                acceptance_threshold: float = 1e-8) -> Tuple[np.ndarray, float, Dict]:
        """
        Run the optimization process
        
        Args:
            fitness_func: Objective function to minimize
            max_evals: Maximum number of function evaluations
            acceptance_threshold: Acceptance threshold for early stopping
            
        Returns:
            Tuple of (best_solution, best_fitness, optimization_info)
        """
        # Initialize fitness values by evaluating each individual in the population
        fitness_values = []
        for ind in self.population:
            # GNBG expects input as 2D array
            ind_reshaped = ind.reshape(1, -1)
            result = fitness_func(ind_reshaped)
            # Extract the scalar value
            fitness = result[0] if isinstance(result, np.ndarray) else result
            fitness_values.append(fitness)
        
        fitness_values = np.array(fitness_values)
        self.fitness_history.extend(fitness_values)
        
        # Find initial best
        best_idx = np.argmin(fitness_values)
        self.best_fitness = fitness_values[best_idx]
        self.best_solution = self.population[best_idx].copy()
        
        # Main optimization loop
        eval_count = self.params.population_size
        gen = 0
        
        while eval_count < max_evals:
            # Check for acceptance threshold
            if abs(self.best_fitness) < acceptance_threshold:
                break
            
            # Generate offspring
            new_population = []
            new_fitness = []
            success_count = 0
            
            for i in range(self.params.population_size):
                # Get adaptive parameters
                f, cr = self._adaptive_parameters(gen)
                
                # Generate trial vector
                trial = self._ensemble_variation(i, f, cr)
                
                # GNBG expects input as 2D array
                trial_reshaped = trial.reshape(1, -1)
                result = fitness_func(trial_reshaped)
                # Extract the scalar value
                trial_fitness = result[0] if isinstance(result, np.ndarray) else result
                
                eval_count += 1
                
                # Selection
                if trial_fitness < fitness_values[i]:
                    new_population.append(trial)
                    new_fitness.append(trial_fitness)
                    success_count += 1
                    
                    # Update archive
                    self._update_archive(trial, trial_fitness)
                    
                    # Update best solution
                    if trial_fitness < self.best_fitness:
                        self.best_fitness = trial_fitness
                        self.best_solution = trial.copy()
                else:
                    new_population.append(self.population[i])
                    new_fitness.append(fitness_values[i])
            
            # Update population
            self.population = np.array(new_population)
            fitness_values = np.array(new_fitness)
            self.fitness_history.extend(fitness_values)
            
            # Update success memory
            success_rate = success_count / self.params.population_size
            self.success_memory[gen % self.params.memory_size] = success_rate
            
            # Check for restart
            diversity = self._calculate_diversity(self.population)
            if diversity < self.params.diversity_threshold:
                logging.info(f"Restarting population at generation {gen} due to low diversity")
                self.population = self._initialize_population()
                
                # Re-evaluate new population
                fitness_values = []
                for ind in self.population:
                    # GNBG expects input as 2D array
                    ind_reshaped = ind.reshape(1, -1)
                    result = fitness_func(ind_reshaped)
                    # Extract the scalar value
                    fitness = result[0] if isinstance(result, np.ndarray) else result
                    fitness_values.append(fitness)
                
                fitness_values = np.array(fitness_values)
                eval_count += self.params.population_size
            
            gen += 1
            
            # Log progress
            if gen % 100 == 0:
                logging.info(f"Generation {gen}, Best fitness: {self.best_fitness:.6e}")
        
        # Prepare optimization info
        opt_info = {
            'generations': gen,
            'function_evaluations': eval_count,
            'final_diversity': self._calculate_diversity(self.population),
            'success_rate': np.mean(self.success_memory),
            'ensemble_weights': self.ensemble_weights
        }
        
        return self.best_solution, self.best_fitness, opt_info

def run_optimization(problem_index: int, 
                    num_runs: int = 31,
                    seed: Optional[int] = None) -> Tuple[List[float], List[np.ndarray]]:
    """
    Run multiple optimization runs for a given problem
    
    Args:
        problem_index: GNBG problem index (1-24)
        num_runs: Number of independent runs
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (best_fitness_values, best_solutions)
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Load problem instance
    from GNBG_instances import GNBG
    gnbg = GNBG(problem_index)
    
    # Set up bounds
    bounds = (gnbg.MinCoordinate, gnbg.MaxCoordinate)
    
    # Initialize results storage
    best_values = []
    best_solutions = []
    
    for run in range(num_runs):
        logging.info(f"Starting run {run + 1}/{num_runs}")
        
        # Initialize algorithm
        params = AEEAParams(
            population_size=min(100, 10 * gnbg.Dimension),
            max_generations=gnbg.MaxEvals // min(100, 10 * gnbg.Dimension)
        )
        
        optimizer = AdaptiveEnsembleEA(
            problem_dim=gnbg.Dimension,
            bounds=bounds,
            params=params
        )
        
        # Run optimization
        best_solution, best_fitness, _ = optimizer.optimize(
            fitness_func=gnbg.fitness,
            max_evals=gnbg.MaxEvals,
            acceptance_threshold=gnbg.AcceptanceThreshold
        )
        
        best_values.append(best_fitness)
        best_solutions.append(best_solution)
        
        logging.info(f"Run {run + 1} completed. Best fitness: {best_fitness:.6e}")
    
    return best_values, best_solutions

if __name__ == "__main__":
    # Example usage
    problem_index = 1
    best_values, best_solutions = run_optimization(problem_index)
    
    print(f"\nResults for Problem {problem_index}:")
    print(f"Best fitness values: {best_values}")
    print(f"Mean fitness: {np.mean(best_values):.6e}")
    print(f"Std fitness: {np.std(best_values):.6e}") 