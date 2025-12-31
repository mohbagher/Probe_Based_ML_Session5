"""
Diversity analysis for probe banks.

Computes diversity metrics:
- Cosine similarity (for continuous probes)
- Hamming distance (for binary probes)
- Statistical summaries (mean, std, min, max)
"""

import numpy as np
from typing import Dict
from .probe_generators import ProbeBank


def compute_cosine_similarity_matrix(probe_bank: ProbeBank) -> np.ndarray:
    """
    Compute pairwise cosine similarity matrix for probe phases.
    
    For continuous probes, we treat phases as vectors and compute
    cosine similarity between them.
    
    Args:
        probe_bank: ProbeBank object
        
    Returns:
        Similarity matrix of shape (K, K)
    """
    K = probe_bank.K
    phases = probe_bank.phases
    
    # Compute dot products
    similarity_matrix = np.zeros((K, K))
    
    for i in range(K):
        for j in range(K):
            # Cosine similarity based on phase vectors
            # We use the complex representation for better similarity measure
            v1 = np.exp(1j * phases[i])
            v2 = np.exp(1j * phases[j])
            
            # Inner product of complex vectors
            inner_prod = np.abs(np.dot(v1.conj(), v2))
            
            # Normalize by magnitudes (both are N)
            similarity = inner_prod / probe_bank.N
            similarity_matrix[i, j] = similarity
    
    return similarity_matrix


def compute_hamming_distance_matrix(probe_bank: ProbeBank) -> np.ndarray:
    """
    Compute pairwise Hamming distance matrix for binary/2-bit probes.
    
    Hamming distance counts the number of positions where probes differ.
    For binary probes, we count phase differences.
    
    Args:
        probe_bank: ProbeBank object
        
    Returns:
        Distance matrix of shape (K, K), normalized to [0, 1]
    """
    K = probe_bank.K
    N = probe_bank.N
    phases = probe_bank.phases
    
    distance_matrix = np.zeros((K, K))
    
    # Tolerance for comparing floating point phases
    tol = 1e-6
    
    for i in range(K):
        for j in range(K):
            # Count positions where phases differ
            diff = np.abs(phases[i] - phases[j])
            # Account for wrap-around (2π = 0)
            diff = np.minimum(diff, 2*np.pi - diff)
            # Count as different if difference > tolerance
            num_different = np.sum(diff > tol)
            # Normalize by N
            distance_matrix[i, j] = num_different / N
    
    return distance_matrix


def compute_diversity_metrics(probe_bank: ProbeBank) -> Dict[str, float]:
    """
    Compute diversity metrics for a probe bank.
    
    Returns mean, std, min, max of pairwise diversity measures.
    Uses cosine similarity for continuous probes, Hamming distance for others.
    
    Args:
        probe_bank: ProbeBank object
        
    Returns:
        Dictionary with diversity statistics
    """
    if probe_bank.probe_type == "continuous":
        # Use cosine similarity
        similarity_matrix = compute_cosine_similarity_matrix(probe_bank)
        
        # Extract upper triangle (excluding diagonal)
        K = probe_bank.K
        mask = np.triu(np.ones((K, K), dtype=bool), k=1)
        pairwise_similarities = similarity_matrix[mask]
        
        return {
            'metric_type': 'cosine_similarity',
            'mean': float(np.mean(pairwise_similarities)),
            'std': float(np.std(pairwise_similarities)),
            'min': float(np.min(pairwise_similarities)),
            'max': float(np.max(pairwise_similarities)),
            'median': float(np.median(pairwise_similarities))
        }
    else:
        # Use Hamming distance for binary/2bit/hadamard
        distance_matrix = compute_hamming_distance_matrix(probe_bank)
        
        # Extract upper triangle (excluding diagonal)
        K = probe_bank.K
        mask = np.triu(np.ones((K, K), dtype=bool), k=1)
        pairwise_distances = distance_matrix[mask]
        
        return {
            'metric_type': 'hamming_distance',
            'mean': float(np.mean(pairwise_distances)),
            'std': float(np.std(pairwise_distances)),
            'min': float(np.min(pairwise_distances)),
            'max': float(np.max(pairwise_distances)),
            'median': float(np.median(pairwise_distances))
        }
