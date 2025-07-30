from itertools import combinations_with_replacement
from re import match
import numpy as np
import pandas as pd
import random
from typing import Dict, List

types = ['Normal', 'Fire', 'Water', 'Electric', 'Grass', 'Ice', 'Fighting', 'Poison',
         'Ground', 'Flying', 'Psychic', 'Bug', 'Rock', 'Ghost', 'Dragon', 'Dark', 'Steel', 'Fairy']

type_chart = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.5, 0, 1, 1, 0.5, 1],  # Normal
    [1, 0.5, 0.5, 1, 2, 2, 1, 1, 1, 1, 1, 2, 0.5, 1, 0.5, 1, 2, 1],  # Fire
    [1, 2, 0.5, 1, 0.5, 1, 1, 1, 2, 1, 1, 1, 2, 1, 0.5, 1, 1, 1],  # Water
    [1, 1, 2, 0.5, 0.5, 1, 1, 1, 0, 2, 1, 1, 1, 1, 0.5, 1, 1, 1],  # Electric
    [1, 0.5, 2, 1, 0.5, 1, 1, 0.5, 2, 0.5, 1, 0.5, 2, 1, 0.5, 1, 0.5, 1],  # Grass
    [1, 0.5, 0.5, 1, 2, 0.5, 1, 1, 2, 2, 1, 1, 1, 1, 2, 1, 0.5, 1],  # Ice
    [2, 1, 1, 1, 1, 2, 1, 0.5, 1, 0.5, 0.5, 0.5, 2, 0, 1, 2, 2, 0.5],  # Fighting
    [1, 1, 1, 1, 2, 1, 1, 0.5, 0.5, 1, 1, 1, 0.5, 0.5, 1, 1, 0, 2],  # Poison
    [1, 2, 1, 2, 0.5, 1, 1, 2, 1, 0, 1, 0.5, 2, 1, 1, 1, 2, 1],  # Ground
    [1, 1, 1, 0.5, 2, 1, 2, 1, 1, 1, 1, 2, 0.5, 1, 1, 1, 0.5, 1],  # Flying
    [1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 0.5, 1, 1, 1, 1, 0, 0.5, 1],  # Psychic
    [1, 0.5, 1, 1, 2, 1, 0.5, 0.5, 1, 0.5, 2, 1, 1, 0.5, 1, 2, 0.5, 0.5],  # Bug
    [1, 2, 1, 1, 1, 2, 0.5, 1, 0.5, 2, 1, 2, 1, 1, 1, 1, 0.5, 1],  # Rock
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 0.5, 1, 1],  # Ghost
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 0.5, 0],  # Dragon
    [1, 1, 1, 1, 1, 1, 0.5, 1, 1, 1, 2, 1, 1, 2, 1, 0.5, 1, 0.5],  # Dark
    [1, 0.5, 0.5, 0.5, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 0.5, 2],  # Steel
    [1, 0.5, 1, 1, 1, 1, 2, 0.5, 1, 1, 1, 1, 1, 1, 2, 2, 0.5, 1]  # Fairy
]
A_single = np.array(type_chart)

dual_types = []
for type1, type2 in combinations_with_replacement(types, 2):
    dual_types.append(f"{type1}/{type2}" if type1 != type2 else type1)

def calculate_max_effectiveness(type1: str, type2: str) -> float:
    
    type_effectiveness = []
    for a in type1:
        effectiveness = 1.0
        for d in type2:
            effectiveness *= A_single[types.index(a)][types.index(d)]
        type_effectiveness.append(effectiveness)
    
    return max(type_effectiveness)

A_dual = np.zeros((171, 171))
for i, attacker in enumerate(dual_types):
    a_types = attacker.split('/')
    for j, defender in enumerate(dual_types):
        d_types = defender.split('/')   
        A_dual[i][j] = calculate_max_effectiveness(a_types, d_types)

def calculate_matchup_score(type1: str, type2: str) -> float:
    """Calculate matchup score between two type combinations"""
    a_types = type1.split('/')
    b_types = type2.split('/')

    a_max = calculate_max_effectiveness(a_types, b_types)
    b_max = calculate_max_effectiveness(b_types, a_types)

    # Handle special cases to prevent divide by zero
    if a_max == 0 and b_max == 0:
        return 1  # Both completely immune to each other
    elif a_max == 0:
        a_max = 0.25
    elif b_max == 0:
        b_max = 0.25

    matchup = a_max / b_max
    return matchup

    # Only return positive/negative advantage
    if matchup < 1:
        return .5
    elif matchup > 1:
        return 2
    else:
        return 1
    
A_dual_matchup = np.zeros((171, 171))
for i, attacker in enumerate(dual_types):
    for j, defender in enumerate(dual_types):
        matchup_score = calculate_matchup_score(attacker, defender)
        A_dual_matchup[i][j] = np.log2(matchup_score)

def calculate_eigenvector(A, offense, type_names) -> float:
    if not offense:
        A = A.T  # Transpose for defensive perspective
        
    eigenvalues, eigenvectors = np.linalg.eig(A)
    dominant_idx = np.argmax(eigenvalues)
    dominant_eigenvector = eigenvectors[:, dominant_idx]
    
    # Normalize and scale
    dominant_eigenvector = dominant_eigenvector / np.sum(dominant_eigenvector) * A.shape[0]
    
    if not offense:
        dominant_eigenvector = 2 - dominant_eigenvector  # Invert scale for defense
    
    return dominant_eigenvector

def print_top(n, A, type_names, offense=True, reversed=False) -> None:
    """
    Print top n type combinations based on eigenvector analysis
    
    Args:
        A: The matchup matrix
        type_names: List of type names
        offense: If True, prints offensive rankings; if False, defensive
    """
    vec = calculate_eigenvector(A, offense, type_names)
    
    # Pair type names with their scores
    ranked = sorted(zip(type_names, vec), key=lambda x: x[1], reverse=not reversed)

    print("\nBottom" if reversed else "Top", n, "Offensive" if offense else "Defensive", "Type Combinations:")
    for i, (type_combo, score) in enumerate(ranked[:n], 1):
        print(f"{i}. {type_combo:20} {score:.4f}")

def calculate_overall_scores(A, type_names, offense_weight=0.5, defense_weight=0.5) -> Dict[str, float]:
    offensive_scores = calculate_eigenvector(A, True, type_names)
    defensive_scores = calculate_eigenvector(A, False, type_names)
    
    # Combine scores with weights
    overall_scores = {}
    for i, type_combo in enumerate(type_names):
        overall = (offense_weight * offensive_scores[i] + 
                   defense_weight * defensive_scores[i])
        overall_scores[type_combo] = overall
        
    return overall_scores

def print_top_overall(n, A, type_names, offense_weight=0.5, defense_weight=0.5) -> None:
    scores = calculate_overall_scores(A, type_names, offense_weight, defense_weight)
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\nTop {n} Overall Type Combinations (Offense {offense_weight}:Defense {defense_weight}):")
    for i, (type_combo, score) in enumerate(ranked[:n], 1):
        print(f"{i}. {type_combo:20} {score:.4f}")

def print_random_type(n) -> None:
    for _ in range(n):
        print(random.choice(dual_types))

def main():
    # print_top(18, A_single, types, True, False)
    # print_top(18, A_single, types, False, False)
    # print_top_overall(18, A_single, types, offense_weight=0.5, defense_weight=0.5)

    # print_top(171, A_dual, dual_types, True, False)
    # print_top(171, A_dual, dual_types, False, False)
    # print_top_overall(171, A_dual, dual_types, offense_weight=0.5, defense_weight=0.5)

    # net_advantage = A_dual_matchup.sum(axis=1)
    # matchup_ranking = sorted(zip(dual_types, net_advantage), key=lambda x: x[1], reverse=True)
    # for i, (type_combo, score) in enumerate(matchup_ranking[:171], 1):
    #     print(f"{i}. {type_combo:20} {score:.4f}")
    # print("end")

    print_random_type(10)
    
if __name__ == "__main__":
    main()
