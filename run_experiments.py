#!/usr/bin/env python3
"""
Example script showing how to run different experiments with the enhanced Main.py
"""

import subprocess
import sys
import os

def run_command(description, command):
    """Run a command and print its description"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {command}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        print("STDOUT:")
        print(result.stdout)
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        return result.returncode == 0
    except Exception as e:
        print(f"Error running command: {e}")
        return False

def main():
    """Run various experiment configurations"""
    
    # Change to the directory containing Main.py
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    print("Layer Specialization Analysis Experiments")
    print("=" * 50)
    
    # Experiment 1: Basic run with default parameters
    run_command(
        "Experiment 1: Default configuration (classes 0-3, correct predictions only)",
        "python Main.py"
    )
    
    # Experiment 2: Run with all classes and specialization analysis
    run_command(
        "Experiment 2: All classes with layer specialization analysis",
        "python Main.py --selected_classes 0 1 2 3 4 5 6 7 8 9 --run_specialization"
    )
    
    # Experiment 3: Filter by specific layer predictions
    run_command(
        "Experiment 3: Classes 0-7, filter by layer 1 predictions of class 0",
        "python Main.py --selected_classes 0 1 2 3 4 5 6 7 --filter_by_layer 0 --target_layer 1"
    )
    
    # Experiment 4: Different target layer with specialization
    run_command(
        "Experiment 4: Classes 0-5, target layer 0, with specialization analysis",
        "python Main.py --selected_classes 0 1 2 3 4 5 --target_layer 0 --run_specialization"
    )
    
    # Experiment 5: No filtering, all data
    run_command(
        "Experiment 5: All classes, no correct-only filtering, with specialization",
        "python Main.py --selected_classes 0 1 2 3 4 5 6 7 8 9 --target_layer 2 --run_specialization"
    )
    
    # Experiment 6: Energy analysis on all classes
    run_command(
        "Experiment 6: Energy consumption analysis on all classes",
        "python Main.py --selected_classes 0 1 2 3 4 5 6 7 8 9 --run_energy_analysis"
    )
    
    # Experiment 7: Combined specialization and energy analysis
    run_command(
        "Experiment 7: Combined specialization and energy analysis",
        "python Main.py --selected_classes 0 1 2 3 4 5 6 7 8 9 --run_specialization --run_energy_analysis"
    )

if __name__ == "__main__":
    main()
