#!/usr/bin/env python3
"""
Training Task Runner
Execute different ML training tasks
"""

import argparse
import sys
from pathlib import Path

def run_house_price_task():
    """Run house price prediction training"""
    from train_house_price_model import main as house_price_main
    return house_price_main()

def run_logistic_regression_task():
    """Run logistic regression training (placeholder)"""
    print("🔄 Logistic regression training task not implemented yet")
    print("Implement this following the same pattern as house price task")
    return 0

AVAILABLE_TASKS = {
    'house-price': {
        'function': run_house_price_task,
        'description': 'Train linear regression model for house price prediction'
    },
    'logistic': {
        'function': run_logistic_regression_task,
        'description': 'Train logistic regression model (placeholder)'
    }
}

def main():
    parser = argparse.ArgumentParser(description='ML Training Task Runner')
    parser.add_argument('task', choices=AVAILABLE_TASKS.keys(), 
                       help='Training task to run')
    parser.add_argument('--list', action='store_true', 
                       help='List available training tasks')
    
    args = parser.parse_args()
    
    if args.list:
        print("Available Training Tasks:")
        print("=" * 30)
        for task_name, task_info in AVAILABLE_TASKS.items():
            print(f"  {task_name}: {task_info['description']}")
        return 0
    
    # Run selected task
    task_info = AVAILABLE_TASKS[args.task]
    print(f"🚀 Running training task: {args.task}")
    
    return task_info['function']()

if __name__ == "__main__":
    exit(main())