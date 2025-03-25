# Banzhaf Ensemble Learning

This repository contains the source code and implementation details for my master's thesis with [Dr. Jajati Keshari Sahoo](https://www.bits-pilani.ac.in/goa/jajat-keshari-sahoo/). We explored the application of game theoretic methods in Ensemble Learning. The work was completed in 2017 and is now open-sourced under a Creative Commons License.

## Overview

This research investigates novel approaches to ensemble learning by incorporating concepts from game theory, specifically the Banzhaf Power Index and Borda count methods. The project introduces innovative techniques for feature selection and ensemble pruning, demonstrating their effectiveness in improving classification performance.

For detailed methodology, results, and theoretical background, refer to the full thesis: [report.pdf](report.pdf)


## Key Concepts

### Game Theoretic Methods
* **Banzhaf Power Index**: A measure of voting power used to evaluate the importance of individual classifiers in an ensemble
* **Borda Count**: A voting method applied to aggregate predictions from multiple classifiers
* **Coalition Games**: Framework for analyzing classifier interactions and contributions

### Technical Innovations
* **Feature Selection**: Novel method using conditional mutual information for feature pruning
* **Ensemble Pruning**: Implementation of Banzhaf Random Forests with strategic classifier selection
* **Voting Mechanisms**: Integration of Borda count for ensemble prediction aggregation

## Project Structure

```
.
├── Programs/                    # Source code implementation
│   ├── Banzhaf Decision Tree/  # Implementation of Banzhaf-based decision trees
│   │   ├── Banzhaf_Decision_Tree.ipynb  # Jupyter notebook with examples
│   │   ├── banzhaf_dt.py      # Core Banzhaf decision tree implementation
│   │   ├── banzhaf_rf.py      # Banzhaf Random Forests implementation
│   │   └── entropy_estimators.py  # Entropy calculation utilities
│   │
│   ├── Borda Count/           # Borda count ensemble implementation
│   │   ├── Borda Ensemble.ipynb  # Jupyter notebook with examples
│   │   ├── banzhaf_dt.py      # Banzhaf decision tree for Borda ensemble
│   │   └── entropy_estimators.py  # Entropy calculation utilities
│   │
│   └── SCG Pruning with LAE/  # Strategic Classifier Grouping with Local Accuracy Estimates
│       └── WMG with LAC.ipynb # Weighted Majority Game with Local Accuracy
│
├── report.pdf                 # Detailed thesis document
└── LICENSE                    # Creative Commons License
```

### Code Components

#### 1. Banzhaf Decision Tree
- `banzhaf_dt.py`: Core implementation of decision trees using Banzhaf Power Index for feature selection
- `banzhaf_rf.py`: Implementation of Banzhaf Random Forests
- `entropy_estimators.py`: Utilities for calculating entropy and information gain
- `Banzhaf_Decision_Tree.ipynb`: Interactive examples and experiments

#### 2. Borda Count Ensemble
- `Borda Ensemble.ipynb`: Implementation and experiments with Borda count voting
- `banzhaf_dt.py`: Banzhaf decision trees used as base classifiers
- `entropy_estimators.py`: Shared entropy calculation utilities

#### 3. SCG Pruning with LAE
- `WMG with LAC.ipynb`: Implementation of Weighted Majority Game with Local Accuracy Estimates
- Focuses on ensemble pruning using game theoretic approaches

## Key Findings

1. The Banzhaf Power Index provides an effective framework for evaluating classifier importance
2. Feature selection based on conditional mutual information improves ensemble performance
3. Borda count method offers a robust approach to ensemble prediction aggregation
4. Banzhaf Random Forests demonstrate competitive performance against traditional ensemble methods

## Future Work

Several promising directions for future research include:

1. **Feature Selection Comparison**
   * Compare the proposed feature selection method with other established methods
   * Evaluate performance across different datasets and domains

2. **Ensemble Pruning Optimization**
   * Compare SCG pruning performance with LAE in AdaBoost contexts
   * Investigate hybrid approaches combining multiple pruning strategies

3. **Scalability Improvements**
   * Optimize computational efficiency for large-scale datasets
   * Develop parallel processing capabilities

4. **Theoretical Analysis**
   * Deepen theoretical understanding of game theoretic approaches
   * Establish mathematical bounds for performance guarantees

## License

This project is licensed under a Creative Commons License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this code or reference the work in your research, please cite the original thesis:

```
@thesis{banzhaf-ensemble-2017,
  title={Game Theoretic Approaches in Ensemble Learning},
  author={Iyer, Sethu},
  year={2017},
  type={Master's Thesis},
  institution={BITS-Pilani},
  keywords={ensemble learning, game theory, Banzhaf Power Index, feature selection, random forests}
}
```


