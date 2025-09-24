# EdgeFF PyTorch Project - Copilot Instructions

## Code Style Guidelines

When working with this EdgeFF PyTorch project, follow these specific coding conventions and patterns:

### Import Organization
- Place PyTorch imports first: `torch`, `torch.nn`, `torch.utils.data`, `torchvision`
- Follow with sklearn imports: `from sklearn.metrics import ...`, `from sklearn.model_selection import ...`
- Then numpy: `import numpy as np` 
- Standard library imports last: `import os`, `import matplotlib.pyplot as plt`
- Local module imports: `import Train`, `import Evaluation`, `import tools`

### Variable Naming Conventions
- Use snake_case for variables and functions: `train_loader`, `test_loader`, `num_epochs`
- Use descriptive names for data processing: `x_pos`, `x_neg`, `x_neutral`, `y_predicted`

### Class and Function Structure
- Use descriptive docstrings for complex functions:
  ```python
  def overlay_y_on_x(x, y):
      """Replace the first 10 pixels of data [x] with one-hot-encoded label [y]
      """
  ```

### PyTorch Patterns
- Model saving: `torch.save(model, path)` with relative paths

### Device Management (GPU/CPU)
- Always use GPU by default with CPU fallback:
  ```python
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  print(f'Using device: {device}')
  ```

### Training Loop Patterns
- Use `tqdm` for progress bars: `for epoch in tqdm(range(num_epochs)):`
- Optimizer pattern: `self.opt.zero_grad()`, `loss.backward()`, `self.opt.step()`
- Learning rates: typically `lr=0.03` for Adam optimizer

### Evaluation and Analysis
- Consistent result printing format:
  ```python
  print("\nResults for the {}SET_NAME{} set: ".format('\033[1m', '\033[0m'))
  print_results(targets.detach().cpu().numpy(), y_predicted)
  print('\tError:', 1.0 - torch.eq(torch.tensor(y_predicted), targets.detach().cpu()).float().mean().item())
  ```
- F1-score and accuracy metrics using sklearn
- Batch-wise evaluation for memory efficiency

### Specific Architecture Features
- Channel-based data overlay: 3 channels at indices 0, 1024, 2048
- Confidence-based early exit mechanisms
- Goodness calculations: `.pow(2).mean(1)` for layer activations
- Threshold-based training: `self.threshold = 2.0`

### Code Organization
- Separate files for different functionalities: `Train.py`, `Evaluation.py`, `tools.py`
- Main execution in `Main.py` with clear train/test flags
- Model persistence in `/model/` directory
- Utility functions in separate modules


When generating code for this project, maintain consistency with these patterns and ensure all new code follows the established conventions for readability and maintainability.