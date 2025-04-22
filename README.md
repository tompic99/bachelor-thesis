# Bachelor Thesis: Contagion Model Simulation

## Overview
This project simulates the spread of awareness and adoption in a network using an agent-based contagion model. It supports different network types and seeding strategies, and provides both interactive and batch analysis modes.

## Requirements
- Python 3.10+
- mesa
- solara
- matplotlib
- numpy
- pandas
- seaborn
- networkx

Install dependencies with:
```
pip install mesa solara matplotlib numpy pandas seaborn networkx
```

## Usage

### Interactive Visualization
Run the interactive browser visualization:
```
python interactive_visualization.py
```

### Batch Analysis
Run batch experiments and generate plots:
```
python app.py
```

or
```
python parameters.py
```

## Files
- `agent.py`: Agent logic for the contagion model
- `model.py`: Model logic and network/agent setup
- `interactive_visualization.py`: Interactive browser visualization (Solara)
- `app.py`, `parameters.py`: Batch experiments and plotting
- `params1.py`: Example parameter sets for batch runs

## Output
Batch runs generate PNG plots and CSVs for further analysis.

---

For questions, contact the project author.
