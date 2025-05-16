# car-routing-optimization
This repository contains the final project for the course **Bioinspired Algorithms for Optimization** at **ETSISI - Universidad Politécnica de Madrid (UPM)**.

## Project Structure

The repository contains the following folders and files:

---

### `data/`
Contains the datasets and scripts used to prepare the routing graph of Madrid and define the optimization instances.

- `datasets.json`: Defines 5 origin–destination route scenarios over which all algorithms are evaluated.  
- `edges_clean.json`: JSON representation of the road network’s edges in Madrid, including added attributes (fuel consumption, LEZ restriction, signals, lanes, etc.).  
- `nodes_clean.json`: JSON with all nodes of the Madrid road network, enriched with custom information for the routing problem.  
- `madrid_graph.ml`: Full OpenStreetMap-based graph of Madrid saved as a backup before filtering/cleaning.  
- `clean.py`: Script used to generate the cleaned version of the graph, applying preprocessing and computing both hard and soft constraints for each edge or node.  

---

### `experiments/`
Stores the results of the optimization experiments for each algorithm.

- `nsga/`: JSON files with results from the NSGA-II experiments.  
- `paco/`: JSON files with results from the PACO experiments.  
- `spea/`: JSON files with results from the SPEA2 experiments.  

---

### `metaheuristics/`
Contains the core implementations of the optimization algorithms.

#### `aco/`
- `paco_graph.py`: Implementation of the PACO algorithm for graph-based routing problems.

#### `nsga/`
- `nsga_graph.py`: Implementation of the NSGA-II algorithm for the routing problem.

#### `spea/`
- `spea_graph.py`: Implementation of the SPEA2 algorithm for the routing problem.

> ⚠️ **Note:** Although SPEA2 is implemented here for completeness, it consistently failed to generate any valid or diverse routes. Its archive remained static in all runs, so SPEA2 results have been excluded from the final analysis.

---

### `metaheuristics/experiment/`
Contains the runners and analysis utilities.

- `aco/paco_runner.py`: Executes the PACO optimizer across all datasets (31 repeats each) and writes JSON results to `experiments/paco/`.  
- `nsga/nsga_runner.py`: Executes the NSGA-II optimizer across all datasets (31 repeats each) and writes JSON results to `experiments/nsga/`.  
- `spea/spea_runner.py`: Executes the SPEA2 optimizer across all datasets (31 repeats each) and writes JSON results to `experiments/spea/` (no valid solutions).  
- `analyzer.py`: Loads and analyzes all JSON results from `experiments/`, computes summary metrics (hypervolume, spread), and generates comparative plots.

---

### Notebooks

- **`run_nsga.ipynb`**, **`run_paco.ipynb`**, **`run_spea.ipynb`**  
  Each notebook executes its respective algorithm **31 times** over the 5 predefined routes, evaluates results, adjusts hyperparameters, and re-runs to explore performance.

- **`compare_results.ipynb`**  
  Loads all experiment outputs, performs statistical comparisons (e.g. Wilcoxon tests), computes aggregate metrics (hypervolume, spread, runtime) and generates comparative plots.

---

## Getting Started

1. **Clone the repository and navigate into it**  
   ```bash
   git clone https://github.com/JCelayaRdz/car-routing-optimization.git
   cd car-routing-optimization
   ```

2. **(Optional but recommended) Create and activate a virtual environment**  
   ```bash
   python -m venv .venv
   source .venv/bin/activate        # Linux/macOS  
   .venv\Scripts\activate           # Windows
   ```

3. **Install the dependencies**  
   ```bash
   pip install -r requirements.txt
   ```
   > ⚠️ If you encounter installation errors, remove the following two lines from `requirements.txt`:
   > ```
   > ipython==9.1.0
   > ipython_pygments_lexers==1.1.1
   > ```
