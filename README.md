# Parallel Feature Selection Engine

A parallel, population-based feature selection system built across three milestones for the Parallel and Distributed Computing course. The system evolves binary feature masks using two operators running concurrently, evaluated with a Naive Bayes classifier on three datasets: MNIST, Madelon, and CIFAR-10 (grayscale).

**Team:** Aqib Ali, Ayesha Salahuddin, Syed Hasan Imam

---

## Milestone 1 — Core Building Blocks

Established the foundational classes used throughout all milestones.

- `DatasetLoader` — loads, subsamples, normalizes, and splits MNIST (784 features), Madelon (500 features), and CIFAR-10 grayscale (1024 features)
- `NaiveBayesEvaluator` — thread-safe classifier; creates a fresh `GaussianNB` instance per call
- `BinaryMaskEncoder` — generates and encodes feature masks (all-ones, random density)
- `FitnessFunction` — scores a mask as `accuracy - alpha * (n_selected / n_total)`, penalizing feature count
- `ResultsLogger` — logs every evaluated mask with score, accuracy, feature count, and timing; exports to CSV

**Baseline (all features, no selection):**

| Dataset     | Features | Accuracy |
|-------------|----------|----------|
| MNIST       | 784      | 58.70%   |
| Madelon     | 500      | 84.00%   |
| CIFAR-10    | 1024     | 22.70%   |

---

## Milestone 2 — Parallel Operators and Population Search

Introduced the two operators that evolve the population, and a coordinator that runs them in parallel each generation.

- `FeatureBiasComputer` — computes information gain scores; guides the addition operator to prefer high-IG features
- `AdditionOperator` — proposes adding features to each mask, accepts if fitness improves
- `SparsityBiasComputer` — inverts IG scores; guides the removal operator to drop low-IG features first
- `RemovalOperator` — proposes removing features from each mask, accepts if fitness improves
- `merge_populations` — deduplicates and re-ranks the two operators' outputs by fitness score
- `ParallelCoordinator` — runs AdditionOp and RemovalOp in parallel threads via `ThreadPoolExecutor`; merges their outputs each generation

Each generation: both operators receive a deep copy of the population, run concurrently, and their outputs are merged into one ranked population of fixed size.

**Results after 5 generations (population = 20):**

| Dataset     | Best Score | Features Selected |
|-------------|------------|-------------------|
| MNIST       | 0.772589   | 189 / 784         |
| Madelon     | 0.858478   | 153 / 500         |
| CIFAR-10    | 0.232334   | 273 / 1024        |

---

## Milestone 3 — Elite Protection, Exchange Protocol, and Coordinated Search

Extended the framework with two mechanisms to prevent regression and improve convergence, then ran a full comparative analysis.

- `EliteVault` — stores the top-k masks seen across all generations; reinjects them into the population after each merge to prevent loss of best solutions
- `ExchangeProtocol` — every `exchange_interval` generations, the top-n masks from each operator's snapshot are injected into the other's population, cross-pollinating the two search directions
- `run_coordinated_search` — full loop combining ParallelCoordinator + EliteVault + ExchangeProtocol over n generations

**Final results (8 generations, vault size = 5, exchange every 2 gens):**

| Dataset     | Baseline Accuracy | Final Accuracy | Accuracy Change | Features Used   | Feature Reduction |
|-------------|-------------------|----------------|-----------------|-----------------|-------------------|
| MNIST       | 58.70%            | 79.30%         | +20.60%         | 198 / 784       | 74.7% fewer       |
| Madelon     | 84.00%            | 87.31%         | +3.31%          | 159 / 500       | 68.2% fewer       |
| CIFAR-10    | 22.70%            | 23.60%         | +0.90%          | 288 / 1024      | 71.9% fewer       |

The Removal operator was dominant across all three datasets (higher average masks improved per generation). All datasets reached 95% of their final score by generation 0, with steady incremental gains through generation 7.

---

## Repository Structure

```
main/
    Milestone_3.ipynb                  # Complete notebook: all three milestones end-to-end
    README.md
```

Each milestone branch (`milestone-1`, `milestone-2`, `milestone-3`) preserves the incremental development history.

---

## How to Run

Open `Milestone_3.ipynb` in Google Colab. All cells run top-to-bottom. Datasets are downloaded automatically on first run.

**Dependencies:** `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `tensorflow` (for Keras dataset loaders), `opencv-python`
