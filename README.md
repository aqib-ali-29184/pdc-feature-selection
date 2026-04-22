# Milestone 3 — Member 1: Coordination & Elite Protection

## What this adds

This builds directly on top of the Milestone 2 notebook (`milestone2_parallel_framework.ipynb`).
All Milestone 1 and 2 cells are untouched. Four new cells are appended at the bottom.

---

## New Classes

### `EliteVault`

Keeps a ranked archive of the **top-k masks seen across all generations**, not just the current population.

**The problem it solves:** After a merge, the operators start mutating the population again. Without protection, a good mask found in gen 3 can be overwritten and lost by gen 5. `EliteVault` prevents this.

**How it works:**
- After every generation, `update(scored_population)` compares the current population against the vault and retains only the best unique masks (default: top 5).
- At the end of each generation, `inject(population)` physically overwrites the **last `k` slots** of the population with the vault's top masks. Since `merge_populations()` already sorts the population best-first, the last slots are always the weakest — so elites replace the weakest, not the best.
- `vault.summary()` prints the final leaderboard at the end of the run.

```python
vault = EliteVault(vault_size=5)
vault.update(scored_population)   # call after every merge
population = vault.inject(population)  # call after update
best_score, best_mask = vault.best()
```

---

### `ExchangeProtocol`

Every `exchange_interval` generations, the **top-n masks from each operator's population are migrated into the other operator's population**, replacing its weakest slots.

**The problem it solves:** Both operators start each generation from the same merged population. Without exchange, they can converge to the same local optimum independently. Migration forces cross-pollination — the Addition side gets the most compact masks from Removal, and vice versa.

**Exchange rule:**
- Fires at generations: `exchange_interval, 2×exchange_interval, 3×exchange_interval, ...` (gen 0 is skipped).
- Default: every 2 generations, 3 masks migrate per side.
- Migration is **bilateral** — both operators give and receive simultaneously.

```python
exchange = ExchangeProtocol(fitness_fn, exchange_interval=2, n_exchange=3)

if exchange.should_exchange(gen):
    pop_add, pop_rem = exchange.exchange(
        pop_add, pop_rem, X_train, X_test, y_train, y_test, generation=gen
    )
```

---

## Entry Point: `run_coordinated_search()`

Replaces the bare `coordinator.run()` from Milestone 2. The loop per generation is:

```
1. Exchange (if due)  →  ExchangeProtocol cross-pollinates the two operator snapshots
2. Parallel generation  →  ParallelCoordinator runs AdditionOp + RemovalOp in threads
3. Vault update  →  EliteVault absorbs any new best masks
4. Elite injection  →  Vault's top masks are written back into the population
```

```python
final_pop, all_stats, vault = run_coordinated_search(
    coordinator        = coordinator,
    initial_population = init_pop,
    X_train=X_train, X_test=X_test,
    y_train=y_train, y_test=y_test,
    fitness_fn         = fitness_fn,
    n_generations      = 8,
    vault_size         = 5,
    exchange_interval  = 2,
    n_exchange         = 3,
    logger             = logger,
)
```

**Returns:**
- `final_pop` — the last population (list of masks)
- `all_stats` — list of per-generation stat dicts (same format as Milestone 2)
- `vault` — the `EliteVault` instance; call `vault.best()` or `vault.top_masks()` to retrieve the best results

---

## Confirmed Output (8 generations, all 3 datasets)

| Dataset     | Final Vault Best Score | Best Features |
|-------------|------------------------|---------------|
| MNIST       | 0.790474               | 198           |
| MADELON     | 0.869897               | 159           |
| CIFAR10     | 0.233187               | 288           |

Scores improve monotonically across all datasets. `dupes removed` increasing over generations confirms the population is converging rather than drifting randomly.

---

## Parameters to tune

| Parameter          | Default | Effect |
|--------------------|---------|--------|
| `vault_size`       | 5       | How many elite masks to protect. Increase if you want a wider elite pool for Milestone 3 comparative analysis. |
| `exchange_interval`| 2       | How often operators exchange. Lower = more aggressive cross-pollination. |
| `n_exchange`       | 3       | Masks migrated per side per exchange. Keep ≤ 15% of population size. |
| `n_generations`    | 8       | Increase to 15–20 for the final benchmarking run in Milestone 3. |

---

## What Hasan (Member 3) needs to know

`run_coordinated_search()` is a drop-in wrapper around `ParallelCoordinator`. It does not change how threads are spawned or how operators are called — it only wraps the loop with vault and exchange logic. No changes to `ParallelCoordinator`, `AdditionOperator`, or `RemovalOperator` were made.

The `all_stats` list returned has the same structure as before, so any export or analysis code from Milestone 2 will work on it without modification.

## What Ayesha (Member 2) needs to know

`EliteVault` and `ExchangeProtocol` are self-contained and have no dependencies on the operator internals. The vault stores raw `np.ndarray` masks and scores — nothing Removal-specific. The exchange simply calls `fitness_fn.evaluate()` on existing masks, the same call the operators themselves make.