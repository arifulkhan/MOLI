# Multilayer Networks (Research Notes)

## Scope and terminology

A multilayer network models entities connected through multiple relation types, domains, or time slices.

- Multilayer network: multiple adjacency layers A[1..L] plus inter-layer coupling.
- Multiplex: same node set appears in each layer.
- Interdependent network: multiple systems with dependency links.
- Temporal layer: each time window is treated as a layer.
- Supra-graph: full block matrix that combines intra- and inter-layer relations.

Core notation:

`A_l` = adjacency in layer l

`C_lm` = coupling from layer l to m

`A_M` = supra-adjacency built from A_l and C_lm

---

## Expanded major technique taxonomy

### 1) Supra-adjacency and tensor representation

Core idea: jointly represent all layers as one block operator and analyze structure/dynamics in a common space. This is the foundation for most downstream methods.

Formula: `A_M = [ A[1]  C12 ... C1L; C21 A[2] ... CL2; ... ]`.

Example: hospital systems where staff interaction, transfer, and equipment-sharing are separate layers.

Real-life impact: better tracing of infection pathways using all channels together.

### 2) Supra-Laplacian dynamics and diffusion

Core idea: run dynamics on the fused network to capture propagation speed across and between layers.

Formula: `dx/dt = -L_M x`, where `L_M = D_M - A_M`.

Example: rumor spread across social, email, and app messaging layers.

Real-life impact: improved estimates of time-to-awareness and targeted intervention timing.

### 3) Multislice community detection (generalized modularity)

Core idea: clusters are encouraged to persist across layers while adapting to layer-specific structure.

Formula: `Q = sum_{i,j,l} (A[i][j][l] - gamma_l P[i][j][l]) * 1[g[i,l]==g[j,l]] + sum_{i,l,r} omega_l_r * 1[g[i,l]==g[i,r]]`.

Example: neural networks at rest and task conditions.

Real-life impact: more stable cross-condition functional modules for clinical interpretation.

### 4) Tensor decomposition of node-layer structure

Core idea: decompose node-node-layer tensor into low-dimensional factors to separate shared and layer-specific effects.

Formula: `X approx sum_{k=1..K} lambda_k * a_k circ b_k circ c_k`.

Example: protein interaction, regulatory, and metabolic layers from multiple assays.

Real-life impact: robust pathway discovery by filtering assay noise.

### 5) Multilayer spreading models (SIS/SIR)

Core idea: layer-dependent transmissibility and cross-layer transfer rules replace single-layer epidemic equations.

Formula: `dI_i_l/dt = beta_l * sum_j A_l[i,j] S_i_l I_j_l + alpha * sum_m C_lm S_i_l I_i_m - mu I_i_l`.

Example: household, workplace, transport networks for flu modeling.

Real-life impact: identifies bridge nodes that should be targeted by vaccination or isolation.

### 6) Multilayer centrality and supracentrality

Core idea: rank influence in the full supra-network instead of averaging per-layer rankings.

Formula: `c = alpha * A_M * c + (1-alpha) * v`.

Example: influencer who is medium on each platform but high across cross-posting/bridge links.

Real-life impact: superior source ranking for intervention and cybersecurity hardening.

### 7) Random walks and flow metrics

Core idea: define walk transitions over supra-graph transitions to estimate routeability, commute time, and transition probabilities.

Formula: `P_M = row_normalize(A_M)`, then standard Markov-hitting-time metrics on `P_M`.

Example: multimodal transport routing with transfer switching costs.

Real-life impact: more realistic itinerary and disruption planning.

### 8) Interdependent robustness and cascades

Core idea: model mutually dependent failures; failures can cascade across layers through dependency links.

Formula: fixed-point/percolation equations on mutually coupled giant component probabilities.

Example: power grid outage combined with telecom outage.

Real-life impact: thresholds for abrupt collapse and recovery policy design.

### 9) Multilayer optimization and control

Core idea: optimize system variables jointly across layers with coupling constraints.

Formula: `min_x sum_l cost_l(x_l) subject to A_l x_l = b_l and coupling_constraints(x_1..x_L)`.

Example: logistics across road, rail, and air layers with shared inventory.

Real-life impact: fewer bottlenecks and lower delay under disruptions.

### 10) Multilayer motif and mesoscopic pattern mining

Core idea: detect small recurring cross-layer patterns and compare against null models.

Formula: `z_M = (N_obs(M)-mean_null(M))/sd_null(M)`.

Example: signaling motifs appearing across gene-regulation and PPI layers.

Real-life impact: interpretable mechanisms and candidate intervention points.

### 11) Cross-layer link/node prediction

Core idea: predict missing links using signals from each layer and learn cross-layer weights.

Formula: `score(i,j)=sum_l w_l f_l(i,j)` with `w_l >=0, sum w_l =1`.

Example: predicting collaboration edges from communication + output + funding layers.

Real-life impact: better data completion and recommendation.

### 12) Multilayer representation learning (incl. multiplex GNNs)

Core idea: learn embeddings for each node-layer state using messages from both intra-layer and inter-layer neighborhoods.

Formula: `h_{i,l}^{k+1}=sigma(Ws AGG_s(neigh in layer l) + Wc AGG_c(neigh across layers)).`

Example: multi-omics patient graphs.

Real-life impact: stronger disease subtype classification and missing-data robustness.

### 13) Temporal-structural coupling

Core idea: combine structure learning per time with temporal smoothness between snapshots.

Formula: `Loss = sum_t data_loss(X_t, Xhat_t) + lambda * sum_t ||Z_t - Z_{t-1}||^2`.

Example: changing flight networks with seasonality.

Real-life impact: better early anomaly and regime-shift detection.

### 14) Cross-layer causality and intervention modeling

Core idea: estimate whether perturbation in one layer causes effects in another, not just correlation.

Formula: `Y_l = f_l(X_l, X_-l, U), evaluate Y_l(do(T=t)).`

Example: test if reducing one transport layer causally lowers downstream epidemic burden.

Real-life impact: stronger policy planning and counterfactual analysis.

---

## Life science, airline, infrastructure, and other solved applications

- Life science: brain connectivity, protein-protein interaction, and multi-omics used for disease marker discovery and mechanism refinement.
- Airline: scheduling, re-routing, and delay propagation across airline, metro, and rail layers.
- Infrastructure: interdependent robustness used in power-cyber and telecom planning.
- Social systems: multi-platform misinformation and moderation models.
- Finance/supply chain: contagion and disruption risk across payment, credit, and logistics layers.

---

## Application matrix by category

| Category | Representative systems | Typical data | Key method | Known impact |
|---|---|---|---|---|
| Life science | Brain, PPI, multi-omics | Imaging, omics, assay-specific interactions | Supra models, tensor decomposition, cross-layer prediction | Better subtype signatures and pathway understanding |
| Transportation | Air, rail, metro, road | Schedule, OD flow, transfer graph | Multilayer flow optimization, random walks | More resilient routing and disruption response |
| Infrastructure | Power, telecom, water | Dependency graphs, failure logs | Cascade/percolation, centrality | Reduced cascade risk and faster recovery |
| Epidemiology | Household, workplace, mobility | Temporal contact edges | Multilayer SIS/SIR + centrality | Better outbreak prediction and control |
| Social / comms | Social, message, content graph | Layered interaction streams | Motif mining, multilayer GNN | Better moderation/recommendation targeting |
| Finance / supply | Interbank, payment, logistics | Layered financial/supply links | Causality + dependency + optimization | Stronger stress-testing and mitigation planning |

---

## Pitfalls and implementation notes

- Align node IDs, ontology, and time windows before building the model.
- Calibrate layer weights; naive equal weighting often over- or under-emphasizes dense layers.
- Use sparse/symbolic operations for large L*N graphs.
- Watch causal interpretation; most models are still descriptive unless designed as causal designs.
- Validate with layer-aware splits and realistic negative sampling.

---

## Canonical references

- Kivela, M. et al. (2014). Multilayer networks.
- Boccaletti, S. et al. (2014). The structure and dynamics of multilayer networks.
- Mucha, P.J. et al. (2010). Community structure in time-dependent, multiplex, and multiscale networks.
- De Domenico, M. et al. (2013). Mathematical formulation of multilayer networks.
- De Domenico, M. et al. (2015). Ranking in interconnected networks.
- Buldyrev, S.V. et al. (2010). Catastrophic cascade of failures in interdependent networks.
- Halu, A. et al. (2014). Multiplex networks and spreading.

```bib
@article{kivela2014multilayer,
  title={Multilayer networks},
  author={Kivela, Mikko and Arenas, Alex and Barthelemy, Marc and Gleeson, James P and Moreno, Yamir and Porter, Mason A},
  journal={Journal of Complex Networks},
  volume={2},
  number={3},
  pages={203--271},
  year={2014}
}

@article{boccaletti2014structure,
  title={The structure and dynamics of multilayer networks},
  author={Boccaletti, Stefano and Bianconi, Ginestra and Criado, Raquel and Delgenio, Carlos I and Gomez-Gardenes, Juan and Romance, Miguel and Sendina-Nadal, Irene},
  journal={Physics Reports},
  volume={544},
  number={1},
  pages={1--122},
  year={2014}
}

@article{mucha2010community,
  title={Community structure in time-dependent, multiplex, and multiscale networks},
  author={Mucha, Peter J and Richardson, Thomas and Macon, Kevin and Porter, Mason A and Onnela, J-P},
  journal={Science},
  volume={328},
  number={5980},
  pages={876--878},
  year={2010}
}

@article{dedomenico2013mathematical,
  title={Mathematical formulation of multilayer networks},
  author={De Domenico, Mauro and Soler-Ribalta, Alba and Cozzo, Enrico and Kivela, Mikko and Moreno, Yamir and Porter, Mason A},
  journal={Physical Review X},
  volume={3},
  number={4},
  pages={041022},
  year={2013}
}

@article{dedomenico2015versatile,
  title={Ranking in interconnected networks},
  author={De Domenico, Mauro and Nicosia, Vincenzo and Latora, Vito and Porter, Mason A and others},
  journal={Nature Communications},
  volume={6},
  number={1},
  pages={6868},
  year={2015}
}

@article{buldyrev2010cascading,
  title={Catastrophic cascade of failures in interdependent networks},
  author={Buldyrev, Sergey V and Parshani, Roni and Paul, Gerald and Stanley, H Eugene and Havlin, Shlomo},
  journal={Nature},
  volume={464},
  number={7291},
  pages={1025--1028},
  year={2010}
}

@article{halu2014spreading,
  title={Multiplex networks and spreading},
  author={Halu, Alireza and Bianconi, Ginestra and Kivela, Mikko},
  journal={Chaos},
  volume={24},
  number={5},
  pages={054106},
  year={2014}
}
```

## ML/AI and RL coverage check (for repository planning)

Short answer to your question: **yes, ML/AI research is already represented**, but the current literature list is mostly method-focused and not implementation-complete.

### Current ML/AI representation in this file

- Represented in conceptual sections:
  - Multilayer representation learning (multiplex GNN-style encoders).
  - Cross-layer link/node prediction.
  - Temporal-structural coupling and tensor decomposition for learned latent states.
  - Causality/intervention pipelines (causal discovery + effect estimation perspective).
- Not yet represented as an explicit implementation or benchmark section.

### What is implemented in available open-source stacks (from repository scan)

| ML/AI technique | Representative implemented method | Typical language / stack | Notes |
|---|---|---|---|
| Multiplex network embeddings | `GATNE`, `DMGI`, `MultiVERSE`, `MGN-Net` | Python (PyTorch / PyTorch-Geometric) | Focused on node/link prediction and node representation quality on multiplex graphs. |
| Graph representation & ranking | `BraneMF`, tensor-style embedding ideas | Python (PyTorch / NumPy / SciPy) | Bi-relational / matrix-factorization style baselines and learned scores. |
| Structural ML on multilayers | `pymnet`, `multinetx` | Python | Mostly model construction + classic network measures; useful as baseline plumbing. |
| Explainable/visual ML workflows | `muxViz`, `muxVizPy`, `py3plex`, `MNetX` ecosystem | Python, web-based visualization stack in part | Primarily explainability and exploratory analysis rather than end-to-end deep learning pipelines. |
| Temporal / coupled network learning | `JOD`- and project-specific scripts around `PyTorch`, `PyG`, `DGL` ecosystems | Python | Typically custom code in papers; less standardized packaging than classic graph libraries. |

### RL (reinforcement learning) status

- **Not clearly present in the surveyed core multilayer-network libraries.**
- Most available implementations are still:
  - supervised/sampling style representation learning,
  - link prediction,
  - or percolation/diffusion optimization heuristics.
- This leaves RL as a high-value gap for the repository you are planning.

### Real-life impact relevance for your roadmap

- Existing ML implementations already support many practical problems (disease subtype prediction from multi-omics, heterogeneous transport demand forecasting, fraud/link discovery), but:
  - Reproducible benchmark baselines are fragmented.
  - Cross-layer task definitions are often inconsistent (taxonomy differences: layer meaning, negative sampling, temporal indexing).
  - RL methods are almost absent, creating opportunity for adaptive intervention policies on multilayer systems.

### Next-step gap map to add in the codebase design

1. **Layered benchmark module**: standardized dataset loaders, train/validation/test splits by time and layer, and evaluation protocols.
2. **Pluggable method registry**: include both classical methods and ML modules (GNN, embeddings, link prediction) with unified `fit/predict` APIs.
3. **RL extension layer**: start with discrete action environments for intervention selection on interdependent layers (e.g., vaccination, load shedding, rerouting).
4. **Performance contract**: include wall-clock, memory, and scaling curves for each method (nodes, edges, layers, and epochs).
