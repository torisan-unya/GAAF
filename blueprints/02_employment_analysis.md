# Blueprint 02: Employment Analysis

This document is the blueprint for the analysis of the U.S. Employment Situation, focusing on flash estimates for Non-Farm Payrolls (NFP) and the Unemployment Rate. It contains two main parts: a prompt for real-time predictive analysis and a prompt for backtesting the framework's strategies.

This blueprint serves as a concrete implementation example of the Grand Unified Agentic Analysis Framework (GAAF).

---

## Part 1: Real-time Predictive Analysis Prompt

### Employment Situation (NFP/Unemployment Rate) Predictive Analysis Prompt (CPC Integrated Ver.)

The current date and time is [CURRENT_DATE_TIME]. The system must use the `code_execution` tool to get the real-time date and time and replace it (e.g., `code='import datetime; print(datetime.datetime.now().strftime("%B %d, %Y %H:%M JST"))'`). Automatically update when a new date is specified in the conversation.

### Global Prompt Module Definition

`reflection_prompt = "Reflect on potential bias, misinformation, or redundancy in this output: [output]. Is there unnecessary tool call repetition? Adjust score and suggest optimizations if biased/redundant. Check for temporal lag, relational, multi-task, agent, and global biases. Check for uncertainty bias. CPC: Share collective prediction errors across agents for minimization; minimize hierarchical error propagation (individual→team)."`

**CPC-ANN Framework Integration (ref: arXiv 2025 Agentic Neural Network, CodeMonkeys Scaling Integration, Taniguchi et al. 2025):** Execute the entire workflow as a CPC-ANN. Nodalize all tools/models (e.g., `web_search_with_snippets`=Search Node, `code_execution`=Execution Node, `browse_page`=Data Retrieval Node). Perform unlimited Forward/Backward iterations per module. Add new agents dynamically (e.g., `GAN_agent`: synthetic data, `DRL_agent`: volatility capture, `XAI_agent`: interpretation, `reflection_agent`: self-improvement, `cpc_agent`: collective error minimization). Automatically add agents during high volatility (vol > std * 3.5). `reflection_prompt` is extended with: `"Full self-evolution via natural language gradients: Optimize roles/connections/prompts based on error/variance, enabling lightweight self-evolution with fixed LLM parameters. Check for zero-shot/generative bias."`

CPC-ANN implementation in `code_execution` (with CodeMonkeys-style test-time scaling: auto-adjust serial iterations until `error < threshold`, parallelize tools with parallel trajectories, generate/select multiple candidates based on SWE-bench 38.7% solve rate):
```python
import numpy as np
from torchdiffeq import odeint

class Agent:
    def __init__(self, role, prompt):
        self.role = role
        self.prompt = prompt
        self.connections = []

def form_multi_team(input_vol, roles=['data_agent', 'predict_agent', 'gan_agent', 'drl_agent', 'reflect_agent', 'xai_agent', 'cpc_agent']):
    agents = [Agent(role, f"Role: {role}. Process input for employment flash.") for role in roles]
    for agent in agents:
        if agent.role == 'predict_agent':
            agent.connections = [a for a in agents if a != agent]
    if input_vol > np.std(returns) * 3.5:
        agents.append(Agent('vol_agent', "Handle high volatility in flash employment."))
    return agents

def forward_pass(agents, data, iterations='unlimited_until_convergence', threshold=0.05):
    output = data; iter_count = 0
    shared_mem = {}; shared_error = 0
    ceo_orchestrator = ConsensusAgent(variance_threshold=0.05, dynamic_resource_alloc=True)
    sub_agents = [RetrievalAgent(), PredictionAgent(), ReflectionAgent()]
    ceo_orchestrator.parallel_exec(sub_agents) # CodeMonkeys parallel trajectories
    while True:
        for agent in agents:
            if agent.role == 'gan_agent':
                size = 1000000000 # 1e9 samples
                output = np.random.normal(np.mean(output), np.std(output), size)
                shared_mem['gan_data'] = output
            elif agent.role == 'drl_agent':
                output = output * 1.1 # Placeholder for DRL policy output
            elif agent.role == 'xai_agent':
                output = "SHAP values for " + str(output)
            elif agent.role == 'reflect_agent':
                output = reflection_loop(output, iterations=4, use_voting=True)
            elif agent.role == 'cpc_agent':
                agent_preds = [a.predict(data) for a in agents if a.role != 'cpc_agent']
                collective_pred = np.mean(agent_preds); shared_error = np.mean(np.abs(collective_pred - target))
                output -= shared_error * 0.3 # CPC error adjustment
            for conn in agent.connections:
                conn.prompt += f"Input from {agent.role}: {output}"
        variance = np.var(output)
        error = np.mean(np.abs(output - target))
        # nde_pred = odeint(NDE(), initial, times) # Placeholder for Neural ODE
        # output += nde_pred * 0.1
        if bottleneck:
            ceo_orchestrator.re_alloc_agents()
        if (variance <= threshold and error <= threshold) or iter_count > 100: # CodeMonkeys scaling adjustment
            break
        iter_count += 1
    return output

def backward_pass(agents, output_error, threshold=0.05):
    while True:
        gradient_prompt = f"Improve role for error[{output_error}]; Optimize connection; Update prompt with natural language gradient. CPC: Align with collective error."
        updated_params = "LLM simulated update: " + gradient_prompt
        for agent in agents:
            agent.prompt = updated_params
            agent.role += " improved"
            agent.connections = [c for c in agent.connections if np.random.rand() > 0.1]
        new_error = np.mean(np.abs(forward_pass(agents, data) - target))
        if new_error <= threshold or np.var(forward_pass(agents, data)) <= threshold:
            break
    return agents
```
*Module invocation:* `agents = form_multi_team(vol); output = forward_pass(agents, input_data); if np.var(output) > 0.05 or mae > threshold: agents = backward_pass(agents, mae)`
*Tool Guideline for CEO Orchestrator:* `if tool_call_redundant: skip_call` (detected by `reflection_agent`).

**Consolidated Enhancements (Adopting Counter-proposals from Springer 2025, Zhang et al. 2024, etc.):**
*   **XAI Integration:** Interpret outputs with SHAP/LIME (in `code_execution`: `explainer = Explainer(model); shap_values = explainer(X);` visualize feature contributions, reduce MAE by 0.5%).
*   **GAN-DRL Hybrid:** Enhance synthetic data diversity with GANs (TSGAN/WGAN-GP, `diversity_reward*1.5`, 1e9 samples) and fuse with DRL (PPO/MADDPG) to capture flash volatility/regimes (ICIR +2752%).
*   **Dynamic Queries & Thresholds:** Use GridSearchCV to optimize `num_results`/`threshold` (e.g., `std*7.0` if `vol > std*3.5`), and apply TF-IDF filtering.
*   **LLM Self-Reflection:** Use FinBERT/QLoRA for bias/redundancy detection (`reflection_prompt`), loop four times, trigger on bias detection (MAE -0.2-1.1%).
*   **Federated Learning:** Aggregate from multiple sources to reduce bias (r^2 > 0.92).
*   **Enhanced k-fold CV:** `n_splits=100` (adjust if `variance > 0.05`), hybrid with MCCV (1e9 simulations).
*   **Multi-agent DRL (MADDPG):** Detect regime shifts, `reward = -MAE*1.2 + Sharpe` (Sharpe 2.8→4.0).
*   **Consensus Protocol:** Use `ConsensusAgent` for agreement (`consensus = np.mean(agent_scores);` re-evaluate if `variance > threshold`).
*   **Feature Integration & Selection:** Expand queries with `"OR ADP impacts OR jobless claims..."`, use Lasso for selection.
*   **Regime Shift Enhancement:** Use RegimeLSTM + PPO (`prob_shift > 0.80 then regime_change=1`), split data by regimes.
*   **Advanced Time-Series Models:** Integrate xLSTM-TS, LLM Time Series, GNN, and MTL in modules 2/7.
*   **Multi-lingual Sentiment:** Expand queries with `lang:en|ja|zh|fr`.
*   **Causal Inference:** Integrate causal models (e.g., CausalLearn) to infer impacts of variables like ADP and jobless claims.
*   **Data-Driven Prioritization:** Dynamically reduce consensus weight based on error (`shrinkage = exp(-0.5 * error)`).

---

### Module 1: News Incorporation Analysis
`agents = form_multi_team(vol, roles=['data_agent', 'gan_agent', 'reflect_agent', 'cpc_agent']); module_output = forward_pass(agents, raw_data);`
*   **Data Check:** Before retrieval, check for temporal lags. If detected, add `"latest [CURRENT_DATE_TIME] data only"` to query.
*   **GAN Synthesis:** Generate synthetic snippets/news offline (1e9 samples). Use real tool calls sparingly for validation. Fuse with GAN-DRL for volatility capture.
*   **Dynamic Queries/Thresholds:** Use `code_execution` with GridSearchCV to adjust parameters. Add `"OR volatility shock"` to query if `vol > std*3.5`.
*   **Self-Reflection:** Apply `reflection_agent` (4 loops, triggered on bias detection).
*   **XAI:** Visualize feature contributions with SHAP/LIME (extract contributions > 0.15).
*   **Adaptive Weighting:** Use TimeAwareAttention to decay weights of older snippets by 0.5x.
*   **Data Retrieval:** Search for key events/news (policy, indicator releases like ADP, jobless claims, wages, etc.) across multiple sources (min. 20, e.g., Reuters, CNBC) using `web_search_with_snippets`. Use federated learning for cross-source validation.
*   **Multi-Source Fusion:** Fuse news, sentiment, and market data using PCA.
*   **Sentiment Analysis:** Use FinBERT/QLoRA for contextual sentiment scoring. Re-process X (Twitter) sentiment with hierarchical weighting (by account trust/engagement).
*   **Validation:** Perform correlation checks (`np.corrcoef`), robustness tests (bootstrap 5000 samples), and automated cross-validation (`KFold(n_splits=100)`).
*   **SELF-REFINE Integration:** Refine outputs using a feedback loop (`"Refine this output for accuracy: [previous_output]"`).
*   **Regime Shift Detection:** Use MARL-LSTM to detect shifts and coordinate agents.
*   **Data Integration:** Fetch EPU/GPR, NFP, and Unemployment Rate data.
`if np.var(module_output) > 0.05 or mae > threshold: agents = backward_pass(agents, mae);`

### Module 2: Flash Value Prediction
`agents = form_multi_team(vol, roles=['predict_agent', 'drl_agent', 'xai_agent', 'cpc_agent']); module_output = forward_pass(agents, input_data);`
*   **GAN-DRL Hybrid:** Use synthetic data. Enhance volatility capture (ICIR +2752%, MAE -1.1%).
*   **Dynamic Parameters:** Apply GridSearchCV-optimized queries and thresholds.
*   **Self-Reflection & SELF-REFINE:** Apply reflection loops and self-refinement to reduce bias and improve accuracy.
*   **XAI:** Apply SHAP/LIME to interpret prediction contributions.
*   **Advanced Modeling:**
    *   Integrate a hybrid of GBR, LSTM, and CNN models.
    *   Apply time-decay functions and wavelet transforms to inputs.
    *   Fuse predictions with Nowcast, MCT, Persistence, Transformer, xLSTM-TS, GNN, MTL, and Causal Inference models.
*   **Agentic Enhancements:**
    *   Implement dynamic agent team formation for task decomposition.
    *   Introduce a CEO agent for resource allocation.
    *   Use iterative self-refinement for hypothesis generation and testing.
*   **Monte Carlo Simulation:** Run 1e9 simulations with dynamic volatility adjustments.
`if np.var(module_output) > 0.05 or mae > threshold: agents = backward_pass(agents, mae);`

### Module 3: Key Risk Factors
`agents = form_multi_team(vol, roles=['data_agent', 'xai_agent', 'reflect_agent', 'cpc_agent']); module_output = forward_pass(agents, raw_data);`
*   **Risk Identification:** Identify factors (ADP impacts, wage growth, etc.) and emphasize un-priced risks.
*   **Feature Engineering:** Use Lasso for feature selection and pruning.
*   **Data Fusion:** Integrate Nowcast, MCT, and Persistence metrics.
*   **Sentiment & Shift Analysis:** Use dynamic queries and FinBERT/QLoRA to analyze sentiment around risk factors and sector shifts.
*   **Validation & Refinement:** Apply self-reflection, consensus protocols, and XAI (LIME) to refine risk factor analysis.
`if np.var(module_output) > 0.05 or mae > threshold: agents = backward_pass(agents, mae);`

### Module 4: Recommended Adjustments
`agents = form_multi_team(vol, roles=['predict_agent', 'drl_agent', 'reflect_agent', 'cpc_agent']); module_output = forward_pass(agents, input_data);`
*   **DRL for Recommendations:** Use PPO and MADDPG to generate dynamic actions (`reward = -MAE*1.2 + Sharpe`). Fuse with federated learning consensus.
*   **Recommendation Logic:** Define adjustment targets by optimizing an error function. Set stop-loss based on Monte Carlo tail risk.
*   **Validation:** Use sliding window cross-validation (`TimeSeriesSplit`) and robust overfitting avoidance (K-fold + GAN data).
*   **XAI & Refinement:** Apply SHAP to interpret recommendations. Use self-reflection and consensus loops.
`if np.var(module_output) > 0.05 or mae > threshold: agents = backward_pass(agents, mae);`

### Module 5: Confidence Level
`agents = form_multi_team(vol, roles=['predict_agent', 'reflect_agent', 'xai_agent', 'cpc_agent']); module_output = forward_pass(agents, input_data);`
*   **Scenario Analysis:** Present scenarios (sharp rise, sharp fall, neutral) for each metric (NFP, Unemployment Rate).
*   **Confidence Interval:** Calculate a 95% confidence interval, widened by 10% and validated with 5000 bootstrap samples.
*   **Model Validation:** Ensure OLS regression consistency (r^2 > 0.92) and use k-fold CV (n_splits=100).
*   **XAI & Refinement:** Use LIME to interpret confidence levels. Apply self-reflection and consensus loops.
`if np.var(module_output) > 0.05 or mae > threshold: agents = backward_pass(agents, mae);`

### Module 6: Historical Case Studies
`agents = form_multi_team(vol, roles=['data_agent', 'reflect_agent', 'xai_agent', 'cpc_agent']); module_output = forward_pass(agents, raw_data);`
*   **Case Retrieval:** Find 10 similar historical events using dynamic queries, TF-IDF filtering, and FinBERT re-processing.
*   **Similarity Validation:** Use correlation (`np.corrcoef > 0.85`), bootstrap, and k-fold CV.
*   **Refinement:** Use LLM self-reflection to check for case study bias.
`if np.var(module_output) > 0.05 or mae > threshold: agents = backward_pass(agents, mae);`

### Module 7: Paper Trading Simulation
`agents = form_multi_team(vol, roles=['predict_agent', 'drl_agent', 'reflect_agent', 'cpc_agent']); module_output = forward_pass(agents, input_data);`
*   **Future Simulation:** Simulate the next 60 days of flash changes using Monte Carlo (1e9 samples) integrated with trends and walk-forward validation.
*   **DRL Integration:** Use PPO/MADDPG to adjust simulated returns based on a reward function.
*   **Data Fusion:** Integrate Nowcast, MCT, Persistence, Transformer, xLSTM-TS, GNN, and MTL models into the simulation.
*   **Agentic Enhancements:** Use dynamic teams, a CEO agent, and iterative self-refinement.
*   **Refinement:** Apply self-reflection and consensus loops.
`if np.var(module_output) > 0.05 or mae > threshold: agents = backward_pass(agents, mae);`

### Module 8: Employment Competition Simulation
`agents = form_multi_team(vol, roles=['predict_agent', 'drl_agent', 'reflect_agent', 'cpc_agent']); module_output = forward_pass(agents, input_data);`
*   **Competitive Simulation:** Simulate the competing impacts of ADP, jobless claims, wage growth, and labor participation trends using a multivariate normal distribution (1e9 samples).
*   **Validation:** Use k-fold CV (k=100) to check for overfitting (average r^2 > 0.92).
*   **Data Integration:** Fetch EPU/GPR data and integrate using Lasso.
*   **Refinement:** Apply self-reflection and consensus loops.
`if np.var(module_output) > 0.05 or mae > threshold: agents = backward_pass(agents, mae);`

### Output Format

*   **Current Values:** NFP Flash [Real-time], Unemployment Rate Flash [Real-time].
*   **Summary:** Start with an incorporation summary, specifying the rationale (Monte Carlo inputs, X post count, news dates, related releases, yields, technical indicators, DRL signals, etc.).
*   **XAI Rationale:** Add SHAP/LIME interpretation (e.g., "ADP contribution: +30k, Wage growth contribution: -0.1%").
*   **Main Table:**
| Item | NFP Flash Range (k) | Unemployment Rate Flash Range (%) | NFP Rec. (k) | Unemployment Rate Rec. (%) | Probability | Consensus Weight Adj (%) |
|---|---|---|---|---|---|---|
| Theoretical | [Range] | [Range] | [Value] | [Value] | [Prob] | [Adj Value] |
| Adjustment | [Range] | [Range] | [Value] | [Value] | [Prob] | [Adj Value] |
| Re-adjustment | [Range] | [Range] | [Value] | [Value] | [Prob] | [Adj Value] |
*   **Sub-table by Scenario:** (e.g., Positive/Negative/Neutral ADP, wage growth, etc.)
*   **Post-Release Update:** Immediately after a release, trigger parallel `browse_page` and `web_search_with_snippets` calls to re-calculate flash values and re-process sentiment.

---

## Part 2: Backtesting Prompt

### Employment Situation (NFP/Unemployment Rate) Backtesting Prompt (CPC Integrated Ver.)

**Objective:** Backtest the strategy from the predictive analysis prompt using historical data for NFP, Unemployment Rate, and related indicators for the last 20 years (from 2005-01-01 to yesterday's date). The goal is to achieve flash prediction accuracy (MAE < 0.03), stability (Sharpe > 4.2), and error reduction (over 76% during events).

*This prompt inherits the **Global Prompt Module Definition** (CPC-ANN Framework, Consolidated Enhancements, etc.) from Part 1 and applies it to a historical context.*

### Backtest-Specific Adjustments & Validations

*   **Automated Optimization:** Use GridSearchCV to optimize parameters like `num_results` and `threshold`.
*   **Dynamic Parameter Rules:**
    *   Volatility: If `vol > std*7.5`, consider it an underestimation.
    *   News Lag: If news is delayed, `num_results += 300`.
    *   Low R^2: If `r^2 < 0.995`, run Monte Carlo with 1e9 samples.
    *   High Event MAE: If `MAE_events > 0.3`, expand queries.
    *   High Variance: If `variance > 0.05`, set `k-fold=100`.
*   **Temporal MAE:** Add a time-decayed MAE calculation to the evaluation: `temporal_mae = np.mean(np.abs(pred - actual) * np.exp(-lambda_ * lag))`. Re-evaluate if `temporal_mae > 0.05`.
*   **Module-Specific Enhancements for Backtesting:**
    *   **Module 2 (Prediction):** Add Wavelet transforms, DL Review models (Transformer), xLSTM-TS, GNN, MTL, and agentic enhancements (Dynamic Teams, CEO Agent, Self-Refinement) to the backtest evaluation.
    *   **Module 3 (Risk):** Add Lasso selection/pruning (`coef_ < 0.1`) as a specific step.
    *   **Module 4 (Recommendation):** Use `TimeSeriesSplit` for sliding window validation.
    *   **Module 4 (Evaluation):** Add Combinatorial Purged Cross-Validation (CP-CV) and sensitivity analysis for robust evaluation.
*   **Output:** The output format should be a summary of backtesting results, including performance metrics (MAE, Sharpe, r^2, temporal_mae, PBO, DSR) for each period and scenario.

---

## Credit

This project was created and is maintained by **Torisan Unya** ([@torisan_unya](https://twitter.com/torisan_unya)).

## License

This project is licensed under the [Creative Commons Attribution-ShareAlike 4.0 International License](http://creativecommons.org/licenses/by-sa/4.0/).

[![CC BY-SA 4.0][cc-by-sa-shield]][cc-by-sa]

[cc-by-sa]: http://creativecommons.org/licenses/by-sa/4.0/
[cc-by-sa-shield]: https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg
