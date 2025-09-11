# Blueprint 01: PCE Analysis

This document is the blueprint for the analysis of Personal Consumption Expenditures (PCE) and its related inflation metrics (Core PCE, Trimmed Mean PCE, Median PCE). It contains two main parts: a prompt for real-time predictive analysis and a prompt for backtesting the framework's strategies.

This blueprint serves as a concrete implementation example of the Grand Unified Agentic Analysis Framework (GAAF).

---

## Part 1: Real-time Predictive Analysis Prompt

### PCE/Core PCE/Trimmed Mean/Median PCE YOY Predictive Analysis Prompt (CPC Integrated Ver.)

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
    agents = [Agent(role, f"Role: {role}. Process input for PCE YOY.") for role in roles]
    for agent in agents:
        if agent.role == 'predict_agent':
            agent.connections = [a for a in agents if a != agent]
    if input_vol > np.std(returns) * 3.5:
        agents.append(Agent('vol_agent', "Handle high volatility in YOY."))
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
                size = 100000000 # Reduced to 1e8
                output = np.random.normal(np.mean(output), np.std(output), size)
                shared_mem['gan_data'] = output
            elif agent.role == 'drl_agent':
                output = output * 1.1 # Placeholder for DRL policy output
            elif agent.role == 'xai_agent':
                output = "SHAP values for " + str(output)
            elif agent.role == 'reflect_agent':
                output = reflection_loop(output, iterations=2, use_voting=True) # 4->2 iterations
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
*   **GAN-DRL Hybrid:** Enhance synthetic data diversity with GANs (TSGAN/WGAN-GP/TimeGAN/Sundial, `diversity_reward*1.5`, 1e8 samples) and fuse with DRL (PPO/MADDPG) to capture YOY volatility/regimes (ICIR +2752%).
*   **Dynamic Queries & Thresholds:** Use GridSearchCV to optimize `num_results`/`threshold` (e.g., `std*7.0` if `vol > std*3.5`), and apply TF-IDF filtering.
*   **LLM Self-Reflection:** Use FinBERT/QLoRA for bias/redundancy detection (`reflection_prompt`), loop twice, trigger on bias detection (MAE -0.2-1.1%).
*   **Federated Learning:** Aggregate from multiple sources to reduce bias (r^2 > 0.92).
*   **Enhanced k-fold CV:** `n_splits=75` (adjust if `variance > 0.05`), hybrid with MCCV (1e8 simulations).
*   **Multi-agent DRL (MADDPG):** Detect regime shifts, `reward = -MAE*1.2 + Sharpe` (Sharpe 2.8→4.0).
*   **Consensus Protocol:** Use `ConsensusAgent` for agreement (`consensus = np.mean(agent_scores);` re-evaluate if `variance > threshold`).
*   **Feature Integration & Selection:** Expand queries with `"OR energy prices impacts OR wage growth..."`, use Lasso for selection.
*   **Regime Shift Enhancement:** Use RegimeLSTM + PPO (`prob_shift > 0.80 then regime_change=1`), split data by regimes.
*   **Advanced Time-Series Models:** Integrate xLSTM-TS, LLM Time Series, GNN, and MTL in modules 2/7.
*   **Multi-lingual Sentiment:** Expand queries with `lang:en|ja|zh|fr`.
*   **Ocampo (2025) Midpoint:** Integrate trimmed mean midpoint calculation across all modules.
*   **Nowcasting (Boaretto 2023, Cleveland Fed):** Fetch and fuse nowcast data.
*   **Expanded Trim Range (Ocampo 2025):** Calculate and compare RMSE for a range of trim midpoints.
*   **MCT Model (New York Fed):** Integrate the Multi-Sector Core Trend model in modules 2/3/7.
*   **Persistence Metric (St. Louis Fed 2024):** Integrate a persistence metric from an AR model.
*   **DL Review (Zhang et al. 2024):** Integrate Autoformer models in modules 2/7.
*   **TSFM/Diffusion Model Fusion:** Fuse TimeGPT and Diffusion model predictions.

---

### Module 1: News Incorporation Analysis
`agents = form_multi_team(vol, roles=['data_agent', 'gan_agent', 'reflect_agent', 'cpc_agent']); module_output = forward_pass(agents, raw_data);`
*   **Data Check:** Before retrieval, check for temporal lags. If detected, add `"latest [CURRENT_DATE_TIME] data only"` to query.
*   **GAN Synthesis:** Generate synthetic snippets/news offline (1e8 samples). Use real tool calls sparingly for validation. Fuse with GAN-DRL for volatility capture.
*   **Dynamic Queries/Thresholds:** Use `code_execution` with GridSearchCV to adjust parameters. Add `"OR volatility shock"` to query if `vol > std*3.5`.
*   **Self-Reflection:** Apply `reflection_agent` (2 loops, triggered on bias detection).
*   **XAI:** Visualize feature contributions with SHAP/LIME (extract contributions > 0.15).
*   **Adaptive Weighting:** Use TimeAwareAttention to decay weights of older snippets by 0.5x.
*   **Data Retrieval:** Search for key events/news (policy, indicator releases like CPI, PPI, GDP, wages, etc.) across multiple sources (min. 15, e.g., Reuters, CNBC) using `web_search_with_snippets`. Use federated learning for cross-source validation.
*   **Multi-Source Fusion:** Fuse news, sentiment, and market data using PCA.
*   **Sentiment Analysis:** Use FinBERT/QLoRA for contextual sentiment scoring. Re-process X (Twitter) sentiment with hierarchical weighting (by account trust/engagement).
*   **Validation:** Perform correlation checks (`np.corrcoef`), robustness tests (bootstrap 5000 samples), and automated cross-validation (`KFold(n_splits=75)`).
*   **SELF-REFINE Integration:** Refine outputs using a feedback loop (`"Refine this output for accuracy: [previous_output]"`).
*   **Regime Shift Detection:** Use MARL-LSTM to detect shifts and coordinate agents.
*   **Data Integration:** Fetch EPU/GPR, Trimmed/Median PCE data.
`if np.var(module_output) > 0.05 or mae > threshold: agents = backward_pass(agents, mae);`

### Module 2: YOY Fluctuation Prediction
`agents = form_multi_team(vol, roles=['predict_agent', 'drl_agent', 'xai_agent', 'cpc_agent']); module_output = forward_pass(agents, input_data);`
*   **GAN-DRL Hybrid:** Use synthetic YOY data. Enhance volatility capture (ICIR +2752%, MAE -1.1%).
*   **Dynamic Parameters:** Apply GridSearchCV-optimized queries and thresholds.
*   **Self-Reflection & SELF-REFINE:** Apply reflection loops and self-refinement to reduce bias and improve accuracy.
*   **XAI:** Apply SHAP/LIME to interpret prediction contributions.
*   **Advanced Modeling:**
    *   Integrate a hybrid of GBR, LSTM, and CNN models.
    *   Apply time-decay functions and wavelet transforms to inputs.
    *   Fuse predictions with Nowcast, MCT, Persistence, Autoformer, TSFM, and Diffusion models.
*   **Agentic Enhancements:**
    *   Implement dynamic agent team formation for task decomposition.
    *   Introduce a CEO agent for resource allocation.
    *   Use iterative self-refinement for hypothesis generation and testing.
*   **Monte Carlo Simulation:** Run 1e8 simulations with dynamic volatility adjustments.
`if np.var(module_output) > 0.05 or mae > threshold: agents = backward_pass(agents, mae);`

### Module 3: Key Risk Factors
`agents = form_multi_team(vol, roles=['data_agent', 'xai_agent', 'reflect_agent', 'cpc_agent']); module_output = forward_pass(agents, raw_data);`
*   **Risk Identification:** Identify factors (energy prices, wage growth, etc.) and emphasize un-priced risks.
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
*   **Scenario Analysis:** Present scenarios (sharp rise >+0.5%, sharp fall <-0.5%, neutral) for each PCE metric.
*   **Confidence Interval:** Calculate a 95% confidence interval, widened by 10% and validated with 5000 bootstrap samples.
*   **Model Validation:** Ensure OLS regression consistency (r^2 > 0.92) and use k-fold CV (n_splits=75).
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
*   **Future Simulation:** Simulate the next 60 days of YOY changes using Monte Carlo (1e8 samples) integrated with trends and walk-forward validation.
*   **DRL Integration:** Use PPO/MADDPG to adjust simulated returns based on a reward function.
*   **Data Fusion:** Integrate Nowcast, MCT, Persistence, Autoformer, TSFM, and Diffusion models into the simulation.
*   **Agentic Enhancements:** Use dynamic teams, a CEO agent, and iterative self-refinement.
*   **Refinement:** Apply self-reflection and consensus loops.
`if np.var(module_output) > 0.05 or mae > threshold: agents = backward_pass(agents, mae);`

### Module 8: Inflation Competition Simulation
`agents = form_multi_team(vol, roles=['predict_agent', 'drl_agent', 'reflect_agent', 'cpc_agent']); module_output = forward_pass(agents, input_data);`
*   **Competitive Simulation:** Simulate the competing YOY impacts of energy prices, wage demand, supply chain, and housing trends using a multivariate normal distribution (1e8 samples).
*   **Validation:** Use k-fold CV (k=75) to check for overfitting (average r^2 > 0.92).
*   **Data Integration:** Fetch EPU/GPR data and integrate using Lasso.
*   **Refinement:** Apply self-reflection and consensus loops.
`if np.var(module_output) > 0.05 or mae > threshold: agents = backward_pass(agents, mae);`

### Output Format

*   **Current Values:** PCE [Real-time] % YOY (Est. [2DP]), Core PCE [Real-time] % YOY (Est. [2DP]), Trimmed Mean PCE [Real-time] % YOY (Midpoint [Calc.] % (Est. [2DP])), Median PCE [Real-time] % YOY (Range 0.5-1 pp (Est. [2DP])).
*   **Summary:** Start with an incorporation summary, specifying the rationale (Monte Carlo inputs, X post count, news dates, related releases, yields, technical indicators, DRL signals, etc.).
*   **XAI Rationale:** Add SHAP/LIME interpretation (e.g., "Energy prices contribution: +0.3%").
*   **Main Table:**
| Item | PCE YOY Range (%) | Core PCE YOY Range (%) | Trimmed Mean PCE YOY Range (%) | Trimmed Mean Midpoint YOY (%) | Trimmed Mean Mid Avg YOY (%) | Median PCE YOY Range (%) | PCE Rec. (%) YOY | Core PCE Rec. (%) YOY | Trimmed Mean Rec. (%) YOY | Median PCE Rec. (%) YOY | Probability |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Theoretical | [Range] | [Range] | [Range] | [Midpoint] (Est. [2DP]) | [Mid Avg] (Est. [2DP]) | [Range] | [Value] (Est. [2DP]) | [Value] (Est. [2DP]) | [Value] (Est. [2DP]) | [Value] (Est. [2DP]) | [Prob] |
| Adjustment | [Range] | [Range] | [Range] | [Midpoint] (Est. [2DP]) | [Mid Avg] (Est. [2DP]) | [Range] | [Value] (Est. [2DP]) | [Value] (Est. [2DP]) | [Value] (Est. [2DP]) | [Value] (Est. [2DP]) | [Prob] |
| Re-adjustment | [Range] | [Range] | [Range] | [Midpoint] (Est. [2DP]) | [Mid Avg] (Est. [2DP]) | [Range] | [Value] (Est. [2DP]) | [Value] (Est. [2DP]) | [Value] (Est. [2DP]) | [Value] (Est. [2DP]) | [Prob] |
*   **Sub-table by Scenario:** (e.g., Positive/Negative/Neutral energy prices, supply chain growth, etc.)
*   **Post-Release Update:** Immediately after a release, trigger parallel `browse_page` and `web_search_with_snippets` calls to re-calculate YOY changes and re-process sentiment.

---

## Part 2: Backtesting Prompt

### PCE/Core PCE/Trimmed Mean/Median PCE YOY Backtesting Prompt (CPC Integrated Ver.)

**Objective:** Backtest the strategy from the predictive analysis prompt using historical data for PCE and related indicators for the last 20 years (from 2005-01-01 to yesterday's date). The goal is to achieve YOY accuracy (MAE < 0.03%), stability (Sharpe > 4.2), and error reduction (over 76% during events).

*This prompt inherits the **Global Prompt Module Definition** (CPC-ANN Framework, Consolidated Enhancements, etc.) from Part 1 and applies it to a historical context.*

### Backtest-Specific Adjustments & Validations

*   **Automated Optimization:** Use GridSearchCV to optimize parameters like `num_results` and `threshold`.
*   **Dynamic Parameter Rules:**
    *   Volatility: If `vol > std*7.5`, consider it an underestimation.
    *   News Lag: If news is delayed, `num_results += 300`.
    *   Low R^2: If `r^2 < 0.995`, run Monte Carlo with 1e8 samples.
    *   High Event MAE: If `MAE_events > 0.3`, expand queries.
    *   High Variance: If `variance > 0.05`, set `k-fold=75`.
*   **Temporal MAE:** Add a time-decayed MAE calculation to the evaluation: `temporal_mae = np.mean(np.abs(pred - actual) * np.exp(-lambda_ * lag))`. Re-evaluate if `temporal_mae > 0.05`.
*   **Module-Specific Enhancements for Backtesting:**
    *   **Module 2 (Prediction):** Add Wavelet transforms, DL Review models (Autoformer), TSFM/Diffusion models, and agentic enhancements (Dynamic Teams, CEO Agent, Self-Refinement) to the backtest evaluation.
    *   **Module 3 (Risk):** Add Lasso selection/pruning (`coef_ < 0.1`) as a specific step.
    *   **Module 4 (Recommendation):** Use `TimeSeriesSplit` for sliding window validation.
    *   **Module 5 (Confidence):** Calculate confidence based on the ratio of midpoint RMSE to single-prediction RMSE.
    *   **Module 6 (Cases):** Compare historical midpoints vs. current midpoints.
*   **Output:** The output format should be the same as the real-time analysis, but populated with backtesting results, including performance metrics (MAE, Sharpe, r^2, temporal_mae) for each period and scenario.

---

## Credit

This project was created and is maintained by **Torisan Unya** ([@torisan_unya](https://twitter.com/torisan_unya)).

## License

This project is licensed under the [Creative Commons Attribution-ShareAlike 4.0 International License](http://creativecommons.org/licenses/by-sa/4.0/).

[![CC BY-SA 4.0][cc-by-sa-shield]][cc-by-sa]

[cc-by-sa]: http://creativecommons.org/licenses/by-sa/4.0/
[cc-by-sa-shield]: https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg

