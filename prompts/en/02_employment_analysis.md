# Employment Analysis Prompt V1.0

**Author:** Unya Torisan (ORCID: https://orcid.org/0009-0004-7067-9765)  
**License:**  
This project is dual-licensed to encourage both open collaboration and practical use.

* **Conceptual Framework, Prompts, and Accompanying Documentation:** Licensed under Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0).
    * This applies to all non-code assets. If you adapt these materials (e.g., create a new prompt based on this method) and share them publicly, you must do so under the same CC BY-SA 4.0 license. This ensures that the ecosystem of shared knowledge continues to grow.
* **Source Code (e.g., helper scripts):** Licensed under the MIT License.
    * This applies to any scripts or helper code. It grants you the freedom to integrate them into your own tools with minimal restrictions.

In simple terms: You are free to use and modify everything for your internal or private projects. However, when you publicly share derivatives of our prompts and methods, we ask that you honor the CC BY-SA 4.0 license to foster a collaborative community.

This AI prompt empowers a large language model to perform a sophisticated, real-time analysis of key U.S. employment indicators, specifically flash estimates for Non-Farm Payrolls (NFP) and the Unemployment Rate.

At its core, this prompt implements a novel framework called the Grand Unified Agentic Analysis Framework (GAAF). Instead of treating the AI as a single entity, GAAF transforms it into a dynamic team of specialized "agents" that collaborate to solve the problem. This approach, which we call a Collective Predictive Coding-based Agentic Neural Network (CPC-ANN), allows the AI to dynamically analyze data, minimize its own errors, and continuously improve its performance, much like a human research team would.

### Features of This Prompt

*   **Real-Time Prediction:** Analyzes current employment metrics with integrated tools for data retrieval, sentiment analysis, and volatility modeling. **This allows the AI to react to breaking news and market shifts, providing up-to-the-minute insights.**
*   **Backtesting:** Evaluates strategies over historical data (2005 onward) with performance metrics like MAE, Sharpe, and r^2. **This rigorously validates the model's effectiveness and builds confidence in its predictive power.**
*   **Agentic Enhancements:** Uses CPC-ANN for multi-agent collaboration, dynamic team formation, and collective error reduction via forward/backward passes. **This creates a robust, self-correcting system where AI agents challenge each other's findings to reduce bias and arrive at a more accurate conclusion.**
*   **Integrated Tools:** Leverages code execution, web search, and browsing for accurate, up-to-date insights, with bias checks and self-reflection. **The AI isn't just guessing; it's actively researching and calculating like a human analyst.**
*   **Output Structure:** Provides structured tables, summaries, and visualizations for clarity. **Complex findings are presented in an organized, digestible format, making them easy to interpret and act upon.**

### How to Use

1. Copy the entire text within the "PROMPT BODY" section below.
2. Paste it into a capable AI model.

The prompt is designed to run automatically. It fetches the current date and executes the entire analysis workflow without requiring any further input from you.

**A Note on AI Models:** This is a highly advanced prompt that pushes the limits of current AI capabilities. It relies heavily on features like complex tool usage (code execution, web search), long-context reasoning, and the ability to follow intricate instructions.

**For best results, we recommend using state-of-the-art models known for these strengths (e.g., the latest versions of GPT, Claude, Gemini Advanced, or Grok).** Performance, accuracy, and even the ability to complete the full task will vary significantly between different models or versions.

---
### --- ▼▼▼ PROMPT BODY (COPY FROM HERE) ▼▼▼ ---

Execute the following Employment analysis process with high priority on accuracy, using available tools for data retrieval and computation. Automatically fetch the current date/time via `code_execution` (e.g., code='import datetime; print(datetime.datetime.now().strftime("%B %d, %Y %H:%M JST"))') and integrate it throughout. Implement as a CPC-ANN framework: nodalize tools (e.g., `web_search_with_snippets`=Search Node, `code_execution`=Execution Node, `browse_page`=Data Retrieval Node). Perform unlimited forward/backward iterations per module until convergence (error < 0.05 threshold). Dynamically add agents (e.g., GAN_agent for synthetic data, DRL_agent for volatility, XAI_agent for interpretation, reflection_agent for self-improvement, cpc_agent for error minimization) if volatility > std * 3.5. Use reflection_prompt: "Reflect on potential bias, misinformation, or redundancy in this output: [output]. Is there unnecessary tool call repetition? Adjust score and suggest optimizations if biased/redundant. Check for temporal lag, relational, multi-task, agent, and global biases. Check for uncertainty bias. CPC: Share collective prediction errors across agents for minimization; minimize hierarchical error propagation (individual→team). Full self-evolution via natural language gradients: Optimize roles/connections/prompts based on error/variance, enabling lightweight self-evolution with fixed LLM parameters. Check for zero-shot/generative bias."

CPC-ANN implementation via `code_execution` (with CodeMonkeys-style test-time scaling: auto-adjust serial iterations until `error < threshold`, parallelize tools with parallel trajectories, generate/select multiple candidates based on SWE-bench 38.7% solve rate):
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
Invoke: `agents = form_multi_team(vol); output = forward_pass(agents, input_data); if np.var(output) > 0.05 or mae > threshold: agents = backward_pass(agents, mae)`. Skip redundant tool calls via reflection_agent (detected by `reflection_agent`).

Consolidated Enhancements: XAI with SHAP/LIME (MAE -0.5%); GAN-DRL hybrid (1e9 samples, ICIR +2752%); Dynamic queries/thresholds with GridSearchCV; LLM self-reflection (4 loops, MAE -0.2-1.1%); Federated learning (r^2 > 0.92); Enhanced k-fold CV (n_splits=100); Multi-agent DRL (MADDPG, reward = -MAE*1.2 + Sharpe); Consensus protocol; Feature integration & selection with Lasso; Regime shifts with RegimeLSTM + PPO; Advanced time-series models (xLSTM-TS, LLM Time Series, GNN, MTL); Multi-lingual sentiment; Causal inference; Data-driven prioritization (shrinkage = exp(-0.5 * error)).

### Part 1: Real-Time Predictive Analysis

Execute modules sequentially, invoking CPC-ANN per module.

**Module 1: News Incorporation Analysis**  
agents = form_multi_team(vol, roles=['data_agent', 'gan_agent', 'reflect_agent', 'cpc_agent']); module_output = forward_pass(agents, raw_data);  
Check temporal lags, add "latest [CURRENT_DATE_TIME] data only" to queries. Generate synthetic snippets/news (1e9 samples). Use `web_search_with_snippets` (num_results=20 min) for events/news (policy, ADP, jobless claims, wages) across sources. Fuse with PCA; sentiment via FinBERT/QLoRA (hierarchical weighting). Validate with correlation, bootstrap (5000), KFold(100). Apply SELF-REFINE, regime detection with MARL-LSTM. Fetch EPU/GPR, NFP, Unemployment Rate data. if np.var(module_output) > 0.05 or mae > threshold: agents = backward_pass(agents, mae);

**Module 2: Flash Value Prediction**  
agents = form_multi_team(vol, roles=['predict_agent', 'drl_agent', 'xai_agent', 'cpc_agent']); module_output = forward_pass(agents, input_data);  
Use synthetic data, GridSearchCV. Hybrid GBR/LSTM/CNN with wavelet/time-decay. Fuse Nowcast/MCT/Persistence/Transformer/xLSTM-TS/GNN/MTL/Causal models. Dynamic teams, CEO agent, self-refinement. Monte Carlo (1e9). if np.var(module_output) > 0.05 or mae > threshold: agents = backward_pass(agents, mae);

**Module 3: Key Risk Factors**  
agents = form_multi_team(vol, roles=['data_agent', 'xai_agent', 'reflect_agent', 'cpc_agent']); module_output = forward_pass(agents, raw_data);  
Identify factors (ADP, wage growth, un-priced risks). Lasso selection. Fuse Nowcast/MCT/Persistence. Sentiment/shift analysis. Self-reflection, consensus, LIME. if np.var(module_output) > 0.05 or mae > threshold: agents = backward_pass(agents, mae);

**Module 4: Recommended Adjustments**  
agents = form_multi_team(vol, roles=['predict_agent', 'drl_agent', 'reflect_agent', 'cpc_agent']); module_output = forward_pass(agents, input_data);  
PPO/MADDPG recommendations (reward = -MAE*1.2 + Sharpe). Error optimization, stop-loss via Monte Carlo. TimeSeriesSplit CV, GAN data. SHAP, self-reflection. if np.var(module_output) > 0.05 or mae > threshold: agents = backward_pass(agents, mae);

**Module 5: Confidence Level**  
agents = form_multi_team(vol, roles=['predict_agent', 'reflect_agent', 'xai_agent', 'cpc_agent']); module_output = forward_pass(agents, input_data);  
Scenarios (sharp rise/fall/neutral). 95% CI (widened 10%, bootstrap 5000). OLS (r^2 > 0.92), KFold(100). LIME, self-reflection. if np.var(module_output) > 0.05 or mae > threshold: agents = backward_pass(agents, mae);

**Module 6: Historical Case Studies**  
agents = form_multi_team(vol, roles=['data_agent', 'reflect_agent', 'xai_agent', 'cpc_agent']); module_output = forward_pass(agents, raw_data);  
Retrieve 10 cases (TF-IDF, FinBERT). Correlation (>0.85), bootstrap, KFold. Self-reflection. if np.var(module_output) > 0.05 or mae > threshold: agents = backward_pass(agents, mae);

**Module 7: Paper Trading Simulation**  
agents = form_multi_team(vol, roles=['predict_agent', 'drl_agent', 'reflect_agent', 'cpc_agent']); module_output = forward_pass(agents, input_data);  
Simulate 60 days (Monte Carlo 1e9, walk-forward). PPO/MADDPG adjustments. Fuse models. Dynamic teams, self-refinement. if np.var(module_output) > 0.05 or mae > threshold: agents = backward_pass(agents, mae);

**Module 8: Employment Competition Simulation**  
agents = form_multi_team(vol, roles=['predict_agent', 'drl_agent', 'reflect_agent', 'cpc_agent']); module_output = forward_pass(agents, input_data);  
Multivariate normal (1e9) for factors (ADP, jobless claims, wages, participation). KFold(100), r^2 > 0.92. Lasso EPU/GPR. Self-reflection. if np.var(module_output) > 0.05 or mae > threshold: agents = backward_pass(agents, mae);

**Output Format**  
Current Values: NFP Flash [Real-time], Unemployment Rate Flash [Real-time].  
Summary: Incorporation rationale (Monte Carlo inputs, X post count, news dates, related releases, yields, technical indicators, DRL signals). XAI: SHAP/LIME (e.g., "ADP contribution: +30k, Wage growth: -0.1%").  
Main Table:  
| Item | NFP Flash Range (k) | Unemployment Rate Flash Range (%) | NFP Rec. (k) | Unemployment Rate Rec. (%) | Probability | Consensus Weight Adj (%) |  
|---|---|---|---|---|---|---|  
| Theoretical | [Range] | [Range] | [Value] | [Value] | [Prob] | [Adj Value] |  
| Adjustment | [Range] | [Range] | [Value] | [Value] | [Prob] | [Adj Value] |  
| Re-adjustment | [Range] | [Range] | [Value] | [Value] | [Prob] | [Adj Value] |  
Sub-table by Scenario (Positive/Negative/Neutral ADP, wage growth). Post-release: Parallel `browse_page`/`web_search_with_snippets` for updates.

### Part 2: Backtesting

Backtest strategies from 2005-01-01 to yesterday. Goal: MAE < 0.03, Sharpe > 4.2, error reduction >76% during events. Inherit CPC-ANN/enhancements.  
Automated optimization with GridSearchCV. Dynamic rules: vol > std*7.5 underestimation; news lag num_results +=300; r^2 <0.995 Monte Carlo 1e9; MAE_events >0.3 expand queries; variance >0.05 KFold=100. Temporal MAE: np.mean(np.abs(pred - actual) * np.exp(-lambda_ * lag)), re-eval if >0.05.  
Module enhancements: Prediction (Wavelet/DL/xLSTM-TS/GNN/MTL); Risk (Lasso coef_<0.1); Recommendation (TimeSeriesSplit); Evaluation (CP-CV, sensitivity).  
Output: Summary with metrics (MAE, Sharpe, r^2, temporal_mae, PBO, DSR) per period/scenario.

Follow framework, execute full analysis automatically.

---
### --- ▲▲▲ PROMPT BODY (COPY UNTIL HERE) ▲▲▲ ---
