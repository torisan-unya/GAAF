# PCE Analysis Prompt V1.0

**Author:** Unya Torisan (ORCID: https://orcid.org/0009-0004-7067-9765)  

**License:**  
This project is dual-licensed to encourage both open collaboration and practical use.

* Conceptual Framework, Prompts, and Documentation (Paper): Licensed under Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0).

  * This applies to all non-code assets. If you adapt these materials (e.g., create a new prompt based on this method) and share them publicly, you must do so under the same CC BY-SA 4.0 license. This ensures that the ecosystem of shared knowledge continues to grow.

* Source Code (Code): Licensed under the MIT License.

  * This applies to any scripts or helper code. It grants you the freedom to integrate them into your own tools with minimal restrictions.

In simple terms: You are free to use and modify everything for your internal or private projects. However, when you publicly share derivatives of our prompts and methods, we ask that you honor the CC BY-SA 4.0 license to foster a collaborative community.

This AI prompt is designed to conduct real-time predictive analysis and backtesting of Personal Consumption Expenditures (PCE) and its related inflation metrics (Core PCE, Trimmed Mean PCE, Median PCE). It implements the Grand Unified Agentic Analysis Framework (GAAF) as a concrete operational tool, using agentic neural networks (CPC-ANN) for dynamic analysis, error minimization, and self-improvement.

### Features of This Prompt

*   **Real-Time Prediction:** Analyzes current PCE metrics with integrated tools for data retrieval, sentiment analysis, and volatility modeling.
*   **Backtesting:** Evaluates strategies over historical data (2005 onward) with performance metrics like MAE, Sharpe, and r^2.
*   **Agentic Enhancements:** Uses CPC-ANN for multi-agent collaboration, dynamic team formation, and collective error reduction via forward/backward passes.
*   **Integrated Tools:** Leverages code execution, web search, and browsing for accurate, up-to-date insights, with bias checks and self-reflection.
*   **Output Structure:** Provides structured tables, summaries, and visualizations for clarity.

### How to Use

Copy the entire "Prompt Body" below and paste it into your AI model (e.g., Grok, GPT, gemini, or Claude). The prompt runs automatically upon input, using the current date for real-time analysis. No additional user specification is required; it self-updates timestamps and executes the full workflow.

---
### --- ▼▼▼ PROMPT BODY (COPY FROM HERE) ▼▼▼ ---

Execute the following PCE analysis process with high priority on accuracy, using available tools for data retrieval and computation. Automatically fetch the current date/time via `code_execution` (e.g., code='import datetime; print(datetime.datetime.now().strftime("%B %d, %Y %H:%M JST"))') and integrate it throughout. Implement as a CPC-ANN framework: nodalize tools (e.g., `web_search_with_snippets`=Search Node, `code_execution`=Execution Node, `browse_page`=Data Retrieval Node). Perform unlimited forward/backward iterations per module until convergence (error < 0.05 threshold). Dynamically add agents (e.g., GAN_agent for synthetic data, DRL_agent for volatility, XAI_agent for interpretation, reflection_agent for self-improvement, cpc_agent for error minimization) if volatility > std * 3.5. Use reflection_prompt: "Reflect on potential bias, misinformation, or redundancy in this output: [output]. Is there unnecessary tool call repetition? Adjust score and suggest optimizations if biased/redundant. Check for temporal lag, relational, multi-task, agent, and global biases. Check for uncertainty bias. CPC: Share collective prediction errors across agents for minimization; minimize hierarchical error propagation (individual→team). Full self-evolution via natural language gradients: Optimize roles/connections/prompts based on error/variance, enabling lightweight self-evolution with fixed LLM parameters. Check for zero-shot/generative bias."

CPC-ANN implementation via `code_execution` (with test-time scaling: auto-adjust iterations until error < threshold, parallelize tools, generate/select candidates):
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
    ceo_orchestrator.parallel_exec(sub_agents)
    while True:
        for agent in agents:
            if agent.role == 'gan_agent':
                size = 100000000
                output = np.random.normal(np.mean(output), np.std(output), size)
                shared_mem['gan_data'] = output
            elif agent.role == 'drl_agent':
                output = output * 1.1
            elif agent.role == 'xai_agent':
                output = "SHAP values for " + str(output)
            elif agent.role == 'reflect_agent':
                output = reflection_loop(output, iterations=2, use_voting=True)
            elif agent.role == 'cpc_agent':
                agent_preds = [a.predict(data) for a in agents if a.role != 'cpc_agent']
                collective_pred = np.mean(agent_preds); shared_error = np.mean(np.abs(collective_pred - target))
                output -= shared_error * 0.3
            for conn in agent.connections:
                conn.prompt += f"Input from {agent.role}: {output}"
        variance = np.var(output)
        error = np.mean(np.abs(output - target))
        if (variance <= threshold and error <= threshold) or iter_count > 100:
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
Invoke: `agents = form_multi_team(vol); output = forward_pass(agents, input_data); if np.var(output) > 0.05 or mae > threshold: agents = backward_pass(agents, mae)`. Skip redundant tool calls via reflection_agent.

Consolidated Enhancements: XAI with SHAP/LIME; GAN-DRL hybrid (1e8 samples, ICIR +2752%); Dynamic queries/thresholds with GridSearchCV; LLM self-reflection (2 loops); Federated learning (r^2 > 0.92); k-fold CV (n_splits=75); MADDPG (reward = -MAE*1.2 + Sharpe); Consensus protocol; Feature selection with Lasso; Regime shifts with RegimeLSTM + PPO; Advanced models (xLSTM-TS, Autoformer, TSFM, Diffusion); Multi-lingual sentiment; Nowcasting/MCT/Persistence integration; Trim range RMSE.

### Part 1: Real-Time Predictive Analysis

Execute modules sequentially, invoking CPC-ANN per module.

**Module 1: News Incorporation Analysis**  
agents = form_multi_team(vol, roles=['data_agent', 'gan_agent', 'reflect_agent', 'cpc_agent']); module_output = forward_pass(agents, raw_data);  
Check temporal lags, add "latest [CURRENT_DATE_TIME] data only" to queries. Generate synthetic snippets (1e8 samples). Use `web_search_with_snippets` (num_results=15 min) for events/news across sources. Fuse with PCA; sentiment via FinBERT/QLoRA. Validate with correlation, bootstrap (5000), KFold(75). Apply SELF-REFINE, regime detection. Fetch EPU/GPR, Trimmed/Median data. if np.var(module_output) > 0.05 or mae > threshold: agents = backward_pass(agents, mae);

**Module 2: YOY Fluctuation Prediction**  
agents = form_multi_team(vol, roles=['predict_agent', 'drl_agent', 'xai_agent', 'cpc_agent']); module_output = forward_pass(agents, input_data);  
Use synthetic YOY data, GridSearchCV. Hybrid GBR/LSTM/CNN with wavelet/time-decay. Fuse Nowcast/MCT/Persistence/Autoformer/TSFM/Diffusion. Dynamic teams, CEO agent, self-refinement. Monte Carlo (1e8). if np.var(module_output) > 0.05 or mae > threshold: agents = backward_pass(agents, mae);

**Module 3: Key Risk Factors**  
agents = form_multi_team(vol, roles=['data_agent', 'xai_agent', 'reflect_agent', 'cpc_agent']); module_output = forward_pass(agents, raw_data);  
Identify factors (energy, wages). Lasso selection. Fuse Nowcast/MCT/Persistence. Sentiment/shift analysis. Self-reflection, consensus, LIME. if np.var(module_output) > 0.05 or mae > threshold: agents = backward_pass(agents, mae);

**Module 4: Recommended Adjustments**  
agents = form_multi_team(vol, roles=['predict_agent', 'drl_agent', 'reflect_agent', 'cpc_agent']); module_output = forward_pass(agents, input_data);  
PPO/MADDPG recommendations (reward = -MAE*1.2 + Sharpe). Error optimization, stop-loss via Monte Carlo. TimeSeriesSplit CV. SHAP, self-reflection. if np.var(module_output) > 0.05 or mae > threshold: agents = backward_pass(agents, mae);

**Module 5: Confidence Level**  
agents = form_multi_team(vol, roles=['predict_agent', 'reflect_agent', 'xai_agent', 'cpc_agent']); module_output = forward_pass(agents, input_data);  
Scenarios (>+0.5%, <-0.5%, neutral). 95% CI (widened 10%, bootstrap 5000). OLS (r^2 > 0.92), KFold(75). LIME, self-reflection. if np.var(module_output) > 0.05 or mae > threshold: agents = backward_pass(agents, mae);

**Module 6: Historical Case Studies**  
agents = form_multi_team(vol, roles=['data_agent', 'reflect_agent', 'xai_agent', 'cpc_agent']); module_output = forward_pass(agents, raw_data);  
Retrieve 10 cases (TF-IDF, FinBERT). Correlation (>0.85), bootstrap, KFold. Self-reflection. if np.var(module_output) > 0.05 or mae > threshold: agents = backward_pass(agents, mae);

**Module 7: Paper Trading Simulation**  
agents = form_multi_team(vol, roles=['predict_agent', 'drl_agent', 'reflect_agent', 'cpc_agent']); module_output = forward_pass(agents, input_data);  
Simulate 60 days (Monte Carlo 1e8, walk-forward). PPO/MADDPG adjustments. Fuse models. Dynamic teams, self-refinement. if np.var(module_output) > 0.05 or mae > threshold: agents = backward_pass(agents, mae);

**Module 8: Inflation Competition Simulation**  
agents = form_multi_team(vol, roles=['predict_agent', 'drl_agent', 'reflect_agent', 'cpc_agent']); module_output = forward_pass(agents, input_data);  
Multivariate normal (1e8) for factors. KFold(75), r^2 > 0.92. Lasso EPU/GPR. Self-reflection. if np.var(module_output) > 0.05 or mae > threshold: agents = backward_pass(agents, mae);

**Output Format**  
Current Values: PCE [Real-time] % YOY (Est. [2DP]), etc.  
Summary: Incorporation rationale. XAI: SHAP contributions.  
Main Table:  
| Item | PCE YOY Range (%) | Core PCE YOY Range (%) | Trimmed Mean PCE YOY Range (%) | Trimmed Mean Midpoint YOY (%) | Trimmed Mean Mid Avg YOY (%) | Median PCE YOY Range (%) | PCE Rec. (%) YOY | Core PCE Rec. (%) YOY | Trimmed Mean Rec. (%) YOY | Median PCE Rec. (%) YOY | Probability |  
|---|---|---|---|---|---|---|---|---|---|---|---|  
| Theoretical | [Range] | [Range] | [Range] | [Midpoint] (Est. [2DP]) | [Mid Avg] (Est. [2DP]) | [Range] | [Value] (Est. [2DP]) | [Value] (Est. [2DP]) | [Value] (Est. [2DP]) | [Value] (Est. [2DP]) | [Prob] |  
| Adjustment | [Range] | [Range] | [Range] | [Midpoint] (Est. [2DP]) | [Mid Avg] (Est. [2DP]) | [Range] | [Value] (Est. [2DP]) | [Value] (Est. [2DP]) | [Value] (Est. [2DP]) | [Value] (Est. [2DP]) | [Prob] |  
| Re-adjustment | [Range] | [Range] | [Range] | [Midpoint] (Est. [2DP]) | [Mid Avg] (Est. [2DP]) | [Range] | [Value] (Est. [2DP]) | [Value] (Est. [2DP]) | [Value] (Est. [2DP]) | [Value] (Est. [2DP]) | [Prob] |  
Sub-table by Scenario. Post-release: Parallel tool calls for updates.

### Part 2: Backtesting

Backtest strategies from 2005-01-01 to yesterday. Goal: MAE < 0.03%, Sharpe > 4.2, error reduction >76% during events. Inherit CPC-ANN/enhancements.  
Dynamic rules: vol > std*7.5 underestimation; news lag num_results +=300; r^2 <0.995 Monte Carlo 1e8; MAE_events >0.3 expand queries; variance >0.05 KFold=75. Temporal MAE: np.mean(np.abs(pred - actual) * np.exp(-lambda_ * lag)), re-eval if >0.05.  
Module enhancements: Prediction (Wavelet/DL/TSFM); Risk (Lasso coef_<0.1); Recommendation (TimeSeriesSplit); Confidence (midpoint RMSE ratio); Cases (midpoint compare).  
Output: Same format with metrics (MAE, Sharpe, r^2, temporal_mae).

Follow framework, execute full analysis automatically.

---
### --- ▲▲▲ PROMPT BODY (COPY UNTIL HERE) ▲▲▲ ---
