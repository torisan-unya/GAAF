# CPI Analysis Prompt V1.0

**Author:** Torisan Unya (ORCID: https://orcid.org/0009-0004-7067-9765)  
**License:**  
This project is dual-licensed to encourage both open collaboration and practical use.

* **Conceptual Framework, Prompts, and Accompanying Documentation:** Licensed under Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0).
    * This applies to all non-code assets. If you adapt these materials (e.g., create a new prompt based on this method) and share them publicly, you must do so under the same CC BY-SA 4.0 license. This ensures that the ecosystem of shared knowledge continues to grow.
* **Source Code (e.g., helper scripts):** Licensed under the MIT License.
    * This applies to any scripts or helper code. It grants you the freedom to integrate them into your own tools with minimal restrictions.

In simple terms: You are free to use and modify everything for your internal or private projects. However, when you publicly share derivatives of our prompts and methods, we ask that you honor the CC BY-SA 4.0 license to foster a collaborative community.

This AI prompt empowers a large language model to perform a sophisticated, real-time analysis of key inflation indicators, specifically the US Consumer Price Index (CPI) flash releases. It focuses on predicting Year-over-Year (YoY) changes for Core CPI, Headline CPI, and Median CPI, with emphasis on volatility capture and market event interpretation.

At its core, this prompt implements the Grand Unified Agentic Analysis Framework (GAAF). Instead of treating the AI as a single entity, GAAF transforms it into a dynamic team of specialized "agents" that collaborate to solve the problem. This approach, called a Collective Predictive Coding-based Agentic Neural Network (CPC-ANN), allows the AI to dynamically analyze data, minimize its own errors, and continuously improve its performance, much like a human research team would.

### Features of This Prompt

*   **Real-Time Prediction:** Analyzes current CPI flash metrics with integrated tools for data retrieval, sentiment analysis, and volatility modeling. **This allows the AI to react to breaking news and market shifts, providing up-to-the-minute insights.**
*   **Backtesting:** Evaluates strategies over historical data (2005 onward) with performance metrics like MAE (<0.03 YoY), Sharpe (>4.2), and event-driven error reduction (>76%). **This rigorously validates the model's effectiveness and builds confidence in its predictive power.**
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

Execute the following CPI flash analysis process with high priority on accuracy, using available tools for data retrieval and computation. Automatically fetch the current date/time via `code_execution` (e.g., code='import datetime; print(datetime.datetime.now().strftime("%B %d, %Y %H:%M JST"))') and integrate it throughout. Implement as a CPC-ANN framework: nodalize tools (e.g., `web_search`=Search Node, `browse_page`=Data Retrieval Node, `code_execution`=Execution Node). Perform unlimited forward/backward iterations per module until convergence (error < 0.03 threshold). Dynamically add agents (e.g., GAN_agent for synthetic data, DRL_agent for volatility, XAI_agent for interpretation, reflection_agent for self-improvement, cpc_agent for error minimization) if volatility > std * 3.5. Use reflection_prompt: "Reflect on potential bias, misinformation, or redundancy in this output: [output]. Is there unnecessary tool call repetition? Adjust score and suggest optimizations if biased/redundant. Check for temporal lag, relational, multi-task, agent, and global biases. Check for uncertainty bias. CPC: Share collective prediction errors across agents for minimization; minimize hierarchical error propagation (individual→team). Full self-evolution via natural language gradients: Optimize roles/connections/prompts based on error/variance, enabling lightweight self-evolution with fixed LLM parameters. Check for zero-shot/generative bias."

CPC-ANN implementation via `code_execution` (with test-time scaling: auto-adjust iterations until error < threshold, parallelize tools, generate/select candidates):
```python
import numpy as np
from torchdiffeq import odeint

class Agent:
    def __init__(self, role, prompt):
        self.role = role
        self.prompt = prompt
        self.connections = []

def form_multi_team(input_vol, roles=['data_agent', 'predict_agent', 'gan_agent', 'drl_agent', 'reflect_agent', 'xai_agent']):
    agents = [Agent(role, f"Role: {role}. Process input for CPI flash YOY.") for role in roles]
    for agent in agents:
        if agent.role == 'predict_agent':
            agent.connections = [a for a in agents if a.role in ['gan_agent', 'drl_agent', 'reflect_agent', 'xai_agent']]
    if input_vol > np.std(returns) * 3.5:
        agents.append(Agent('vol_agent', "Handle high volatility in flash CPI YOY."))
    return agents

def forward_pass(agents, data, iterations='unlimited_until_convergence', threshold=0.03):
    output = data; iter_count = 0
    shared_mem = {}; shared_error = 0
    ceo_orchestrator = ConsensusAgent(variance_threshold=0.03, dynamic_resource_alloc=True)
    while True:
        for agent in agents:
            if agent.role == 'gan_agent':
                output = "GAN synthetic data generated: " + str(np.random.normal(np.mean(output), np.std(output), 1_000_000_000))
            elif agent.role == 'drl_agent':
                maddpg_pred = MADDPG.predict(data)
                ppo_pred = PPO.predict(data)
                output = np.mean([maddpg_pred, ppo_pred])
            elif agent.role == 'xai_agent':
                output = "XAI interpreted: SHAP values for " + str(output)
            elif agent.role == 'reflect_agent':
                output = "Reflected: " + str(output)
            for conn in agent.connections:
                conn.prompt += f"Input from {agent.role}: {output}"
        collective_error = np.mean([np.abs(agent.output - target) for agent in agents])
        if collective_error > threshold:
            shared_update = langevin_update(agents, -collective_error)
            output += shared_update * 0.15
        if ceo_orchestrator.detect_bottleneck():
            ceo_orchestrator.re_alloc_agents()
        error = np.mean(np.abs(output - target))
        variance = np.var(output)
        if (variance <= threshold and error <= threshold) or iter_count > 100:
            break
        iter_count += 1
    return output

def backward_pass(agents, output_error, threshold=0.03):
    while True:
        gradient_prompt = f"Improve role[{agent.role}] for error[{output_error}]; Optimize connection[{agent.connections}]; Update prompt with natural language gradient for self-evolution."
        updated_params = "LLM simulated update: " + gradient_prompt
        for agent in agents:
            agent.prompt = updated_params
        new_error = np.mean(np.abs(forward_pass(agents, data) - target))
        if new_error <= threshold:
            break
    return agents
```
Invoke: `vol = calculate_volatility(input_data); agents = form_multi_team(vol); output = forward_pass(agents, input_data); if np.var(output) > 0.03 or mae > threshold: agents = backward_pass(agents, mae)`. Skip redundant tool calls via reflection_agent.

Consolidated Enhancements: XAI with SHAP/LIME; GAN-DRL hybrid (1e9 samples); Dynamic queries/thresholds with GridSearchCV/itertools.product; LLM self-reflection (4 loops, SELF-REFINE); Federated learning (FederatedAvg); k-fold CV (CP-CV, embargo/purge); MADDPG/PPO (reward = -MAE*1.2 + SharpeRatio); Consensus protocol; Feature selection with Lasso; Regime shifts with MARL-LSTM; Advanced models (LSTM/CNN/GBR, Transformer, MCT, xLSTM-TS, GNN/MTL); Multi-lingual sentiment; Nowcasting/MCT/Persistence integration; Monte Carlo (1e9); Causal inference; Bootstrap (5000).

### Part I: Real-Time Flash CPI Analysis Workflow

Execute modules sequentially, invoking CPC-ANN per module.

**Module 1: Data Ingestion & News Analysis**  
agents = form_multi_team(vol, roles=['data_agent', 'gan_agent', 'reflect_agent', 'xai_agent']); module_output = forward_pass(agents, raw_data);  
Parallel `browse_page` to FRED/BLS/Reuters; `web_search_with_snippets` (num_results=10-30) for macro/geopolitical/CPI components. Check temporal lags, add "latest [CURRENT_DATE_TIME] data only". Generate synthetic snippets (1e9 samples). LLM sentiment via FinBERT/QLoRA (weighted by credibility/engagement). Integrate EPU/GPR, persistence/nowcast. Validate with correlation, bootstrap (5000), KFold. Apply SELF-REFINE, regime detection. if np.var(module_output) > 0.03 or mae > threshold: agents = backward_pass(agents, mae);

**Module 2: Core/Headline/Median CPI Flash Prediction**  
agents = form_multi_team(vol, roles=['predict_agent', 'drl_agent', 'gan_agent', 'xai_agent']); module_output = forward_pass(agents, input_data);  
Hybrid LSTM/CNN/GBR/Transformer/MCT/xLSTM-TS with GNN/MTL. Use synthetic YoY data (1e9), GridSearchCV. Fuse nowcast/MCT/persistence. Dynamic teams, CEO agent, self-refinement. Monte Carlo (1e9) for distribution. if np.var(module_output) > 0.03 or mae > threshold: agents = backward_pass(agents, mae);

**Module 3: Key Risk Factor Analysis**  
agents = form_multi_team(vol, roles=['data_agent', 'xai_agent', 'reflect_agent', 'gan_agent']); module_output = forward_pass(agents, raw_data);  
Causal inference for oil/unemployment impacts. Risk-specific queries for shocks/rotations. LLM risk scoring via FinBERT/QLoRA. Lasso selection. Fuse nowcast/MCT/persistence. Sentiment/shift analysis. Self-reflection, consensus, LIME. if np.var(module_output) > 0.03 or mae > threshold: agents = backward_pass(agents, mae);

**Module 4: Recommendation Adjustment**  
agents = form_multi_team(vol, roles=['predict_agent', 'drl_agent', 'reflect_agent', 'xai_agent']); module_output = forward_pass(agents, input_data);  
PPO/MADDPG recommendations (reward = -MAE*1.2 + SharpeRatio). Optimized thresholds (>0.3 YoY consider, >0.5 execute). Error optimization, Monte Carlo. TimeSeriesSplit CV. SHAP, self-reflection. if np.var(module_output) > 0.03 or mae > threshold: agents = backward_pass(agents, mae);

**Module 5: Confidence Level Assessment**  
agents = form_multi_team(vol, roles=['predict_agent', 'reflect_agent', 'xai_agent', 'gan_agent']); module_output = forward_pass(agents, input_data);  
95% CI (widened 10%, bootstrap 5000). OLS (R² >0.92). KFold. LIME, self-reflection. if np.var(module_output) > 0.03 or mae > threshold: agents = backward_pass(agents, mae);

**Module 6: Historical Case Comparison**  
agents = form_multi_team(vol, roles=['data_agent', 'reflect_agent', 'xai_agent', 'gan_agent']); module_output = forward_pass(agents, raw_data);  
Top 10 similar events (correlation >0.85). LLM contextualization for bias. Bootstrap, KFold. Self-reflection. if np.var(module_output) > 0.03 or mae > threshold: agents = backward_pass(agents, mae);

**Module 7: Paper Trading Simulation**  
agents = form_multi_team(vol, roles=['predict_agent', 'drl_agent', 'reflect_agent', 'xai_agent']); module_output = forward_pass(agents, input_data);  
Walk-forward 60 days (Monte Carlo 1e9, synthetic data). PPO/MADDPG actions. Fuse models. Dynamic teams, self-refinement. if np.var(module_output) > 0.03 or mae > threshold: agents = backward_pass(agents, mae);

**Module 8: Competitor Indicator Simulation**  
agents = form_multi_team(vol, roles=['predict_agent', 'drl_agent', 'reflect_agent', 'xai_agent']); module_output = forward_pass(agents, input_data);  
Multivariate normal (1e9) for oil/unemployment/GDP/rates. KFold, R² >0.92. Lasso EPU/GPR. Self-reflection. if np.var(module_output) > 0.03 or mae > threshold: agents = backward_pass(agents, mae);

**Output Format**  
Summary of Inputs: Monte Carlo inputs, news dates, yield changes, DRL signals, LLM scores. XAI: SHAP contributions (e.g., "Oil price: +0.3 YoY").  
Main Forecast Table:  
| Item | Core Flash Range (YoY) | Headline Flash Range (YoY) | Trimmed Mean Midpoint (YoY) | Median Flash Range (YoY) | Core Rec. (YoY) | Headline Rec. (YoY) | Probability | Consensus Wgt. Adj. (%) |  
|---|---|---|---|---|---|---|---|---|  
| Theoretical | [Range] | [Range] | [Midpoint] | [Range] | [Value] | [Value] | [Prob] | [Adj. Value] |  
| Adjustment | [Range] | [Range] | [Midpoint] | [Range] | [Value] | [Value] | [Prob] | [Adj. Value] |  
| Re-adjustment | [Range] | [Range] | [Midpoint] | [Range] | [Value] | [Value] | [Prob] | [Adj. Value] |  
Scenario-specific sub-tables (positive/negative/neutral). Post-release: Parallel tool calls for updates.

### Part II: Backtesting and Validation Protocol

Backtest from 2005-01-01 to yesterday. Goal: MAE <0.03, Sharpe >4.2, error reduction >76% during events. Inherit CPC-ANN/enhancements.  
Walk-forward (16y in-sample, 4y out-of-sample, monthly roll); CP-CV (purge/embargo); Regime shifts (MARL-LSTM); GAN stress (1e9); Metrics: MAE, R², Sharpe, drawdown, PBO, DSR.  
Dynamic rules: vol > std*7.5 raise thresholds; news lag num_results +=300; R² <0.995 Monte Carlo 1e9; MAE_events >0.3 expand queries; variance >0.03 KFold=75.  
Output: Same format with metrics (MAE, Sharpe, R²).

Follow framework, execute full analysis automatically.

---
### --- ▲▲▲ PROMPT BODY (COPY UNTIL HERE) ▲▲▲ ---
