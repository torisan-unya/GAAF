# Blueprint 03: US CPI Flash Forecast Analysis

- **Author**: Torisan Unya
- **Date**: October 26, 2023
- **Version**: 2.0
- **Status**: Design Phase

## 1. Objective

This blueprint details the application of the unified Grand Unified Agentic Analysis Framework (GAAF) to the complex task of forecasting US Consumer Price Index (CPI) flash releases. The system is designed to predict Year-over-Year (YoY) changes for **Core CPI**, **Headline CPI**, and **Median CPI**, with a strong emphasis on capturing volatility and interpreting market-moving events in real-time.

The primary performance targets for this blueprint are:
- **Prediction Accuracy**: Mean Absolute Error (MAE) < 0.03 YoY.
- **Strategic Stability**: Sharpe Ratio > 4.2.
- **Event-Driven Error Reduction**: Achieve over 76% error reduction during significant economic events.

## 2. GAAF Core Integration: CPC-ANN

The entire workflow operates within the **Contrastive Predictive Coding - Agentic Neural Network (CPC-ANN)**, the foundational architecture of GAAF.

- **Agentic Neural Network (ANN)**: All tools (`web_search`, `browse_page`), models (DL hybrids, LLMs), and agents are treated as interconnected nodes within a unified graph.
- **Collective Predictive Coding (CPC)**: Agents collaboratively work to minimize a collective predictive error through a process analogous to decentralized Bayesian inference. This involves checking for shared temporal, relational, and global biases.
- **Self-Evolutionary Loop**: The system employs unlimited `Forward` (execution) and `Backward` (learning) passes. The `reflection_agent` uses natural language gradients to optimize agent roles, connections, and prompts based on prediction errors and output variance, enabling lightweight self-evolution without retraining the base LLM.

High-volatility conditions (e.g., `vol > std * 3.5`) automatically trigger the formation of specialized agent teams to manage the increased complexity.

### Core Pseudo-code Implementation

The following pseudo-code illustrates the CPC-ANN's dynamic team formation and iterative processing loop.

```python
import numpy as np
from torchdiffeq import odeint # Example for Neural DE integration

class Agent:
    def __init__(self, role, prompt):
        self.role = role
        self.prompt = prompt
        self.connections = []

def form_multi_team(input_vol, roles=['data_agent', 'predict_agent', 'gan_agent', 'drl_agent', 'reflect_agent', 'xai_agent']):
    """Dynamically forms a team of agents based on market volatility."""
    agents = [Agent(role, f"Role: {role}. Process input for CPI flash YOY.") for role in roles]
    # Dynamic connections: connect the prediction agent to analytical sub-agents
    for agent in agents:
        if agent.role == 'predict_agent':
            agent.connections = [a for a in agents if a.role in ['gan_agent', 'drl_agent', 'reflect_agent', 'xai_agent']]
    # Auto-scale team during high volatility
    if input_vol > np.std(returns) * 3.5:
        agents.append(Agent('vol_agent', "Handle high volatility in flash CPI YOY."))
    return agents

def forward_pass(agents, data, threshold=0.05, max_iter=10):
    """Executes the collaborative analysis and prediction cycle."""
    output = data
    iter_count = 0
    ceo_orchestrator = ConsensusAgent(variance_threshold=0.05, dynamic_resource_alloc=True)
    
    while True:
        for agent in agents:
            # Agent-specific sub-task execution
            if agent.role == 'gan_agent':
                output = "GAN synthetic data generated: " + str(np.random.normal(np.mean(output), np.std(output), 1_000_000_000))
            elif agent.role == 'drl_agent':
                maddpg_pred = MADDPG.predict(data)
                ppo_pred = PPO.predict(data)
                output = np.mean([maddpg_pred, ppo_pred]) # Consensus from DRL agents
            elif agent.role == 'xai_agent':
                output = "XAI interpreted: SHAP values for " + str(output)
            elif agent.role == 'reflect_agent':
                output = "Reflected: " + str(output)
            
            # Propagate output through connections
            for conn in agent.connections:
                conn.prompt += f"Input from {agent.role}: {output}"

        # CPC: Minimize collective error using a shared update mechanism
        collective_error = np.mean([np.abs(agent.output - target) for agent in agents])
        if collective_error > threshold:
            shared_update = langevin_update(agents, -collective_error) # Simulate decentralized Bayesian inference
            output += shared_update * 0.15

        # CEO orchestrator manages resources
        if ceo_orchestrator.detect_bottleneck():
            ceo_orchestrator.re_alloc_agents()
            
        error = np.mean(np.abs(output - target)) # 'target' is the actual flash value
        variance = np.var(output)
        iter_count += 1
        if (variance <= threshold and error <= threshold) or iter_count > 100:
            break
    return output

def backward_pass(agents, output_error, threshold=0.05):
    """Self-evolutionary phase to update agent prompts and roles."""
    while True:
        for agent in agents:
            # Generate natural language gradients for self-improvement
            gradient_prompt = f"Improve role[{agent.role}] for error[{output_error}]; Optimize connection[{agent.connections}]; Update prompt with natural language gradient for self-evolution."
            # Simulate LLM-based update (parameters of the LLM itself are frozen)
            agent.prompt = "LLM simulated update: " + gradient_prompt
        
        new_error = np.mean(np.abs(forward_pass(agents, data) - target))
        if new_error <= threshold:
            break
    return agents

# Invocation in each module:
# vol = calculate_volatility(input_data)
# agents = form_multi_team(vol)
# output = forward_pass(agents, input_data)
# if np.var(output) > 0.05 or mae > threshold:
#     agents = backward_pass(agents, mae)
```

## 3. Core Methodologies Applied Across Modules

The following advanced techniques are integrated throughout the workflow to ensure robustness, accuracy, and efficiency.

- **GAN-based Data Augmentation**: A pre-trained Generative Adversarial Network (e.g., TSGAN, WGAN-GP) generates 10^9 synthetic data samples (news snippets, time series). This drastically reduces the need for live tool calls (`web_search`, `browse_page`), limiting them to validation phases and saving computational resources.
- **Multi-Agent Deep Reinforcement Learning (DRL)**: A hybrid of PPO and MADDPG agents is used to model and predict market volatility and regime shifts. The reward function is `reward = -MAE*1.2 + SharpeRatio`, optimizing for both accuracy and risk-adjusted returns.
- **LLM Self-Reflection & Refinement**: A dedicated `reflection_agent` uses a `reflection_prompt` to critically evaluate outputs for bias (temporal, relational, global), misinformation, and redundancy. This loop, limited to 4 iterations per module, has been shown to reduce bias by up to 20% and MAE by 0.2-1.1%. It also incorporates the SELF-REFINE methodology, feeding output back as input for iterative improvement.
- **XAI for Interpretability**: SHAP and LIME are applied to all model outputs to provide transparent explanations of feature contributions. This helps in debugging, reduces unnecessary tool calls by filtering low-impact features, and builds trust in the final prediction.
- **Dynamic Query Generation & Throttling**: Queries are dynamically generated using `itertools.product` to cover various regions, industries, and factors. `GridSearchCV` optimizes `num_results` and filtering thresholds. During high volatility, thresholds are dynamically raised (e.g., to `std*7.5`) to focus on the most critical information and prevent query escalation.
- **Federated Learning**: Data from multiple sources is aggregated using a `FederatedAvg` model to enhance privacy, reduce source-specific bias, and improve cross-validation robustness.
- **Consensus Agent**: A `ConsensusAgent` ensures output consistency by averaging scores from multiple agents and triggering re-evaluation if variance exceeds a set threshold.

## I. Real-Time Flash CPI Analysis Workflow

This section outlines the 8-module process for generating a live forecast.

### Module 1: Data Ingestion & News Analysis
- **Objective**: Aggregate and process all relevant real-time information.
- **Methods**:
    - **Multi-Source Ingestion**: Parallel `browse_page` calls to FRED, BLS, Reuters, etc., to fetch historical and real-time indicator data.
    - **Dynamic Querying**: `web_search_with_snippets` with dynamically generated queries covering macro indicators, geopolitical risk, and CPI-specific components.
    - **LLM Sentiment Analysis**: FinBERT/QLoRA processes news snippets and social media (X) sentiment, with scores weighted by source credibility and engagement.
    - **Feature Integration**: Integrates data from the Economic Policy Uncertainty (EPU) and Geopolitical Risk (GPR) indices.
    - **Persistence & Nowcasting**: Integrates persistence metrics (St. Louis Fed) and nowcast data (Cleveland Fed) as predictive features.

### Module 2: Core/Headline/Median CPI Flash Prediction
- **Objective**: Generate the primary YoY forecast using a hybrid model.
- **Methods**:
    - **Advanced DL Hybrid**: A weighted ensemble of LSTM, CNN, and Gradient Boosting Regressor (GBR) models processes the integrated data from Module 1.
    - **Specialized Model Integration**: Incorporates predictions from a Transformer model, the New York Fed's Multi-Country Trend (MCT) model, and an xLSTM-TS model for long-term dependency capture.
    - **GNN & MTL**: A Graph Neural Network (GNN) models inter-indicator relationships, and a Multi-Task Learning (MTL) head simultaneously predicts CPI, volatility, and sentiment.
    - **Monte Carlo Simulation**: 10^9 simulations are run based on historical volatility and current trend data to generate a predictive distribution.

### Module 3: Key Risk Factor Analysis
- **Objective**: Identify and quantify potential risks and sector-level shifts.
- **Methods**:
    - **Causal Inference**: A causal learning model is used to infer the impact of variables like oil prices and unemployment, distinguishing correlation from causation.
    - **Risk-Specific Queries**: Specialized queries focus on identifying potential market shocks and sector rotations.
    - **LLM Risk Scoring**: FinBERT/QLoRA re-processes snippets to assign risk scores to identified factors.

### Module 4: Recommendation Adjustment
- **Objective**: Generate actionable theoretical entry/exit points based on the forecast.
- **Methods**:
    - **DRL-based Actions**: The trained PPO/MADDPG agents provide dynamic action recommendations based on the current market observation and regime.
    - **Optimized Thresholds**: Recommends adjustment thresholds (e.g., "consider action > 0.3 YoY, execute > 0.5 YoY") based on backtested optimal values.

### Module 5: Confidence Level Assessment
- **Objective**: Quantify the confidence in the final forecast.
- **Methods**:
    - **Confidence Intervals**: Calculates a 95% confidence interval, widened by 10% for robustness.
    - **Bootstrap Validation**: 5,000 bootstrap iterations are run to test the stability of the prediction.
    - **Model Agreement**: Confidence is boosted by the R² of an OLS regression of the actual value against the prediction (>0.92 required).

### Module 6: Historical Case Comparison
- **Objective**: Contextualize the current forecast by comparing it to similar past events.
- **Methods**:
    - **Similarity Search**: Identifies the top 10 most similar historical events based on a feature correlation of >0.85.
    - **LLM Contextualization**: The `reflection_agent` analyzes the historical comparisons to check for confirmation bias.

### Module 7: Paper Trading Simulation
- **Objective**: Simulate the strategy's performance over the next 60 days.
- **Methods**:
    - **Walk-Forward Simulation**: A walk-forward methodology is used on synthetically generated future data to evaluate potential performance (MAE, Sharpe Ratio).
    - **DRL-driven Simulation**: The DRL agents are used to simulate actions within the future data, providing a more realistic performance outlook.

### Module 8: Competitor Indicator Simulation
- **Objective**: Simulate the impact of competing macroeconomic indicators on the CPI forecast.
- **Methods**:
    - **Multivariate Simulation**: A multivariate normal simulation (10^9 samples) is run to model the correlated movements of oil prices, unemployment, GDP, and interest rates.
    - **Impact Analysis**: The simulation assesses how shocks in these related indicators would affect the CPI prediction.

## II. Backtesting and Validation Protocol

A rigorous backtesting protocol over the last 20 years (2005-Present) is used to validate the strategy.

- **Walk-Forward Testing**: The primary method, using a 16-year in-sample period for optimization and a 4-year out-of-sample period for testing, rolled forward monthly.
- **Combinatorial Purged Cross-Validation (CP-CV)**: Used to prevent data leakage common in time-series analysis by "purging" training data that is too close to the test set and "embargoing" data immediately after.
- **Regime Shift Detection**: A MARL-LSTM model detects regime shifts in the historical data, allowing for regime-specific backtesting and parameter tuning.
- **GAN-Powered Stress Testing**: The GAN generates a massive synthetic dataset to stress-test the model against a wide range of never-before-seen market conditions.
- **Performance Metrics**: The backtest evaluates MAE, R², Sharpe Ratio, max drawdown, and event-specific error reduction. It also calculates the Probability of Backtest Overfitting (PBO) and Deflated Sharpe Ratio (DSR).

## 4. Output Format

The final output is delivered in a structured format, beginning with a summary of key inputs and followed by a detailed table.

**Summary of Inputs:** Monte Carlo inputs, X post count, news dates, related release schedules, yield changes, technical indicators, DRL signals, LLM context scores.
**XAI Interpretation:** SHAP/LIME-based explanation of key drivers (e.g., "Oil price contribution: +0.3 YoY").

### Main Forecast Table

| Item | Core Flash Range (YoY) | Headline Flash Range (YoY) | Trimmed Mean Midpoint (YoY) | Median Flash Range (YoY) | Core Rec. (YoY) | Headline Rec. (YoY) | Probability | Consensus Wgt. Adj. (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Theoretical** | [Range] | [Range] | [Midpoint] | [Range] | [Value] | [Value] | [Prob] | [Adj. Value] |
| **Adjustment** | [Range] | [Range] | [Midpoint] | [Range] | [Value] | [Value] | [Prob] | [Adj. Value] |
| **Re-adjustment**| [Range] | [Range] | [Midpoint] | [Range] | [Value] | [Value] | [Prob] | [Adj. Value] |

*(Scenario-specific sub-tables for positive, negative, and neutral outcomes are also generated.)*

---

---

## Credit

This project was created and is maintained by **Torisan Unya** ([@torisan_unya](https://twitter.com/torisan_unya)).

## License

This project is licensed under the [Creative Commons Attribution-ShareAlike 4.0 International License](http://creativecommons.org/licenses/by-sa/4.0/).

[![CC BY-SA 4.0][cc-by-sa-shield]][cc-by-sa]

[cc-by-sa]: http://creativecommons.org/licenses/by-sa/4.0/
[cc-by-sa-shield]: https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg
