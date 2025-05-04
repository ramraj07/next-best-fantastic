Next Best Fantastic

This is a Streamlit application powered by Anthropic Claude 3.5 Sonnet that performs end-to-end analysis of a scientific paper using a pipeline of intelligent agents. It critically evaluates, generates hypotheses, debates them, and identifies the most promising direction for future research.

⸻

How It Works

The user provides the full text of a scientific paper. The system launches a sequence of autonomous agents, each responsible for a different aspect of reasoning.

The final output is a complete audit trail of reasoning and the winning hypothesis.

⸻

🖼️ Workflow Diagram
```mermaid
    flowchart TD
    A[User Pastes Paper and Clicks Analyze] --> B[Agent 1: Critical Evaluation]
    B --> B1[Low Skepticism]
    B --> B2[Neutral Skepticism]
    B --> B3[High Skepticism]
    B1 --> C[Agent 2: Objective Summary]
    B2 --> C
    B3 --> C

    C --> D[Agent 3: Identify Future Directions]
    D --> E1[Direction 1]
    D --> E2[Direction 2]
    D --> E3[...Direction N]

    subgraph Hypothesis Loop
        E1 --> F1[Agent 4: Mature Hypothesis]
        F1 --> G1[Agent 6: Generate 10 Criticisms]
        subgraph DLC[Debate Loop per criticism]
            direction RL
            FOR[Argue against]
            AGA[Argue for criticism]
            FOR <--3X--> AGA
        end 
        G1 --> DLC
        DLC --> I1[Summarize Debates]
    end

    E2 --> F2
    F2 --> G2
    G2 --> H2
    H2 --> I2

    E3 --> F3
    F3 --> G3
    G3 --> H3
    H3 --> I3

    I1 --> J[Agent 8: Final Judgement]
    I2 --> J
    I3 --> J

    J --> K[Display Best Hypothesis & Reasoning]
```
