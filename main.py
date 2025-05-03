# streamlit_app.py
import streamlit as st
import anthropic  # Anthropic Python SDK
import os
import time
import re # For basic parsing

# --- Configuration ---
# Best practice: Use Streamlit secrets for API keys
# Create a .streamlit/secrets.toml file with:
# ANTHROPIC_API_KEY = "sk-ant-..."
# Or, use environment variables.
ANTHROPIC_API_KEY =  os.environ.get("ANTHROPIC_API_KEY")

# --- Constants ---
# Using Claude 3.5 Sonnet as a powerful and available model.
# Claude 3.7 is not yet released as of my last update.
MODEL_NAME = "claude-3-5-sonnet-20240620"
MAX_PAPER_LEN_FOR_PROMPT = 15000 # Truncate paper text in prompts to avoid excessive token usage

# --- Initialize Anthropic Client ---
client = None
if not ANTHROPIC_API_KEY:
    st.error("Anthropic API key not found. Please configure it using Streamlit secrets (recommended) or the ANTHROPIC_API_KEY environment variable.")
    st.stop()
else:
    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    except Exception as e:
        st.error(f"Failed to initialize Anthropic client: {e}")
        st.stop()

# --- Agent Functions ---

def call_claude(system_prompt, user_message, max_tokens=4000):
    """
    Helper function to call the Claude API.
    Handles API calls, basic error handling, and extracts text content.
    """
    if not client:
        st.error("Anthropic client not initialized.")
        return None

    try:
        # Ensure messages are in the correct format
        messages = [{"role": "user", "content": user_message}]

        # Make the API call
        message = client.messages.create(
            model=MODEL_NAME,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=messages
        )

        # Extract text content safely
        if message.content and isinstance(message.content, list) and len(message.content) > 0:
            # Check if the first block has text content
            if hasattr(message.content[0], 'text'):
                return message.content[0].text
            else:
                st.warning(f"Unexpected response structure from Claude API. First content block: {message.content[0]}")
                # Attempt to find a text block if the first isn't one
                for block in message.content:
                    if hasattr(block, 'text'):
                        return block.text
                st.error("No text block found in Claude API response content.")
                return f"Error: No text block found in response: {message.content}"
        else:
            st.warning("Received empty or unexpected content list from Claude API.")
            return "Error: Empty or unexpected response from Claude API."

    except anthropic.APIConnectionError as e:
        st.error(f"Anthropic API request failed to connect: {e}")
    except anthropic.RateLimitError as e:
        st.error(f"Anthropic API request hit rate limit: {e}. Please wait and try again.")
        # Consider adding a sleep/retry mechanism here if needed
    except anthropic.APIStatusError as e:
        st.error(f"Anthropic API returned an error status: {e.status_code} - {e.response}")
    except anthropic.BadRequestError as e:
         st.error(f"Anthropic API Bad Request Error (check inputs/prompts): {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred during Claude API call: {e}")

    return None # Indicate failure

def agent_1_evaluate(paper_text, skepticism_level):
    """Agent 1: Critically evaluates the paper based on skepticism level."""
    levels_desc = {
        "Low": "You are generally trusting of the paper's findings. Focus on its strengths, potential positive implications, and contributions, while acknowledging only minor or obvious limitations.",
        "Neutral": "Maintain a balanced and objective perspective. Assess both the strengths and weaknesses, methodology, evidence, and conclusions impartially. Avoid taking an overly positive or negative stance.",
        "High": "You are highly skeptical and actively seeking flaws. Focus intensely on inconsistencies, methodological weaknesses, unsupported claims, logical fallacies, potential biases, and alternative explanations. Challenge assertions rigorously."
    }
    if skepticism_level not in levels_desc:
        return "Error: Invalid skepticism level provided."

    system_prompt = f"""
You are Agent 1, a specialized assistant for critically evaluating scientific papers.
Your assigned skepticism level for this task is: **{skepticism_level}**.

**Your Task:**
Analyze the provided paper text thoroughly based *only* on your assigned skepticism level: {levels_desc[skepticism_level]}

**Evaluation Criteria (Consider these through your skepticism lens):**
* **Research Question/Objective:** Clarity, significance, focus.
* **Literature Review:** Comprehensiveness, relevance, critical appraisal (if applicable).
* **Methodology:** Soundness, appropriateness, detailed description, reproducibility, controls, sample size, potential biases.
* **Data Presentation & Analysis:** Clarity, rigor, statistical validity, appropriate techniques.
* **Results:** Clear presentation, direct relation to research question, support from data.
* **Discussion:** Coherent interpretation, comparison with existing literature, limitations acknowledged adequately (or inadequately, from a high skepticism view).
* **Conclusion:** Validity based on results, justified claims, implications discussed appropriately.
* **Overall:** Logical flow, writing clarity, potential conflicts of interest.

**Output:**
Provide a detailed critical evaluation reflecting your specific skepticism level. Structure your points clearly. Do NOT simply summarize the paper; CRITIQUE it according to your assigned stance. Use Markdown for formatting.
"""
    user_message = f"Here is the paper text to evaluate:\n\n```text\n{paper_text}\n```"
    # print(f"--- Agent 1 Prompt (Skepticism: {skepticism_level}) ---\nSystem: {system_prompt}\nUser: {user_message[:200]}...\n---") # Debugging
    evaluation = call_claude(system_prompt, user_message, max_tokens=3000) # Allow sufficient tokens
    return evaluation

def agent_2_summarize(eval_low, eval_neutral, eval_high):
    """Agent 2: Creates an objective summary from the three evaluations."""
    system_prompt = """
You are Agent 2, an Objective Summarizer AI. You have received three distinct critical evaluations of the *same* scientific paper. Each evaluation was written from a specific skepticism standpoint: Low, Neutral, and High.

**Your Task:**
Synthesize these three evaluations into a single, objective summary that highlights the *range of critical perspectives* on the paper. Your goal is to represent the key points of agreement and disagreement found across the different evaluations regarding the paper's perceived strengths and weaknesses.

**Instructions:**
* Focus ONLY on the content of the provided evaluations.
* Do NOT inject your own opinion or evaluate the paper yourself.
* Do NOT make assumptions about the paper's actual quality or veracity. Your summary must reflect *only* what the evaluations state.
* Identify common themes (e.g., methodology, results interpretation, conclusions) and describe how the different skepticism levels addressed them.
* Highlight areas where evaluations converged (e.g., all noted a specific strength/weakness) and diverged (e.g., low skepticism saw a point as minor, high saw it as critical).
* Structure the summary logically and concisely. Use Markdown for formatting.
* Aim for clarity and neutrality in your language.
"""
    user_message = f"""
Here are the three critical evaluations:

**Evaluation (Low Skepticism):**
```text
{eval_low}
```

---

**Evaluation (Neutral Skepticism):**
```text
{eval_neutral}
```

---

**Evaluation (High Skepticism):**
```text
{eval_high}
```

---

Based *only* on these evaluations, synthesize an objective summary of the critical perspectives presented.
"""
    # print(f"--- Agent 2 Prompt ---\nSystem: {system_prompt}\nUser: Inputs provided...\n---") # Debugging
    summary = call_claude(system_prompt, user_message, max_tokens=2000)
    return summary

def agent_3_find_directions(paper_text, objective_summary):
    """Agent 3: Identifies 3-5 future hypothesis directions."""
    system_prompt = """
You are Agent 3, a Hypothesis Master AI. You are analyzing a scientific paper and an objective summary of its critiques to identify promising avenues for future research.

**Your Task:**
Identify 3 to 5 distinct, general **directions** for future research hypotheses that logically follow from or are inspired by the provided paper and the summary of its critiques.

**Considerations for Directions:**
* **Gaps & Unanswered Questions:** What questions remain open after reading the paper and critiques?
* **Limitations:** How can the identified limitations be addressed in future work?
* **Extensions & Applications:** Can the findings/methods be extended to new contexts, populations, or problems?
* **Refutations & Challenges:** Can alternative hypotheses be formulated to challenge the paper's claims based on the critiques?
* **Novel Connections:** Are there links to other fields or broader concepts suggested by the work?
* **Methodological Improvements:** Can new hypotheses be tested using improved or different methodologies?

**Output Format:**
Provide a list of 3 to 5 directions. For each direction:
1.  Start with a number (e.g., "1.", "2.").
2.  Provide a concise, bolded **Title:** summarizing the direction.
3.  Follow with a brief description (1-3 sentences) outlining the core idea or question for that direction.

**Example:**
1.  **Investigating Mechanism X:** Explore the underlying biological or computational mechanism responsible for the observed effect Y, which was not fully elucidated in the original paper.
2.  **Generalizability to Population Z:** Test whether the findings reported for population A hold true for population Z, addressing a limitation noted in the critiques regarding sample specificity.

Output ONLY the numbered list of directions in the specified format.
"""
    # Truncate paper text for the prompt
    paper_excerpt = paper_text[:MAX_PAPER_LEN_FOR_PROMPT]
    if len(paper_text) > MAX_PAPER_LEN_FOR_PROMPT:
        paper_excerpt += "\n... (paper text truncated)"

    user_message = f"""
**Original Paper Text (Excerpt):**
```text
{paper_excerpt}
```

---

**Objective Summary of Critical Perspectives:**
```text
{objective_summary}
```

---

Based on the paper excerpt and the summary of critiques, identify 3-5 general directions for future hypotheses using the specified output format.
"""
    # print(f"--- Agent 3 Prompt ---\nSystem: {system_prompt}\nUser: Inputs provided...\n---") # Debugging
    directions_text = call_claude(system_prompt, user_message, max_tokens=1000)

    # --- Parsing Logic ---
    directions = []
    if directions_text:
        # Regex to find lines starting with number, dot, optional space, then bold text for title
        # Allows for flexibility in spacing and handles potential markdown variations
        pattern = re.compile(r"^\s*(\d+)\.?\s*\*\*(.*?)\*\*\s*[:\-]?\s*(.*)", re.MULTILINE)
        matches = pattern.finditer(directions_text)

        parsed_any = False
        for match in matches:
            parsed_any = True
            title = match.group(2).strip()
            description = match.group(3).strip()
            # Try to capture subsequent lines belonging to the same description
            # This is tricky; simple approach: assume description continues until next number
            current_pos = match.end()
            next_match = pattern.search(directions_text, current_pos)
            end_pos = next_match.start() if next_match else len(directions_text)
            description += "\n" + directions_text[current_pos:end_pos].strip()
            directions.append({"title": title, "description": description.strip()})

        # Fallback if regex fails or format is unexpected
        if not parsed_any:
            st.warning("Could not parse directions using primary pattern. Trying simpler split.")
            lines = [line.strip() for line in directions_text.split('\n') if line.strip()]
            current_direction = None
            for line in lines:
                # Simpler check for list markers
                if re.match(r"^\s*\d+\.?\s+", line) or re.match(r"^\s*[\*\-]\s+", line):
                    if current_direction:
                         # Heuristic: If description seems empty, merge title/desc
                         if not current_direction['description'] and ':' in current_direction['title']:
                              parts = current_direction['title'].split(':', 1)
                              current_direction['title'] = parts[0].strip().lstrip('0123456789.*- ').strip('**')
                              current_direction['description'] = parts[1].strip()
                         directions.append(current_direction)

                    # Extract title (potentially including description if no clear separator)
                    title_part = re.sub(r"^\s*\d+\.?\s*", "", line).strip()
                    title_part = re.sub(r"^\s*[\*\-]\s+", "", title_part).strip()
                    current_direction = {"title": title_part, "description": ""}
                elif current_direction:
                    current_direction["description"] += " " + line
            if current_direction: # Add the last one
                 if not current_direction['description'] and ':' in current_direction['title']:
                     parts = current_direction['title'].split(':', 1)
                     current_direction['title'] = parts[0].strip().lstrip('0123456789.*- ').strip('**')
                     current_direction['description'] = parts[1].strip()
                 directions.append(current_direction)

            # Final cleanup if parsing was rough
            for i, d in enumerate(directions):
                 if not d.get("title"): d["title"] = f"Direction {i+1}"
                 if not d.get("description"): d["description"] = f"Further research based on '{d['title']}'."
                 # Clean title if it still has list markers / bold markers
                 d["title"] = d["title"].strip('**').strip()


    if not directions and directions_text:
         st.error("Failed to parse directions from Agent 3. Raw output:")
         st.text(directions_text)
         return [] # Return empty list on failure

    if not directions:
         st.warning("Agent 3 did not return any directions.")
         return []

    return directions


def agent_4_mature_hypothesis(direction_title, direction_description, paper_text):
    """Agent 4: Matures a direction into a detailed hypothesis abstract."""
    system_prompt = f"""
You are Agent 4, a Hypothesis Maturing AI assistant. You are tasked with transforming a general research direction into a concrete, testable hypothesis and outlining a potential study in the form of a detailed abstract.

**Research Direction Provided:**
* **Title:** {direction_title}
* **Description:** {direction_description}

**Your Task:**
Write a detailed abstract (target: 500-1000 words) for a hypothetical research paper based *specifically* on the provided direction. The abstract must include the following sections, clearly delineated (e.g., using bold headings):

1.  **Background:** Briefly introduce the context, referencing the original paper's findings or limitations that motivate this new research direction. State the knowledge gap this study aims to fill.
2.  **Hypothesis/Research Question:** Formulate a single, clear, specific, and *testable* hypothesis (or a primary research question) directly derived from the given direction.
3.  **Proposed Methodology:** Outline the key aspects of the study design. Include:
    * Participants/Sample (source, size justification if possible).
    * Design (e.g., experimental, correlational, longitudinal).
    * Key variables/measures (how will constructs be operationalized?).
    * Procedure (brief overview of steps).
    * Proposed data analysis techniques.
    Be specific enough to demonstrate feasibility but concise.
4.  **Expected Outcomes & Interpretation:** Describe the plausible results. What findings would support the hypothesis? What findings would refute it? How would results be interpreted in either case?
5.  **Significance & Novelty:** Explain the potential contribution of this research. How does it advance knowledge, build upon the original paper, address critiques, or offer practical implications? Emphasize what makes this proposed study novel.
6.  **Potential Challenges/Limitations:** Briefly acknowledge 1-2 key potential hurdles or limitations inherent in the proposed study design or hypothesis.

**Constraints:**
* The abstract must be between 500 and 1000 words.
* Ensure all sections listed above are included and clearly marked.
* The content must directly relate to maturing the provided research direction.
* Use Markdown for formatting (especially headings).
"""
    # Truncate paper text for context
    paper_excerpt = paper_text[:MAX_PAPER_LEN_FOR_PROMPT]
    if len(paper_text) > MAX_PAPER_LEN_FOR_PROMPT:
        paper_excerpt += "\n... (paper text truncated)"

    user_message = f"""
**Original Paper Text (Excerpt for Context):**
```text
{paper_excerpt}
```

---

**Research Direction to Mature:**
* **Title:** {direction_title}
* **Description:** {direction_description}

---

Based on this direction and the context, please generate the detailed 500-1000 word abstract following all instructions in the system prompt.
"""
    # print(f"--- Agent 4 Prompt (Hypothesis: {direction_title}) ---\nSystem: Prompt defined...\nUser: Inputs provided...\n---") # Debugging
    # Increased tokens needed for a long abstract
    abstract = call_claude(system_prompt, user_message, max_tokens=1800)
    return abstract

def agent_6_criticize(hypothesis_title, hypothesis_abstract):
    """Agent 6: Generates 5-10 criticisms for the matured hypothesis abstract."""
    system_prompt = """
You are Agent 6, a Critical Reviewer AI. You specialize in identifying potential weaknesses and flaws in proposed research plans.

**Your Task:**
Analyze the provided research hypothesis abstract and generate 5 to 10 distinct, insightful, and specific criticisms of the proposed study.

**Focus Areas for Criticism:**
* **Hypothesis:** Is it truly clear, specific, testable, falsifiable? Are there hidden assumptions? Is the novelty overstated?
* **Methodology:** Feasibility issues? Sample size/representativeness concerns? Appropriateness of design/measures? Potential confounds or biases not addressed? Lack of controls? Ethical concerns? Vague descriptions?
* **Analysis Plan:** Appropriate statistical methods? Potential for p-hacking or misinterpretation?
* **Expected Outcomes/Interpretation:** Are alternative explanations for expected outcomes ignored? Is the link between potential results and hypothesis conclusion sound?
* **Significance/Novelty:** Is the claimed contribution realistic? Does it truly address a significant gap or just an incremental step?
* **Overall Logic:** Are there inconsistencies or logical gaps in the proposal?

**Instructions:**
* Generate between 5 and 10 distinct criticisms.
* Each criticism should be specific and actionable (i.e., point to a particular aspect of the abstract). Avoid vague or generic complaints.
* Present the criticisms as a numbered list.
* Be rigorous but constructive.
"""
    user_message = f"""
**Hypothesis Title:** {hypothesis_title}

**Hypothesis Abstract to Critique:**
```text
{hypothesis_abstract}
```

---

Generate 5-10 specific criticisms of this proposed research, formatted as a numbered list.
"""
    # print(f"--- Agent 6 Prompt (Critiquing: {hypothesis_title}) ---\nSystem: {system_prompt}\nUser: Abstract provided...\n---") # Debugging
    criticisms_text = call_claude(system_prompt, user_message, max_tokens=1500)

    # --- Parsing Logic ---
    criticisms = []
    if criticisms_text:
        # Split by lines that start with a number and a dot.
        lines = criticisms_text.split('\n')
        current_criticism = ""
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Check if the line starts with a number followed by '.', ')', or ':'
            if re.match(r"^\s*\d+[\.\):]\s+", line):
                if current_criticism: # Save the previous criticism
                    criticisms.append(current_criticism.strip())
                # Start new criticism, removing the number marker
                current_criticism = re.sub(r"^\s*\d+[\.\):]\s*", "", line).strip()
            elif current_criticism: # Append to the current criticism
                current_criticism += " " + line
            # Handle cases where the first line might not have a number (less ideal)
            # elif not criticisms and not current_criticism:
            #     current_criticism = line

        if current_criticism: # Add the last one
            criticisms.append(current_criticism.strip())

        # Fallback if parsing fails
        if not criticisms and criticisms_text:
            st.warning("Could not parse criticisms using standard list format. Splitting by newline.")
            criticisms = [c for c in criticisms_text.split('\n') if c.strip() and len(c) > 10] # Basic filter

    if not criticisms:
        st.warning(f"Agent 6 did not return parseable criticisms for '{hypothesis_title}'. Raw output (if any):")
        st.text(criticisms_text if criticisms_text else "None")
        return []

    return criticisms


def agent_7_debate(criticism, hypothesis_abstract, debate_history, role):
    """Agent 7: Argues FOR or AGAINST a criticism based on the role."""
    if role == "support":
        system_prompt = """
You are Agent 7, a Debate Agent. Your **current role** is to ARGUE STRONGLY **IN SUPPORT** of a specific criticism leveled against a research hypothesis abstract.

**Your Task:**
Read the hypothesis abstract, the specific criticism you must support, and the debate history so far. Present a concise (1-2 paragraphs) but compelling argument that *reinforces* the validity and importance of this criticism. You can build on previous points made in support (if any) or introduce new angles to strengthen the case FOR the criticism.

**Instructions:**
* Be assertive and persuasive in your support of the criticism.
* Directly address why the criticism is valid and significant.
* Do NOT argue against the criticism. Do NOT be neutral. Your sole purpose is to make the best case FOR the criticism.
* Reference specific parts of the hypothesis abstract if helpful.
"""
        role_instruction = "Present a strong argument IN SUPPORT of the criticism, considering the debate history."
    elif role == "refute":
        system_prompt = """
You are Agent 7, a Debate Agent. Your **current role** is to ARGUE STRONGLY **AGAINST** a specific criticism leveled against a research hypothesis abstract.

**Your Task:**
Read the hypothesis abstract, the specific criticism you must refute, and the debate history so far (which includes arguments supporting the criticism). Present a concise (1-2 paragraphs) but compelling argument that *refutes* the criticism and defends the hypothesis abstract. Address the points raised in support of the criticism and provide counter-arguments or justifications for the abstract's approach.

**Instructions:**
* Be assertive and persuasive in your refutation of the criticism.
* Directly counter the points made in support of the criticism.
* Defend the choices made in the hypothesis abstract relevant to the criticism.
* Do NOT agree with the criticism. Do NOT be neutral. Your sole purpose is to make the best case AGAINST the criticism.
* Reference specific parts of the hypothesis abstract if helpful.
"""
        role_instruction = "Present a strong argument REFUTING the criticism, responding to the points raised in the debate history."
    else:
        return "Error: Invalid role for Agent 7."

    user_message = f"""
**Hypothesis Abstract:**
```text
{hypothesis_abstract}
```

**Criticism Being Debated:**
```text
{criticism}
```

**Debate History So Far:**
```text
{debate_history if debate_history else "No previous arguments in this debate."}
```

---

**Your Turn ({role.upper()}):** {role_instruction} Keep your argument concise (1-2 paragraphs).
"""
    # print(f"--- Agent 7 Prompt (Role: {role}) ---\nSystem: Prompt defined...\nUser: Inputs provided...\n---") # Debugging
    argument = call_claude(system_prompt, user_message, max_tokens=600) # Moderate length for arguments
    return argument

def summarize_debate(criticism, debate_transcript):
     """Summarizes a single debate thread using Claude."""
     system_prompt = """
You are a Summarization Assistant AI. You are given a transcript of a debate focused on a specific criticism of a research hypothesis. The transcript contains arguments presented both for and against the criticism over several rounds.

**Your Task:**
Provide a concise, neutral summary of the debate. Your summary should capture:
* The core point of the criticism being debated.
* The main arguments presented *in support* of the criticism.
* The main arguments presented *in refutation* of the criticism.
* The apparent outcome or key remaining points of contention (e.g., was a strong counter-argument made? Is the issue still unresolved?).

**Instructions:**
* Be objective and neutral. Do not take sides or add your own opinion on the validity of the arguments.
* Focus on the substance of the arguments, not just the back-and-forth structure.
* Keep the summary concise (target 100-200 words). Use Markdown for clarity.
"""
     user_message = f"""
**Criticism Debated:**
```text
{criticism}
```

**Full Debate Transcript:**
```text
{debate_transcript}
```

---
Please provide a concise, neutral summary of this debate.
"""
     # print(f"--- Summarize Debate Prompt ---\nSystem: {system_prompt}\nUser: Inputs provided...\n---") # Debugging
     summary = call_claude(system_prompt, user_message, max_tokens=400)
     return summary


def agent_8_judge(hypotheses_data):
    """Agent 8: Judges the best hypothesis based on novelty, validity post-debate, significance, feasibility."""
    system_prompt = """
You are Agent 8, the Final Judging AI. You have been presented with several refined research hypotheses, each accompanied by its abstract, a list of criticisms raised against it, and summaries of debates held on those criticisms.

**Your Task:**
Evaluate all the presented hypotheses and select the **SINGLE 'best' hypothesis** to pursue further. Your judgment must be based *primarily* on the following criteria, weighed in this order:

1.  **Novelty (Highest Priority):** How original, unique, and potentially groundbreaking is the core idea? Does it explore genuinely uncharted territory, challenge existing paradigms, or connect concepts in a surprising way? Strongly prefer hypotheses that seem highly novel and less explored.
2.  **Validity (Post-Debate):** Considering the criticisms and the debate summaries, how plausible and scientifically sound does the hypothesis appear *after* scrutiny? Were the major criticisms effectively addressed or mitigated during the debates? Does the core logic hold up?
3.  **Significance:** If the hypothesis were supported, what would be the potential impact or importance of the findings? Would it solve a major problem, open significant new research avenues, or have practical applications?
4.  **Feasibility:** Based on the abstract's proposed methodology and any relevant criticisms/debates, how realistic is it that the study could be successfully conducted? Are the methods sound and practical?

**Input Format:**
You will receive a summary of each hypothesis, including its title, an excerpt of its abstract, and summaries of the debates surrounding its criticisms.

**Output Format:**
Structure your response clearly as follows:

1.  **Chosen Hypothesis:** State the exact title of the hypothesis you have selected as the best.
2.  **Reasoning for Selection (Detailed):** Provide a comprehensive justification (minimum 200 words, ideally more) explaining *why* you chose this specific hypothesis over the others.
    * Explicitly address how the chosen hypothesis excels according to the four criteria (Novelty, Validity Post-Debate, Significance, Feasibility), emphasizing Novelty and Validity.
    * Compare it, at least implicitly, to the other candidates, explaining why they were ranked lower (e.g., less novel, critical flaws remained after debate, lower significance).
    * Reference specific aspects from the abstract, criticism summaries, or debate outcomes to support your points.
    * Acknowledge any remaining weaknesses of the chosen hypothesis but argue why its strengths (especially novelty and post-debate validity) outweigh them.
3.  **Summary of Chosen Hypothesis & Debates:** Briefly summarize the core concept of the hypothesis you selected and the overall outcome of the debates concerning its criticisms (e.g., "The hypothesis proposes X. Debates focused on Y and Z; criticism Y was largely refuted, while Z remains a minor concern regarding methodology detail...").

**Final Goal:** Identify the most promising, novel, and defensible research direction based on the entire analysis pipeline.
"""

    # --- Prepare Input Summary for Agent 8 ---
    input_data_summary = "Here is the data for the hypotheses you need to judge:\n\n"
    if not hypotheses_data:
         return "Error: No hypothesis data provided to Agent 8."

    for i, data in enumerate(hypotheses_data):
        input_data_summary += f"--- Hypothesis {i+1} ---\n"
        input_data_summary += f"**Title:** {data.get('title', 'N/A')}\n"
        # Include a larger abstract excerpt for better context
        abstract_excerpt = data.get('abstract', 'N/A')[:1000]
        if len(data.get('abstract', '')) > 1000: abstract_excerpt += "..."
        input_data_summary += f"**Abstract Excerpt:**\n{abstract_excerpt}\n\n"
        input_data_summary += f"**Criticism & Debate Summaries:**\n"
        if data.get('criticisms'):
            for j, crit_data in enumerate(data['criticisms']):
                 criticism_excerpt = crit_data.get('criticism', 'N/A')[:200]
                 if len(crit_data.get('criticism', '')) > 200: criticism_excerpt += "..."
                 debate_summary_excerpt = crit_data.get('debate_summary', 'No summary available.')[:300]
                 if len(crit_data.get('debate_summary', '')) > 300: debate_summary_excerpt += "..."

                 input_data_summary += f"  * **Criticism {j+1}:** {criticism_excerpt}\n"
                 input_data_summary += f"    * **Debate Summary:** {debate_summary_excerpt}\n"
        else:
            input_data_summary += "  * No criticisms were generated or debated for this hypothesis.\n"
        input_data_summary += "---\n\n"

    user_message = input_data_summary + "\nPlease evaluate these hypotheses based on the criteria provided in the system prompt (Novelty, Validity Post-Debate, Significance, Feasibility) and provide your final judgement in the specified output format."

    # print(f"--- Agent 8 Prompt ---\nSystem: Prompt defined...\nUser: Summarized data provided...\n---") # Debugging
    judgement = call_claude(system_prompt, user_message, max_tokens=2500) # Allow ample tokens for reasoning
    return judgement

# --- Streamlit App UI ---

st.set_page_config(layout="wide", page_title="Paper Analysis Agent System")
st.title("🔬 Multi-Agent Paper Analysis & Hypothesis Generation")

st.info("""
**Instructions:**
1.  Paste the full text of a scientific paper into the text area below.
2.  Click "Analyze Paper".
3.  The system will run a sequence of AI agents (powered by Claude 3.5 Sonnet) to analyze the paper, generate, critique, and debate new hypotheses, and finally judge the most promising one based on novelty and validity.
4.  Results will appear progressively below. Please be patient, as multiple AI calls are involved.

**Workflow:**
* **Agent 1:** Critical Evaluation (Low, Neutral, High Skepticism)
* **Agent 2:** Objective Summary of Critiques
* **Agent 3:** Identify Hypothesis Directions
* **Agent 4:** Mature Directions into Abstracts
* **Agent 6:** Criticize Abstracts
* **Agent 7:** Debate Criticisms (3 Rounds) -> Summarize Debates
* **Agent 8:** Judge Best Hypothesis (Novelty & Validity focus)
""")

# --- Initialize Session State ---
# Use keys that are unlikely to clash with user inputs if they modify code
if 'paper_analysis_text' not in st.session_state:
    st.session_state.paper_analysis_text = ""
if 'analysis_running' not in st.session_state:
    st.session_state.analysis_running = False
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = {} # Store all intermediate and final results


# --- Input Area ---
paper_text_input = st.text_area("Paste Full Paper Text Here:", height=350, key="paper_text_area",
                                value=st.session_state.paper_analysis_text,
                                disabled=st.session_state.analysis_running)

# --- Control Button ---
if st.button("Analyze Paper", type="primary", disabled=st.session_state.analysis_running):
    if not paper_text_input:
        st.warning("Please paste the paper text before analyzing.")
    elif not client:
         st.error("Analysis cannot start: Anthropic client not initialized (check API key).")
    else:
        # --- Start Analysis ---
        st.session_state.paper_analysis_text = paper_text_input # Store current text
        st.session_state.analysis_running = True
        st.session_state.analysis_complete = False
        st.session_state.analysis_results = {} # Clear previous results
        st.rerun() # Rerun to disable input and show progress

# --- Analysis Execution Block ---
if st.session_state.analysis_running:
    st.info("Analysis in progress... Please wait. This may take several minutes.")
    # Use placeholders for status updates and results display
    status_placeholder = st.empty()
    results_container = st.container()

    try: # Wrap the whole analysis in a try block for robustness
        paper_text = st.session_state.paper_analysis_text
        results = {} # Temporary dict for this run

        with results_container:
            # --- Agent 1 ---
            st.subheader("Agent 1: Critical Evaluations")
            evaluations = {}
            agent1_success = True
            for level in ["Low", "Neutral", "High"]:
                status_placeholder.info(f"Running Agent 1 (Skepticism: {level})...")
                with st.spinner(f"Agent 1 evaluating ({level} skepticism)..."):
                    eval_result = agent_1_evaluate(paper_text, level)
                if not eval_result or eval_result.startswith("Error:"):
                    st.error(f"Agent 1 failed for skepticism level: {level}. {eval_result}")
                    agent1_success = False
                    break # Stop if one level fails
                evaluations[level] = eval_result
            results['agent_1_evaluations'] = evaluations

            if not agent1_success:
                 raise Exception("Agent 1 failed. Aborting analysis.")

            tabs = st.tabs(["Low Skepticism", "Neutral Skepticism", "High Skepticism"])
            with tabs[0]: st.markdown(evaluations.get("Low", "N/A"))
            with tabs[1]: st.markdown(evaluations.get("Neutral", "N/A"))
            with tabs[2]: st.markdown(evaluations.get("High", "N/A"))
            st.success("Agent 1 finished.")
            time.sleep(1)

            # --- Agent 2 ---
            st.subheader("Agent 2: Objective Summary of Critical Perspectives")
            status_placeholder.info("Running Agent 2 (Objective Summarizer)...")
            with st.spinner("Agent 2 summarizing evaluations..."):
                 objective_summary = agent_2_summarize(
                     evaluations["Low"], evaluations["Neutral"], evaluations["High"]
                 )
            if not objective_summary or objective_summary.startswith("Error:"):
                 st.error(f"Agent 2 failed. {objective_summary}")
                 raise Exception("Agent 2 failed. Aborting analysis.")
            results['agent_2_summary'] = objective_summary
            st.markdown(objective_summary)
            st.success("Agent 2 finished.")
            time.sleep(1)

            # --- Agent 3 ---
            st.subheader("Agent 3: Future Hypothesis Directions")
            status_placeholder.info("Running Agent 3 (Hypothesis Master)...")
            with st.spinner("Agent 3 identifying hypothesis directions..."):
                 hypothesis_directions = agent_3_find_directions(paper_text, objective_summary)

            if not hypothesis_directions: # Agent 3 function handles errors/warnings internally
                 st.error("Agent 3 failed to return valid directions. Aborting analysis.")
                 raise Exception("Agent 3 failed. Aborting analysis.")

            results['agent_3_directions'] = hypothesis_directions
            for i, direction in enumerate(hypothesis_directions):
                 st.markdown(f"**{i+1}. {direction.get('title', 'N/A')}**")
                 st.markdown(f"   > {direction.get('description', 'N/A')}")
            st.success(f"Agent 3 finished identifying {len(hypothesis_directions)} directions.")
            time.sleep(1)

            # --- Agents 4, 6, 7 Loop ---
            st.subheader("Agents 4, 6, 7: Hypothesis Maturation, Criticism & Debate")
            results['hypotheses_analysis'] = []
            num_directions = len(hypothesis_directions)
            any_hypothesis_processed = False

            for i, direction in enumerate(hypothesis_directions):
                direction_title = direction.get('title', f'Direction {i+1}')
                direction_desc = direction.get('description', 'N/A')
                st.markdown(f"--- Processing Direction {i+1}/{num_directions}: **{direction_title}** ---")
                hypothesis_data = {"title": direction_title, "direction_description": direction_desc, "abstract": None, "criticisms": []}

                # --- Agent 4 ---
                status_placeholder.info(f"Running Agent 4 (Maturing Hypothesis {i+1}/{num_directions}: {direction_title})...")
                with st.spinner(f"Agent 4 maturing '{direction_title}' into abstract..."):
                     abstract = agent_4_mature_hypothesis(direction_title, direction_desc, paper_text)

                if not abstract or abstract.startswith("Error:"):
                     st.warning(f"Agent 4 failed for direction: '{direction_title}'. Skipping this direction. {abstract}")
                     # Store placeholder data indicating failure for this direction
                     hypothesis_data['abstract'] = f"Error: Agent 4 failed to generate abstract for this direction."
                     results['hypotheses_analysis'].append(hypothesis_data)
                     continue # Skip to next direction

                hypothesis_data['abstract'] = abstract
                with st.expander(f"Agent 4: Abstract for '{direction_title}'", expanded=False):
                     st.markdown(abstract)
                st.success(f"Agent 4 finished for '{direction_title}'.")
                time.sleep(0.5)

                # --- Agent 6 ---
                status_placeholder.info(f"Running Agent 6 (Criticizing Hypothesis {i+1}/{num_directions}: {direction_title})...")
                with st.spinner(f"Agent 6 criticizing '{direction_title}'..."):
                    criticisms = agent_6_criticize(direction_title, abstract)

                # Agent 6 handles parsing errors internally and returns [] if needed
                if not criticisms:
                     st.warning(f"Agent 6 generated no valid criticisms for: '{direction_title}'. Proceeding without debates for this hypothesis.")
                     hypothesis_data['criticisms'] = [] # Ensure it's an empty list
                else:
                     hypothesis_data['criticisms'] = [{"criticism": c, "debate_transcript": "", "debate_summary": "Not yet generated"} for c in criticisms]
                     with st.expander(f"Agent 6: Criticisms for '{direction_title}' ({len(criticisms)} found)", expanded=False):
                         for idx, crit in enumerate(criticisms):
                             st.markdown(f"{idx+1}. {crit}")
                     st.success(f"Agent 6 finished for '{direction_title}'.")
                     time.sleep(0.5)

                # --- Agent 7 ---
                if hypothesis_data['criticisms']:
                    num_criticisms = len(hypothesis_data['criticisms'])
                    st.markdown(f"**Initiating Debates for '{direction_title}' ({num_criticisms} criticisms)**")
                    for j, crit_data in enumerate(hypothesis_data['criticisms']):
                        criticism = crit_data['criticism']
                        st.markdown(f"*Debating Criticism {j+1}/{num_criticisms}*")
                        # Use an inner expander for the debate itself
                        with st.expander(f"Debate on Criticism {j+1}: '{criticism[:80]}...'", expanded=False):
                            st.markdown(f"**Criticism:**\n> {criticism}")
                            debate_history = f"**Criticism:**\n{criticism}"
                            max_rounds = 3
                            debate_failed = False

                            for round_num in range(max_rounds):
                                st.markdown(f"--- Round {round_num + 1} ---")
                                # Support Argument
                                status_placeholder.info(f"Agent 7 debating Criticism {j+1}, Rd {round_num+1} (Support) for Hyp {i+1}...")
                                with st.spinner(f"Agent 7 arguing FOR criticism {j+1} (Round {round_num+1})..."):
                                     support_arg = agent_7_debate(criticism, abstract, debate_history, role="support")
                                if not support_arg or support_arg.startswith("Error:"):
                                     st.warning(f"Agent 7 (Support) failed in Round {round_num+1} for criticism {j+1}. {support_arg}")
                                     debate_history += f"\n\n**Round {round_num + 1} - Argument For:**\n*Agent failed to generate argument.*"
                                     # Decide whether to break or continue if agent fails
                                     # debate_failed = True; break
                                else:
                                     st.markdown(f"**Argument For:** {support_arg}")
                                     debate_history += f"\n\n**Round {round_num + 1} - Argument For:**\n{support_arg}"

                                time.sleep(0.2) # Small delay

                                # Refute Argument
                                status_placeholder.info(f"Agent 7 debating Criticism {j+1}, Rd {round_num+1} (Refute) for Hyp {i+1}...")
                                with st.spinner(f"Agent 7 arguing AGAINST criticism {j+1} (Round {round_num+1})..."):
                                     refute_arg = agent_7_debate(criticism, abstract, debate_history, role="refute")
                                if not refute_arg or refute_arg.startswith("Error:"):
                                     st.warning(f"Agent 7 (Refute) failed in Round {round_num+1} for criticism {j+1}. {refute_arg}")
                                     debate_history += f"\n\n**Round {round_num + 1} - Argument Against:**\n*Agent failed to generate argument.*"
                                     # debate_failed = True; break
                                else:
                                     st.markdown(f"**Argument Against:** {refute_arg}")
                                     debate_history += f"\n\n**Round {round_num + 1} - Argument Against:**\n{refute_arg}"

                                time.sleep(0.2) # Small delay

                            crit_data['debate_transcript'] = debate_history

                            # Summarize Debate
                            status_placeholder.info(f"Summarizing debate for Criticism {j+1} of Hyp {i+1}...")
                            with st.spinner(f"Summarizing debate for criticism {j+1}..."):
                                 debate_summary = summarize_debate(criticism, debate_history)
                            if not debate_summary or debate_summary.startswith("Error:"):
                                 st.warning(f"Failed to summarize debate for criticism {j+1}. {debate_summary}")
                                 crit_data['debate_summary'] = "Error: Failed to generate summary."
                            else:
                                 crit_data['debate_summary'] = debate_summary
                                 st.markdown("**Debate Summary:**")
                                 st.markdown(debate_summary)

                        # End of debate expander
                        st.success(f"Debate concluded for criticism {j+1}.")
                        time.sleep(0.5) # Pause between debates

                # Store the fully processed data for this hypothesis
                results['hypotheses_analysis'].append(hypothesis_data)
                any_hypothesis_processed = True # Mark that at least one was processed successfully

            # --- Agent 8 ---
            if not any_hypothesis_processed or not results.get('hypotheses_analysis'):
                 st.error("No hypotheses were successfully processed through abstract generation and criticism. Cannot proceed to judgement.")
                 raise Exception("No valid hypotheses to judge.")

            st.subheader("🏆 Agent 8: Final Judgement")
            status_placeholder.info("Running Agent 8 (Judge)...")
            # Filter out hypotheses that failed Agent 4
            valid_hypotheses_for_judging = [h for h in results['hypotheses_analysis'] if h.get('abstract') and not h['abstract'].startswith("Error:")]

            if not valid_hypotheses_for_judging:
                 st.error("No hypotheses had successfully generated abstracts. Cannot proceed to judgement.")
                 raise Exception("No valid abstracts generated for judgement.")

            with st.spinner("Agent 8 judging the best hypothesis..."):
                 final_judgement = agent_8_judge(valid_hypotheses_for_judging)

            if not final_judgement or final_judgement.startswith("Error:"):
                 st.error(f"Agent 8 failed to produce a judgement. {final_judgement}")
                 raise Exception("Agent 8 failed.")

            results['agent_8_judgement'] = final_judgement
            st.markdown(final_judgement)
            st.success("Agent 8 finished.")

            # --- Analysis Complete ---
            st.session_state.analysis_results = results
            st.session_state.analysis_complete = True
            st.session_state.analysis_running = False
            status_placeholder.empty() # Clear status message
            st.balloons()
            st.success("Analysis Complete!")
            st.rerun() # Rerun to re-enable input and show final state

    except Exception as e:
         st.error(f"An error occurred during the analysis: {e}")
         st.session_state.analysis_running = False
         st.session_state.analysis_complete = False # Mark as incomplete due to error
         # Optionally save partial results if desired
         # st.session_state.analysis_results = results
         status_placeholder.error("Analysis aborted due to an error.")
         st.rerun()


# --- Display Results (if analysis completed successfully) ---
elif st.session_state.analysis_complete:
    st.success("Analysis previously completed. Displaying results:")
    results = st.session_state.analysis_results

    # Display Agent 1 results
    if 'agent_1_evaluations' in results:
        st.subheader("Agent 1: Critical Evaluations")
        evals = results['agent_1_evaluations']
        tabs = st.tabs(["Low Skepticism", "Neutral Skepticism", "High Skepticism"])
        with tabs[0]: st.markdown(evals.get("Low", "N/A"))
        with tabs[1]: st.markdown(evals.get("Neutral", "N/A"))
        with tabs[2]: st.markdown(evals.get("High", "N/A"))
        st.divider()

    # Display Agent 2 results
    if 'agent_2_summary' in results:
        st.subheader("Agent 2: Objective Summary of Critical Perspectives")
        st.markdown(results['agent_2_summary'])
        st.divider()

    # Display Agent 3 results
    if 'agent_3_directions' in results:
        st.subheader("Agent 3: Future Hypothesis Directions")
        for i, direction in enumerate(results['agent_3_directions']):
            st.markdown(f"**{i+1}. {direction.get('title', 'N/A')}**")
            st.markdown(f"   > {direction.get('description', 'N/A')}")
        st.divider()

    # Display Agents 4, 6, 7 results (grouped by hypothesis)
    if 'hypotheses_analysis' in results:
        st.subheader("Agents 4, 6, 7: Hypothesis Development, Criticism & Debate Details")
        for i, data in enumerate(results['hypotheses_analysis']):
            with st.expander(f"Details for Hypothesis based on: '{data.get('title', f'Direction {i+1}')}'", expanded=False):
                st.markdown(f"**Agent 4: Abstract**")
                abstract = data.get('abstract', 'N/A')
                if abstract.startswith("Error:"):
                    st.error(abstract) # Show error if Agent 4 failed
                else:
                    st.markdown(abstract)

                st.markdown(f"---")
                st.markdown(f"**Agent 6 & 7: Criticisms & Debates**")
                if data.get('criticisms'):
                    for j, crit_data in enumerate(data['criticisms']):
                         st.markdown(f"**Criticism {j+1}:** {crit_data.get('criticism', 'N/A')}")
                         # Optionally display full debate transcript too
                         # with st.expander(f"Full Debate Transcript for Criticism {j+1}", expanded=False):
                         #    st.text(crit_data.get('debate_transcript', 'N/A'))
                         st.markdown(f"**Debate Summary:**")
                         summary = crit_data.get('debate_summary', 'N/A')
                         if summary.startswith("Error:"):
                             st.warning(summary)
                         else:
                             st.markdown(summary)
                         st.markdown("---") # Separator between criticisms
                elif not abstract.startswith("Error:"): # Only show 'no criticisms' if abstract was generated
                    st.markdown("_No criticisms were generated or debated for this hypothesis._")
        st.divider()

    # Display Agent 8 results
    if 'agent_8_judgement' in results:
        st.subheader("🏆 Agent 8: Final Judgement")
        st.markdown(results['agent_8_judgement'])
        st.divider()

# --- Footer ---
st.markdown("---")
st.caption("Powered by Anthropic Claude 3.5 Sonnet & Streamlit")
