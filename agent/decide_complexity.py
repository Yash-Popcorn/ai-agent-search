from verdict import Pipeline
from verdict.schema import Schema
from verdict.common.judge import JudgeUnit
from verdict.scale import DiscreteScale
from verdict.util import ratelimit
ratelimit.disable()

def decide_complexity(prompt, conversation_history_str=""):
    """Judges the complexity of a task given the prompt and conversation history."""

    context_text = f"\n\nConversation History:\n{conversation_history_str}" if conversation_history_str else ""

    pipeline = Pipeline() \
    >> JudgeUnit(DiscreteScale([
        'CRITICAL_COMPLEXITY',
        'HIGH_COMPLEXITY',
        'MODERATE_COMPLEXITY',
        'LOW_COMPLEXITY',
        'MINIMAL_COMPLEXITY',
        'TRIVIAL'
    ]), explanation=True).prompt("""
        Evaluate the complexity and difficulty of executing this task and categorize it into one of these complexity levels:

        CRITICAL_COMPLEXITY:
        - Highly intricate processes
        - Expert-level knowledge required
        - Multiple dependencies and constraints
        - Critical decision points
        - High risk management needs
        - Requires extensive planning

        HIGH_COMPLEXITY:
        - Multiple sophisticated steps
        - Deep technical knowledge required
        - Significant coordination needed
        - Complex problem-solving
        - Risk management considerations

        MODERATE_COMPLEXITY:
        - Multiple interconnected steps
        - Specific domain expertise needed
        - Requires planning and organization
        - Some complex decision making

        LOW_COMPLEXITY:
        - Multiple basic steps
        - Basic domain knowledge needed
        - Minor coordination required
        - Simple problem-solving

        MINIMAL_COMPLEXITY:
        - Basic, straightforward tasks
        - Single-step operations
        - Common everyday actions
        - No special knowledge required

        TRIVIAL:
        - Extremely basic tasks
        - No special knowledge or skills needed
        - Can be done without thought
        - Instantaneous execution

        Task to evaluate: {source.prompt}
        {source.context_text}

        Provide the complexity category and brief justification for the classification.
    """)

    response, leaf_node_prefixes = pipeline.run(Schema.of(prompt=prompt, context_text=context_text))
    # print(response) # Removed for cleaner terminal output
    return response['Pipeline_root.block.unit[DirectScoreJudge]_score']