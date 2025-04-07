from verdict import Pipeline
from verdict.schema import Schema
from verdict.common.judge import JudgeUnit
from verdict.scale import DiscreteScale
from verdict.util import ratelimit
ratelimit.disable()

def judge_possibility(prompt, conversation_history_str=""):
    """Judges if a task is possible for the software, considering conversation history."""

    context_text = f"\n\nConversation History:\n{conversation_history_str}" if conversation_history_str else ""

    pipeline = Pipeline() \
    >> JudgeUnit(DiscreteScale([
        'YES',
        'NO',
    ]), explanation=True).prompt("""
        Evaluate whether the following software task is possible to accomplish within the scope of programmatic capabilities. Assume that the system has access to files or text storage that the user has explicitly provided or saved.

        Answer YES if the task:
        - Can be accomplished purely through software/programming
        - Requires only digital/computational resources
        - Can be done with available programming languages, frameworks, or APIs
        - Is within scope of data processing, automation, or digital operations
        - Doesn't require physical world interactions or human intervention
        - Can be achieved with proper API access and authentication where needed
        - Involves searching for information or context that an AI system can access
        - Requires document or information retrieval from accessible digital sources
        - Retrieving and searching for specific information based on the user's prior interaction and storage/context

        Answer NO if the task:
        - Requires physical world manipulation (e.g., building hardware, 3D printing)
        - Needs human physical intervention
        - Involves purchasing or financial transactions without proper API access
        - Requires real-world sensing or actuating without proper interfaces
        - Goes beyond pure software capabilities
        - Requires AGI-level capabilities or general world knowledge
        - Involves unauthorized access or illegal operations

        Task to evaluate: {source.prompt}
        {source.context_text}

        Provide a YES/NO answer and brief justification for whether this is a valid software task.
    """)

    response, leaf_node_prefixes = pipeline.run(Schema.of(prompt=prompt, context_text=context_text))
    # print(response) # Removed for cleaner terminal output
    # Return both score and explanation for potential use in experiment.py
    return {
        'score': response['Pipeline_root.block.unit[DirectScoreJudge]_score'],
        'explanation': response.get('Pipeline_root.block.unit[DirectScoreJudge]_explanation', '')
    }