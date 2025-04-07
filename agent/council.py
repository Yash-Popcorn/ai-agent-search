from verdict.common.judge import JudgeUnit
from verdict.scale import DiscreteScale
from verdict.transform import MaxPoolUnit
from verdict import Pipeline
from verdict import Layer
from verdict.schema import Schema
from openai import OpenAI
from verdict.util import ratelimit
ratelimit.disable()

client = OpenAI()

# Define scales based on the required checks
hallucination_scale = DiscreteScale(["hallucination", "no_hallucination"])
verification_scale = DiscreteScale(["verified", "not_verified"])
alignment_scale = DiscreteScale(["aligned", "not_aligned"])

# Define prompts for each judge, referencing input fields including conversation context
hallucination_prompt = """
Evaluate the following response for hallucination based on the conversation history and user query. A hallucination includes making up facts, citing non-existent sources, or making incorrect assumptions about code or context.

Conversation History:
{source.conversation_context}

User Query:
{source.user_query}

Response:
{source.response}

Respond with only 'hallucination' or 'no_hallucination'.
Judgement:
"""

verification_prompt = """
Verify the claims and code correctness in the following response based on the conversation history and user query. Check if facts, library usages, function calls, or module imports are accurate.

Conversation History:
{source.conversation_context}

User Query:
{source.user_query}

Response:
{source.response}

Respond with only 'verified' or 'not_verified'.
Judgement:
"""

alignment_prompt = """
Evaluate if the following response aligns with the user's query, considering the full conversation history.

Conversation History:
{source.conversation_context}

User Query:
{source.user_query}

Response:
{source.response}

Respond with only 'aligned' or 'not_aligned'.
Judgement:
"""

# Define the evaluation pipeline using VERDICT components
evaluation_pipeline = Pipeline("EvaluationCouncil") \
    >> Layer(
        [
            JudgeUnit(scale=hallucination_scale, explanation=True)
                .prompt(hallucination_prompt)
                .via("gpt-4o-mini", retries=3, temperature=0.9),
            JudgeUnit(scale=verification_scale, explanation=True)
                .prompt(verification_prompt)
                .via("gpt-4o-mini", retries=3, temperature=0.9),
            JudgeUnit(scale=alignment_scale, explanation=True)
                .prompt(alignment_prompt)
                .via("gpt-4o-mini", retries=3, temperature=0.9),
        ]
    ) \
    >> MaxPoolUnit()

def council(response: str, user_query: str, conversation_context: str):
    """
    Evaluates a response using a council pipeline, considering the full conversation context.

    Args:
        response (str): The aggregated agent responses to evaluate.
        user_query (str): The original user query for alignment checking.
        conversation_context (str): The full history of the conversation.

    Returns:
        dict: A dictionary with 'judgement' (e.g., 'aligned', 'hallucination') and 'explanation'.
    """
    pipeline_input_schema = Schema.of(
        response=response,
        user_query=user_query,
        conversation_context=conversation_context
    )

    try:
        judgement_dict, _ = evaluation_pipeline.run(pipeline_input_schema)

        # Extract the most relevant judgement information
        # Prioritize alignment, then verification, then hallucination for the primary score
        # Explanation might come from MaxPool or individual judges
        final_judgement = "Unknown"
        explanation = "No explanation provided."

        # Check individual judge results if MaxPool doesn't give a clear winner explanation
        alignment_score = judgement_dict.get('EvaluationCouncil_root.block.block[0].unit[DirectScoreJudge]_score')
        verification_score = judgement_dict.get('EvaluationCouncil_root.block.block[1].unit[DirectScoreJudge]_score')
        hallucination_score = judgement_dict.get('EvaluationCouncil_root.block.block[2].unit[DirectScoreJudge]_score')

        alignment_expl = judgement_dict.get('EvaluationCouncil_root.block.block[0].unit[DirectScoreJudge]_explanation', '')
        verification_expl = judgement_dict.get('EvaluationCouncil_root.block.block[1].unit[DirectScoreJudge]_explanation', '')
        hallucination_expl = judgement_dict.get('EvaluationCouncil_root.block.block[2].unit[DirectScoreJudge]_explanation', '')

        # Determine overall judgement and combine explanations
        combined_explanation = f"Alignment: {alignment_expl} | Verification: {verification_expl} | Hallucination: {hallucination_expl}"

        print(judgement_dict)

        if alignment_score == 'not_aligned':
            final_judgement = 'not_aligned'
            explanation = alignment_expl or combined_explanation
        elif verification_score == 'not_verified':
            final_judgement = 'not_verified'
            explanation = verification_expl or combined_explanation
        elif hallucination_score == 'hallucination':
            final_judgement = 'hallucination'
            explanation = hallucination_expl or combined_explanation
        else:
            final_judgement = 'passed'
            explanation = combined_explanation

        return {
            "judgement": final_judgement,
            "explanation": explanation
        }

    except AttributeError as ae:
         print(f"❌ Council Error: Pipeline structure issue or missing method: {ae}")
         return {"judgement": "error", "explanation": f"AttributeError: {ae}"}
    except KeyError as ke:
        print(f"❌ Council Error: Missing expected key in judgement dictionary: {ke}")
        print("Full judgement dict:", judgement_dict)
        return {"judgement": "error", "explanation": f"KeyError: Missing key {ke}"}
    except Exception as e:
        print(f"❌ An unexpected error occurred during council evaluation: {e}")
        return {"judgement": "error", "explanation": str(e)}


