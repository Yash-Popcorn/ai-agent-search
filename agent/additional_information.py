from verdict import Pipeline
from verdict.schema import Schema
from verdict.common.judge import JudgeUnit
from verdict.scale import BooleanScale
from openai import OpenAI
import re
import time
from verdict.util import ratelimit
ratelimit.disable()

client = OpenAI()
# Constant to indicate no info needed
NO_INFO_NEEDED = "NO_INFO_NEEDED"

def ask_additional_information(prompt, conversation_history_str="", max_questions=2):
    """
    Determines if more info is needed and returns questions or NO_INFO_NEEDED.
    
    Args:
        prompt: The user's task description or query
        conversation_history_str: String representation of the conversation history
        max_questions: Maximum number of rounds of questions to ask before proceeding
    
    Returns:
        Either a formatted string of questions or NO_INFO_NEEDED constant
    """
    # Count how many times questions have already been asked
    question_count = 0
    if conversation_history_str:
        # Find occurrences of the question pattern in the history
        question_pattern = r"To proceed with this task, I need clarification"
        question_count = len(re.findall(question_pattern, conversation_history_str))
    
    # If we've already asked max_questions, don't ask more
    if question_count >= max_questions:
        print(f"⚠️ Already asked {question_count} rounds of questions, proceeding with available information")
        return NO_INFO_NEEDED

    context_text = f"\n\nConversation History:\n{conversation_history_str}" if conversation_history_str else ""

    pipeline = Pipeline() \
    >> JudgeUnit(BooleanScale(), explanation=True).prompt(f"""
        Determine if additional information or clarification from the user would be genuinely helpful or necessary to properly address the task, considering the conversation history.

        IMPORTANT: You MUST analyze the full conversation history to determine if relevant information might have already been provided in previous messages. Avoid asking for information already present.

        Consider the user's request and the context. Would asking for clarification significantly improve the quality, relevance, or feasibility of the response?

        Answer True if:
        - The request is ambiguous and requires clarification on scope, intent, or specific requirements.
        - Key details necessary for a tailored or accurate response are missing and are likely user-specific (e.g., personal preferences, specific configurations, private data).
        - The task could be interpreted in multiple ways, and user input is needed to choose the correct path.
        - Information seems potentially contradictory or incomplete based on the conversation.
        - User input is needed to narrow down broad topics or choose among options.

        Answer False if:
        - The request is clear enough to make a reasonable attempt.
        - Missing information consists of general knowledge or facts that can be easily researched or reasonably assumed.
        - The user has already provided sufficient context or clarification in the conversation history.
        - Asking for more details would be overly pedantic or unlikely to significantly change the outcome.

        Task to evaluate: {{source.prompt}}
        {{source.context_text}}

        Provide a True/False decision with a brief justification based on whether clarification would be substantially beneficial.
    """).via("gpt-4o", retries=1)

    response, leaf_node_prefixes = pipeline.run(Schema.of(prompt=prompt, context_text=context_text))
    # print(response) # Removed internal print

    needs_info_key = 'Pipeline_root.block.unit[DirectScoreJudge]_score'
    explanation_key = 'Pipeline_root.block.unit[DirectScoreJudge]_explanation'

    if needs_info_key in response and response[needs_info_key] == True and question_count < max_questions:
        justification = response.get(explanation_key, "No justification provided.")
        # print(f"   Decision: Need more info. Justification: {justification}") # Removed internal print

        question_generation_prompt = f"""
        Based on the following task and conversation history, please formulate ONLY CRITICAL questions to ask the user.
        The initial check indicated information is missing with the justification: '{justification}'.

        This is question round {question_count + 1} of {max_questions} maximum, so focus ONLY on the most important missing information.
        
        IMPORTANT: First thoroughly analyze all previous messages in the conversation history. DO NOT ask questions about information that has already been provided.

        Task: {prompt}
        {context_text}

        Rules for questions:
        1. First carefully review the conversation history to avoid asking for information already provided.
        2. Ask a MAXIMUM of 3 questions, even if more information could be helpful.
        3. Only ask about missing details that are ABSOLUTELY REQUIRED and CANNOT be researched.
        4. DO NOT ask about information that could be found through online research (e.g., tuition costs, program details).
        5. DO NOT ask about preferences unless they are critical to completing the task.
        6. If the user has already provided substantial information, consider whether more is truly needed.
        
        Format the response as:
        "To proceed with this task, I need clarification on the following points:"
        1. [Critical question 1]
        2. [Critical question 2]
        3. [Critical question 3]
        """

        try:
            # Add retry logic for the OpenAI API call
            max_retries = 1
            retry_count = 0
            backoff_time = 1
            
            while retry_count < max_retries:
                try:
                    message = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": "You are an assistant helping to gather only the most critical information needed. Be concise and ask as few questions as possible."}, 
                            {"role": "user", "content": question_generation_prompt}
                        ],
                        temperature=0.3
                    )
                    ai_question = message.choices[0].message.content.strip()
                    break  # Success, exit the retry loop
                except Exception as e:
                    if "rate limit" in str(e).lower():
                        retry_count += 1
                        if retry_count >= max_retries:
                            raise
                        print(f"   ⚠️ Rate limit hit, retrying in {backoff_time} seconds (attempt {retry_count}/{max_retries})")
                        time.sleep(backoff_time)
                        backoff_time *= 2  # Exponential backoff
                    else:
                        raise  # Not a rate limit issue, re-raise

            # Return the formatted questions string directly
            return ai_question

        except Exception as e:
            print(f"   ❌ Error generating clarification questions: {e}") # Keep error print for debugging
            return NO_INFO_NEEDED # If there's an error, just proceed

    else:
        explanation = response.get(explanation_key, "No justification provided.")
        # print(f"   Decision: No additional info needed. Justification: {explanation}") # Removed internal print
        return NO_INFO_NEEDED # Return constant