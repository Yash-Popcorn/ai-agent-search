from openai import OpenAI
import asyncio
import litellm

client = OpenAI()

async def synthesize_response(original_query: str, conversation_context: str, aggregated_agent_responses: str, council_judgement: dict):
    """Synthesizes the final response based on query, context, agent outputs, and council judgement."""
    judgement_text = f"Judgement: {council_judgement.get('judgement', 'N/A')}. Explanation: {council_judgement.get('explanation', 'N/A')}"
    
    prompt = f"""
    You are a helpful assistant synthesizing information into a final response for the user.
    
    **Original User Query:**
    {original_query}
    
    **Full Conversation History (User and AI):**
    {conversation_context}
    
    **Aggregated Results from Executed Agents:**
    {aggregated_agent_responses}
    
    **Internal Quality Check Result (DO NOT MENTION THIS TO THE USER):**
    {judgement_text}
    
    **Instructions:**
    1. Create a DETAILED and comprehensive response that directly addresses each specific aspect of the user's original query.
    2. **Crucially, DO NOT mention the internal quality check (judgement) or the agents involved in your final response.** The user should only see the final result.
    3. Extract and include specific, detailed information from the aggregated agent results rather than generalizing.
    4. Use the conversation history to maintain context, but prioritize answering the current query in detail.
    5. If the user requested specific details, facts, or examples, be sure to include them explicitly.
    6. Balance coherence with specificity - don't sacrifice important details for the sake of a smooth narrative.
    7. If the quality check indicated issues (e.g., 'not_aligned', 'hallucination', 'not_verified'), be *extra careful* to rely *only* on verified information from the agent results and context. If necessary, state that certain information could not be verified or obtained.
    8. Always cite your sources when providing information. Include clear references to where information was obtained from within the aggregated results.
    9. Use appropriate citation formats like [Source: X] or include footnotes when presenting facts, data, or specific claims.
    
    **Final Synthesized Response for the User:**
    """

    try:
        # Use litellm with Gemini model
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None,
            lambda: litellm.completion(
                model="gemini/gemini-2.5-pro-preview-03-25",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3  # Add a lower temperature for more focused responses
            )
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"❌ Error during synthesis with Gemini: {e}")
        # Fallback to OpenAI if Gemini fails
        try:
            print("⚠️ Falling back to OpenAI model...")
            loop = asyncio.get_running_loop()
            fallback_response = await loop.run_in_executor(
                None,
                lambda: client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": prompt}],
                )
            )
            return fallback_response.choices[0].message.content.strip()
        except Exception as fallback_e:
            print(f"❌ Fallback also failed: {fallback_e}")
            return "I encountered an error while trying to synthesize the final response."

