from openai import OpenAI
import json

client = OpenAI()

def planning(task, complexity="N/A", additional_info="", conversation_history_str=""):
    """Generates an agent execution plan based on task, complexity, additional info, and history."""
    # Include additional information in the prompt if provided
    additional_info_text = f"\nAdditional information provided by user: {additional_info}" if additional_info else ""
    # Include conversation history if provided
    history_context = f"\n\nConversation History:\n{conversation_history_str}" if conversation_history_str else ""

    prompt_content = f"""
        You are a task planner. You will be given a task, its assessed complexity, any additional info provided by the user, and the conversation history. Plan out the AI Agents required to complete the task.
        {history_context}

        Task: {task}
        Complexity: {complexity}
        {additional_info_text}

        IMPORTANT: Balance efficiency with parallelization:
        - For trivial tasks (like basic math or yes/no questions), use only ONE agent
        - For tasks that can be broken down into independent subtasks, use MULTIPLE agents working in parallel
        - Prioritize SPEED over minimal agent count when tasks can be parallelized
        - Look for opportunities to split work into concurrent streams
        - Example: For a presentation about history, use multiple researchers to gather different time periods simultaneously

        Your response must be in valid JSON format with the following schema:
        {{
            "task": "the original task description",
            "agents": [
                {{
                    "type": "agent type name",
                    "order": number indicating execution order (same order number means parallel execution),
                    "purpose": "description of what this agent instance will do",
                    "dependencies": [list of agent orders this depends on, if any],
                    "query": "specific query for researcher or analyst agents (only include for these types)"
                }}
            ]
        }}

        Available agent types are:
        - "researcher": Can do research on the internet for information based on a query
        - "qa": Can write text based content to answer questions based on provided context
        - "contextualizer": Can search through user's files/documents/database to find relevant information (specify search terms and folder/source if known)
        - "analyst": Can analyze data and provide insights using mermaid.js diagrams (specify diagram type and data)

        PARALLELIZATION GUIDELINES:
        - Agents with the same order number will run in parallel
        - Break down large research/analysis tasks into multiple concurrent agents
        - Use multiple agents of the same type if they can work on different aspects simultaneously
        - Consider data dependencies when assigning order numbers
        - For complex tasks, aim to have multiple execution streams running in parallel
        - IMPORTANT: Limit researcher agents to AT MOST 2 with the same order number to avoid API rate limits
        - For multiple researcher agents, assign them different order numbers in sequence (1, 2, 3, etc.) 
          rather than running too many in parallel

        QA AGENT GUIDELINES:
        - IMPORTANT: QA agents should NOT be used for synthesizing or consolidating information to be concise or comprehensive.
        - There is a separate post-processing step after agent execution that handles synthesis and summarization.
        - QA agents should ONLY extract and provide direct, factual information from the context they receive.
        - Use QA agents for straightforward information retrieval and direct question answering.
        - QA agents should ONLY be used for queries that can be solved very quickly without requiring multiple agents.
        - For complex questions requiring extensive research or analysis, use researcher or analyst agents instead.
        - DO NOT assign tasks like "synthesize findings" or "create a comprehensive summary" to QA agents.
        - QA agents should focus on extracting specific information based on clear queries.
        - For multiple research findings, prefer to use separate QA agents to extract specific subsets of information rather than a single agent for comprehensive synthesis.

        CONTEXTUALIZER AGENT GUIDELINES:
        - ALWAYS include at least one contextualizer agent when the task involves finding or recalling personal documents/files
        - Contextualizer agents MUST be used for any request containing phrases like "find that document", "that file about...", etc.
        - Each contextualizer agent MUST have a "query" field with specific search terms to look for in files
        - When users refer to documents they've previously seen or created, ALWAYS use a contextualizer agent
        - Contextualizer agents should be among the FIRST agents created for any document retrieval request
        - Document search queries MUST NOT be routed to researcher agents - use contextualizer agents instead
        - Example scenarios requiring contextualizer agents:
          * When users ask to find documents they've previously mentioned
          * When users ask about specific content they remember but need help finding
          * When users use phrases like "can you find that document talking about X"
          * When users can't remember exact details but want to locate previously seen information
        - Contextualizer query format:
          * Provide clear search terms extracted from the user's request
          * BAD QUERY: "Find document" (too vague)
          * GOOD QUERIES:
            - "Search for documents containing information about Marco Polo or similar historical figures"
            - "Find files mentioning fashion trends with keywords: styles, clothing, seasonal"
            - "Locate documents with financial data including terms: budget, expenses, Q2 results"
          * Include all potential relevant keywords, especially names, dates, topics, and unique terms
          * When user is uncertain, include multiple possible search terms to increase chances of finding the right document

        RESEARCHER AGENT LIMITATIONS:
        - Each researcher agent can only handle ONE specific, focused query at a time
        - Each query MUST be answerable with a single search
        - Complex research tasks MUST be broken down into multiple researcher agents
        - NEVER combine multiple aspects into one query
        - BAD QUERY EXAMPLE: "Compare course offerings, programs, and research opportunities"
        - GOOD QUERY EXAMPLES:
          - "What undergraduate CS programs does Boston University offer?"
          - "What is Northeastern's computer science research focus areas?"
          - "Compare admission requirements for CS at both universities"
        - Each researcher agent must have a "query" field with a specific, focused search query
        - RATE LIMIT WARNING: No more than 2 researcher agents should have the same order number

        ANALYST AGENT LIMITATIONS:
        - Each analyst agent can only handle ONE specific analysis or visualization at a time
        - Each analysis MUST focus on a single metric, trend, or relationship
        - Complex analysis tasks MUST be broken down into multiple analyst agents
        - NEVER combine multiple analyses into one task
        - BAD ANALYSIS EXAMPLE: "Create charts showing user growth, retention, and engagement over time"
        - GOOD ANALYSIS EXAMPLES:
          - "Create a mermaid.js line chart showing monthly user growth rate for 2023"
          - "Generate a mermaid.js bar chart comparing retention rates across user segments"
          - "Plot a mermaid.js heatmap of user engagement by time of day"
        - Each analyst agent must have a "query" field that EXPLICITLY includes "mermaid.js" and specifies:
          1. Exact type of mermaid.js diagram (flowchart, sequence, gantt, etc.)
          2. Single metric/relationship being visualized
          3. Clear data requirements

        DEPENDENCIES FORMAT:
        - Dependencies must be a list of INTEGER order numbers, not strings
        - Example: "dependencies": [1, 2, 3] is correct, "dependencies": ["1", "2", "3"] is incorrect
        - Dependencies should reference the order numbers of agents that must complete before this agent starts
        - If an agent has no dependencies, use an empty list: "dependencies": []
        - Agents that process or analyze data MUST list ALL order numbers of agents they depend on
        - For example, if a QA agent synthesizes data from 5 researcher agents with order=1, it must list [1] as its dependency

        Consider the conversation history for context and avoid redundant actions if information was already gathered or provided.
    """

    message = client.chat.completions.create(
            model="o3-mini", # Updated model
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert AI agent planner."
                },
                {
                    "role": "user",
                    "content": prompt_content
                }
            ],
            response_format={"type": "json_object"} # Request JSON output
        )
    
    try:
        plan = json.loads(message.choices[0].message.content)
        
        # Validate and fix dependencies format
        for agent in plan.get("agents", []):
            # Ensure dependencies is a list of integers
            if "dependencies" in agent:
                if not isinstance(agent["dependencies"], list):
                    agent["dependencies"] = [] # Default to empty list if invalid format
                else:
                    # Ensure all items in the list are integers
                    agent["dependencies"] = [int(dep) for dep in agent["dependencies"] if isinstance(dep, (int, str)) and str(dep).isdigit()]
            else:
                 agent["dependencies"] = [] # Add empty list if missing
            
            # Ensure analyst agents specify mermaid.js in their query
            if agent.get("type") == "analyst":
                if "query" not in agent:
                    raise ValueError("Analyst agents must have a query field")
                if not any(phrase in agent["query"].lower() for phrase in ["mermaid.js", "mermaid diagram"]):
                    raise ValueError("Analyst agents must explicitly specify mermaid.js diagram type in query")
        
        # Enforce rate limit by ensuring no more than 2 researcher agents have the same order number
        order_counts = {}
        for agent in plan.get("agents", []):
            if agent.get("type") == "researcher":
                order = agent.get("order", 0)
                order_counts[order] = order_counts.get(order, 0) + 1
        
        # If any order has more than 2 researcher agents, redistribute them
        next_order = max(order_counts.keys(), default=0) + 1
        for order, count in order_counts.items():
            if count > 2:
                # Find agents to redistribute
                extras = count - 2
                for agent in plan.get("agents", []):
                    if extras > 0 and agent.get("type") == "researcher" and agent.get("order") == order:
                        # Update this agent's order to the next available
                        agent["order"] = next_order
                        # Update any dependencies that might have referenced the old order
                        # This is a simplified approach - in a real system you'd want to update
                        # all interdependencies more carefully
                        print(f"⚠️ Redistributed researcher agent from order {order} to {next_order} to avoid rate limits")
                        extras -= 1
                        next_order += 1
        
        # print(plan) # Removed internal print for cleaner output
        return plan
    except json.JSONDecodeError as e: # Catch JSON errors specifically
        print(f"❌ Error decoding JSON plan: {e}")
        print("Raw response content:")
        print(message.choices[0].message.content) # Print raw response on error
        raise ValueError("Failed to generate valid JSON plan")
    except ValueError as ve:
        print(f"❌ Validation Error in plan: {ve}") # Print validation errors
        raise ve # Re-raise validation errors
    except Exception as ex:
        print(f"❌ Unexpected error during planning: {ex}") # Catch other potential errors
        raise ex
    