#!/usr/bin/env python3
"""
Agentic Chat System - Terminal-Only Interface

This script orchestrates the conversation between the user and AI agents
through a terminal interface, handling task analysis, planning, and execution.
"""

import sys
import asyncio
import time
from collections import defaultdict
from typing import Dict, List, Any, Tuple

import decide_complexity
import judge_possibility
from additional_information import ask_additional_information, NO_INFO_NEEDED
from planning import planning
from agent import create_agent, Agent
from council import council
from synthesizer import synthesize_response


class AgentSystem:
    """Main orchestrator for the terminal-based agentic system."""
    
    def __init__(self):
        """Initialize with empty conversation history."""
        self.conversation_history = []  # List of (role, message) tuples
        self.exit_keywords = ["exit", "quit", "bye"]

    def _update_history(self, role: str, message: str):
        """Add a message to conversation history."""
        self.conversation_history.append((role, message))

    def _get_history_str(self) -> str:
        """Convert history to a formatted string for context."""
        history_str = ""
        for role, message in self.conversation_history:
            history_str += f"{role}: {message}\n\n"
        return history_str

    async def _execute_plan(self, plan: Dict[str, Any]) -> str:
        """
        Execute a plan by creating and running agents in the correct order.
        
        Args:
            plan: The planning dict with task and agents
            
        Returns:
            Aggregated string of all agent outputs
        """
        if not plan or "agents" not in plan:
            return "🤖 No valid plan to execute."
            
        # Group agents by execution order
        order_groups = defaultdict(list)
        for i, agent_config in enumerate(plan["agents"]):
            order = agent_config.get("order", i)  # Use index as fallback
            order_groups[order].append(agent_config)
        
        # Maps agent order number to its output
        agent_outputs = {}
        
        print(f"🚀 Executing plan with {len(plan['agents'])} agents...")
        
        # Process agents in order
        for order in sorted(order_groups.keys()):
            agents_in_group = order_groups[order]
            group_size = len(agents_in_group)
            
            if group_size > 1:
                print(f"⏳ Executing {group_size} agents in parallel at order {order}...")
            else:
                print(f"⏳ Executing agent at order {order}...")
            
            # Create and initialize all agents in this order group
            agent_tasks = []
            for agent_config in agents_in_group:
                agent = create_agent(
                    agent_config,
                    self._get_history_str()
                )
                
                # Collect outputs from dependencies
                dependency_outputs = {}
                for dep_order in agent_config.get("dependencies", []):
                    if dep_order in agent_outputs:
                        dependency_outputs[dep_order] = agent_outputs[dep_order]
                
                # Schedule agent for execution
                agent_tasks.append(
                    self._run_agent(agent, agent_config.get("order", order), dependency_outputs)
                )
            
            # Run all agents in this order group concurrently
            results = await asyncio.gather(*agent_tasks)
            
            # Store outputs by order
            for agent_order, output in results:
                agent_outputs[agent_order] = output
        
        # Combine all outputs
        aggregated_output = "\n\n".join(
            [f"--- Agent Order {order} Output ---\n{output}" for order, output in sorted(agent_outputs.items())]
        )
        
        return aggregated_output

    async def _run_agent(self, agent: Agent, order: int, dependency_outputs: Dict[int, str]) -> Tuple[int, str]:
        """Run a single agent and return its output with order number."""
        try:
            output = await agent.run(dependency_outputs)
            return order, output
        except Exception as e:
            error_msg = f"❌ Error running {agent.type} agent (order {order}): {e}"
            print(error_msg)
            return order, error_msg

    async def process_query(self, user_query: str) -> str:
        """
        Process a user query through the entire agentic workflow.
        
        Args:
            user_query: The user's query/request
            
        Returns:
            The final synthesized response
        """
        # Add user query to history
        self._update_history("User", user_query)
        
        # 1. Determine complexity
        print("🧠 Analyzing complexity...")
        complexity = decide_complexity.decide_complexity(user_query, self._get_history_str())
        print(f"✅ Complexity determined: {complexity}")
        
        # 2. Judge possibility
        print("⚖️ Judging feasibility...")
        possibility_result = judge_possibility.judge_possibility(user_query, self._get_history_str())
        possibility = possibility_result['score']
        explanation = possibility_result['explanation']
        
        if possibility != "YES":
            print(f"❌ Task not feasible: {explanation}")
            research_response = input("🔍 This doesn't seem technically possible. Would you like me to research this topic instead? (yes/no): ")
            
            if research_response.lower() in ["yes", "y"]:
                print("📚 Switching to research mode...")
                # Create a simple research plan instead
                research_plan = {
                    "task": f"Research about: {user_query}",
                    "agents": [
                        {
                            "type": "researcher",
                            "order": 1,
                            "purpose": f"Research about {user_query}",
                            "dependencies": [],
                            "query": user_query
                        },
                        {
                            "type": "qa",
                            "order": 2,
                            "purpose": f"Summarize research findings about {user_query}",
                            "dependencies": [1],
                            "query": f"Based on the research, what are the key findings about '{user_query}'?"
                        }
                    ]
                }
                
                # Execute the research plan
                agent_outputs = await self._execute_plan(research_plan)
                
                # Council evaluation and synthesis - use synchronous call (signal handling limitation)
                judgement = council(agent_outputs, user_query, self._get_history_str())
                final_response = await synthesize_response(user_query, self._get_history_str(), agent_outputs, judgement)
                
                # Add response to history
                self._update_history("AI", final_response)
                return final_response
            else:
                response = f"I'm sorry, but I can't execute that task as it doesn't appear to be technically possible. {explanation}"
                self._update_history("AI", response)
                return response
        
        print("✅ Task is feasible")
        
        # 3. Check if additional information is needed
        print("🔍 Checking if additional information is needed...")
        additional_info = ask_additional_information(user_query, self._get_history_str())
        
        # If additional info is needed, ask the user
        if additional_info != NO_INFO_NEEDED:
            print("❓ Need more information")
            print(additional_info)
            
            # Keep asking until no more info is needed
            while additional_info != NO_INFO_NEEDED:
                # Get user's additional information
                user_response = input("🙋 Your response: ")
                self._update_history("User", user_response)
                
                # Check if the additional info is now sufficient
                print("🔄 Checking if this information is sufficient...")
                # No need to create enriched_query - we'll rely on the conversation history
                # that already includes the new response
                additional_info = ask_additional_information(user_query, self._get_history_str())
                
                if additional_info != NO_INFO_NEEDED:
                    print("❓ Still need more information")
                    print(additional_info)
                else:
                    print("✅ Got all the information I need")
                    
                    # Don't change the original query - the conversation history 
                    # already contains all the additional information
                    # Just add a note for planning
                    user_query = f"{user_query}\n[Additional information incorporated from conversation history]"
        else:
            print("✅ No additional information needed")
        
        # 4. Generate execution plan
        print("📋 Generating execution plan...")
        plan = planning(user_query, complexity, "", self._get_history_str())
        
        if not plan or "agents" not in plan:
            error_msg = "Failed to generate a valid execution plan."
            print(f"❌ {error_msg}")
            self._update_history("AI", error_msg)
            return error_msg
        
        # Display plan summary
        agent_count = len(plan["agents"])
        agent_types = set(agent["type"] for agent in plan["agents"])
        print(f"✅ Plan created with {agent_count} agents: {', '.join(agent_types)}")
        
        # 5. Execute the plan
        agent_outputs = await self._execute_plan(plan)
        
        # 6. Run council for evaluation
        print("⚖️ Evaluating results with council...")
        # Use synchronous call directly (signal handling limitation)
        judgement = council(agent_outputs, user_query, self._get_history_str())
        judgement_result = judgement.get("judgement", "unknown")
        print(f"✅ Council judgement: {judgement_result}")
        
        # 7. Synthesize final response
        print("🔄 Synthesizing final response...")
        final_response = await synthesize_response(user_query, self._get_history_str(), agent_outputs, judgement)
        
        # Add response to history
        self._update_history("AI", final_response)
        
        return final_response

    async def run_conversation_loop(self):
        """Main conversation loop to handle user interactions."""
        print("🤖 Welcome to the Agentic Assistant! Type 'exit' to quit at any time.")
        
        while True:
            user_query = input("\n🙋 What can I help you with today? ")
            
            # Check for exit command
            if user_query.lower() in self.exit_keywords:
                print("👋 Goodbye! Have a great day!")
                break
            
            # Process the query
            try:
                start_time = time.time()
                response = await self.process_query(user_query)
                end_time = time.time()
                
                # Display the response
                print(f"\n🤖 {response}")
                print(f"\n⏱️ Total processing time: {end_time - start_time:.2f} seconds")
            except Exception as e:
                print(f"❌ An error occurred while processing your request: {e}")


def main():
    """Main function to run the agentic system."""
    agent_system = AgentSystem()
    
    try:
        asyncio.run(agent_system.run_conversation_loop())
    except KeyboardInterrupt:
        print("\n👋 Program interrupted. Goodbye!")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
