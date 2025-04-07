from openai import OpenAI
from verdict.common.judge import BestOfKJudgeUnit
from exa_py import Exa
import asyncio
import json
from verdict import Pipeline
from verdict.extractor import TokenProbabilityExtractor
from verdict.schema import Schema
import os
import glob
import textract
import abc
import time
from typing import Dict, List, Any
from verdict.util import ratelimit
import litellm
ratelimit.disable()

client = OpenAI()
# Ensure EXA_API_KEY environment variable is set or add fallback
exa = Exa(api_key=os.getenv("EXA_API_KEY"))

# Try importing mermaid, handle potential import error gracefully
try:
    import mermaid as md
    from mermaid.graph import Graph
    MERMAID_AVAILABLE = True
except ImportError:
    print("⚠️ Warning: `mermaid` library not found. Analyst agent cannot generate PNG files.")
    MERMAID_AVAILABLE = False

class Agent(abc.ABC):
    """Abstract base class for all agents."""
    def __init__(self, agent_type: str, purpose: str, dependencies: List[int], query: str, conversation_history: str):
        self.type = agent_type
        self.purpose = purpose
        self.dependencies = dependencies # List of order numbers this agent depends on
        self.query = query # Specific query/instruction for this agent
        self.conversation_history = conversation_history # Full conversation history

    def display_agent_info(self) -> None:
        """Displays the agent type, purpose and query before execution."""
        print(f"\n⏳ Starting {self.type.capitalize()} Agent ⏳")
        print(f"Purpose: {self.purpose}")
        print(f"Query: {self.query}")
        print("------------------------------")

    @abc.abstractmethod
    async def run(self, dependency_outputs: Dict[int, str]) -> str:
        """
        Executes the agent's task asynchronously.

        Args:
            dependency_outputs: A dictionary mapping dependency agent order numbers
                                to their string outputs.

        Returns:
            A formatted string summarizing the agent's actions and results.
        """
        # Display agent information before execution
        self.display_agent_info()
        pass

    def _get_dependencies_context(self, dependency_outputs: Dict[int, str]) -> str:
        """Helper to format context from dependencies."""
        if not dependency_outputs:
            return "No dependencies provided context."
        context = "\n--- Context from Dependencies ---\n"
        for order, output in dependency_outputs.items():
            context += f"Output from Agent Order {order}:\n{output}\n---\n"
        return context

class ResearcherAgent(Agent):
    """Agent responsible for external research using Exa."""
    def __init__(self, purpose: str, dependencies: List[int], query: str, conversation_history: str):
        super().__init__("researcher", purpose, dependencies, query, conversation_history)
        self.max_queries_to_generate = 6 # Reduced for speed/cost
        self.max_queries_to_use = 2     # Reduced to 2 to avoid rate limits
        self.num_search_results = 5
        self.rate_limit_pause = 0.2     # Seconds to pause between API calls
        self.use_sequential_search = True  # Set to True to avoid parallel search rate limits
        self.max_retries = 3  # Maximum number of retries for rate-limited requests

    async def _create_search_queries(self) -> str:
        """Generates potential search queries using an LLM."""
        prompt = f"""
        You are a search query generator. Your task is to create {self.max_queries_to_generate} diverse and effective search queries for researching the following purpose, based on the initial query and conversation history.

        Purpose: {self.purpose}
        Initial Query: {self.query}
        Conversation History:
        {self.conversation_history}

        Guidelines for creating queries:
        1. EXTREMELY IMPORTANT: Keep all queries short and concise (maximum 5-7 words).
        2. Focus on specific technical terms only.
        3. Eliminate all unnecessary words and context.
        4. Avoid complete sentences - use keyword phrases instead.
        5. Each query should be direct and to the point.
        6. Consider the conversation history to avoid redundancy.

        Return ONLY a JSON array of exactly {self.max_queries_to_generate} short string queries without any explanation or commentary. Example format:
        ["verdict library overview", "python verdict judge usage", ... "verdict tokenizer example"]
        """
        try:
            # Use loop.run_in_executor to run the blocking OpenAI call asynchronously
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                None,
                lambda: client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"}
                )
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"❌ Error generating search queries: {e}")
            return json.dumps({"error": "Failed to generate queries", "queries": []}) # Return valid JSON on error

    async def _process_query_group(self, queries_list: List[str], group: int, prompt: str, group_size: int) -> Dict[int, float]:
        """Processes a group of queries using BestOfKJudgeUnit."""
        start_idx = group * group_size
        end_idx = min(start_idx + group_size, len(queries_list))
        group_queries = queries_list[start_idx:end_idx]
        if not group_queries:
            return {}

        group_prompt = prompt + "\n"
        for i, query in enumerate(group_queries):
            letter = chr(65 + i)  # A, B, C, etc.
            group_prompt += f"{letter}: {query}\n"

        try:
            # Verdict library may use signals - run synchronously
            pipeline = Pipeline() >> BestOfKJudgeUnit(k=len(group_queries)).prompt(group_prompt).extract(TokenProbabilityExtractor())
            response, _ = pipeline.run(Schema.of(options=group_queries))
            
            distribution = response.get('Pipeline_root.block.unit[BestOfKJudge]_distribution', {})
            result = {}
            for letter, prob in distribution.items():
                letter_idx = ord(letter) - ord('A')
                orig_idx = start_idx + letter_idx
                if 0 <= orig_idx < len(queries_list): # Bounds check
                     result[orig_idx] = prob
            return result
        except Exception as e:
            print(f"❌ Error processing query group {group}: {e}")
            return {} # Return empty dict on error

    async def _decide_on_queries_async(self, search_queries_raw: str) -> Dict[str, Any]:
        """Ranks and selects the best search queries asynchronously."""
        try:
            search_queries_data = json.loads(search_queries_raw)
            # Handle potential dict format {'queries': [...]}
            if isinstance(search_queries_data, dict):
                 queries_list = search_queries_data.get('queries', [])
            elif isinstance(search_queries_data, list):
                 queries_list = search_queries_data
            else:
                 print(f"⚠️ Unexpected format for search queries: {type(search_queries_data)}")
                 queries_list = []

            if not queries_list:
                 return {"top_queries": [], "explanation": "No valid search queries generated."}

            # Ensure we don't process more queries than generated
            num_queries_to_process = min(self.max_queries_to_generate, len(queries_list))
            queries_list = queries_list[:num_queries_to_process]

        except json.JSONDecodeError as e:
            print(f"❌ Error decoding search queries JSON: {e}")
            print(f"Raw content: {search_queries_raw}")
            return {"top_queries": [], "explanation": "Failed to decode generated queries JSON."}
        except Exception as e:
            print(f"❌ Unexpected error processing generated queries: {e}")
            return {"top_queries": [], "explanation": f"Error processing queries: {e}"}

        if len(queries_list) < self.max_queries_to_use:
            print(f"⚠️ Warning: Generated only {len(queries_list)} queries, using all.")
            return {"top_queries": queries_list, "explanation": "Used all generated queries due to low count."}

        prompt = f"""
        You are a search query evaluator. Below are search queries for researching: {self.purpose}
        Context: {self.query}
        Conversation History: {self.conversation_history}

        Evaluate each query's potential effectiveness. Consider:
        1. CONCISENESS - Strongly prefer the shortest, most direct queries (5-7 words is ideal)
        2. Relevance to the research purpose
        3. Use of specific technical terms
        4. Precision and clarity
        5. Likelihood of retrieving useful information

        Select the single best query from the options provided below, with a strong preference for the most concise options that maintain relevance:
        """

        group_size = 5 # Process in groups of 5
        num_groups = (len(queries_list) + group_size - 1) // group_size

        tasks = [self._process_query_group(queries_list, group, prompt, group_size) for group in range(num_groups)]
        group_results = await asyncio.gather(*tasks)

        all_distributions = {}
        for result in group_results:
            all_distributions.update(result)

        if not all_distributions:
             print("⚠️ No query distributions obtained from evaluator.")
             # Fallback: just take the first few queries
             top_queries = queries_list[:self.max_queries_to_use]
             explanation = f"Fell back to using the first {len(top_queries)} generated queries due to evaluation error."
             return {"top_queries": top_queries, "explanation": explanation}

        # Sort queries by probability score, highest first
        # Filter out invalid indices before sorting
        valid_sorted_indices = sorted(
            [(idx, score) for idx, score in all_distributions.items() if 0 <= idx < len(queries_list)],
            key=lambda item: item[1],
            reverse=True
        )

        # Get top N queries based on sorted indices
        top_indices = [idx for idx, _ in valid_sorted_indices[:self.max_queries_to_use]]
        top_queries = [queries_list[idx] for idx in top_indices]

        return {
            "top_queries": top_queries,
            "explanation": f"Selected top {len(top_queries)} queries based on evaluation."
        }

    async def _search_query_async(self, query: str) -> List[Dict[str, str]]:
        """Executes a single search query using Exa asynchronously with retries and backoff."""
        for retry in range(self.max_retries + 1):
            try:
                # Use asyncio.to_thread for the blocking SDK call
                loop = asyncio.get_running_loop()
                search_response = await loop.run_in_executor(
                    None,
                    lambda: exa.search_and_contents(
                        query,
                        num_results=self.num_search_results,
                        text=True # Request text content
                    )
                )
                
                # Format results
                results = []
                if hasattr(search_response, 'results'):
                     for res in search_response.results:
                         results.append({
                             "title": getattr(res, 'title', 'N/A'),
                             "url": getattr(res, 'url', 'N/A'),
                             "text_snippet": getattr(res, 'text', 'N/A')[:5000] + "..." # Include snippet
                         })
                return results if results else [{"info": "No results found"}]
                
            except Exception as e:
                error_message = str(e)
                # Check if it's a rate limit error
                if "429" in error_message and retry < self.max_retries:
                    backoff_time = self.rate_limit_pause * (2 ** retry)  # Exponential backoff
                    print(f"⚠️ Rate limit hit, retrying in {backoff_time:.2f}s (attempt {retry+1}/{self.max_retries})")
                    await asyncio.sleep(backoff_time)
                    continue
                else:
                    print(f"❌ Error during Exa search for query '{query}': {e}")
                    return [{"error": f"Search failed: {e}"}]
        
        # If we got here, we've exhausted all retries
        return [{"error": "Search failed after maximum retries due to rate limits"}]

    async def _search_sequential(self, queries: List[str]) -> List[List[Dict[str, str]]]:
        """Execute searches sequentially with pauses to avoid rate limits."""
        results = []
        for query in queries:
            # Add a pause between requests to avoid rate limiting
            await asyncio.sleep(self.rate_limit_pause)
            result = await self._search_query_async(query)
            results.append(result)
        return results

    async def run(self, dependency_outputs: Dict[int, str]) -> str:
        """Runs the researcher agent's workflow."""
        # Display agent information before execution
        self.display_agent_info()
        
        start_time = time.time()
        # 1. Generate queries
        queries_raw = await self._create_search_queries()

        # 2. Decide on best queries
        ranked_queries_result = await self._decide_on_queries_async(queries_raw)
        top_queries = ranked_queries_result["top_queries"]
        ranking_explanation = ranked_queries_result["explanation"]

        # 3. Execute searches - either in parallel or sequentially based on setting
        if self.use_sequential_search:
            # Sequential execution with pauses to avoid rate limits
            print(f"🔄 Running {len(top_queries)} searches sequentially (with rate limiting)...")
            search_results_list = await self._search_sequential(top_queries)
        else:
            # Parallel execution (may hit rate limits with many queries)
            print(f"🔄 Running {len(top_queries)} searches in parallel...")
            search_tasks = [self._search_query_async(query) for query in top_queries]
            search_results_list = await asyncio.gather(*search_tasks) # Wait for all searches

        # 4. Format output
        output = f"🔎 **Researcher Agent Summary** 🔎\n"
        output += f"Purpose: {self.purpose}\n"
        output += f"Query: {self.query}\n"
        output += f"Selected Queries ({ranking_explanation}):\n"
        if top_queries:
            for i, q in enumerate(top_queries):
                output += f"  {i+1}. {q}\n"
        else:
             output += "  - No queries selected or generated.\n"

        output += f"\n**Search Results:**\n"
        any_results = False
        for i, results in enumerate(search_results_list):
            query_used = top_queries[i] if i < len(top_queries) else "N/A"
            output += f"\n*Results for query: '{query_used}'*\n"
            if results and not results[0].get("error") and not results[0].get("info"):
                 any_results = True
                 for j, res_dict in enumerate(results):
                     output += f"  {j+1}. Title: {res_dict.get('title', 'N/A')}\n"
                     output += f"     URL: {res_dict.get('url', 'N/A')}\n"
                     output += f"     Snippet: {res_dict.get('text_snippet', 'N/A')}\n" # Optional: uncomment to include snippets
            elif results and results[0].get("error"):
                 output += f"  - Error: {results[0]['error']}\n"
            else:
                 output += f"  - No results found or error occurred.\n"

        if not any_results and not top_queries:
             output += "- No searches performed.\n"
        elif not any_results:
             output += "- No valid results obtained from searches.\n"

        end_time = time.time()
        output += f"\nExecution Time: {end_time - start_time:.2f} seconds\n"
        output += "--------------------\n"
        return output

class QAAgent(Agent):
    """Agent responsible for answering questions based on provided context."""
    def __init__(self, purpose: str, dependencies: List[int], query: str, conversation_history: str):
        super().__init__("qa", purpose, dependencies, query, conversation_history)

    async def _answer_question(self, context: str) -> str:
        """Generates an answer using an LLM based on the query and context."""
        prompt = f"""
        You are a Question Answering Assistant. Your purpose is: {self.purpose}.
        Answer the following question based *only* on the provided context and conversation history.

        Conversation History:
        {self.conversation_history}

        Context from Previous Steps:
        {context}

        Question: {self.query}

        Instructions:
        - IMPORTANT: DO NOT synthesize, summarize, or consolidate information to be concise or comprehensive. This will be done in a separate post-processing step.
        - Your role is to provide direct, factual responses based exactly on the information in the context.
        - Focus on extracting and presenting the relevant information as-is from the provided context.
        - If the context doesn't contain sufficient information, state that clearly.
        - Do not add your own analysis, opinions, or external knowledge not found in the context.
        - Include specific details and maintain the technical accuracy of the information.

        Answer:
        """
        try:
            # Use loop.run_in_executor to run the blocking OpenAI call asynchronously
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                None,
                lambda: client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": f"You are a helpful QA assistant that provides direct, factual information without synthesizing or summarizing. Purpose: {self.purpose}"},
                        {"role": "user", "content": prompt}
                    ]
                )
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"❌ Error during QA generation: {e}")
            return f"Error generating answer: {e}"

    async def run(self, dependency_outputs: Dict[int, str]) -> str:
        """Runs the QA agent's workflow."""
        # Display agent information before execution
        self.display_agent_info()
        
        start_time = time.time()
        context = self._get_dependencies_context(dependency_outputs)
        answer = await self._answer_question(context)
        end_time = time.time()

        output = f"🗣️ **QA Agent Summary** 🗣️\n"
        output += f"Purpose: {self.purpose}\n"
        output += f"Query/Question: {self.query}\n"
        output += f"Generated Answer:\n{answer}\n"
        output += f"\nExecution Time: {end_time - start_time:.2f} seconds\n"
        output += "--------------------\n"
        return output


class ContextExtractorAgent(Agent):
    """Agent responsible for extracting information from local files."""
    def __init__(self, purpose: str, dependencies: List[int], query: str, conversation_history: str, context_folder: str = "context"):
        super().__init__("contextualizer", purpose, dependencies, query, conversation_history)
        self.context_folder = context_folder
        # Define supported extensions (consider security implications of executing textract)
        self.supported_extensions = ["*.csv", "*.doc", "*.docx", "*.eml", "*.epub", "*.gif", "*.jpg", "*.jpeg",
                                      "*.json", "*.html", "*.htm", "*.mp3", "*.msg", "*.odt", "*.ogg", "*.pdf",
                                      "*.png", "*.pptx", "*.ps", "*.rtf", "*.tiff", "*.tif", "*.txt", "*.wav",
                                      "*.xlsx", "*.xls", "*.md"]

    def _extract_text_from_file(self, file_path: str) -> str:
        """Synchronous function to extract text from a single file."""
        try:
            if file_path.endswith('.md'):
                with open(file_path, 'r', encoding='utf-8') as file:
                    return file.read()
            else:
                # textract can be slow and resource-intensive
                return textract.process(file_path).decode('utf-8')
        except Exception as e:
            print(f"⚠️ Error extracting text from {os.path.basename(file_path)}: {str(e)}")
            return f"[Error extracting text: {e}]"

    async def _extract_all_text(self) -> Dict[str, str]:
        """Extracts text from all supported files asynchronously."""
        file_texts = {}
        if not os.path.exists(self.context_folder):
            print(f"⚠️ Context folder '{self.context_folder}' not found.")
            return file_texts

        # Run synchronous extraction in threads
        loop = asyncio.get_running_loop()
        tasks = []
        found_files = []

        for ext in self.supported_extensions:
            file_pattern = os.path.join(self.context_folder, ext)
            found_files.extend(glob.glob(file_pattern))

        for file_path in found_files:
             # Pass file_path to the function using lambda to capture it correctly
             task = loop.run_in_executor(None, lambda p=file_path: (p, self._extract_text_from_file(p)))
             tasks.append(task)

        results = await asyncio.gather(*tasks)

        for file_path, text in results:
             file_texts[os.path.basename(file_path)] = text # Store by basename

        return file_texts


    async def _extract_relevant_information(self, all_text: Dict[str, str]) -> str:
        """Uses LLM to extract relevant information based on query and purpose."""
        if not all_text:
            return "No text could be extracted from the context folder."

        all_content = ""
        total_len = 0
        max_len = 30000 # Limit context to LLM

        for file_name, content in all_text.items():
            # Truncate individual file content first
            truncated_content = content[:max(2000, max_len // len(all_text))] + "... [truncated]" if len(content) > 2000 else content
            entry = f"--- File: {file_name} ---\n{truncated_content}\n\n"
            if total_len + len(entry) > max_len:
                all_content += "... [Overall context truncated]\n"
                break
            all_content += entry
            total_len += len(entry)


        prompt = f"""
        You are an information extraction expert. Extract the most relevant information from the provided documents based on the user's query, purpose, and conversation history.

        Purpose: {self.purpose}
        Query: {self.query}
        Conversation History:
        {self.conversation_history}

        Documents Content (potentially truncated):
        {all_content}

        Instructions:
        1. Identify and extract information *highly relevant* to the query and purpose.
        2. Prioritize relevance.
        3. Summarize key findings concisely.
        4. Mention source file names if quoting specific details.
        5. If information is contradictory, note it.
        6. If no relevant information is found, state that.

        Relevant Information Summary:
        """
        try:
            # Use loop.run_in_executor to run the blocking OpenAI call asynchronously
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                None,
                lambda: client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You extract relevant information from documents based on context."},
                        {"role": "user", "content": prompt}
                    ]
                )
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"❌ Error during context relevance analysis: {e}")
            return f"Error analyzing context: {e}"

    async def run(self, dependency_outputs: Dict[int, str]) -> str:
        """Runs the context extractor agent's workflow."""
        # Display agent information before execution
        self.display_agent_info()
        
        start_time = time.time()
        # Note: Dependencies are ignored by this agent as it reads from the filesystem
        # context = self._get_dependencies_context(dependency_outputs) # Can be added if needed

        # 1. Extract text from files
        extracted_texts = await self._extract_all_text()
        num_files_processed = len(extracted_texts)
        num_files_errors = sum(1 for text in extracted_texts.values() if text.startswith("[Error"))

        # 2. Analyze for relevance
        relevant_info_summary = await self._extract_relevant_information(extracted_texts)
        end_time = time.time()

        output = f"📁 **Context Extractor Agent Summary** 📁\n"
        output += f"Purpose: {self.purpose}\n"
        output += f"Query: {self.query}\n"
        output += f"Scanned Folder: '{self.context_folder}'\n"
        output += f"Files Processed: {num_files_processed} (Errors: {num_files_errors})\n"
        output += f"Relevant Information Summary:\n{relevant_info_summary}\n"
        output += f"\nExecution Time: {end_time - start_time:.2f} seconds\n"
        output += "--------------------\n"
        return output


class AnalystAgent(Agent):
    """Agent responsible for generating analysis and diagrams (Mermaid)."""
    def __init__(self, purpose: str, dependencies: List[int], query: str, conversation_history: str, diagram_type: str = "flowchart", output_name: str = None):
        super().__init__("analyst", purpose, dependencies, query, conversation_history)
        # Use provided diagram type or default, no query detection
        self.diagram_type = diagram_type.lower() # Normalize
        base_output_name = output_name or f"{self.diagram_type}_analysis"
        # Clean output name for filesystem
        safe_output_name = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in base_output_name)
        self.output_name = safe_output_name
        self.diagrams_dir = "diagrams" # Directory to save diagrams
    
    async def _generate_mermaid_code(self, context: str, attempt_number: int, previous_errors: List[str]) -> str:
        """Generates Mermaid code using Gemini via LiteLLM, incorporating context and error feedback."""
        error_context = ""
        if attempt_number > 1 and previous_errors:
            last_error = previous_errors[-1]
            error_context = f"\nPrevious attempt failed with error: {last_error}\nPlease analyze this error and generate corrected Mermaid code."
        
        prompt = f"""
        You are an expert Mermaid diagram generator specializing in '{self.diagram_type}' diagrams.
        Your purpose is: {self.purpose}
        User Query: {self.query}
        Conversation History:
        {self.conversation_history}
        Context from Previous Steps:
        {context}
        {error_context}
        
        Instructions:
        - Analyze the query, history, and context.
        - Generate ONLY the raw Mermaid code compatible with mermaid.js syntax for a '{self.diagram_type}' diagram based on the provided context.
        - Do NOT include explanations, comments, backticks (```mermaid or ```), or any text other than the valid Mermaid code itself.
        
        Generate the Mermaid '{self.diagram_type}' code now:
        """
        try:
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                None,
                lambda: litellm.completion(
                    model="gemini/gemini-2.5-pro-preview-03-25", # Use Gemini 2.5 Pro via LiteLLM
                    messages=[
                        {"role": "system", "content": f"You are an expert Mermaid {self.diagram_type} diagram generator."},
                        {"role": "user", "content": prompt}
                    ]
                )
            )
            mermaid_code = response.choices[0].message.content.strip()
            
            # Basic cleanup (keep this part)
            if mermaid_code.startswith("```mermaid"):
                mermaid_code = mermaid_code.split("\n", 1)[1]
            if mermaid_code.endswith("```"):
                mermaid_code = mermaid_code.rsplit("\n", 1)[0]
            if mermaid_code.startswith("```"):
                mermaid_code = mermaid_code.strip("`")
                
            mermaid_code = mermaid_code.strip()
            
            # Check for explicit error from LLM
            if "ERROR:" in mermaid_code or not mermaid_code: # Also check if empty
                error_detail = mermaid_code if mermaid_code else "LLM returned empty response."
                print(f"⚠️ LLM indicated diagram generation error: {error_detail}")
                return f"ERROR: {error_detail}" # Propagate error
            
            return mermaid_code
        except Exception as e:
            print(f"❌ Error during Mermaid code generation with LiteLLM (Attempt {attempt_number}): {e}")
            # Check if the error is specific, e.g., API key missing
            error_str = str(e)
            if "authentication" in error_str.lower() or "api key" in error_str.lower():
                print("   Hint: Ensure necessary API keys (e.g., GOOGLE_API_KEY) are set for LiteLLM.")
            return f"ERROR: LiteLLM call failed: {e}"
    
    async def _save_diagram_png(self, mermaid_code: str, png_path: str) -> bool:
        """Saves the Mermaid code as a PNG image using the library."""
        if not MERMAID_AVAILABLE:
            print("❌ Cannot save PNG: Mermaid library not installed.")
            return False
        try:
            # Run synchronous/blocking mermaid call in a thread
            loop = asyncio.get_running_loop()
            # Use a local function for the synchronous part to avoid self issues in lambda
            def save_sync():
                graph = Graph(self.output_name, mermaid_code)
                mermaid_diagram = md.Mermaid(graph)
                mermaid_diagram.to_png(png_path)
            
            await loop.run_in_executor(None, save_sync)
            return True
        except Exception as e:
            print(f"❌ Error saving diagram to {png_path}: {e}")
            # Attempt to provide more specific feedback if possible
            if "syntax error" in str(e).lower():
                 print("   Hint: The generated Mermaid code likely has a syntax error.")
            return False

    async def run(self, dependency_outputs: Dict[int, str]) -> str:
        """Runs the analyst agent's workflow."""
        # Display agent information before execution
        self.display_agent_info()
        
        start_time = time.time()
        context = self._get_dependencies_context(dependency_outputs)
        previous_errors = []
        max_attempts = 3 # Limit attempts
        mermaid_code = ""
        png_path = os.path.join(self.diagrams_dir, f"{self.output_name}.png")
        success = False
        final_message = "Diagram generation failed."

        # Create diagrams directory if it doesn't exist
        if not os.path.exists(self.diagrams_dir):
            try:
                os.makedirs(self.diagrams_dir)
            except OSError as e:
                print(f"❌ Failed to create diagrams directory '{self.diagrams_dir}': {e}")
                # Cannot proceed without directory
                output = f"📊 **Analyst Agent Summary** 📊\n"
                output += f"Purpose: {self.purpose}\n"
                output += f"Query: {self.query}\n"
                output += f"Result: Failed - Could not create output directory '{self.diagrams_dir}'.\n"
                output += f"Execution Time: {time.time() - start_time:.2f} seconds\n"
                output += "--------------------\n"
                return output

        for attempt in range(1, max_attempts + 1):
            # 1. Generate Mermaid code
            mermaid_code = await self._generate_mermaid_code(context, attempt, previous_errors)

            if mermaid_code.startswith("ERROR:"):
                error_msg = f"Attempt {attempt}: Failed to generate valid Mermaid code. Reason: {mermaid_code}"
                print(f"⚠️ {error_msg}")
                previous_errors.append(error_msg)
                final_message = f"Failed to generate Mermaid code after {attempt} attempts. Last error: {mermaid_code}"
                if attempt < max_attempts:
                     await asyncio.sleep(1) # Wait before retrying
                     continue # Try again
                else:
                     break # Max attempts reached


            # 2. Attempt to save as PNG (if library available)
            if MERMAID_AVAILABLE:
                save_success = await self._save_diagram_png(mermaid_code, png_path)
                if save_success:
                    final_message = f"Successfully created '{self.diagram_type}' diagram and saved to '{png_path}'."
                    success = True
                    break # Exit loop on success
                else:
                    # Use the generated code even if PNG saving fails, but note the error
                    error_msg = f"Attempt {attempt}: Generated Mermaid code but failed to save diagram PNG (check Mermaid syntax/logs)."
                    print(f"⚠️ {error_msg}")
                    previous_errors.append(error_msg) # Add error for next attempt's context
                    final_message = f"Generated Mermaid code for '{self.diagram_type}' but failed to save PNG to '{png_path}'. Last error: {previous_errors[-1]}"
                    # Let's consider code generation a success but PNG saving a failure
                    # If we want to retry code generation on PNG failure, change logic here
                    if attempt < max_attempts:
                         await asyncio.sleep(1) # Wait before retrying generation
                         continue # Try generating different code
                    else:
                         success = False # Mark overall as failure if PNG save fails after retries
                         break # Max attempts reached for generation/saving cycle

            else:
                 # Mermaid library not available, only provide the code
                 final_message = f"Generated Mermaid code for '{self.diagram_type}' (PNG generation skipped as library is missing)."
                 success = True # Consider success as code was generated
                 break

        end_time = time.time()
        output = f"📊 **Analyst Agent Summary** 📊\n"
        output += f"Purpose: {self.purpose}\n"
        output += f"Query: {self.query}\n"
        output += f"Result: {final_message}\n"
        
        # Include generated code if it exists and isn't an error message, even if PNG failed
        if mermaid_code and not mermaid_code.startswith("ERROR:"):
            output += f"Generated Mermaid Code:\n```mermaid\n{mermaid_code}\n```\n"
        
        # Report errors if the overall process failed
        if not success:
            output += f"Errors encountered during generation/saving:\n"
            for i, err in enumerate(previous_errors):
                output += f" - {err}\n"

        output += f"\nExecution Time: {end_time - start_time:.2f} seconds\n"
        output += "--------------------\n"
        return output


# --- Agent Factory ---
def create_agent(agent_config: Dict[str, Any], conversation_history: str) -> Agent:
    """Factory function to create agent instances."""
    agent_type = agent_config.get("type")
    purpose = agent_config.get("purpose", "N/A")
    dependencies = agent_config.get("dependencies", [])
    query = agent_config.get("query", "") # Default query to empty string if missing

    if agent_type == "researcher":
        return ResearcherAgent(purpose, dependencies, query, conversation_history)
    elif agent_type == "qa":
        return QAAgent(purpose, dependencies, query, conversation_history)
    elif agent_type == "context" or agent_type == "contextualizer":
        # Default context folder can be configured here if needed
        context_folder = agent_config.get("context_folder", "context")
        return ContextExtractorAgent(purpose, dependencies, query, conversation_history, context_folder)
    elif agent_type == "analyst":
        diagram_type = agent_config.get("diagram_type", "flowchart") # Default if not specified
        output_name = agent_config.get("output_name")
        # Ensure query exists for analyst
        if not query:
             raise ValueError(f"Analyst agent (Purpose: {purpose}) is missing required 'query' field in plan.")
        return AnalystAgent(purpose, dependencies, query, conversation_history, diagram_type, output_name)
    else:
        raise ValueError(f"Unknown agent type specified in plan: {agent_type}")




