# Import LangChain/LangGraph libraries
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from typing import Optional, Type, Dict, List, Any, Tuple
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from typing import TypedDict, Annotated, Sequence
from langgraph.graph.message import add_messages
import json
import re
from collections import defaultdict
import io
import uuid
from dotenv import load_dotenv
import os
import google.generativeai as genai


load_dotenv()
gemini_api_key = os.getenv('GOOGLE_API_KEY')

class RegexExtractionTool(BaseTool):
    """Tool for extracting unit readings using regex"""
    name: str = "regex_extraction"
    description: str = "Extract unit readings from natural language text using regex pattern"
    pattern: str = r'(?:Unit\s+)?(\d+[A-Z])\s+(?:reads|is|reading)\s+(\d+(?:\.\d+)?)\s+(?:cubic\s+meter|m\^3|m3)'

    def _run(self, input_text: str) -> List[Tuple[str, str]]: # Ex. Output = [('Unit 1', '123.45'), ('Unit 2', '67.89')]
        """Extract unit and reading pairs from input text"""
        matches = re.findall(self.pattern, input_text, re.IGNORECASE)
        
        return matches

    async def _arun(self, input_text: str) -> List[Tuple[str, str]]:
        """Async version of _run"""
        return self._run(input_text)

class ValidationTool(BaseTool):

    """Tool for validating extracted unit readings"""
    name: str = "validation"
    description: str = "Validate extracted unit readings for duplicates and conflicts"

    def _run(self, matches: List[Tuple[str, str]]) -> Dict[str, Any]: # Ex. Output = {'is_valid': True, 'errors': [], 'duplicates_found': False, 'conflicts_found': False, 'conflicts': {}}
        """Validate matches and return validation results"""
        validation_result = {
            "is_valid": True,
            "errors": [],
            "duplicates_found": False,
            "conflicts_found": False,
            "conflicts": {},
            "duplicates": {}
        }

        # Check validity of matches
        if not matches:
            validation_result["is_valid"] = False
            validation_result["errors"].append("No unit readings found")
            return validation_result

        # Check for exact duplicates
        seen_pairs = set()
        duplicates = {}
        for unit, reading in matches:
            pair = (unit, reading)
            if pair in seen_pairs:
                duplicates[unit] = reading
                validation_result["duplicates_found"] = True
                validation_result["errors"].append(f"Duplicate reading found: Unit {unit} with reading {reading}")
            seen_pairs.add(pair)
        validation_result["duplicates"] = duplicates

        # Check for conflicting values
        unit_readings = defaultdict(list)
        for unit, reading in matches:
            unit_readings[unit].append(reading)

        conflicts = {}
        for unit, readings in unit_readings.items():
            unique_readings = list(set(readings))
            if len(unique_readings) > 1:
                conflicts[unit] = unique_readings
                validation_result["conflicts_found"] = True

        if conflicts:
            validation_result["is_valid"] = False
            validation_result["conflicts"] = conflicts
            for unit, values in conflicts.items():
                validation_result["errors"].append(
                    f"Conflicting readings for Unit {unit}: {', '.join(values)} cubic meters"
                )

        return validation_result

    async def _arun(self, matches: List[Tuple[str, str]]) -> Dict[str, Any]:
        """Async version of _run"""
        return self._run(matches)

class RemoveDuplicates(BaseTool):
    """Tool for processing validated data into JSON format"""
    name: str = "remove_duplicates"
    description: str = "Clean unit reading duplicates from input matches"

    def _run(self, matches: List[Tuple[str, str]]) -> List[Tuple[str, Any]]:
        """Process matches into structured data format"""
        result = []

        # Remove duplicates while preserving order
        seen_unit_readings = set()
        unique_matches = []
        for unit, reading in matches:
            key = (unit, reading)
            if key not in seen_unit_readings:
                unique_matches.append((unit, reading))
                seen_unit_readings.add(key)
    
        return unique_matches

class RemoveConflicts(BaseTool):
    """Tool for removing conflicts from validated data"""
    name: str = "remove_conflicts"
    description: str = "Remove conflicts from validated data"
    
    def _run(self, matches: List[Tuple[str, str]]) -> List[Tuple[str, Any]]:
        """Remove duplicats from the matches"""

        unit_readings = defaultdict(set)
        for unit, reading in matches:
            unit_readings[unit].add(reading)

        valid_units = {unit for unit, readings in unit_readings.items() if len(readings) == 1}

        filtered = [(unit, reading) for unit, reading in matches if unit in valid_units]
        
        return filtered

class CreateJSON(BaseTool):
    """Convert validated matches into JSON file"""

    name: str = "data_processing"
    description: str = "Process validated unit readings into structured JSON format"

    def _run(self, matches: List[Tuple[str, str]]) -> List[Dict[str, Any]]: 

        result = []

        for unit, reading in matches:
            unit_data = {
                "unit": unit,
                "reading": float(reading)
            }
            result.append(unit_data)

        return result

# Pre-requisites for Workflow
class UnitReadingReasoner: # Custom Reasoner class for unit reading processing
    def __init__(self, tools: List[BaseTool]):
        self.tools = {tool.name: tool for tool in tools}
        self.processing_steps = [
            "regex_extraction",
            "validation",
            "remove_duplicates",
            "remove_conflicts",
            "data_processing"
        ]
        # Initialize log for thoughts, actions, and observations
        self.log = []
        self.step = 0
        self.standardized_input = ""
        self.validation_result = None

    def log_tao(self, type: str, description: str):
        """Log thought, action, and observation."""
        self.log.append(f"{type}: {description}")
        print(f"{type}: {description}")

    def _think_initial(self, user_input: str) -> str:
            
            """Initial reasoning about the input."""
            if self._contains_unit_readings(user_input):
                return "I can see unit readings in the input. I need to extract them first."
            else:
                return "I don't see any unit readings in this input. I should inform the user."
    
    def reason(self, user_input: str) -> Tuple[str, List[Dict[str, Any]]]:
        self.log.clear()
        """Main reasoning method that decides processing steps for unit readings."""
        # Check if input contains unit reading patterns
        print(f"{80 * '='}\n ReAct Simulation\n{80 * '='}")
    
        self.log_tao("💭 Thought", "I need to check if there is a viable input") 

        if self._contains_unit_readings(user_input):

            tool_calls = [
                {
                    'tool': 'regex_extraction',
                    'args': {'input_text': self.standardized_input},
                    'step': 1
                }
            ]
            self.log_tao("💭 Thought", "I'll process your unit readings step by step.") 
            self.step = 1
            return "I'll process your unit readings step by step.", tool_calls
        else:
            return "I don't detect any unit readings in your input. Please provide text containing unit readings in the format 'Unit 1A reads 123.45 cubic meter'.", []

    def _contains_unit_readings(self, text: str) -> bool: 
        """Check if text contains unit reading patterns and standardize input."""
        # Log the action of checking for inputs
        self.log_tao("🔧 Action", "Checking for unit readings in input with Gemini")

        # Try to extract and standardize using Gemini AI
        try:
            prompt = (
                """
                You are a helpful assistant that extracts unit readings from text.  
                Please extract all unit readings from the following text and clean it by returning them in the standard format: 'Unit 1A reads 24 m3, Unit 2B reads 3 m3, Unit 2A reads 24 m3'" 
                Check for cubic meter unit reading synonyms, convert units to cubic meter and make readings two decimal places.
                Return it as the final text only, no other text or code
                """ 

                f"Text: {text}"
            )

            genai.configure(api_key=gemini_api_key)

            model = genai.GenerativeModel("gemini-2.5-flash")

            response = model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.0
                }
            )
            
            result = response.text

            if 'Unit' in result or 'm3' in result:
                self.log_tao("👁️ Observation", f"Gemini response: Code for ready for extraction. {result}")
                self.standardized_input = result
                return True
            else:
                self.log_tao("👁️ Observation", f"Gemini response: Ambigious input found. {result}")
                self.standardized_input = result
                return False
            
        except Exception as e:
            self.log_tao("👁️ Observation", f"Gemini API error: {str(e)}")
            return False

    def execute_tool_call(self, tool_name: str, args: Dict[str, Any]) -> Any:
        """Execute a tool call with given arguments."""
        if tool_name not in self.tools:
            return f"Error: Tool '{tool_name}' not found"

        tool = self.tools[tool_name]
        try:
            # Pass args directly to the tool's _run method
            if hasattr(tool, '_run'):
                self.log_tao("🔧 Action", f"Executing tool '{tool_name}' with args: {args}")
                if self.step == 5:
                    self.log_tao("👁️ Observation", "Final JSON is ready to export")
                    print(f"{80 * '='}\n")
                return tool._run(**args)
            else:
                return f"Error: Tool '{tool_name}' does not have a _run method."
        except Exception as e:
            return f"Error executing {tool_name}: {str(e)}"

    def get_next_step(self, current_step: int, validation_result: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """Determine the next processing step based on current results."""
        
        if current_step == 1:  # After regex extraction
            self.log_tao("👁️ Observation", "Unit and Water Meter Reading pairs found")
            self.log_tao("💭 Thought", "I'll validate the extracted unit readings")
            self.step = 2
            
            return {
                'tool': 'validation',
                'args': {'matches': None},  # Will be filled with extraction results
                'step': 2
            }
            
        elif current_step == 2:  # After validation
            self.validation_result = validation_result
            if validation_result:
                errors = validation_result.get('errors', [])
                self.log_tao("👁️ Observation", f"Validation Result, {errors}")
                # for error in errors:
                #     if "Duplicate" in error:
                #         self.log_tao("👁️ Observation", f"Duplicates found, {error}")
                #     else:
                #         self.log_tao("👁️ Observation", f"Conflicts found, {error}")   

                if validation_result.get('duplicates_found', False):
                    duplicates = validation_result.get('duplicates', {})
                    self.log_tao("👁️ Observation", f"Duplicates found, {duplicates}")
                    self.log_tao("💭 Thought", "I'll remove extraction matches duplicates")
                    self.step = 3

                    return {
                    'tool': 'remove_duplicates',
                    'args': {'matches': None},  # Will be filled with extraction results
                    'step': 3,
                    'duplicates_found': duplicates
                    } 

                elif validation_result.get('conflicts_found', False):
                    conflicts = validation_result.get('conflicts', {})
                    self.log_tao("👁️ Observation", f"Conflicts found, {conflicts}")
                    self.log_tao("💭 Thought", "I'll remove extraction matches with conflicts")

                    # for unit, readings in conflicts.items():
                    #     self.log_tao("👁️ Observation", f"Conflicting values found, Unit {unit}: {', '.join(readings)}")
                    #     self.log_tao("💭 Thought", "I'll remove extraction matches with conflicts")
                        
                    return {
                        'tool': 'remove_conflicts',
                        'args': {'matches': None},  # Will be filled with extraction results
                        'step': 4,
                        'conflicts_found': conflicts
                        }
                    
                else:
                    self.log_tao("👁️ Observation", "Validation done, no issues found")
                    self.log_tao("💭 Thought", "I'll process the validated matches into JSON file")
                    self.step = 5
                    
                    return {
                    'tool': 'data_processing',
                    'args': {'matches': None},  # Will be filled with extraction results
                    'step': 5
                    }
                
            else:
                if self.validation_result.get('conflicts_found', False):
                    conflicts = validation_result.get('conflicts', {})
                    self.log_tao("👁️ Observation", f"Conflicts found, {conflicts}")
                    self.log_tao("💭 Thought", "I'll remove extraction matches with conflicts")

                    return {
                        'tool': 'remove_conflicts',
                        'args': {'matches': None},  # Will be filled with extraction results
                        'step': 4,
                        'conflicts_found': conflicts
                        }

        elif current_step == 3: # After removing duplicates
            if self.validation_result and self.validation_result.get('conflicts_found', False):
                conflicts = self.validation_result.get('conflicts', {})
                self.log_tao("👁️ Observation", f"Conflicts found, {conflicts}")
                self.log_tao("💭 Thought", "I'll remove extraction matches with conflicts")

                return {
                    'tool': 'remove_conflicts',
                    'args': {'matches': None},  # Will be filled with extraction results
                    'step': 4,
                    'conflicts_found': conflicts
                    }
                
            else:
                if self.log[-1].startswith("👁️ Observation: Final JSON is ready to export"):
                    return None  # No further steps needed

                self.log_tao("👁️ Observation", "Validation done, no issues found")
                self.log_tao("💭 Thought", "I'll process the validated matches into JSON file")
                self.step = 5
                
                return {
                    'tool': 'data_processing',
                    'args': {'matches': None},  # Will be filled with extraction results
                    'step': 5
                }
        
        else:
            # If no duplicates were found or after processing duplicate:
            if self.log[-1].startswith("👁️ Observation: Final JSON is ready to export"):
                return None  # No further steps needed
            else:
                self.log_tao("👁️ Observation", "All input errors were removed")
                self.log_tao("💭 Thought", "I'll process the validated matches into JSON file")
                self.step = 5

                return {
                    'tool': 'data_processing',
                    'args': {'matches': None},  # Will be filled with extraction results
                    'step': 5
                }

    def synthesize_results(self, user_input: str, all_results: List[Dict[str, Any]]) -> str:
        """Synthesize all processing results into a coherent response."""
        if not all_results:
            return "No results to process."

        response_parts = []

        # Find results by step
        extraction_result = next((r for r in all_results if r.get('step') == 1), None)
        validation_result = next((r for r in all_results if r.get('step') == 2), None)
        duplicates_result = next((r for r in all_results if r.get('step') == 3), None)
        conflicts_result = next((r for r in all_results if r.get('step') == 4), None)
        processing_result = next((r for r in all_results if r.get('step') == 5), None)

        if extraction_result:
            matches = extraction_result.get('output')
            response_parts.append(f"📊 **Extraction Results:**")
            if matches:
                response_parts.append(f"Found {len(matches)} unit readings:")
                for unit, reading in matches:
                    response_parts.append(f"  - Unit {unit}: {reading} cubic meters")
            else:
                response_parts.append("No unit readings found in the input.")

        if validation_result:
            validation_data = validation_result.get('output')
            response_parts.append(f"\n✅ **Validation Results:**")

            if validation_data and validation_data.get('is_valid', False):
                response_parts.append("All unit readings are valid!")
            else:
                response_parts.append("Validation issues found:")
                errors = validation_data.get('errors', []) if validation_data else []
                for error in errors:
                    response_parts.append(f"  ⚠️ {error}")

            if validation_data and validation_data.get('duplicates_found', False) and duplicates_result:
                response_parts.append("  🔄 Duplicates detected")

                unique_matches = duplicates_result.get('output')
                response_parts.append(f"\n🔄 **Duplicates Removed:**")
                if unique_matches and processing_result:
                    response_parts.append(f"Found {len(unique_matches)} unique unit readings:")
                    for unit, reading in unique_matches:
                        response_parts.append(f"  - Unit {unit}: {reading} cubic meters")
                else:
                    response_parts.append("No unique unit readings found after removing duplicates.")

                
            if validation_data and validation_data.get('conflicts_found', False):
                response_parts.append("  ⚡ Conflicts detected:")
                conflicts = validation_data.get('conflicts', {}) if validation_data else {}
                for unit, readings in conflicts.items():
                    response_parts.append(f"    - Unit {unit}: {', '.join(readings)}")

        if duplicates_result:
            unique_matches = duplicates_result.get('output')
            response_parts.append(f"\n🔄 **Duplicates Removed:**")
            if unique_matches:
                response_parts.append(f"Found {len(unique_matches)} unique unit readings:")
                for unit, reading in unique_matches:
                    response_parts.append(f"  - Unit {unit}: {reading} cubic meters")
            else:
                response_parts.append("No unique unit readings found after removing duplicates.")
        
        if conflicts_result:
            conflicts_matches = conflicts_result.get('output')
            response_parts.append(f"\n🔄 **Conflicts Removed:**")
            if conflicts_matches:
                response_parts.append(f"Found {len(conflicts_matches)} conflicts:")
                for unit, reading in conflicts_matches:
                    response_parts.append(f"  - Unit {unit}: {reading} cubic meters")

        if processing_result:
            processed_data = processing_result.get('output')
            response_parts.append(f"\n🔧 **Final Processed Data:**")
            if processed_data:
                response_parts.append("```json")
                response_parts.append(json.dumps(processed_data, indent=2))
                response_parts.append("```")

                response_parts.append(f"\n📈 **Summary:**")
                response_parts.append(f"Successfully processed {len(processed_data)} unique unit readings")
            else:
                response_parts.append("No data processed.")


        return "\n".join(response_parts)

class WorkflowState(TypedDict): # Workflow State
    messages: Annotated[Sequence[BaseMessage], add_messages]
    current_step: int
    extraction_results: List[Tuple[str, str]]
    validation_results: Dict[str, Any]
    duplicates_found: List[Dict[str, Any]]
    conflicts_found: List[Dict[str, Any]]
    processing_results: List[Dict[str, Any]]
    pending_tool_calls: List[Dict[str, Any]]
    all_results: List[Dict[str, Any]]
    logs: List[List[str]] # Changed to List[List[str]] to store copies of logs


# Initialize tools and reasoner
regex_tool = RegexExtractionTool()
validation_tool = ValidationTool()
duplicates_tool = RemoveDuplicates()
conflicts_tool = RemoveConflicts()
processing_tool = CreateJSON()
tools = [regex_tool, validation_tool, duplicates_tool, conflicts_tool, processing_tool]

reasoner = UnitReadingReasoner(tools)

# Create the workflow
workflow = StateGraph(WorkflowState)

def agent_node(state: WorkflowState):
    """Main agent node using the UnitReadingReasoner class."""
    messages = state["messages"]
    last_message = messages[-1]

    if isinstance(last_message, HumanMessage):
        # Initial reasoning
        response_text, tool_calls = reasoner.reason(last_message.content)

        if tool_calls:
            # Create AI message indicating tool usage
            ai_message = AIMessage(
                content=response_text,
                additional_kwargs={"tool_calls": tool_calls}
            )
            return {
                "messages": [ai_message],
                "pending_tool_calls": tool_calls,
                "current_step": 1,
                "all_results": []
            }
        else:
            # Direct response without tools
            return {"messages": [AIMessage(content=response_text)]}

    # Handle completion of processing steps
    elif state.get("all_results"):
        # Check if we need to continue processing
        current_step = state.get("current_step", 0)

        if current_step > 0 and current_step < 5:  # Still have steps to process
            # Get the last result to determine next step
            last_result = state["all_results"][-1] if state["all_results"] else None

            if current_step == 1:  # After extraction
                extraction_matches = last_result['output'] if last_result and 'output' in last_result else []
                state["extraction_results"].append(extraction_matches)

                next_step = reasoner.get_next_step(current_step)
                if next_step:
                    next_step['args']['matches'] = extraction_matches
                    return {
                        "pending_tool_calls": [next_step],
                        "current_step": current_step + 1
                    }

            elif current_step == 2:  # After validation
                # Get the original dictionary output from the validation tool
                validation_result_dict = last_result['output'] if last_result and 'output' in last_result and isinstance(last_result['output'], dict) else {}

                next_step = reasoner.get_next_step(current_step, validation_result_dict)

                
                if next_step:
                    # Get original extraction results to pass to processing
                    extraction_result = next((r for r in state["all_results"] if r.get('step') == 1), None)
                    extraction_matches = extraction_result['output'] if extraction_result and 'output' in extraction_result else []
                    
                    if 'duplicates_found' in next_step:
                        duplicates = next_step['duplicates_found'] 
                        state["duplicates_found"].append(duplicates)
                    else:
                        duplicates = []

                    next_step['args']['matches'] = extraction_matches

                    return {
                        "pending_tool_calls": [next_step],
                        "current_step": current_step + 1
                    }
                else:
                    # Validation failed, provide final response
                    user_input = next(msg.content for msg in messages if isinstance(msg, HumanMessage))
                    final_response = reasoner.synthesize_results(user_input, state["all_results"])
                    return {"messages": [AIMessage(content=final_response)]}

            elif current_step == 3:  # After removing duplicates
                # Get the unique matches from the last result  
                unique_matches = last_result['output'] if last_result and 'output' in last_result else []
                

                next_step = reasoner.get_next_step(current_step)
                if next_step:
                    if 'conflicts_found' in next_step:
                        conflicts = next_step['conflicts_found']
                        state["conflicts_found"].append(conflicts)
                    else:
                        conflicts = []

                    next_step['args']['matches'] = unique_matches
                    
                    return {
                        "pending_tool_calls": [next_step],
                        "current_step": current_step + 1
                    }

            elif current_step == 4:  # After removing conflicts
                conflicts_matches = last_result['output'] if last_result and 'output' in last_result else []
                next_step = reasoner.get_next_step(current_step)
                if next_step:
                    next_step['args']['matches'] = conflicts_matches
                    return {
                        "pending_tool_calls": [next_step],
                        "current_step": current_step + 1
                    }

        # Processing complete, synthesize final response
        logs = reasoner.log.copy()
        state["logs"].append(logs)

        user_input = next(msg.content for msg in messages if isinstance(msg, HumanMessage))
        final_response = reasoner.synthesize_results(user_input, state["all_results"])
        return {"messages": [AIMessage(content=final_response)]}

    return {"messages": [AIMessage(content="Processing complete.")]}

def tool_node(state: WorkflowState):
    """Execute pending tool calls."""
    pending_calls = state.get("pending_tool_calls", [])
    tool_results = []
    tool_messages = []
    current_step = state.get("current_step", 1)

    for call in pending_calls:
        tool_name = call["tool"]
        args = call["args"]
        step = call.get("step", current_step)

        result = reasoner.execute_tool_call(tool_name, args)
        tool_result = {
            "tool": tool_name,
            "args": args,
            "output": result,
            "step": step
        }
        tool_results.append(tool_result)

        # Create tool message (convert result to string for message content)
        tool_message_content = json.dumps(result) if isinstance(result, (dict, list)) else str(result)
        tool_message = ToolMessage(
            content=f"Step {step} ({tool_name}): {tool_message_content}",
            tool_call_id=f"{tool_name}_{step}_{hash(str(args))}"
        )
        tool_messages.append(tool_message)

    # Update all_results with new results
    all_results = state.get("all_results", [])
    all_results.extend(tool_results)

    return {
        "messages": tool_messages,
        "pending_tool_calls": [],
        "all_results": all_results
    }

def should_continue(state: WorkflowState) -> str:
    """Determine whether to continue with tools or end."""
    if state.get("pending_tool_calls"):
        return "tools"

    # Check if we need more processing steps
    current_step = state.get("current_step", 0)
    if current_step > 0 and current_step < 5:
        # Check if the last step was successful and requires a follow-up
        if state.get("all_results"):
            last_result = state["all_results"][-1]
            if last_result.get('step') == 1: # After extraction, always go to validation
                 return "agent"
            elif last_result.get('step') == 2: # After validation, check if valid for processing
                 validation_output = last_result.get('output')
                 if isinstance(validation_output, dict):
                     # Continue if there are duplicates or conflicts to process
                     if validation_output.get('duplicates_found', False) or validation_output.get('conflicts_found', False):
                         return "agent"
                     # Continue if validation passed
                     elif validation_output.get('is_valid', False):
                         return "agent"
                     else:
                         return END # Validation failed, end workflow
                 else:
                     return END # Invalid validation output
            elif last_result.get('step') in [3, 4]: # After removing duplicates or conflicts
                 return "agent" # Continue to final processing
            else:
                 return END # unexpected step

    return END

# Add nodes and edges
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)

workflow.add_edge(START, "agent")
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        END: END,
    }
)
workflow.add_edge("tools", "agent")


# Compile the workflow
app = workflow.compile()

def run_workflow(user_input: str):
    """Run the workflow with user input."""
    initial_state = {
        "messages": [HumanMessage(content=user_input)],
        "current_step": 0,
        "extraction_results": [],
        "validation_results": {},
        "processing_results": [],
        "duplicates_found": [],
        "conflicts_found": [],
        "pending_tool_calls": [],
        "all_results": [],
        "logs": []
    }

    result = app.invoke(initial_state)
    return result 
    
    # print(f"\nUser: {user_input}")
    # for message in result["messages"]:
    # #     # if isinstance(message, AIMessage):
    # #     #     print(f"Assistant: {message.content}")
    # #     # elif isinstance(message, ToolMessage):
    # #     #     print(f"Tool: {message.content}")
    #     print(f"{message.__class__.__name__}: {message.content}")
    # # print("=" * 80)


# result = run_workflow("Unit 12A reads 2 m3, Unit 3B reads 2 m3, Unit 12A reads 2 m3, 3B is 3 m3")

# print(f'RESULT: {result["logs"]}')


# for message in result["messages"]:
# #     # if isinstance(message, AIMessage):
# #     #     print(f"Assistant: {message.content}")
# #     # elif isinstance(message, ToolMessage):
# #     #     print(f"Tool: {message.content}")
#     print(f"{message.__class__.__name__}: {message.content}")


