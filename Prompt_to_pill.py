import asyncio
import json
import logging
import aiohttp
import os
from typing import List
from autogen_ext.models.ollama import OllamaChatCompletionClient
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_ext.tools.mcp import StdioServerParams, StreamableHttpServerParams, mcp_server_tools
from autogen_core.tools import FunctionTool
from Orchestrator import DrugDiscoveryOrchestrator

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler()])

logging.getLogger("autogen_core.events").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


DEFAULT_NUM_MOLECULES = 4
PATIENT_DATA_FILEPATH = "/path/to/patients.xml"


def select_best_smiles(smiles: List[str], scores: List[float]) -> str:

    if not smiles or not scores or len(smiles) != len(scores):
        return json.dumps({"error": "Invalid input for selection", "retry": True})

    try:
        best_index = scores.index(min(scores))
        best_smiles = smiles[best_index]
        best_score = scores[best_index]

        result_dict = {
            "ranked_smiles": [best_smiles],
            "ranked_scores": [best_score]
        }
        return json.dumps(result_dict)
    except Exception as e:
        return json.dumps({"error": f"Failed to select best SMILES: {e}", "retry": True})


async def check_server_health(params):

    if isinstance(params, StreamableHttpServerParams):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(params.url, headers=params.headers, timeout=10) as response:
                    logger.debug(f"Health check for {params.url}: {response.status}")
                    return response.status == 200
        except Exception as e:
            logger.warning(f"Server health check failed for {params.url}: {e}")
            return False
    elif isinstance(params, StdioServerParams):
        try:
            script_path = params.args[0]
            if not os.path.exists(script_path):
                logger.error(f"Script not found: {script_path}")
                return False
            return True
        except Exception as e:
            logger.warning(f"Server health check failed for {params.args}: {e}")
            return False
    return False


async def main():
    logger.info("Starting Drug Discovery Workflow Main Execution.")

    uniprot_id = input("Please enter the UniProt ID (e.g., P35354): ").strip().upper()
    if not uniprot_id:
        logger.critical("UniProt ID cannot be empty. Exiting.")
        return
    num_molecules_input = input(
        f"Please enter the number of molecules to generate (default: {DEFAULT_NUM_MOLECULES}): ").strip()
    num_molecules = DEFAULT_NUM_MOLECULES
    if num_molecules_input:
        try:
            num_molecules = int(num_molecules_input)
            if num_molecules <= 0:
                logger.warning(f"Number of molecules must be positive. Using default: {DEFAULT_NUM_MOLECULES}")
                num_molecules = DEFAULT_NUM_MOLECULES
        except ValueError:
            logger.warning(f"Invalid number of molecules provided. Using default: {DEFAULT_NUM_MOLECULES}")

    logger.info(f"Received input: UniProt ID = {uniprot_id}, Number of molecules = {num_molecules}")

    model_client = OllamaChatCompletionClient(model="llama3.1:latest", timeout=600, json_output=True)
    logger.info("OllamaChatCompletionClient initialized.")

    druggen_params = StdioServerParams(command="python", args=["druggen_mcp_server.py"], read_timeout_seconds=1000)
    docking_params = StdioServerParams(command="python", args=["docking_mcp_server.py"], read_timeout_seconds=2000)
    cloud_parms = StreamableHttpServerParams(
        url="https://60fe2e609582.ngrok-free.app/mcp",
        timeout=1000,
        headers={"ngrok-skip-browser-warning": "true"}
    )
    logger.info("MCP Server parameters defined.")

    server_checks = await asyncio.gather(
        check_server_health(druggen_params),
        check_server_health(docking_params),
        check_server_health(cloud_parms),
        return_exceptions=True
    )
    for i, (params, result) in enumerate(zip([druggen_params, docking_params, cloud_parms], server_checks)):

        if not isinstance(result, bool) or not result:
            server_info = params.url if isinstance(params, StreamableHttpServerParams) else ' '.join(params.args)
            logger.warning(f"Server {i + 1} may be unavailable: {server_info}. Result: {result}")
        else:
            server_info = params.url if isinstance(params, StreamableHttpServerParams) else ' '.join(params.args)
            logger.info(f"Server {i + 1} ({server_info}) is healthy.")

    druggen_tools = []
    docking_tools = []
    txgemma_tools = []
    meditab_tools = []
    try:
        logger.info("Attempting to load tools from MCP servers...")
        druggen_tools = await mcp_server_tools(druggen_params)
        docking_tools = await mcp_server_tools(docking_params)
        cloud_tools = await mcp_server_tools(cloud_parms)
        txgemma_tools = [t for t in cloud_tools if t.name in ["predict_properties", "toxicity_screening"]]
        meditab_tools = [t for t in cloud_tools if t.name in ["match_patient_trial", "predict_trial_success"]]

        logger.info(f"Loaded {len(druggen_tools)} tools from Druggen server.")
        logger.info(f"Loaded {len(docking_tools)} tools from Docking server.")
        logger.info(f"Loaded {len(txgemma_tools)} tools from TxGemma server.")
        logger.info(f"Loaded {len(meditab_tools)} tools from Meditab server.")

        if not druggen_tools or not docking_tools or not txgemma_tools or not meditab_tools:
            raise ValueError("One or more MCP servers returned no tools (or empty list). Ensure all tools are exposed.")

    except Exception as e:
        logger.critical(
            f"Failed to load tools from MCP servers: {e}. Please ensure servers are running and exposing tools correctly.")
        return

    selector_tool = FunctionTool(
        select_best_smiles,
        name="select_best_smiles",
        description="Selects the SMILES string with the lowest docking score from a list of SMILES and their corresponding scores. It identifies and returns the single best SMILES string along with its optimal score.",
    )
    logger.info("Custom selector tool defined.")


    logger.info("Initializing Autogen agents...")
    workflow_initiator = UserProxyAgent(
        name="workflow_initiator",
        description="Controls the drug discovery workflow by sending messages to other agents."
    )

    agents = {
        "druggen_agent": AssistantAgent(
            name="druggen_agent",
            model_client=model_client,
            tools=druggen_tools,
            system_message=(
                "You are the molecule generation agent. Your sole task is to generate SMILES strings.\n"
                "Your input will be a user message specifying the number of molecules and UniProt ID.\n"
                f"You MUST call ONLY the 'run_druggen' tool with arguments: {{'uniprot_id': '{uniprot_id}', 'num_generated': {num_molecules}}}.\n"
                "After calling 'run_druggen', you MUST respond with the tool's output formatted as valid JSON: {'smiles': ['SMILES1', ...]}.\n"
                "DO NOT include any conversational text, explanations, or extraneous JSON. Provide ONLY the final JSON result.\n"
                "If 'run_druggen' fails or no SMILES are generated by it, return: {'error': 'Failed to generate SMILES', 'retry': true}."
            )
        ),
        "toxscreen_agent": AssistantAgent(
            name="toxscreen_agent",
            model_client=model_client,
            tools=[*txgemma_tools],
            system_message=(
                "You are the toxicity screening agent. Your SOLE purpose is to screen molecules for toxicity.\n"
                "Your input will always be a JSON string containing a list of SMILES, like: `{'smiles': ['SMILES1', 'SMILES2']}`.\n"
                "You MUST IMMEDIATELY and WITHOUT FAIL call the 'toxicity_screening' tool.\n"
                "The 'toxicity_screening' tool requires a 'smiles_list' argument, which MUST be the list of SMILES from your input.\n"
                "Example of the tool call you MUST generate: `toxicity_screening(smiles_list=['SMILES1', 'SMILES2'])`.\n"
                "Once the 'toxicity_screening' tool executes, you will receive its output. This output will be a dictionary, e.g., `{'non_toxic_smiles': ['SMILES_A', 'SMILES_B']}` or `{'error': ...}`.\n"
                "Your FINAL response MUST be ONLY this exact tool output, formatted as valid JSON. If the tool output is `{'non_toxic_smiles': [...]}` then return `{'non_toxic_smiles': [...]}`. If it's `{'error': ...}` then return that. Do NOT add any extra text or wrapping.\n"
                "DO NOT include any conversational text, explanations, or any other JSON structure. Respond ONLY with the final, required JSON."
            )
        ),
        "docking_agent": AssistantAgent(
            name="docking_agent",
            model_client=model_client,
            tools=docking_tools,
            system_message=(
                "You are the molecule docking agent. Your role is to perform molecular docking simulations.\n"
                "Your input will be a JSON string with the key 'non_toxic_smiles' containing a list of SMILES strings.\n\n"
                "Your task is to:\n"
                "1. Parse the input JSON to extract the list of non-toxic SMILES.\n"
                "2. Call the 'run_docking' tool. The arguments for this tool MUST be:\n"
                f"   - 'smiles': the list of SMILES extracted from your input.\n"
                f"   - 'uniprot_id': '{uniprot_id}' (This is the target protein ID for docking).\n"  
                "3. After successfully calling 'run_docking', you MUST respond with the tool's output formatted as valid JSON.\n"
                "   The expected output JSON format is: {'smiles': ['SMILES1', ...], 'scores': [score1, ...]}.\n"
                "DO NOT include any conversational text, explanations, or extraneous JSON. Provide ONLY the final JSON result.\n"
                "If 'run_docking' fails or returns empty results, return: {'error': 'Failed to perform docking or no results', 'retry': true}."
            )
        ),
        "selector_agent": AssistantAgent(
            name="selector_agent",
            model_client=model_client,
            tools=[selector_tool],
            system_message=(
                "You are the selection agent.\n"
                "Your input will be a JSON string like: {'smiles': [...], 'scores': [...]}.\n"
                "1. Validate input: it must contain 'smiles' and 'scores' keys with equal-length lists.\n"
                "2. Call 'select_best_smiles' with the provided {'smiles': ..., 'scores': ...}.\n"
                "3. Once the 'select_best_smiles' tool executes, you will receive its output. This output will be a dictionary, e.g., `{'ranked_smiles': ['SMILES_string'], 'ranked_scores': [score_value]}` or `{'error': ...}`.\n"
                "4. Your FINAL response MUST be ONLY this exact tool output, formatted as valid JSON. If the tool output is `{'ranked_smiles': [...]}` then return that. If it's `{'error': ...}` then return that. Do NOT add any extra text or wrapping.\n"
                "5. If input is invalid or selection fails, return: {'error': 'Invalid input for selection or selection failed', 'retry': true}."
            )
        ),
        "profiling_agent": AssistantAgent(
            name="profiling_agent",
            model_client=model_client,
            tools=txgemma_tools,
            system_message=(
                "You are the molecule profiling agent. Your sole task is to predict properties for a single SMILES string.\n"
                "Your input will be a JSON string like: `{'smiles': 'SMILES_string'}`.\n"
                "You MUST call the 'predict_properties' tool with the 'smiles' argument.\n"
                "After calling the tool, you will receive a detailed text output for each property. "
                "You MUST parse this output carefully and extract the 'Answer:' for each property.\n"
                "Specifically:\n"
                "- For 'BBB Permeability': if Answer is '(A)', set value to 'Does not cross BBB'; if '(B)', set value to 'Crosses BBB'.\n"
                "- For 'Toxicity': if Answer is '(A)', set value to 'Not toxic'; if '(B)', set value to 'Toxic'.\n"
                "- For 'Lipophilicity', 'Solubility', 'PPBR', 'Half-Life': Extract the numerical value directly after 'Answer:' and convert it to an integer.\n"
                "Your FINAL response MUST be ONLY the extracted properties, formatted as valid JSON, with precise string matches for categorical values and correct integer values for numerical ones.\n"
                "Example expected output: {'BBB_Permeability': 'Does not cross BBB', 'Toxicity': 'Not toxic', 'Lipophilicity': 533, 'Solubility': 801, 'PPBR': 342, 'Half-Life': 1}.\n"
                "Do NOT add any extra text, explanations, or wrapping beyond this JSON structure."
            )
        ),
        "summary_agent": AssistantAgent(
            name="summary_agent",
            model_client=model_client,
            system_message=(
                "You are the summarization agent.\n"
                "Your input will be a JSON string like: {'smiles': '...', 'score': ..., 'properties': {...}}.\n"  
                "Generate a concise summary report for the selected molecule, including its SMILES, docking score, and predicted properties.\n"  
                "Return ONLY valid JSON in the format: {'report': {'smiles': '...', 'score': ..., 'properties': {...}}, 'completed': true}.\n"  
                "If the input is invalid or missing critical information, return: {'error': 'Invalid input for summary', 'completed': false}."
            )
        ),
        "trial_generator_agent": AssistantAgent(
            name="trial_generator_agent",
            model_client=model_client,
            system_message="""
                You are a clinical trial design expert specializing in metabolic diseases.
                Your task is to generate a realistic Phase I clinical trial protocol for a novel drug targeting diabetes mellitus type 2.
                Input will be a JSON string with the format:
                {'drug_smiles': 'SMILES_STRING', 'target_uniprot_id': 'UNIPROT_ID', 'basic_eligibility': 'ELIGIBILITY_TEXT'}
                
                You MUST respond with a valid JSON object containing a 'trial_protocol_text' key with the following fields:
                - "title": Descriptive trial title.
                - "condition": "Diabetes Mellitus Type 2"
                - "phase": "Phase 1"
                - "intervention": Description of the drug (referencing SMILES) and administration.
                - "arms": List of trial arms (e.g., dose groups, placebo).
                - "number_of_participants": Number of participants (20-100).
                - "duration": Trial duration (e.g., "6 months").
                - "primary_outcomes": List of safety/pharmacokinetic outcomes.
                - "secondary_outcomes": List of exploratory outcomes.
                
                Example output:
                {
                  "trial_protocol_text": {
                    "title": "Phase I Trial of Novel Drug for Type 2 Diabetes",
                    "condition": "Diabetes Mellitus Type 2",
                    "phase": "Phase 1",
                    "intervention": "Drug (SMILES: [SMILES]) administered orally at escalating doses.",
                    "arms": [
                      {"name": "Dose Group 1", "dose": "2.5 mg"},
                      {"name": "Placebo", "dose": "Matching placebo"}
                    ],
                    "number_of_participants": 40,
                    "duration": "6 months",
                    "primary_outcomes": ["Safety (adverse events)", "Pharmacokinetics (Cmax, AUC)"],
                    "secondary_outcomes": ["Exploratory HbA1c reduction"]
                  }
                }
                
                If input is invalid, return:
                {'error': 'Invalid input for trial protocol generation', 'retry': true}
                
                Return ONLY the JSON object, with no additional text.
            """
        ),
        "patient_matching_agent": AssistantAgent(
        name="patient_matching_agent",
        model_client=model_client,
        tools=meditab_tools,
        system_message=(
            "You are the patient-trial matching agent.\n"
            "Your input is a JSON string: {'xml_path': 'STRING', 'trial_text': 'STRING'}.\n"
            "You MUST call the 'match_patient_trial' tool with 'xml_path' and 'trial_text' as arguments.\n"
            "Return ONLY the tool's JSON output without any additional text.\n"
            "If the tool fails or returns no output, return: {'error': 'Cannot match patients', 'retry': true}\n"
        )
    ),

        "trial_outcome_prediction_agent": AssistantAgent(
            name="trial_outcome_prediction_agent",
            model_client=model_client,
            tools=meditab_tools,
            system_message=(
                "You are the trial outcome prediction agent.\n"
                "You ONLY use the 'predict_trial_success' tool.\n"
                "Input format: {'trial_text': 'STRING'}\n"
                "Call 'predict_trial_success' with this argument and return its JSON output directly.\n"
                "If the tool fails, return: {'error': 'Cannot predict trial outcome', 'retry': true}\n"
                "Do NOT include any other text."
            )
    ),
        "clinical_report_generator_agent": AssistantAgent(
            name="clinical_report_generator_agent",
            model_client=model_client,
            system_message=(
                "You are the Clinical Study Report (CSR) generator.\n"
                "Your task is to compile a standardized report summarizing the drug discovery process and clinical trial feasibility.\n"
                "You MUST reply ONLY in valid JSON with no explanations, no markdown, and no commentary.\n\n"
                "**Input format:**\n"
                "{\n"
                "  \"drug_discovery_summary\": {...},\n"
                "  \"trial_protocol\": {...},\n"
                "  \"tabular_trial_data\": [...],\n"
                "  \"patients_meditab\": [...],\n"
                "  \"patient_trial_pairs\": [...],\n"
                "  \"matched_patients_summary\": {\n"
                "       \"total_patients_parsed\": INT,\n"
                "       \"matched_patients_count\": INT,\n"
                "       \"top_matches\": [...]\n"
                "  },\n"
                "  \"trial_outcome_prediction\": {\n"
                "      \"predicted_success_probability\": FLOAT,\n"
                "      \"prediction_notes\": \"STRING\"\n"
                "  },\n"
                "  \"uniprot_id\": \"STRING\"\n"
                "}\n\n"
                "**Your JSON output MUST strictly follow this schema:**\n"
                "{\n"
                "  \"completed\": true,\n"
                "  \"report\": {\n"
                "     \"report_title\": \"Integrated Drug Discovery and Clinical Trial Feasibility Report for [Drug SMILES / Target ID]\",\n"
                "     \"date_generated\": \"2025-07-07\",\n"
                "     \"summary_synopsis\": \"STRING\",\n"
                "     \"drug_candidate_profile\": {\n"
                "         \"smiles\": \"STRING\",\n"
                "         \"docking_score\": FLOAT,\n"
                "         \"predicted_properties\": {...}\n"
                "     },\n"
                "     \"clinical_trial_protocol_summary\": {\n"
                "         \"title\": \"STRING\",\n"
                "         \"objectives\": \"STRING\",\n"
                "         \"design_overview\": \"STRING\",\n"
                "         \"key_eligibility_criteria\": {\n"
                "             \"inclusion\": [...],\n"
                "             \"exclusion\": [...]\n"
                "         },\n"
                "         \"treatment_plan_overview\": \"STRING\"\n"
                "     },\n"
                "     \"patient_matching_results\": {\n"
                "         \"total_patients_parsed\": INT,\n"
                "         \"matched_patients_count\": INT,\n"
                "         \"top_matches\": [\n"
                "            {\"pid\": \"STRING\", \"match_probability\": FLOAT}, ...\n"
                "         ]\n"
                "     },\n"
                "     \"trial_outcome_prediction\": {\n"
                "         \"predicted_success_probability\": FLOAT,\n"
                "         \"prediction_notes\": \"STRING\"\n"
                "     },\n"
                "     \"conclusion_and_recommendations\": \"STRING\"\n"
                "  }\n"
                "}\n\n"
                "**Rules:**\n"
                "1. Wrap your response in a JSON object with a top-level key `completed` set to true, and a `report` key as shown above.\n"
                "2. Always populate the JSON fields using the input data, do not fabricate.\n"
                "3. If the input is invalid, reply with:\n"
                "{ \"error\": \"Invalid input for report generation\", \"completed\": false }\n"
                "4. Do not wrap the JSON in markdown.\n"
                "5. Do not provide any additional commentary or explanation outside the JSON.\n"

            )

        )
    }
    logger.info("Autogen agents initialized.")

    orchestrator = DrugDiscoveryOrchestrator(
        agents=agents,
        uniprot_id=uniprot_id,
        num_molecules=num_molecules,
        orchestrator_agent=workflow_initiator,
        patient_data_filepath=PATIENT_DATA_FILEPATH
    )

    logger.info("Running drug discovery workflow...")
    final_result = await orchestrator.run_workflow()

    if final_result.get("completed"):
        logger.info("\n--- Workflow Completed Successfully! ---")
        if final_result.get("report"):
            logger.info("Final Report:")
            print(json.dumps(final_result["report"], indent=2))
        else:
            logger.info("Workflow completed, but no report was generated.")
    else:
        logger.error(f"\n--- Workflow Failed! Error: {final_result.get('error', 'Unknown error')} ---")


if __name__ == "__main__":
    asyncio.run(main())
