import asyncio
import logging
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.ui import Console
from autogen_ext.tools.mcp import StdioServerParams, StreamableHttpServerParams, mcp_server_tools

DRUGGEN_SYS = (
    '''
    You are the Drug Generation Specialist. Your purpose is to design novel SMILES based on a specified biological target.
    You have a tool run_druggen which returns drug smiles for given target or targets. Always generate 7 molecules.
    DO NOT in any case perform properties prediction, docking, or free text generation. 
    Run only tool run_druggen and return its result. If SMILES is present in the task DO NOT run the tool.
    '''
)

CHEM_PROPERTIES_SYS = (
    '''
    You are the Chemical Properties Specialist. Your purpose is to predict and report on the chemical properties of molecules.
    DO NOT in any case perform trial generation, drug generation, ADMET prediction, or docking.
    You have these tools:
        - select_leads_from_smiles(smiles_list, n): selects n lead compounds based on Lipinski (MW <=500, logP <=5, HBD <=5, HBA <=10) and Veber (RotB <=10, TPSA <=140) rules. n is the number of compounds to select as leads.
        - predict_pka_batch: predicts pKa values for SMILES lists.  
        - logd_acid_batch / logd_base_batch: calculates logD for acids or bases at a given pH.  
        - rdkit_physchem_batch: computes physicochemical descriptors (MW, logP, TPSA, HBD, HBA, RotB, QED, etc.) plus pKa and logD(acid/base).  
        - predict_all_batch: integrates all the above, selecting the relevant logD based on acid/base flags.  
    '''
)

ADMET_PROPERTIES_SYS = (
    '''
    You are the ADMET Properties Specialist. Your purpose is to predict and report on the ADMET properties of molecules.
    DO NOT in any case generate clinical trials, drugs, or free text. Always start with chemfm_list_properties to get exact property names.
    Select 10 most relevant ADMET properties (e.g., oral bioavailability, solubility, clearance, BBB permeability, hERG inhibition, hepatotoxicity).
    You have these tools:
        - get_drug_name_from_smiles(smiles): resolves a list of SMILES to PubChem CID and returns {"cid","preferred_name","synonyms","iupac_name"}. 
                Use only when a SMILES is provided and check if names for generated smiles are available; do NOT invent names. If no CID is found, return the tool’s error message.
        - get_smiles_from_drug_name - finds smiles from drug name. DO NOT invent name, only if drug name is available in task run this tool
        - chemfm_list_properties: list all properties supported by the ChemFM Space.
        - chemfm_get_description(property_name): get the ChemFM description for a property.
        - chemfm_predict_single(smiles, property_name): predict ONE property for ONE SMILES. DO NOT take list of smiles
        - chemfm_predict_many(smiles, properties): predict MANY properties for ONE SMILES. DO NOT take list of smiles
        - run_docking - performs docking and returns docking scores. Can be used for selection for the best drug candidates for some target. Lower the score better is a drug candidate. 
                Never predict docking score without running the tool. Use for initial hit filtering and re-evaluation after optimization.
    '''
)

MOL_OPT_SYS = (
    '''
    You are Molecule Optimization specialist. You have one tool:
      - molecule_optimizer(smiles, properties, action): It takes one SMILES, one property that needs to be optimized
    and action (increase or decrease) wanted for the property
    Do not generate trials or drugs, do not predict properties or docking scores. Never generate free text, only return
    the tool result it returns. Do not run if ADMET and chemical properties are not predicted and evaluated.
    '''
)

# Unchanged
TRIAL_SYS = (
    """
    You are a clinical trial design expert.
    Your task is to generate a realistic clinical trial protocol for a drug. If the drug is generated with druggen agent, use the drug with the lowest
    docking score. You can't predict properties, perform docking, or generate molecules. DO NOT try those tasks.
    If a drug name is available in the task, then run it with that name. You must generate the trial with SMILES or drug name available.
    If drug name or drug SMILES is not available, do not generate a trial.

    First, output the initial trial text in this format:
- drug: SMILES (NAME if available)

CLINICAL TRIAL:
- acronym: string, short study name
- brief_title: string
- official_title: string, descriptive trial title
- study_status: string
- study_start_date: string, ISO format (e.g., "2026-03")
- primary_completion_date: string, ISO format
- completion_date: string, ISO format
- condition: string, clinical condition studied
- study_type: string
- phase: string
- enrollment: integer
    Then, call the panacea_extract_components tool with that entire text as the trial_text parameter.

    Finally, use the tool's output to construct and output the final report in this structured format:
- drug: SMILES (NAME if available)

CLINICAL TRIAL:
- acronym: string, short study name
- brief_title: string
- official_title: string, descriptive trial title
- study_status: string (e.g., "Recruiting", "Completed")
- study_start_date: string, ISO format (e.g., "2026-03")
- primary_completion_date: string, ISO format
- completion_date: string, ISO format
- condition: string, clinical condition studied
- study_type: string
- phase: string
- intervention_model: string
- allocation: string
- masking: string
- enrollment: integer
- arms: [list from tool output arms]
- intervention_description: string, must reference the molecule by its SMILES
- primary_outcomes: [list from tool output outcomes]
- secondary_outcomes: [list from tool output outcomes]
- other_outcomes: []  # or from secondary if needed
- eligibility_criteria: {"inclusion": [list from tool], "exclusion": [list from tool]}
- study_documents: list of strings
- brief_summary: string, concise description of trial purpose, design, intervention, and eligibility.

    Phases rules:
        - Phase 1: Safety First
            The transition from laboratory testing to human trials marks a critical milestone in medical research.
            Phase 1 represents the first time a new treatment is tested in humans, with safety as the primary concern.
            As stated by the University of Cincinnati Medical Center:

                "Phase I trials are concerned primarily with establishing a new drug's safety and dose range in about 20-100 healthy
                volunteers."

            Key characteristics of Phase 1:
                Focus on safety and side effects
                Determines optimal dosing
                Usually involves healthy volunteers
                Takes several months to complete
        - Phase 2: Testing Effectiveness
            After establishing basic safety parameters, researchers move to evaluate the treatment's effectiveness. Phase 2 trials represent a crucial step where scientists begin to understand how well a treatment works for its intended purpose. This phase involves carefully selected participants who have the specific condition the treatment aims to address.

            Key aspects include:
                100-300 participants
                Tests effectiveness against specific conditions
                Continues monitoring for side effects
                Usually lasts several months to two years
        - Phase 3: Comparative Testing
            Phase 3 trials mark the most comprehensive evaluation stage, where researchers compare new treatments against current standard therapies. This phase involves the largest number of participants and provides the most detailed evidence of a treatment's value. The FDA explains:

                "Study Participants: 300 to 3,000 volunteers who have the disease or condition. Length of Study: 1 to 4 years. Purpose: Efficacy and monitoring of adverse reactions"

            This phase involves:
                Large-scale testing (300-3,000 participants)
                Comparison with standard treatments
                Multiple testing locations
                Randomized control groups
                Duration of 1-4 years
    """
)

PATIENT_MATCHING_SYS = (
    '''
    You are a Patient Matching Expert. You have one tool:
        - match_patient_trial(xml_path: str, trial_text: str): this tool matches patients with
        trials and it needs a patient summary xml file and the trial summary text. Never run this tool
        without path provided for xml patients summary.
    Do not invent patients or trial text. Use the exact structured trial text provided from the trial_generation_agent.
    Return the number of matched patients and a list of matched patient IDs.
    '''
)
TRIAL_PRED_SYS = (
    '''
    You are a Trial Prediction Agent. You predict trial success according to the trial text it is provided. 
    You return only the probability score (e.g., 0.75). Do not generate trial or drugs, do not predict properties or docking scores. Never generate free text, only return
    the tool result it returns.
    '''
)

SELECTOR_PROMPT = (
    '''
    Select an agent to perform task.

    {roles}

    Current conversation context:
    {history}

    Read the above conversation, then select an agent from {participants} to perform the next task. 
    Follow strictly the plan from planning_agent, including the phased workflow (Drug Discovery → Preclinical → Clinical) and optimization loops.
    Make sure the planning_agent has assigned tasks before other agents start working. 
    Select only the agents as is in plan generated by planning_agent.
    Never select trial_generation_agent before final optimized lead with good ADMET, chemical properties, docking, and one selected lead compound.
    Never select molecule_optimization_agent if ADMET properties are not predicted with values.
    Never skip docking or property re-evaluation after optimization.
    If an agent fails to produce output after one retry, log the failure and select the planning_agent to handle the error and proceed.
    Once the planning_agent outputs the final report reeturn "FINAL" and select no further agents and terminate the process.
    Only select one agent. DO NOT select agents that are NOT in plan generated by planning_agent. 
    When you call the agent once and it generated output, DO NOT call it multiple times. Read the conversation history and use that response. 
    '''
)

async def main():
    model_client = OpenAIChatCompletionClient(
        model="o4-mini",
        api_key="OPENAI-API-KEY", #add your openai api key here
        max_output_tokens=1500
    )

    try:
        druggen_tools = await mcp_server_tools(StdioServerParams(
            command="python", args=["druggen_mcp_server.py"], read_timeout_seconds=600))
        docking_tools = await mcp_server_tools(StdioServerParams(
            command="python", args=["docking_mcp_server.py"], read_timeout_seconds=1800))
        chemical_properties_tools = await mcp_server_tools(StdioServerParams(
            command="python", args=["chemical_properties_mcp_server.py"], read_timeout_seconds=1500))
        admet_properties_tools = await mcp_server_tools(StdioServerParams(
            command="python", args=["admet_prediction_mcp_server.py"], read_timeout_seconds=1500))
        name_tools = await mcp_server_tools(
            StdioServerParams(command="python", args=["name2smiles_mcp_server.py"], read_timeout_seconds=150))
        optimization_tools = await mcp_server_tools(
            StdioServerParams(command="python", args=["mol_opt_mcp_server.py"], read_timeout_seconds=1500))
        panacea_patient_tools = await mcp_server_tools(
            StdioServerParams(command="python", args=["patient_matching_mcp_server.py"], read_timeout_seconds=3000))
        panacea_trial_tools = await mcp_server_tools(
            StdioServerParams(command="python", args=["trialgen_mcp_server.py"], read_timeout_seconds=3000))
        meditab_tools = await mcp_server_tools(
            StdioServerParams(command="python", args=["trialpred_mcp_server.py"], read_timeout_seconds=3000))
    planning_agent = AssistantAgent(
        name="planning_agent",
        description="An agent for planning tasks, this agent should be the first to engage when given a new task.",
        model_client=model_client,
        system_message="""
        You are a planning agent. 
        Your job is to break down complex tasks into smaller, manageable subtasks.
        Your team members are:
            1. druggen_agent: Only generates drug SMILES for given UniProt ID or IDs from target, so use it if drugs are not given in task.
            2. chemical_agent — Chemical Properties Specialist
                 Scope: predict/report chemical properties and return lead compounds (no trial/drug generation).
                 Tools it can use:
                   - select_leads_from_smiles(smiles_list)
                   - predict_pka_batch(smiles_list)
                   - logd_acid_batch(smiles_list, pH=7.4)
                   - logd_base_batch(smiles_list, pH=7.4)
                   - rdkit_physchem_batch(smiles_list, pH=7.4)
                   - predict_all_batch(smiles_list, is_acid=None|bool, is_base=None|bool, pH=7.4)
            3. admet_properties_agent — ADMET Properties Specialist
                 Scope: predict/report ADMET properties and docking; may resolve names/SMILES (no trial/drug generation).
                 For choosing the main lead compound, only one (THE BEST).
                 Tools it can use:
                   - get_drug_name_from_smiles(smiles_list)
                   - get_smiles_from_drug_name(drug_name)
                   - chemfm_list_properties()
                   - chemfm_get_description(property_name)
                   - chemfm_predict_single(smiles, property_name)    
                   - chemfm_predict_many(smiles, properties_list)        
                   - run_docking(target, smiles_list)
                 Always plan chemfm_list_properties tool first so you can take the exact names of the properties to run other 
                 admet tools.
            4. molecule_optimization_agent: Optimize lead molecule by a property it needs to be increased/decreased. It needs to be called
               with admet or chemical property, never docking.         
            5. trial_generation_agent: Generates trial for given lead compound with ADMET and chemical properties and/or disease/target so use it when trial generation is required.
               It always needs to be the last agent (after admet agent and chem agent) to be called, except if it is not the only agent that is required. Always return structured
               format from this agent like this
               - drug: SMILES and NAME  

CLINICAL TRIAL: 
- acronym: string, short study name  
- brief_title: string  
- official_title: string, descriptive trial title  
- study_status: string (e.g., "Recruiting", "Completed")  
- study_start_date: string, ISO format (e.g., "2026-03")  
- primary_completion_date: string, ISO format  
- completion_date: string, ISO format    
- condition: string, clinical condition studied  
- study_type: string 
- phase: string  
- intervention_model: string  
- allocation: string   
- masking: string   
- enrollment: integer
- arms: use output from panacea_extract_components tool, give it drug SMILES, brief_title, official_title, phase, and condition as input
- intervention_description: string, must reference the molecule by its SMILES  
- primary_outcomes: use output from panacea_extract_components tool, give it drug SMILES, brief_title, official_title, phase, and condition as input  
- secondary_outcomes: use output from panacea_extract_components tool, give it drug SMILES, brief_title, official_title, phase, and condition as input  
- other_outcomes: use output from panacea_extract_components tool, give it drug SMILES, brief_title, official_title, phase, and condition as input 
- eligibility_criteria: use output from panacea_extract_components tool, give it drug SMILES, brief_title, official_title, phase, and condition as input
- study_documents: list of strings 
- brief_summary: string, concise description of trial purpose, design, intervention, and eligibility.
            6. patient_matching_agent: Matches patients with trials. Do not plan agent without path provided for xml patients summary.
               And do not give it an invented trial summary text, use strictly the trial summary text generated from trial_generation_agent.
            7. trial_prediction_agent: Predicts trial success from structured output from trial generation agent.
        Rules:
            Strictly follow this workflow for drug development tasks:
            - Drug Discovery Phase:
              - Hits Generation: druggen_agent to generate >10 SMILES.
              - Docking: admet_properties_agent with run_docking to score and filter hits (keep top with lowest scores).
              - Leads Identification: chemical_agent for physchem and select_leads_from_smiles (apply Lipinski/Veber filters); admet_properties_agent for 10 relevant ADMET properties.
              - Select single best lead (lowest docking + passes filters + best ADMET, e.g., bioavailability >0.5, low hERG/hepatotoxicity).
            - Lead Optimization Phase:
              - molecule_optimization_agent on best lead, targeting weak properties (e.g., increase bioavailability if <0.5, decrease toxicity risks).
              - Re-evaluate: admet_properties_agent (docking + ADMET), chemical_agent (physchem/Lipinski/Veber).
              - Loop: If not satisfactory (fails Lipinski/Veber, bioavailability <0.5, high toxicity risks, docking score worse than initial lead), plan another optimization targeting issues. Cap at 3 iterations; if still unsatisfactory, select best available and proceed.
            - Preclinical Phase: Final ADMET re-assessment on optimized lead with admet_properties_agent.
            - Clinical Phase: trial_generation_agent on final lead; if XML path, patient_matching_agent; then trial_prediction_agent.
            - When drug generation is asked, generate just candidate drugs, NEVER generate trial, and check with get_drug_name_from_smiles if names for generated SMILES are available.
            - run_docking must be run after hits and after each optimization.
            - Final Report format should be like this:
                Drug Discovery Properties:
                - Hits: [list of initial SMILES]
                - Leads: [list of selected leads with properties]
                - Optimized Lead: SMILES with optimized properties
                - Chemical Properties: [dict or list from chem agent]
                - ADMET Properties: [dict or list from admet agent]
                - Docking Scores: [scores]

                Clinical Trial Report:
                [full structured trial from trial agent]

                Patient Matching: [results if applicable]

                Trial Success Probability: [probability]

                Summary: [overall summary]
            - Never invent your response from the agent. If a response from the agent is not present in the history, plan the agent to be called.
            - Always check the ADMET and chemical properties of the optimized new molecule, and if they are not good run optimization up to 3 times.
            - For ADMET prediction use 10 most relevant properties form the ADMET properties list (e.g., bioavailability, solubility, clearance, BBB permeability, hERG inhibition, hepatotoxicity).
            - If any agent or tool fails to produce output, log the failure and proceed with the best available data, selecting the best lead or optimized molecule to continue.
            - After all tasks are complete or if no further progress is possible, output the final report in the specified format.
        Do not invent new chemical or ADMET tests, use only the properties that are available and according to them choose the next steps
        and make the report. If additional tests are needed, just suggest them, but mention you do not have the tools to do them and continue
        with what you have. NEVER invent docking scores.
        For planning the trial generation, you must have a lead compound, or a compound must be specified in the task. 
        Do not invent or choose SMILES by yourself, and never prioritize SMILES with name available. Never plan get_drug_name_from_smiles for filtering SMILES lists
        in any way, only if that is explicitly asked. 
        You only plan and delegate tasks - you do not execute them yourself. You do not need to incorporate all agents in your plan.
        Always reason how and why you choose the drug before planning the trial generation agent.
        When assigning tasks, use this format:
        1. <agent> : <task>

        After all tasks are complete, the required tools are executed, or if any step fails and no further progress is possible, output the final report.
        """
    )

    druggen_agent = AssistantAgent(
        name="druggen_agent",
        model_client=model_client,
        tools=list(druggen_tools),
        system_message=DRUGGEN_SYS
    )
    chemical_agent = AssistantAgent(
        name="chemical_agent",
        model_client=model_client,
        tools=list(chemical_properties_tools),
        system_message=CHEM_PROPERTIES_SYS
    )
    admet_properties_agent = AssistantAgent(
        name="admet_properties_agent",
        model_client=model_client,
        tools=list(admet_properties_tools) + list(docking_tools) + list(name_tools),
        system_message=ADMET_PROPERTIES_SYS
    )
    molecule_optimization_agent = AssistantAgent(
        name="molecule_optimization_agent",
        model_client=model_client,
        tools=list(optimization_tools),
        system_message=MOL_OPT_SYS
    )
    trial_generator_agent = AssistantAgent(
        name="trial_generator_agent",
        model_client=model_client,
        tools=list(panacea_trial_tools),
        system_message=TRIAL_SYS
    )
    patient_matching_agent = AssistantAgent(
        name="patient_matching_agent",
        model_client=model_client,
        tools=list(panacea_patient_tools),
        system_message=PATIENT_MATCHING_SYS
    )
    trial_prediction_agent = AssistantAgent(
        name="trial_prediction_agent",
        model_client=model_client,
        tools=list(meditab_tools),
        system_message=TRIAL_PRED_SYS
    )

    termination = TextMentionTermination("FINAL")

    task = input("Enter your task: ").strip() or \
           "Simulate drug development for DPP4(P27487)"

    participants = [
        planning_agent, druggen_agent, chemical_agent, admet_properties_agent,
        molecule_optimization_agent, trial_generator_agent,
        patient_matching_agent, trial_prediction_agent
    ]
    team = SelectorGroupChat(
        participants=participants,
        model_client=model_client,
        selector_prompt=SELECTOR_PROMPT,
        termination_condition=termination,
        max_turns=30
    )

    try:
        result = await Console(team.run_stream(task=task))
        text = result.messages[-1].content
        return text
    except Exception as e:
        return f"Error running team: {str(e)}"

if __name__ == "__main__":
    asyncio.run(main())
