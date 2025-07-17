import asyncio
import json
import logging
from typing import Dict, Any

from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken

logger = logging.getLogger(__name__)


class DrugDiscoveryOrchestrator:
    def __init__(self,
                 agents: Dict[str, AssistantAgent],
                 uniprot_id: str,
                 num_molecules: int,
                 orchestrator_agent: UserProxyAgent,
                 patient_data_filepath: str):
        self.agents = agents
        self.uniprot_id = uniprot_id
        self.num_molecules = num_molecules
        self.orchestrator_agent = orchestrator_agent
        self.patient_data_filepath = patient_data_filepath
        self.max_workflow_retries = 8

    def _parse_agent_response_content(self, response_content: Any) -> Dict[str, Any]:
        logger.debug(f"Attempting to parse agent response. Raw content type: {type(response_content)}")
        if not response_content:
            return {"error": "Agent returned empty content", "retry": True}

        raw_string_content = (
            response_content.content if isinstance(response_content, TextMessage)
            else response_content
        )
        if not raw_string_content:
            return {"error": "No parseable content from agent", "retry": True}

        try:
            outer_parsed = json.loads(raw_string_content)
            if isinstance(outer_parsed, list) and len(outer_parsed) > 0:
                first_item = outer_parsed[0]
                if isinstance(first_item, dict) and first_item.get("type") == "text" and "text" in first_item:
                    return json.loads(first_item["text"])
                else:
                    return outer_parsed
            if isinstance(outer_parsed, dict):
                return outer_parsed
            return outer_parsed
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON: {e}")
            return {"error": f"Agent response not valid JSON: {e}", "retry": True}
        except Exception as e:
            logger.error(f"Internal parsing error: {e}", exc_info=True)
            return {"error": f"Internal parsing error: {str(e)}", "retry": True}

    async def _run_agent_step(self, agent_name: str, message_content: str, cancellation_token: CancellationToken) -> Dict[str, Any]:
        agent = self.agents.get(agent_name)
        if not agent:
            logger.critical(f"Agent '{agent_name}' not found.")
            return {"error": f"Agent '{agent_name}' not found", "completed": False}

        msg_str = json.dumps(message_content) if isinstance(message_content, (dict, list)) else str(message_content)
        logger.info(f"Orchestrator: Sending message to {agent_name}: {msg_str[:100]}...")

        messages = [TextMessage(content=msg_str, source=self.orchestrator_agent.name)]

        try:
            response_obj = await agent.on_messages(messages, cancellation_token=cancellation_token)
            chat_message = response_obj.chat_message
            if not chat_message or not chat_message.content:
                return {"error": f"{agent_name} returned no content", "retry": True}

            parsed_output = self._parse_agent_response_content(chat_message.content)
            if not isinstance(parsed_output, dict):
                return {"error": f"{agent_name} returned unexpected output type: {type(parsed_output)}", "retry": True}

            if parsed_output.get("error"):
                logger.warning(f"{agent_name} error: {parsed_output['error']}")
                return parsed_output

            return parsed_output
        except Exception as e:
            logger.error(f"Unexpected error in {agent_name}: {e}", exc_info=True)
            return {"error": f"Unexpected error in agent '{agent_name}': {e}", "retry": True}

    async def run_workflow(self) -> Dict[str, Any]:
        logger.info("Starting drug discovery workflow execution within Orchestrator.")

        final_report: Dict[str, Any] = {"report": None, "completed": False, "error": "Workflow did not complete."}

        for current_workflow_attempt in range(1, self.max_workflow_retries + 1):
            logger.info(f"\n Starting Workflow Attempt {current_workflow_attempt}/{self.max_workflow_retries}")
            workflow_failed_this_attempt = False
            cancellation_token = CancellationToken()
            current_message_content = f"Generate {self.num_molecules} SMILES for UniProt ID {self.uniprot_id}."

            try:
                logger.info("Orchestrator: Sending message to druggen_agent to generate SMILES...")
                druggen_output = await self._run_agent_step("druggen_agent", current_message_content,
                                                            cancellation_token)
                if druggen_output.get("error") or not druggen_output.get("smiles"):
                    logger.warning(f"Druggen agent step failed: {druggen_output.get('error', 'No SMILES generated')}")
                    workflow_failed_this_attempt = True
                    continue

                smiles_list = druggen_output["smiles"]
                if not smiles_list:
                    logger.warning("Druggen agent generated an empty SMILES list.")
                    workflow_failed_this_attempt = True
                    continue
                logger.info(f"Druggen agent generated {len(smiles_list)} SMILES.")
                current_message_content = json.dumps({"smiles": smiles_list})

                logger.info("Orchestrator: Sending message to toxscreen_agent to filter toxic SMILES...")
                toxscreen_output = await self._run_agent_step("toxscreen_agent", current_message_content,
                                                              cancellation_token)
                if toxscreen_output.get("error") or not toxscreen_output.get("non_toxic_smiles"):
                    logger.warning(
                        f"Toxscreen agent step failed or all compounds are toxic: {toxscreen_output.get('error', 'No non-toxic SMILES')}")
                    workflow_failed_this_attempt = True
                    continue

                non_toxic_smiles = toxscreen_output["non_toxic_smiles"]
                if not non_toxic_smiles:
                    logger.warning("Toxscreen agent returned an empty non_toxic_smiles list.")
                    workflow_failed_this_attempt = True
                    continue
                logger.info(f"Toxscreen agent filtered to {len(non_toxic_smiles)} non-toxic SMILES.")
                current_message_content = json.dumps({"non_toxic_smiles": non_toxic_smiles})

                logger.info("Orchestrator: Sending message to docking_agent for non-toxic SMILES...")
                docking_output = await self._run_agent_step("docking_agent", current_message_content,
                                                            cancellation_token)

                if docking_output.get("error") or not docking_output.get("smiles") or not docking_output.get("scores"):
                    logger.warning(f"Docking agent step failed: {docking_output.get('error', 'No docking results')}")
                    workflow_failed_this_attempt = True
                    continue

                docked_smiles = docking_output["smiles"]
                docking_scores = docking_output["scores"]
                if not docked_smiles or not docking_scores or len(docked_smiles) != len(docking_scores):
                    logger.warning("Docking agent returned inconsistent or empty SMILES/scores lists.")
                    workflow_failed_this_attempt = True
                    continue
                logger.info(f"Docking agent completed for {len(docked_smiles)} SMILES.")
                current_message_content = json.dumps({"smiles": docked_smiles, "scores": docking_scores})

                logger.info("Orchestrator: Sending message to selector_agent to pick the best SMILES...")
                selector_output = await self._run_agent_step("selector_agent", current_message_content,
                                                             cancellation_token)
                if selector_output.get("error") or not selector_output.get("ranked_smiles") or not selector_output.get(
                        "ranked_scores"):
                    logger.warning(
                        f"Selector agent step failed: {selector_output.get('error', 'No best SMILES selected')}")
                    workflow_failed_this_attempt = True
                    continue
                if not selector_output["ranked_smiles"]:
                    logger.warning("Selector agent returned an empty ranked_smiles list.")
                    workflow_failed_this_attempt = True
                    continue

                selected_smiles = selector_output["ranked_smiles"][0]
                selected_score = selector_output["ranked_scores"][0]
                logger.info(f"Selector agent selected SMILES: {selected_smiles} with score: {selected_score}.")

                logger.info("Orchestrator: Sending message to profiling_agent for properties...")
                profiling_message_content = json.dumps({"smiles": selected_smiles})
                profiling_output = await self._run_agent_step("profiling_agent", profiling_message_content,
                                                              cancellation_token)
                if profiling_output.get("error"):
                    logger.warning(f"Profiling agent step failed: {profiling_output.get('error')}")
                    workflow_failed_this_attempt = True
                    continue

                properties = profiling_output
                logger.info(f"Profiling agent completed. Properties: {properties}")

                drug_discovery_summary_data = {
                    "smiles": selected_smiles,
                    "score": selected_score,
                    "properties": properties,
                }
                logger.info("Orchestrator: Combined profiling results for summary.")
                current_message_content = json.dumps(drug_discovery_summary_data)

                logger.info("Orchestrator: Sending message to summary_agent to generate initial drug report...")
                summary_output = await self._run_agent_step("summary_agent", current_message_content,
                                                            cancellation_token)
                if summary_output.get("completed") is not True:
                    logger.warning(
                        f"Summary agent did not signal completion: {summary_output.get('error', 'Unknown error')}. Retrying workflow.")
                    workflow_failed_this_attempt = True
                    continue

                drug_discovery_report = summary_output.get('report', {})
                if not isinstance(drug_discovery_report, dict):
                    logger.warning(
                        f"Summary agent returned non-dictionary report: {drug_discovery_report}. Using empty dict.")
                    drug_discovery_report = {}

                logger.info(f"Initial Drug Discovery Summary: {json.dumps(drug_discovery_report, indent=2)}")

                logger.info("Orchestrator: Proceeding to clinical trial generation and matching steps.")

                trialgen_input = json.dumps({
                    "drug_smiles": selected_smiles,
                    "target_uniprot_id": self.uniprot_id,
                    "basic_eligibility": "Adults, any gender, target disease related to UniProt ID."
                })
                trial_generator_output = await self._run_agent_step("trial_generator_agent", trialgen_input,
                                                                    cancellation_token)
                if trial_generator_output.get("error") or not trial_generator_output.get("trial_protocol_text"):
                    logger.warning(f"Trial generation agent failed: {trial_generator_output.get('error')}")
                    workflow_failed = True
                    continue
                trial_protocol_dict = trial_generator_output["trial_protocol_text"]
                logger.info("Trial protocol dictionary generated by trial_generator_agent.")
                logger.debug(f"Trial protocol dictionary: {trial_protocol_dict}")

                normalized_patient_path = self.patient_data_filepath.replace("\\", "/")

                arms_string = ', '.join([f"{arm['name']} ({arm['dose']})" for arm in trial_protocol_dict['arms']])

                trial_summary = trial_protocol_dict.get("summary", "")

                trial_text_for_outcome = (
                    f"{trial_protocol_dict['brief_title']}. "
                    f"{trial_protocol_dict['condition']}. "
                    f"{trial_protocol_dict['phase']}. "
                    f"{trial_protocol_dict['intervention_description']}. "
                    f"Arms: {arms_string}. "
                    f"Enrollment: {trial_protocol_dict['enrollment']}. "
                    f"Start date: {trial_protocol_dict['study_start_date']}. "
                    f"Primary outcomes: {', '.join(trial_protocol_dict['primary_outcomes'])}. "
                    f"Secondary outcomes: {', '.join(trial_protocol_dict['secondary_outcomes'])}.\n\n"
                    f"Summary: {trial_summary}"
                )

                match_patient_trial_output = await self._run_agent_step(
                    "patient_matching_agent",
                    json.dumps({
                        "xml_path": normalized_patient_path,
                        "trial_summary": trial_summary,
                        "outfile": "/path/to/result/file.json"
                    }),
                    cancellation_token
                )

                if match_patient_trial_output.get("error"):
                    logger.warning(f"Patient matching failed: {match_patient_trial_output['error']}")
                    workflow_failed_this_attempt = True
                    continue

                matches = match_patient_trial_output.get("matches", [])
                matched_patients_count = match_patient_trial_output.get("matched_patients_count", 0)
                total_patients_parsed = match_patient_trial_output.get("total_patients_parsed", len(matches))
                matched_patients_file = match_patient_trial_output.get("matched_patients_file")

                if not matches:
                    logger.warning("Patient-matching agent returned no eligible patients.")
                    workflow_failed_this_attempt = True
                    continue

                logger.info(f"Found {matched_patients_count} matched patients out of {total_patients_parsed}.")

                predict_trial_outcome_output = await self._run_agent_step(
                    "trial_outcome_prediction_agent",
                    json.dumps({
                        "trial_text": trial_text_for_outcome
                    }),
                    cancellation_token
                )

                if predict_trial_outcome_output.get("error"):
                    logger.warning(f"Trial outcome prediction failed: {predict_trial_outcome_output.get('error')}")
                    workflow_failed = True
                    continue

                if isinstance(predict_trial_outcome_output, dict):
                    success_probability = predict_trial_outcome_output.get("success_probability", 0.0)
                elif isinstance(predict_trial_outcome_output, list) and len(predict_trial_outcome_output) > 0:
                    first_prob = predict_trial_outcome_output[0]
                    success_probability = first_prob.get("success_probability", 0.0)
                else:
                    logger.warning(
                        f"Trial outcome prediction agent returned unexpected format: {predict_trial_outcome_output}")
                    workflow_failed = True
                    continue

                logger.info(f"Trial outcome predicted with success probability: {success_probability:.2f}")

                patient_summary = {
                    "total_patients_parsed": total_patients_parsed,
                    "matched_patients_count": matched_patients_count,
                    "matched_patients_file": matched_patients_file
                }

                trial_outcome = {
                    "predicted_success_probability": success_probability,
                    "prediction_notes": "Prediction based on prior data and compound properties."
                }

                final_clinical_report_input = {
                    "drug_discovery_summary": drug_discovery_report,
                    "trial_protocol": trial_protocol_dict,
                    "matched_patients_summary": patient_summary,
                    "trial_outcome_prediction": trial_outcome
                }

                standardized_report_output = await self._run_agent_step(
                    "clinical_report_generator_agent",
                    final_clinical_report_input,
                    cancellation_token
                )

                if standardized_report_output.get("error") or standardized_report_output.get("completed") is not True:
                    logger.warning(f"Clinical report generation failed: {standardized_report_output.get('error')}")
                    workflow_failed = True
                    continue

                final_report = {
                    "completed": True,
                    "report": standardized_report_output.get("report", final_clinical_report_input)
                }
                break


            except Exception as e:
                workflow_failed = True
                cancellation_token.cancel()

            if workflow_failed and current_workflow_attempt == self.max_workflow_retries:
                final_report["error"] = "Workflow failed after max retries."
                break

        return final_report
