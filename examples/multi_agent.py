#!/usr/bin/env python
"""
Multi-Agent Collaboration Example

This example demonstrates the multi-agent capabilities of the Cognitive Engine,
showcasing how multiple cognitive agents can collaborate to solve complex problems.
The example implements a collaborative problem-solving scenario where agents with
different expertise domains work together to analyze a complex interdisciplinary challenge.

Key aspects demonstrated:
1. Agent creation with specialized domains of expertise
2. Communication between agents using symbolic and neural representations
3. Collective reasoning and problem decomposition
4. Knowledge sharing and integration
5. Consensus building and conflict resolution
6. Emergent collaborative intelligence

The example creates a team of specialized agents to analyze the impact of climate
change on agriculture and propose adaptation strategies.
"""

import os
import time
import logging
from typing import Dict, List, Tuple, Any

from cognitive_engine import HybridCognitiveEngine
from cognitive_engine.fractal import Symbol, SymbolRelation
from cognitive_engine.memory import PervasiveMemory
from cognitive_engine.peff import EthicalFramework

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CognitiveAgent:
    """
    A specialized agent built on the Cognitive Engine architecture with
    domain-specific knowledge and capabilities.
    """
    
    def __init__(self, name: str, expertise_domain: str, knowledge_bias: float = 0.7):
        """
        Initialize a cognitive agent with a specific expertise domain.
        
        Args:
            name: The agent's identifier
            expertise_domain: The primary domain of expertise (e.g., "climate", "agriculture")
            knowledge_bias: How strongly the agent favors its domain of expertise (0.0-1.0)
        """
        self.name = name
        self.expertise_domain = expertise_domain
        self.knowledge_bias = knowledge_bias
        
        # Initialize the cognitive engine for this agent
        self.engine = HybridCognitiveEngine()
        
        # Initialize agent's private memory
        self.private_memory = PervasiveMemory(storage_path=f"./agent_memory/{name}")
        
        # Initialize communication channels
        self.message_queue = []
        
        logger.info(f"Agent {name} initialized with expertise in {expertise_domain}")
    
    def initialize_knowledge_base(self, domain_knowledge: Dict[str, Any]):
        """
        Set up the agent's initial knowledge base according to its domain of expertise.
        
        Args:
            domain_knowledge: Dictionary containing domain-specific knowledge
        """
        logger.info(f"Initializing knowledge base for agent {self.name}")
        
        # Create symbols for domain concepts
        for concept, properties in domain_knowledge.get("concepts", {}).items():
            symbol = Symbol(
                name=concept,
                properties=properties
            )
            # Add to fractal system at the appropriate layer (concepts at layer 3)
            self.engine.symbolic_system.add_symbol(symbol, layer=3)
        
        # Create relations between concepts
        for relation in domain_knowledge.get("relations", []):
            source = self.engine.symbolic_system.find_symbol_by_name(relation["source"])
            target = self.engine.symbolic_system.find_symbol_by_name(relation["target"])
            if source and target:
                self.engine.symbolic_system.add_relation(source, target, relation["type"])
        
        # Add factual knowledge to memory
        for fact in domain_knowledge.get("facts", []):
            self.engine.memory_system.store(
                content=fact["content"],
                metadata={
                    "domain": self.expertise_domain,
                    "confidence": fact.get("confidence", 0.9),
                    "source": fact.get("source", "domain_knowledge"),
                    "type": "fact"
                }
            )
        
        logger.info(f"Knowledge base initialized for agent {self.name} with "
                  f"{len(domain_knowledge.get('concepts', {}))} concepts, "
                  f"{len(domain_knowledge.get('relations', []))} relations, and "
                  f"{len(domain_knowledge.get('facts', []))} facts")
    
    def receive_message(self, sender: str, content: str, attachments: Dict = None):
        """
        Receive a message from another agent.
        
        Args:
            sender: The name of the sending agent
            content: The text content of the message
            attachments: Optional structured data or symbol references
        """
        message = {
            "sender": sender,
            "timestamp": time.time(),
            "content": content,
            "attachments": attachments or {},
            "processed": False
        }
        self.message_queue.append(message)
        logger.info(f"Agent {self.name} received message from {sender}: {content[:50]}...")
    
    def process_messages(self):
        """Process all unprocessed messages in the queue."""
        for i, message in enumerate(self.message_queue):
            if not message["processed"]:
                self._process_single_message(message)
                self.message_queue[i]["processed"] = True
    
    def _process_single_message(self, message: Dict):
        """
        Process a single message and integrate its content.
        
        Args:
            message: The message dictionary
        """
        # Store in memory
        self.engine.memory_system.store(
            content=message["content"],
            metadata={
                "sender": message["sender"],
                "timestamp": message["timestamp"],
                "type": "communication"
            }
        )
        
        # Process any attached symbols or data
        if "symbols" in message["attachments"]:
            for symbol_data in message["attachments"]["symbols"]:
                # Create a local copy of the symbol
                symbol = Symbol(
                    name=symbol_data["name"],
                    properties=symbol_data["properties"]
                )
                # Add to local system with reference to source
                symbol.properties["source_agent"] = message["sender"]
                self.engine.symbolic_system.add_symbol(symbol, layer=symbol_data.get("layer", 3))
                
                logger.info(f"Agent {self.name} integrated symbol {symbol.name} from {message['sender']}")
    
    def analyze_problem(self, problem_statement: str) -> Dict:
        """
        Analyze a problem from the agent's specialized perspective.
        
        Args:
            problem_statement: Description of the problem to analyze
            
        Returns:
            Dictionary containing analysis results
        """
        logger.info(f"Agent {self.name} analyzing problem: {problem_statement[:50]}...")
        
        # Process the problem through the cognitive engine
        result = self.engine.process(
            query=f"Analyze this problem from the perspective of {self.expertise_domain} expertise: {problem_statement}",
            reasoning_depth="deep",
            use_memory=True
        )
        
        # Extract key concepts relevant to the problem
        relevant_symbols = self.engine.symbolic_system.find_symbols_by_relevance(
            query=problem_statement,
            domain=self.expertise_domain,
            threshold=0.6,
            max_symbols=10
        )
        
        # Store the analysis in agent's memory
        self.private_memory.store(
            content=result["response"],
            metadata={
                "type": "analysis",
                "problem": problem_statement[:100],
                "timestamp": time.time()
            }
        )
        
        # Prepare the analysis result
        analysis = {
            "agent": self.name,
            "domain": self.expertise_domain,
            "summary": result["response"],
            "confidence": result.get("confidence", 0.8),
            "key_concepts": [{"name": s.name, "properties": s.properties} for s in relevant_symbols],
            "reasoning_trace": result.get("reasoning_trace", [])
        }
        
        logger.info(f"Agent {self.name} completed analysis with {len(relevant_symbols)} key concepts")
        return analysis
    
    def generate_proposal(self, problem_statement: str, team_analyses: List[Dict]) -> Dict:
        """
        Generate a proposal based on the agent's expertise and team analyses.
        
        Args:
            problem_statement: The problem being addressed
            team_analyses: Analyses from all team members
            
        Returns:
            Dictionary containing the agent's proposal
        """
        logger.info(f"Agent {self.name} generating proposal based on team analyses")
        
        # Integrate team analyses into the reasoning process
        team_input = ""
        for analysis in team_analyses:
            if analysis["agent"] != self.name:  # Skip own analysis
                team_input += f"\n\nAnalysis from {analysis['agent']} (expertise: {analysis['domain']}):\n"
                team_input += analysis["summary"]
        
        # Generate proposal through the cognitive engine
        query = (f"Generate a proposal to address this problem from the perspective of "
                f"{self.expertise_domain} expertise, taking into account the following team analyses: "
                f"{team_input}\n\nProblem statement: {problem_statement}")
        
        result = self.engine.process(
            query=query,
            reasoning_depth="deep",
            use_memory=True
        )
        
        # Store the proposal in agent's memory
        self.private_memory.store(
            content=result["response"],
            metadata={
                "type": "proposal",
                "problem": problem_statement[:100],
                "timestamp": time.time()
            }
        )
        
        # Prepare the proposal
        proposal = {
            "agent": self.name,
            "domain": self.expertise_domain,
            "proposal_text": result["response"],
            "confidence": result.get("confidence", 0.8),
            "key_points": result.get("key_points", []),
            "supporting_evidence": result.get("supporting_evidence", [])
        }
        
        logger.info(f"Agent {self.name} generated proposal with confidence {proposal['confidence']}")
        return proposal
    
    def evaluate_proposal(self, proposal: Dict) -> Dict:
        """
        Evaluate another agent's proposal from this agent's expertise perspective.
        
        Args:
            proposal: The proposal to evaluate
            
        Returns:
            Evaluation results
        """
        logger.info(f"Agent {self.name} evaluating proposal from {proposal['agent']}")
        
        # Evaluate the proposal through the cognitive engine
        query = (f"Evaluate this proposal from the perspective of {self.expertise_domain} expertise. "
                f"Identify strengths, weaknesses, and potential improvements:\n\n"
                f"Proposal from {proposal['agent']} (expertise: {proposal['domain']}):\n"
                f"{proposal['proposal_text']}")
        
        result = self.engine.process(
            query=query,
            reasoning_depth="moderate",
            use_memory=True
        )
        
        # Extract the evaluation aspects
        evaluation = {
            "evaluator": self.name,
            "proposal_agent": proposal["agent"],
            "overall_assessment": result["response"],
            "agreement_level": result.get("agreement_level", 0.5),
            "strengths": result.get("strengths", []),
            "weaknesses": result.get("weaknesses", []),
            "suggestions": result.get("suggestions", [])
        }
        
        logger.info(f"Agent {self.name} evaluated proposal with agreement level {evaluation['agreement_level']}")
        return evaluation
    
    def share_knowledge(self, recipient: str, topic: str) -> Dict:
        """
        Share relevant knowledge with another agent.
        
        Args:
            recipient: The name of the recipient agent
            topic: The topic to share knowledge about
            
        Returns:
            The knowledge package to send
        """
        logger.info(f"Agent {self.name} sharing knowledge about {topic} with {recipient}")
        
        # Retrieve relevant memories
        memories = self.engine.memory_system.retrieve(
            query=topic,
            filter={"domain": self.expertise_domain},
            limit=5
        )
        
        # Find relevant symbols
        symbols = self.engine.symbolic_system.find_symbols_by_relevance(
            query=topic,
            domain=self.expertise_domain,
            threshold=0.7,
            max_symbols=5
        )
        
        # Package knowledge for sharing
        knowledge_package = {
            "sender": self.name,
            "recipient": recipient,
            "topic": topic,
            "content": f"Knowledge sharing from {self.name} about {topic}",
            "attachments": {
                "memories": [
                    {
                        "content": memory.content,
                        "metadata": memory.metadata
                    } for memory in memories
                ],
                "symbols": [
                    {
                        "name": symbol.name,
                        "properties": symbol.properties,
                        "layer": self.engine.symbolic_system.get_symbol_layer(symbol)
                    } for symbol in symbols
                ]
            }
        }
        
        logger.info(f"Agent {self.name} prepared knowledge package with {len(memories)} memories and {len(symbols)} symbols")
        return knowledge_package
    
    def find_consensus(self, proposals: List[Dict], evaluations: List[Dict]) -> Dict:
        """
        Find areas of consensus among different proposals and evaluations.
        
        Args:
            proposals: List of all team proposals
            evaluations: List of all proposal evaluations
            
        Returns:
            Consensus summary
        """
        logger.info(f"Agent {self.name} finding consensus among {len(proposals)} proposals")
        
        # Prepare consolidated input for the engine
        consolidated_input = "Team Proposals:\n\n"
        for i, proposal in enumerate(proposals):
            consolidated_input += f"Proposal {i+1} from {proposal['agent']} (expertise: {proposal['domain']}):\n"
            consolidated_input += f"{proposal['proposal_text'][:500]}...\n\n"
        
        consolidated_input += "\n\nTeam Evaluations:\n\n"
        for i, evaluation in enumerate(evaluations):
            consolidated_input += f"Evaluation {i+1} from {evaluation['evaluator']} of {evaluation['proposal_agent']}'s proposal:\n"
            consolidated_input += f"{evaluation['overall_assessment'][:300]}...\n\n"
        
        # Process through cognitive engine
        query = (f"Identify areas of consensus and disagreement among these proposals and evaluations. "
                f"Focus on finding common ground and integrating perspectives:\n\n{consolidated_input}")
        
        result = self.engine.process(
            query=query,
            reasoning_depth="deep",
            use_memory=True
        )
        
        # Extract consensus information
        consensus = {
            "facilitator": self.name,
            "consensus_summary": result["response"],
            "areas_of_agreement": result.get("areas_of_agreement", []),
            "areas_of_disagreement": result.get("areas_of_disagreement", []),
            "integrated_perspective": result.get("integrated_perspective", ""),
            "confidence": result.get("confidence", 0.7)
        }
        
        logger.info(f"Agent {self.name} identified consensus with {len(consensus['areas_of_agreement'])} agreement areas")
        return consensus


class AgentTeam:
    """
    Manages a team of cognitive agents working together on a problem.
    """
    
    def __init__(self, team_name: str):
        """
        Initialize a team of cognitive agents.
        
        Args:
            team_name: Name identifier for the team
        """
        self.name = team_name
        self.agents = {}
        self.shared_memory = PervasiveMemory(storage_path=f"./team_memory/{team_name}")
        self.ethical_framework = EthicalFramework()
        logger.info(f"Team {team_name} initialized")
    
    def add_agent(self, agent: CognitiveAgent):
        """
        Add an agent to the team.
        
        Args:
            agent: The CognitiveAgent to add
        """
        self.agents[agent.name] = agent
        logger.info(f"Agent {agent.name} added to team {self.name}")
    
    def broadcast_message(self, sender: str, content: str, attachments: Dict = None):
        """
        Broadcast a message to all team members.
        
        Args:
            sender: The name of the sending agent
            content: The message content
            attachments: Optional attachments
        """
        for agent_name, agent in self.agents.items():
            if agent_name != sender:  # Don't send to self
                agent.receive_message(sender, content, attachments)
        logger.info(f"Message from {sender} broadcasted to team {self.name}")
    
    def send_message(self, sender: str, recipient: str, content: str, attachments: Dict = None):
        """
        Send a message to a specific team member.
        
        Args:
            sender: The name of the sending agent
            recipient: The name of the receiving agent
            content: The message content
            attachments: Optional attachments
        """
        if recipient in self.agents:
            self.agents[recipient].receive_message(sender, content, attachments)
            logger.info(f"Message from {sender} sent to {recipient}")
        else:
            logger.warning(f"Recipient {recipient} not found in team {self.name}")
    
    def solve_problem(self, problem_statement: str) -> Dict:
        """
        Coordinate the team to solve a complex problem.
        
        Args:
            problem_statement: Description of the problem to solve
            
        Returns:
            The solution with supporting information
        """
        logger.info(f"Team {self.name} beginning collaborative problem solving for: {problem_statement[:50]}...")
        
        # Phase 1: Individual Analysis
        logger.info("Phase 1: Individual Analysis")
        analyses = {}
        for agent_name, agent in self.agents.items():
            analyses[agent_name] = agent.analyze_problem(problem_statement)
        
        # Store analyses in shared memory
        for agent_name, analysis in analyses.items():
            self.shared_memory.store(
                content=analysis["summary"],
                metadata={
                    "type": "analysis",
                    "agent": agent_name,
                    "domain": analysis["domain"],
                    "problem": problem_statement[:100],
                    "timestamp": time.time()
                }
            )
        
        # Phase 2: Knowledge Sharing
        logger.info("Phase 2: Knowledge Sharing")
        for agent_name, agent in self.agents.items():
            # Process received messages
            agent.process_messages()
            
            # Share domain knowledge with team
            for recipient_name in self.agents.keys():
                if recipient_name != agent_name:
                    knowledge = agent.share_knowledge(recipient_name, problem_statement)
                    self.send_message(
                        sender=agent_name,
                        recipient=recipient_name,
                        content=knowledge["content"],
                        attachments=knowledge["attachments"]
                    )
        
        # Give agents time to process the shared knowledge
        for agent in self.agents.values():
            agent.process_messages()
        
        # Phase 3: Proposal Generation
        logger.info("Phase 3: Proposal Generation")
        proposals = {}
        analyses_list = list(analyses.values())
        for agent_name, agent in self.agents.items():
            proposals[agent_name] = agent.generate_proposal(problem_statement, analyses_list)
        
        # Store proposals in shared memory
        for agent_name, proposal in proposals.items():
            self.shared_memory.store(
                content=proposal["proposal_text"],
                metadata={
                    "type": "proposal",
                    "agent": agent_name,
                    "domain": proposal["domain"],
                    "problem": problem_statement[:100],
                    "timestamp": time.time()
                }
            )
        
        # Phase 4: Proposal Evaluation
        logger.info("Phase 4: Proposal Evaluation")
        evaluations = []
        for evaluator_name, evaluator in self.agents.items():
            for proposer_name, proposal in proposals.items():
                if evaluator_name != proposer_name:  # Don't evaluate own proposal
                    evaluation = evaluator.evaluate_proposal(proposal)
                    evaluations.append(evaluation)
        
        # Store evaluations in shared memory
        for evaluation in evaluations:
            self.shared_memory.store(
                content=evaluation["overall_assessment"],
                metadata={
                    "type": "evaluation",
                    "evaluator": evaluation["evaluator"],
                    "proposal_agent": evaluation["proposal_agent"],
                    "problem": problem_statement[:100],
                    "timestamp": time.time()
                }
            )
        
        # Phase 5: Consensus Building
        logger.info("Phase 5: Consensus Building")
        # Select a facilitator agent (using the first agent for simplicity)
        facilitator_name = list(self.agents.keys())[0]
        facilitator = self.agents[facilitator_name]
        
        # Find consensus
        proposals_list = list(proposals.values())
        consensus = facilitator.find_consensus(proposals_list, evaluations)
        
        # Store consensus in shared memory
        self.shared_memory.store(
            content=consensus["consensus_summary"],
            metadata={
                "type": "consensus",
                "facilitator": consensus["facilitator"],
                "problem": problem_statement[:100],
                "timestamp": time.time()
            }
        )
        
        # Phase 6: Solution Refinement
        logger.info("Phase 6: Solution Refinement")
        # Have each agent contribute to refining the consensus
        refinements = {}
        for agent_name, agent in self.agents.items():
            result = agent.engine.process(
                query=(f"Refine this consensus solution from your {agent.expertise_domain} expertise perspective:\n\n"
                      f"{consensus['consensus_summary']}\n\nArea to focus on: {agent.expertise_domain} implications"),
                reasoning_depth="moderate",
                use_memory=True
            )
            refinements[agent_name] = {
                "agent": agent_name,
                "domain": agent.expertise_domain,
                "refinement": result["response"]
            }
        
        # Phase 7: Final Integration
        logger.info("Phase 7: Final Integration")
        # Integrate all refinements into a final solution
        refinements_input = "\n\n".join([f"Refinement from {r['agent']} ({r['domain']}): {r['refinement'][:300]}..." 
                                         for r in refinements.values()])
        
        final_integration = facilitator.engine.process(
            query=(f"Create a final integrated solution by combining these refinements with the consensus:\n\n"
                  f"Original consensus: {consensus['consensus_summary']}\n\n"
                  f"Refinements:\n{refinements_input}"),
            reasoning_depth="deep",
            use_memory=True
        )
        
        # Ethical evaluation
        ethical_assessment = self.ethical_framework.evaluate_action({
            "type": "solution_proposal",
            "content": final_integration["response"],
            "domain": "interdisciplinary",
            "context": problem_statement
        })
        
        # Prepare final solution
        solution = {
            "problem": problem_statement,
            "team": self.name,
            "solution": final_integration["response"],
            "ethical_assessment": {
                "verdict": ethical_assessment.verdict,
                "score": ethical_assessment.score,
                "considerations": ethical_assessment.reasoning
            },
            "contributing_agents": list(self.agents.keys()),
            "process": {
                "analyses": analyses,
                "proposals": proposals,
                "evaluations": evaluations,
                "consensus": consensus,
                "refinements": refinements
            },
            "timestamp": time.time()
        }
        
        logger.info(f"Team {self.name} completed problem solving with ethical score {ethical_assessment.score}")
        return solution


def initialize_climate_expert(name: str) -> CognitiveAgent:
    """Create and initialize a climate science expert agent."""
    agent = CognitiveAgent(name=name, expertise_domain="climate_science")
    
    # Define domain knowledge
    domain_knowledge = {
        "concepts": {
            "global_warming": {
                "type": "phenomenon",
                "definition": "Long-term heating of Earth's climate system due to human activities"
            },
            "greenhouse_effect": {
                "type": "process",
                "definition": "Trapping of the sun's heat by atmospheric gases"
            },
            "carbon_dioxide": {
                "type": "substance",
                "definition": "Major greenhouse gas produced by human activities"
            },
            "climate_model": {
                "type": "tool",
                "definition": "Mathematical representation of climate system components"
            },
            "extreme_weather": {
                "type": "phenomenon",
                "definition": "Unusual, severe or unseasonal weather"
            }
        },
        "relations": [
            {"source": "carbon_dioxide", "target": "greenhouse_effect", "type": "contributes_to"},
            {"source": "greenhouse_effect", "target": "global_warming", "type": "causes"},
            {"source": "global_warming", "target": "extreme_weather", "type": "increases"}
        ],
        "facts": [
            {
                "content": "Global average temperatures have increased by about 1.1°C since pre-industrial times.",
                "confidence": 0.95,
                "source": "IPCC"
            },
            {
                "content": "Carbon dioxide concentrations have increased from 280 ppm in pre-industrial times to over 410 ppm today.",
                "confidence": 0.98,
                "source": "NOAA"
            },
            {
                "content": "Climate models project continued warming, with increases of 1.5°C to 4.5°C by 2100 depending on emission scenarios.",
                "confidence": 0.85,
                "source": "IPCC"
            }
        ]
    }
    
    agent.initialize_knowledge_base(domain_knowledge)
    return agent


def initialize_agriculture_expert(name: str) -> CognitiveAgent:
    """Create and initialize an agricultural expert agent."""
    agent = CognitiveAgent(name=name, expertise_domain="agriculture")
    
    # Define domain knowledge
    domain_knowledge = {
        "concepts": {
            "crop_yield": {
                "type": "measure",
                "definition": "Production of crop per unit of land area"
            },
            "irrigation": {
                "type": "practice",
                "definition": "Artificial application of water to land or soil"
            },
            "drought_resistance": {
                "type": "trait",
                "definition": "Ability of crops to withstand periods of water scarcity"
            },
            "sustainable_agriculture": {
                "type": "approach",
                "definition": "Farming practices that protect the environment and public health"
            },
            "soil_health": {
                "type": "condition",
                "definition": "Capacity of soil to function as a living ecosystem"
            }
        },
        "relations": [
            {"source": "irrigation", "target": "crop_yield", "type": "increases"},
            {"source": "drought_resistance", "target": "crop_yield", "type": "stabilizes"},
            {"source": "sustainable_agriculture", "target": "soil_health", "type": "improves"}
        ],
        "facts": [
            {
                "content": "For each 1°C of temperature increase, grain yields decline by about 5%.",
                "confidence": 0.9,
                "source": "Nature Climate Change"
            },
            {
                "content": "Climate change is projected to reduce global agricultural productivity by 2-15% per decade.",
                "confidence": 0.8,
                "source": "FAO"
            },
            {
                "content": "Drought-resistant crop varieties can maintain 70-90% of optimal yield under moderate drought conditions.",
                "confidence": 0.85,
                "source": "Agricultural Research"
            }
        ]
    }
    
    agent.initialize_knowledge_base(domain_knowledge)
    return agent


def initialize_economics_expert(name: str) -> CognitiveAgent:
    """Create and initialize an economics expert agent."""
    agent = CognitiveAgent(name=name, expertise_domain="economics")
    
    # Define domain knowledge
    domain_knowledge = {
        "concepts": {
            "adaptation_cost": {
                "type": "economic_measure",
                "definition": "Expenditures required to adapt to climate change impacts"
            },
            "food_security": {
                "type": "economic_condition",
                "definition": "State in which people have reliable access to sufficient food"
            },
            "supply_chain": {
                "type": "economic_system",
                "definition": "Network between a company and its suppliers to produce and distribute products"
            },
            "market_volatility": {
                "type": "economic_phenomenon",
                "definition": "Rate at which prices increase or decrease over short time periods"
            },
            "subsidies": {
                "type": "policy_tool",
                "definition": "Government payments to businesses or economic sectors"
            }
        },
        "relations": [
            {"source": "adaptation_cost", "target": "food_security", "type": "influences"},
            {"source": "supply_chain", "target": "food_security", "type": "affects"},
            {"source": "market_volatility", "target": "food_security", "type": "threatens"}
        ],
        "facts": [
            {
                "content": "Global economic costs of climate change are estimated to be between 5-20% of global GDP annually.",
                "confidence": 0.75,
                "source": "Stern Review"
            },
            {
                "content": "Agricultural adaptation investments of $7-8 billion per year could help secure future food production.",
                "confidence": 0.8,
                "source": "World Bank"
            },
            {
                "content": "Supply chain disruptions from climate extremes can increase food prices by 10-20% in affected regions.",
                "confidence": 0.85,
                "source": "Economic Research"
            }
        ]
    }
    
    agent.initialize_knowledge_base(domain_knowledge)
    return agent


def initialize_policy_expert(name: str) -> CognitiveAgent:
    """Create and initialize a policy expert agent."""
    agent = CognitiveAgent(name=name, expertise_domain="policy")
    
    # Define domain knowledge
    domain_knowledge = {
        "concepts": {
            "climate_adaptation_policy": {
                "type": "policy_approach",
                "definition": "Government policies addressing adjustment to climate effects"
            },
            "food_security_program": {
                "type": "government_program",
                "definition": "Initiatives ensuring adequate food access for populations"
            },
            "international_cooperation": {
                "type": "governance_mechanism",
                "definition": "Countries working together on shared challenges"
            },
            "regulatory_framework": {
                "type": "legal_structure",
                "definition": "System of regulations and guidelines"
            },
            "stakeholder_engagement": {
                "type": "process",
                "definition": "Involving affected parties in decision-making"
            }
        },
        "relations": [
            {"source": "climate_adaptation_policy", "target": "food_security_program", "type": "incorporates"},
            {"source": "international_cooperation", "target": "climate_adaptation_policy", "type": "strengthens"},
            {"source": "stakeholder_engagement", "target": "regulatory_framework", "type": "improves"}
        ],
        "facts": [
            {
                "content": "Only 5-10% of climate finance is dedicated to adaptation, despite growing needs.",
                "confidence": 0.9,
                "source": "UNFCCC"
            },
            {
                "content": "National Adaptation Plans have been developed by 20+ countries focusing on agricultural resilience.",
                "confidence": 0.95,
                "source": "UNFCCC"
            },
            {
                "content": "Effective climate-smart agricultural policies can increase both productivity and climate resilience by 15-25%.",
                "confidence": 0.8,
                "source": "IPCC"
            }
        ]
    }
    
    agent.initialize_knowledge_base(domain_knowledge)
    return agent


def main():
    """Main function to run the multi-agent collaboration example."""
    # Create the team
    team = AgentTeam("ClimateAgTeam")
    
    # Create and add specialized agents
    climate_expert = initialize_climate_expert("ClimateExpert")
    agriculture_expert = initialize_agriculture_expert("AgricultureExpert")
    economics_expert = initialize_economics_expert("EconomicsExpert")
    policy_expert = initialize_policy_expert("PolicyExpert")
    
    team.add_agent(climate_expert)
    team.add_agent(agriculture_expert)
    team.add_agent(economics_expert)
    team.add_agent(policy_expert)
    
    # Define the problem statement
    problem_statement = """
    Develop a comprehensive strategy to enhance agricultural resilience in the face of 
    climate change, focusing on regions most vulnerable to increased temperature and 
    altered precipitation patterns. The strategy should include technological adaptations, 
    policy frameworks, economic considerations, and implementation approaches that consider 
    local contexts and global food security implications.
    """
    
    # Solve the problem collaboratively
    solution = team.solve_problem(problem_statement)
    
    # Output the solution
    print("\n" + "="*80)
    print(f"TEAM SOLUTION: {team.name}")
    print("="*80)
    print(f"Problem: {problem_statement.strip()}")
    print("\nSOLUTION:")
    print(solution["solution"])
    print("\nEthical Assessment:")
    print(f"Verdict: {solution['ethical_assessment']['verdict']}")
    print(f"Score: {solution['ethical_assessment']['score']}")
    print(f"Considerations: {solution['ethical_assessment']['considerations']}")
    print("="*80)
    
    # Return the solution for potential further processing
    return solution


if __name__ == "__main__":
    main() 