#!/usr/bin/env python3
"""
Multi-Agent Synthetic Data Generation System

This module implements a sophisticated multi-agent approach for generating 
high-quality synthetic scam data. Based on research from:

- MetaSynth: Meta-Prompting-Driven Agentic Scaffolds (arXiv:2504.12563)
- AgentSGEN: Multi-Agent LLM in the Loop for Semantic Collaboration (arXiv:2505.13466)  
- ConvoGen: Multi-Agent Approach for Conversational Data (arXiv:2503.17460)

The system uses multiple specialized agents that collaborate through iterative
generation, evaluation, and refinement cycles to produce highly diverse and
realistic synthetic data.
"""

import asyncio
import logging
import json
import random
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import time
from datetime import datetime

from src.llm_core.api_call import make_api_call
from src.llm_core.api_provider import LLM
from src.synthesize.diversity_manager import DiversityManager, DiversityConfig
from src.synthesize.synthesis_prompts import SynthesisPromptsManager

logger = logging.getLogger(__name__)


class AgentRole(Enum):
    """Different agent roles in the multi-agent system."""
    GENERATOR = "generator"           # Generates initial content
    EVALUATOR = "evaluator"          # Evaluates content quality
    REFINER = "refiner"             # Refines and improves content
    COORDINATOR = "coordinator"      # Orchestrates collaboration
    SPECIALIST = "specialist"        # Domain-specific expertise


@dataclass
class AgentConfig:
    """Configuration for individual agents."""
    role: AgentRole
    name: str
    specialization: str
    system_prompt: str
    evaluation_criteria: List[str] = field(default_factory=list)
    confidence_threshold: float = 0.8
    max_iterations: int = 3


@dataclass
class GenerationResult:
    """Result from multi-agent generation process."""
    content: Dict[str, Any]
    quality_score: float
    agent_feedback: Dict[str, Any]
    iteration_count: int
    improvement_suggestions: List[str]
    final_assessment: str
    generation_metadata: Dict[str, Any]


class MultiAgentOrchestrator:
    """
    Orchestrates collaboration between multiple LLM agents for high-quality 
    synthetic data generation.
    
    This class implements the MetaSynth approach where a coordinator agent
    manages multiple specialized agents that collaborate to generate, evaluate,
    and refine synthetic data through iterative cycles.
    """
    
    def __init__(self, 
                 llm: LLM,
                 prompts_manager: SynthesisPromptsManager,
                 diversity_manager: DiversityManager,
                 enable_specialist_agents: bool = True,
                 enable_iterative_refinement: bool = True,
                 max_collaboration_rounds: int = 3):
        """
        Initialize the multi-agent orchestrator.
        
        Args:
            llm: Language model for agent communication
            prompts_manager: Base prompts manager
            diversity_manager: Diversity enhancement manager
            enable_specialist_agents: Whether to use domain specialists
            enable_iterative_refinement: Whether to enable refinement cycles
            max_collaboration_rounds: Maximum collaboration iterations
        """
        self.llm = llm
        self.prompts_manager = prompts_manager
        self.diversity_manager = diversity_manager
        self.enable_specialist_agents = enable_specialist_agents
        self.enable_iterative_refinement = enable_iterative_refinement
        self.max_collaboration_rounds = max_collaboration_rounds
        
        # Initialize agent configurations
        self.agents = self._initialize_agents()
        
        # Collaboration state
        self.collaboration_history: List[Dict] = []
        self.agent_performance: Dict[str, Dict] = {}
        
        logger.info(f"MultiAgentOrchestrator initialized with {len(self.agents)} agents")
    
    def _initialize_agents(self) -> Dict[str, AgentConfig]:
        """Initialize the multi-agent system configuration."""
        agents = {}
        
        # Generator Agent - Creates initial content
        agents["content_generator"] = AgentConfig(
            role=AgentRole.GENERATOR,
            name="Content Generator",
            specialization="Initial content creation with diversity techniques",
            system_prompt="""You are a Content Generator Agent specializing in creating realistic synthetic scam scenarios.

Your role:
1. Generate initial synthetic content using provided guidelines
2. Apply diversity techniques (personas, contexts, styles)
3. Ensure content follows the specified category requirements
4. Create realistic, believable scenarios that vary from standard patterns

Focus on:
- Authenticity and realism
- Diversity in approach and content
- Adherence to category specifications
- Natural language patterns
- Appropriate complexity for the scenario type

Generate content that will serve as a strong foundation for further refinement.""",
            evaluation_criteria=["realism", "diversity", "category_adherence", "language_quality"]
        )
        
        # Evaluator Agent - Assesses quality and realism
        agents["quality_evaluator"] = AgentConfig(
            role=AgentRole.EVALUATOR,
            name="Quality Evaluator",
            specialization="Content quality assessment and scoring",
            system_prompt="""You are a Quality Evaluator Agent specializing in assessing synthetic scam data quality.

Your role:
1. Evaluate generated content for realism, authenticity, and quality
2. Score content on multiple dimensions (1-10 scale)
3. Identify specific strengths and weaknesses
4. Provide actionable feedback for improvement
5. Ensure content meets training data standards

Evaluation criteria:
- Realism (1-10): How believable is this scenario?
- Authenticity (1-10): Does this feel like real scam/legitimate content?
- Diversity (1-10): How well does this vary from typical patterns?
- Language Quality (1-10): Natural, appropriate language use?
- Training Value (1-10): How useful for ML model training?

Provide specific, actionable feedback for improvement.""",
            evaluation_criteria=["realism", "authenticity", "diversity", "language_quality", "training_value"],
            confidence_threshold=7.5
        )
        
        # Refiner Agent - Improves content based on feedback
        agents["content_refiner"] = AgentConfig(
            role=AgentRole.REFINER,
            name="Content Refiner",
            specialization="Content improvement and polish",
            system_prompt="""You are a Content Refiner Agent specializing in improving synthetic data quality.

Your role:
1. Take initial content and evaluator feedback
2. Apply specific improvements while maintaining core scenario
3. Enhance realism, authenticity, and training value
4. Preserve the essential scam/legitimate characteristics
5. Ensure improvements don't compromise diversity

Refinement techniques:
- Enhance dialogue naturalness
- Improve character consistency
- Add realistic details and context
- Strengthen persuasion/deception techniques (for scams)
- Improve professional authenticity (for legitimate content)

Maintain the original intent while making content more realistic and effective.""",
            evaluation_criteria=["improvement_quality", "authenticity_preservation", "training_effectiveness"]
        )
        
        # Specialist Agents - Domain expertise
        if self.enable_specialist_agents:
            agents["scam_specialist"] = AgentConfig(
                role=AgentRole.SPECIALIST,
                name="Scam Tactics Specialist",
                specialization="Social engineering and scam technique expertise",
                system_prompt="""You are a Scam Tactics Specialist with deep knowledge of social engineering techniques.

Your expertise:
1. Social engineering psychology and tactics
2. Common scam patterns and variations
3. Persuasion and manipulation techniques
4. Authority, urgency, and fear-based approaches
5. Technical scam methodologies

Your role:
1. Review generated scam content for tactical accuracy
2. Suggest improvements in persuasion techniques
3. Ensure scam scenarios are realistic and educationally valuable
4. Identify missing psychological manipulation elements
5. Validate that scam progression feels authentic

Focus on educational value while maintaining ethical boundaries.""",
                evaluation_criteria=["tactical_accuracy", "psychological_realism", "educational_value"]
            )
            
            agents["authenticity_specialist"] = AgentConfig(
                role=AgentRole.SPECIALIST,
                name="Authenticity Specialist", 
                specialization="Real-world accuracy and believability",
                system_prompt="""You are an Authenticity Specialist focused on real-world accuracy and believability.

Your expertise:
1. Real-world communication patterns
2. Institutional procedures and language
3. Demographic and cultural authenticity
4. Technical accuracy in described processes
5. Situational realism and context

Your role:
1. Verify content matches real-world patterns
2. Check for anachronisms or unrealistic elements
3. Ensure demographic and cultural accuracy
4. Validate technical claims and procedures
5. Improve overall believability

Make suggestions that enhance realism without compromising the core scenario.""",
                evaluation_criteria=["factual_accuracy", "cultural_authenticity", "procedural_realism"]
            )
        
        # Coordinator Agent - Orchestrates collaboration
        agents["coordinator"] = AgentConfig(
            role=AgentRole.COORDINATOR,
            name="Collaboration Coordinator",
            specialization="Multi-agent collaboration management",
            system_prompt="""You are a Collaboration Coordinator managing multi-agent synthetic data generation.

Your role:
1. Orchestrate collaboration between generator, evaluator, and refiner agents
2. Synthesize feedback from multiple agents into actionable improvements
3. Determine when content has reached sufficient quality
4. Balance different agent perspectives and priorities
5. Ensure efficient collaboration without excessive iterations

Decision criteria:
- Content quality scores from evaluators
- Consistency of feedback across agents
- Diminishing returns from additional iterations
- Achievement of quality thresholds
- Overall training data value

Make final decisions on content acceptance and suggest meta-improvements to the process.""",
            evaluation_criteria=["collaboration_efficiency", "quality_convergence", "meta_assessment"],
            confidence_threshold=8.0
        )
        
        return agents
    
    async def generate_with_multi_agent_collaboration(self,
                                                    synthesis_type: str,
                                                    category: str,
                                                    diversity_config: DiversityConfig,
                                                    target_schema: Any) -> GenerationResult:
        """
        Generate synthetic content using multi-agent collaboration.
        
        This implements the core MetaSynth approach where multiple agents
        collaborate through iterative generation, evaluation, and refinement.
        
        Args:
            synthesis_type: Type of content to generate
            category: Category within the synthesis type
            diversity_config: Diversity enhancement configuration
            target_schema: Expected output schema
            
        Returns:
            GenerationResult with enhanced content and metadata
        """
        start_time = time.time()
        collaboration_round = 0
        
        logger.info(f"Starting multi-agent collaboration for {synthesis_type}.{category}")
        
        # Phase 1: Initial Generation
        initial_content = await self._generate_initial_content(
            synthesis_type, category, diversity_config, target_schema
        )
        
        current_content = initial_content
        agent_feedback = {}
        improvement_suggestions = []
        
        # Phase 2: Iterative Collaboration
        while collaboration_round < self.max_collaboration_rounds:
            collaboration_round += 1
            logger.debug(f"Collaboration round {collaboration_round}")
            
            # Evaluate current content
            evaluation_result = await self._evaluate_content(
                current_content, synthesis_type, category
            )
            
            agent_feedback[f"round_{collaboration_round}"] = evaluation_result
            
            # Check if quality threshold met
            if evaluation_result["overall_score"] >= self.agents["quality_evaluator"].confidence_threshold:
                logger.info(f"Quality threshold met in round {collaboration_round}")
                break
            
            # Get improvement suggestions from specialists
            if self.enable_specialist_agents:
                specialist_feedback = await self._get_specialist_feedback(
                    current_content, synthesis_type, category
                )
                evaluation_result.update(specialist_feedback)
            
            # Refine content if enabled
            if self.enable_iterative_refinement:
                refined_content = await self._refine_content(
                    current_content, evaluation_result, synthesis_type, category, target_schema
                )
                
                if refined_content:
                    current_content = refined_content
                    improvement_suggestions.extend(evaluation_result.get("suggestions", []))
                else:
                    logger.warning(f"Content refinement failed in round {collaboration_round}")
                    break
            else:
                break
        
        # Phase 3: Final Coordination Assessment
        final_assessment = await self._coordinate_final_assessment(
            current_content, agent_feedback, synthesis_type, category
        )
        
        # Calculate final quality score
        final_quality_score = self._calculate_final_quality_score(agent_feedback)
        
        generation_time = time.time() - start_time
        
        result = GenerationResult(
            content=current_content,
            quality_score=final_quality_score,
            agent_feedback=agent_feedback,
            iteration_count=collaboration_round,
            improvement_suggestions=improvement_suggestions,
            final_assessment=final_assessment,
            generation_metadata={
                "generation_time": generation_time,
                "collaboration_rounds": collaboration_round,
                "agents_used": list(self.agents.keys()),
                "diversity_techniques_applied": diversity_config.__dict__,
                "synthesis_type": synthesis_type,
                "category": category
            }
        )
        
        # Update collaboration history
        self.collaboration_history.append({
            "timestamp": datetime.now().isoformat(),
            "result": result,
            "metadata": result.generation_metadata
        })
        
        logger.info(f"Multi-agent collaboration completed: {collaboration_round} rounds, "
                   f"quality score: {final_quality_score:.2f}")
        
        return result
    
    async def _generate_initial_content(self,
                                      synthesis_type: str,
                                      category: str,
                                      diversity_config: DiversityConfig,
                                      target_schema: Any) -> Dict[str, Any]:
        """Generate initial content using the content generator agent."""
        # Create enhanced prompt using diversity manager
        base_prompt = self.diversity_manager.create_diverse_generation_prompt(
            synthesis_type, category, diversity_config
        )
        
        # Add agent-specific instructions
        agent_prompt = f"""{base_prompt}

=== CONTENT GENERATOR AGENT INSTRUCTIONS ===
You are the Content Generator in a multi-agent collaboration system.
Your generated content will be evaluated and potentially refined by other agents.

Focus on creating a strong foundation that:
1. Captures the essence of the {category} scenario
2. Includes diverse, realistic elements  
3. Provides clear structure for potential refinement
4. Maintains authenticity and believability

This is the initial generation - aim for good quality that can be improved through collaboration."""
        
        try:
            response = await make_api_call(
                self.llm,
                self.agents["content_generator"].system_prompt,
                agent_prompt,
                response_schema=target_schema
            )
            
            return response.dict() if hasattr(response, 'dict') else response
            
        except Exception as e:
            logger.error(f"Initial content generation failed: {e}")
            # Fallback to diversity manager only
            fallback_prompt = self.diversity_manager.create_diverse_generation_prompt(
                synthesis_type, category, diversity_config
            )
            
            response = await make_api_call(
                self.llm,
                self.prompts_manager.get_system_prompt(synthesis_type),
                fallback_prompt,
                response_schema=target_schema
            )
            
            return response.dict() if hasattr(response, 'dict') else response
    
    async def _evaluate_content(self,
                              content: Dict[str, Any],
                              synthesis_type: str,
                              category: str) -> Dict[str, Any]:
        """Evaluate content quality using the evaluator agent."""
        evaluator_prompt = f"""Evaluate the following {synthesis_type} content for category '{category}':

CONTENT TO EVALUATE:
{json.dumps(content, indent=2)}

Provide detailed evaluation scores (1-10) and specific feedback for:
1. Realism - How believable is this scenario?
2. Authenticity - Does this feel like real content?
3. Diversity - How well does this vary from typical patterns?
4. Language Quality - Natural, appropriate language use?
5. Training Value - How useful for ML model training?

Format your response as JSON:
{{
    "realism": <score>,
    "authenticity": <score>, 
    "diversity": <score>,
    "language_quality": <score>,
    "training_value": <score>,
    "overall_score": <average>,
    "strengths": ["strength1", "strength2"],
    "weaknesses": ["weakness1", "weakness2"],
    "suggestions": ["suggestion1", "suggestion2"]
}}
"""
        
        try:
            response = await make_api_call(
                self.llm,
                self.agents["quality_evaluator"].system_prompt,
                evaluator_prompt
            )
            
            # Parse JSON response
            if isinstance(response, str):
                try:
                    data = json.loads(response)
                    if isinstance(data, dict):
                        return data
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse JSON from evaluator: {response}")
            elif hasattr(response, 'dict'):
                return response.dict()
            elif isinstance(response, dict):
                return response

            logger.warning(f"Content evaluation returned non-dict format. Response: {response}")
            # Return basic evaluation
            return {
                "realism": 6.0,
                "authenticity": 6.0,
                "diversity": 6.0,
                "language_quality": 6.0,
                "training_value": 6.0,
                "overall_score": 6.0,
                "strengths": ["Generated successfully"],
                "weaknesses": ["Evaluation failed"],
                "suggestions": ["Manual review recommended"]
            }
                
        except Exception as e:
            logger.error(f"Content evaluation failed: {e}")
            # Return basic evaluation
            return {
                "realism": 6.0,
                "authenticity": 6.0,
                "diversity": 6.0,
                "language_quality": 6.0,
                "training_value": 6.0,
                "overall_score": 6.0,
                "strengths": ["Generated successfully"],
                "weaknesses": ["Evaluation failed"],
                "suggestions": ["Manual review recommended"]
            }
    
    async def _get_specialist_feedback(self,
                                     content: Dict[str, Any],
                                     synthesis_type: str,
                                     category: str) -> Dict[str, Any]:
        """Get feedback from specialist agents."""
        specialist_feedback = {}
        
        # Determine which specialists to consult
        specialists_to_use = []
        if "scam" in category.lower() or "fraud" in category.lower():
            specialists_to_use.append("scam_specialist")
        specialists_to_use.append("authenticity_specialist")
        
        for specialist_name in specialists_to_use:
            if specialist_name in self.agents:
                try:
                    feedback = await self._get_single_specialist_feedback(
                        content, specialist_name, synthesis_type, category
                    )
                    specialist_feedback[specialist_name] = feedback
                except Exception as e:
                    logger.error(f"Specialist {specialist_name} feedback failed: {e}")
                    specialist_feedback[specialist_name] = {"error": str(e)}
        
        return {"specialist_feedback": specialist_feedback}
    
    async def _get_single_specialist_feedback(self,
                                            content: Dict[str, Any],
                                            specialist_name: str,
                                            synthesis_type: str,
                                            category: str) -> Dict[str, Any]:
        """Get feedback from a single specialist agent."""
        specialist = self.agents[specialist_name]
        
        specialist_prompt = f"""Review the following {synthesis_type} content for category '{category}' from your specialized perspective:

CONTENT TO REVIEW:
{json.dumps(content, indent=2)}

Provide expert feedback focusing on your specialization: {specialist.specialization}

Evaluate and provide specific suggestions for improvement in your domain of expertise.
Format as JSON with scores (1-10) and specific recommendations."""
        
        response = await make_api_call(
            self.llm,
            specialist.system_prompt,
            specialist_prompt
        )
        
        if isinstance(response, str):
            try:
                return json.loads(response)
            except:
                return {"feedback": response}
        else:
            return response.dict() if hasattr(response, 'dict') else response
    
    async def _refine_content(self,
                            content: Dict[str, Any],
                            evaluation: Dict[str, Any],
                            synthesis_type: str,
                            category: str,
                            target_schema: Any) -> Optional[Dict[str, Any]]:
        """Refine content based on evaluation feedback."""
        refiner_prompt = f"""Refine the following {synthesis_type} content based on the evaluation feedback:

ORIGINAL CONTENT:
{json.dumps(content, indent=2)}

EVALUATION FEEDBACK:
{json.dumps(evaluation, indent=2)}

Improve the content by addressing the identified weaknesses and suggestions while:
1. Maintaining the core scenario and intent
2. Preserving authenticity and realism
3. Enhancing training value
4. Keeping diversity elements intact

Generate improved content that addresses the feedback while maintaining quality."""
        
        try:
            response = await make_api_call(
                self.llm,
                self.agents["content_refiner"].system_prompt,
                refiner_prompt,
                response_schema=target_schema
            )
            
            return response.dict() if hasattr(response, 'dict') else response
            
        except Exception as e:
            logger.error(f"Content refinement failed: {e}")
            return None
    
    async def _coordinate_final_assessment(self,
                                         content: Dict[str, Any],
                                         agent_feedback: Dict[str, Any],
                                         synthesis_type: str,
                                         category: str) -> str:
        """Coordinate final assessment using the coordinator agent."""
        coordinator_prompt = f"""Provide final assessment of multi-agent collaboration results:

FINAL CONTENT:
{json.dumps(content, indent=2)}

AGENT FEEDBACK HISTORY:
{json.dumps(agent_feedback, indent=2)}

As the Collaboration Coordinator, provide:
1. Overall assessment of the content quality
2. Evaluation of the collaboration process effectiveness
3. Final recommendations for content use
4. Meta-insights about the generation process

Synthesis type: {synthesis_type}
Category: {category}"""
        
        try:
            response = await make_api_call(
                self.llm,
                self.agents["coordinator"].system_prompt,
                coordinator_prompt
            )
            
            return response if isinstance(response, str) else str(response)
            
        except Exception as e:
            logger.error(f"Final coordination failed: {e}")
            return f"Multi-agent generation completed for {synthesis_type}.{category}"
    
    def _calculate_final_quality_score(self, agent_feedback: Dict[str, Any]) -> float:
        """Calculate final quality score from all agent feedback."""
        scores = []
        
        for round_feedback in agent_feedback.values():
            if isinstance(round_feedback, dict) and "overall_score" in round_feedback:
                scores.append(round_feedback["overall_score"])
        
        if scores:
            # Weight later scores more heavily (improvement over time)
            weights = [i + 1 for i in range(len(scores))]
            weighted_average = sum(score * weight for score, weight in zip(scores, weights)) / sum(weights)
            return min(10.0, weighted_average)
        
        return 6.0  # Default reasonable score
    
    def get_collaboration_stats(self) -> Dict[str, Any]:
        """Get statistics about multi-agent collaboration performance."""
        if not self.collaboration_history:
            return {"status": "No collaborations yet"}
        
        recent_scores = [h["result"].quality_score for h in self.collaboration_history[-10:]]
        recent_rounds = [h["result"].iteration_count for h in self.collaboration_history[-10:]]
        recent_times = [h["result"].generation_metadata["generation_time"] for h in self.collaboration_history[-10:]]
        
        return {
            "total_collaborations": len(self.collaboration_history),
            "average_quality_score": sum(recent_scores) / len(recent_scores),
            "average_collaboration_rounds": sum(recent_rounds) / len(recent_rounds),
            "average_generation_time": sum(recent_times) / len(recent_times),
            "agents_configured": len(self.agents),
            "specialist_agents_enabled": self.enable_specialist_agents,
            "iterative_refinement_enabled": self.enable_iterative_refinement
        }


class MultiAgentSynthesisGenerator:
    """
    Enhanced SynthesisGenerator with multi-agent collaboration capabilities.
    
    This class extends the existing synthesis generation with multi-agent
    orchestration for higher quality synthetic data generation.
    """
    
    def __init__(self,
                 base_generator,
                 enable_multi_agent: bool = False,
                 multi_agent_threshold: float = 7.0,
                 specialist_agents: bool = True,
                 iterative_refinement: bool = True):
        """
        Initialize multi-agent synthesis generator.
        
        Args:
            base_generator: Base SynthesisGenerator instance
            enable_multi_agent: Whether to enable multi-agent collaboration
            multi_agent_threshold: Quality threshold for using multi-agent
            specialist_agents: Whether to use specialist agents
            iterative_refinement: Whether to enable iterative refinement
        """
        self.base_generator = base_generator
        self.enable_multi_agent = enable_multi_agent
        self.multi_agent_threshold = multi_agent_threshold
        
        # Initialize multi-agent orchestrator if enabled
        self.orchestrator = None
        if self.enable_multi_agent and hasattr(base_generator, 'llm') and base_generator.llm:
            try:
                self.orchestrator = MultiAgentOrchestrator(
                    llm=base_generator.llm,
                    prompts_manager=base_generator.prompts_manager,
                    diversity_manager=base_generator.diversity_manager,
                    enable_specialist_agents=specialist_agents,
                    enable_iterative_refinement=iterative_refinement
                )
                logger.info("Multi-agent orchestrator initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize multi-agent orchestrator: {e}")
                self.enable_multi_agent = False
    
    async def generate_single_item_with_agents(self, category: str) -> Dict[str, Any]:
        """
        Generate a single item using multi-agent collaboration if enabled.
        
        Args:
            category: Category identifier
            
        Returns:
            Generation result with enhanced metadata
        """
        if not self.enable_multi_agent or not self.orchestrator:
            # Fall back to base generation
            return await self.base_generator.generate_single_item(category)
        
        try:
            # Create diversity configuration
            diversity_config = self.base_generator._create_diversity_config()
            
            # Use multi-agent collaboration
            result = await self.orchestrator.generate_with_multi_agent_collaboration(
                synthesis_type=self.base_generator.synthesis_type,
                category=category,
                diversity_config=diversity_config,
                target_schema=self.base_generator.response_model
            )
            
            # Convert to base generator format
            return {
                'success': True,
                'response': self._convert_to_response_model(result.content),
                'category': category,
                'synthesis_type': self.base_generator.synthesis_type,
                'multi_agent_metadata': {
                    'quality_score': result.quality_score,
                    'iteration_count': result.iteration_count,
                    'agent_feedback': result.agent_feedback,
                    'final_assessment': result.final_assessment,
                    'generation_metadata': result.generation_metadata
                }
            }
            
        except Exception as e:
            logger.error(f"Multi-agent generation failed: {e}")
            # Fall back to base generation
            return await self.base_generator.generate_single_item(category)
    
    def _convert_to_response_model(self, content: Dict[str, Any]):
        """Convert multi-agent content to response model format."""
        try:
            # Create instance of response model with the content
            return self.base_generator.response_model(**content)
        except Exception as e:
            logger.error(f"Failed to convert to response model: {e}")
            # Return content as-is if conversion fails
            return content
    
    def get_multi_agent_stats(self) -> Optional[Dict[str, Any]]:
        """Get multi-agent collaboration statistics."""
        if self.orchestrator:
            return self.orchestrator.get_collaboration_stats()
        return None