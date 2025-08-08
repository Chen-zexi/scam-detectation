"""Token usage tracking and aggregation for LLM API calls."""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class TokenUsageRecord:
    """Single token usage record."""
    timestamp: datetime
    model: str
    operation: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    reasoning_tokens: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class TokenUsageTracker:
    """Tracks and aggregates token usage across multiple API calls."""
    
    def __init__(self, verbose: bool = False):
        """Initialize the token usage tracker.
        
        Args:
            verbose: If True, log token usage to console. If False, silent tracking.
        """
        self.records: List[TokenUsageRecord] = []
        self.model_totals: Dict[str, Dict[str, int]] = {}
        self.operation_totals: Dict[str, Dict[str, int]] = {}
        self.session_start = datetime.now()
        self.verbose = verbose
    
    def add_usage(
        self,
        token_info: Dict[str, Any],
        model: str = "unknown",
        operation: str = "api_call",
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add a token usage record.
        
        Args:
            token_info: Dictionary with token counts
            model: Model name
            operation: Operation description
            metadata: Additional metadata
        """
        if not token_info:
            return
        
        record = TokenUsageRecord(
            timestamp=datetime.now(),
            model=model,
            operation=operation,
            input_tokens=token_info.get('input_tokens', 0),
            output_tokens=token_info.get('output_tokens', 0),
            total_tokens=token_info.get('total_tokens', 0),
            reasoning_tokens=token_info.get('reasoning_tokens'),
            metadata=metadata or {}
        )
        
        self.records.append(record)
        self._update_totals(record)
        
        # Only log if verbose mode is enabled
        if self.verbose:
            log_msg = f"Token Usage [{model}] - {operation}: "
            log_msg += f"Input={record.input_tokens}, Output={record.output_tokens}, Total={record.total_tokens}"
            if record.reasoning_tokens is not None:
                log_msg += f", Reasoning={record.reasoning_tokens}"
            logger.info(log_msg)
    
    def _update_totals(self, record: TokenUsageRecord) -> None:
        """Update running totals."""
        # Update model totals
        if record.model not in self.model_totals:
            self.model_totals[record.model] = {
                'input_tokens': 0,
                'output_tokens': 0,
                'total_tokens': 0,
                'reasoning_tokens': 0,
                'call_count': 0
            }
        
        self.model_totals[record.model]['input_tokens'] += record.input_tokens
        self.model_totals[record.model]['output_tokens'] += record.output_tokens
        self.model_totals[record.model]['total_tokens'] += record.total_tokens
        if record.reasoning_tokens:
            self.model_totals[record.model]['reasoning_tokens'] += record.reasoning_tokens
        self.model_totals[record.model]['call_count'] += 1
        
        # Update operation totals
        if record.operation not in self.operation_totals:
            self.operation_totals[record.operation] = {
                'input_tokens': 0,
                'output_tokens': 0,
                'total_tokens': 0,
                'reasoning_tokens': 0,
                'call_count': 0
            }
        
        self.operation_totals[record.operation]['input_tokens'] += record.input_tokens
        self.operation_totals[record.operation]['output_tokens'] += record.output_tokens
        self.operation_totals[record.operation]['total_tokens'] += record.total_tokens
        if record.reasoning_tokens:
            self.operation_totals[record.operation]['reasoning_tokens'] += record.reasoning_tokens
        self.operation_totals[record.operation]['call_count'] += 1
    
    def get_summary(self, include_details: bool = False) -> Dict[str, Any]:
        """Get a summary of token usage.
        
        Args:
            include_details: Whether to include detailed breakdown
        
        Returns:
            Dictionary with usage summary
        """
        total_input = sum(r.input_tokens for r in self.records)
        total_output = sum(r.output_tokens for r in self.records)
        total_tokens = sum(r.total_tokens for r in self.records)
        total_reasoning = sum(r.reasoning_tokens for r in self.records if r.reasoning_tokens)
        
        session_duration = (datetime.now() - self.session_start).total_seconds()
        
        summary = {
            'session_duration_seconds': session_duration,
            'total_calls': len(self.records),
            'total_input_tokens': total_input,
            'total_output_tokens': total_output,
            'total_tokens': total_tokens,
            'total_reasoning_tokens': total_reasoning,
            'average_tokens_per_call': total_tokens / len(self.records) if self.records else 0
        }
        
        # Only include detailed breakdowns if requested
        if include_details:
            summary['by_model'] = self.model_totals
            summary['by_operation'] = self.operation_totals
            
        return summary
    
    def print_summary(self) -> None:
        """Print a formatted summary of token usage."""
        summary = self.get_summary()
        
        print("\n" + "="*60)
        print("TOKEN USAGE SUMMARY")
        print("="*60)
        
        print(f"\nSession Duration: {summary['session_duration_seconds']:.1f} seconds")
        print(f"Total API Calls: {summary['total_calls']}")
        
        # Show totals and averages
        if summary['total_calls'] > 0:
            avg_input = summary['total_input_tokens'] / summary['total_calls']
            avg_output = summary['total_output_tokens'] / summary['total_calls']
            avg_total = summary['total_tokens'] / summary['total_calls']
            
            print(f"\nTotal Token Usage:")
            print(f"  Total Input:      {summary['total_input_tokens']:,}")
            print(f"  Total Output:     {summary['total_output_tokens']:,}")
            print(f"  Total Combined:   {summary['total_tokens']:,}")
            if summary['total_reasoning_tokens'] > 0:
                print(f"  Total Reasoning:  {summary['total_reasoning_tokens']:,}")
            
            print(f"\nAverage per Call:")
            print(f"  Avg Input:        {avg_input:.0f}")
            print(f"  Avg Output:       {avg_output:.0f}")
            print(f"  Avg Total:        {avg_total:.0f}")
            if summary['total_reasoning_tokens'] > 0:
                avg_reasoning = summary['total_reasoning_tokens'] / summary['total_calls']
                print(f"  Avg Reasoning:    {avg_reasoning:.0f}")
        
        print("\n" + "="*60)
    
    def export_to_json(self, filepath: str) -> None:
        """Export token usage data to JSON file.
        
        Args:
            filepath: Path to save the JSON file
        """
        data = {
            'summary': self.get_summary(),
            'records': [
                {
                    'timestamp': r.timestamp.isoformat(),
                    'model': r.model,
                    'operation': r.operation,
                    'input_tokens': r.input_tokens,
                    'output_tokens': r.output_tokens,
                    'total_tokens': r.total_tokens,
                    'reasoning_tokens': r.reasoning_tokens,
                    'metadata': r.metadata
                }
                for r in self.records
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Token usage data exported to {filepath}")
    
    def estimate_cost(self, pricing: Optional[Dict[str, Dict[str, float]]] = None, use_config: bool = True) -> Dict[str, float]:
        """Estimate cost based on token usage.
        
        Args:
            pricing: Optional pricing dictionary. If not provided, uses pricing from config or defaults.
                    Format: {model: {'input': price_per_1m, 'output': price_per_1m}}
            use_config: If True, tries to load pricing from model config first
        
        Returns:
            Dictionary with cost estimates by model and total
        """
        if pricing is None and use_config:
            # Try to load pricing from model config
            pricing = {}
            try:
                from .api_provider import ModelConfig
                config = ModelConfig()
                
                for model in self.model_totals.keys():
                    # Try to find the model in config
                    for provider in ['openai', 'anthropic', 'gemini']:
                        model_info = config.get_model_info(provider, model)
                        if model_info and 'pricing' in model_info:
                            price_info = model_info['pricing']
                            # Convert from per 1M tokens to per 1K tokens for backward compatibility
                            pricing[model] = {
                                'input': price_info['input'] / 1000,
                                'output': price_info['output'] / 1000
                            }
                            break
            except Exception:
                pass
        
        if pricing is None or not pricing:
            # Fallback to default pricing - updated to match image (per 1K tokens)
            pricing = {
                'gpt-5': {'input': 0.00125, 'output': 0.01},
                'gpt-5-mini': {'input': 0.00025, 'output': 0.002},
                'gpt-5-nano': {'input': 0.00005, 'output': 0.0004},
                'gpt-4.1': {'input': 0.002, 'output': 0.008},
                'gpt-4.1-mini': {'input': 0.0004, 'output': 0.0016},
                'gpt-4.1-nano': {'input': 0.0001, 'output': 0.0004},
            }
        
        total_input_cost = 0.0
        total_output_cost = 0.0
        
        for model, stats in self.model_totals.items():
            if model in pricing:
                input_cost = (stats['input_tokens'] / 1000) * pricing[model]['input']
                output_cost = (stats['output_tokens'] / 1000) * pricing[model]['output']
                total_input_cost += input_cost
                total_output_cost += output_cost
        
        total_cost = total_input_cost + total_output_cost
        
        # Return simplified cost structure
        return {
            'total_cost': total_cost,
            'input_cost': total_input_cost,
            'output_cost': total_output_cost
        }
    
    def print_cost_estimate(self, pricing: Optional[Dict[str, Dict[str, float]]] = None) -> None:
        """Print estimated costs.
        
        Args:
            pricing: Optional pricing dictionary
        """
        costs = self.estimate_cost(pricing)
        
        if costs and costs['total_cost'] > 0:
            print("\n" + "="*60)
            print("ESTIMATED COST")
            print("="*60)
            
            # Calculate average cost per call
            total_calls = len(self.records)
            avg_cost = costs['total_cost'] / total_calls if total_calls > 0 else 0
            
            print(f"\nTotal Cost:     ${costs['total_cost']:.4f}")
            print(f"Average/Call:   ${avg_cost:.4f}")
            print(f"Breakdown:      ${costs['input_cost']:.4f} (input) + ${costs['output_cost']:.4f} (output)")
            
            print("="*60)


# Global token tracker instance (optional, for convenience)
_global_tracker: Optional[TokenUsageTracker] = None


def get_global_tracker() -> TokenUsageTracker:
    """Get or create the global token tracker instance."""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = TokenUsageTracker()
    return _global_tracker


def reset_global_tracker() -> None:
    """Reset the global token tracker."""
    global _global_tracker
    _global_tracker = TokenUsageTracker()