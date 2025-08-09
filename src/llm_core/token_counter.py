"""Token usage tracking and aggregation for LLM API calls."""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class TokenUsageRecord:
    """Single token usage record with comprehensive token tracking."""
    timestamp: datetime
    model: str
    operation: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    reasoning_tokens: Optional[int] = None
    cached_tokens: Optional[int] = None
    accepted_prediction_tokens: Optional[int] = None
    rejected_prediction_tokens: Optional[int] = None
    audio_input_tokens: Optional[int] = None
    audio_output_tokens: Optional[int] = None
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
            cached_tokens=token_info.get('cached_tokens'),
            accepted_prediction_tokens=token_info.get('accepted_prediction_tokens'),
            rejected_prediction_tokens=token_info.get('rejected_prediction_tokens'),
            audio_input_tokens=token_info.get('audio_input_tokens'),
            audio_output_tokens=token_info.get('audio_output_tokens'),
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
                'cached_tokens': 0,
                'call_count': 0
            }
        
        self.model_totals[record.model]['input_tokens'] += record.input_tokens
        self.model_totals[record.model]['output_tokens'] += record.output_tokens
        self.model_totals[record.model]['total_tokens'] += record.total_tokens
        if record.reasoning_tokens:
            self.model_totals[record.model]['reasoning_tokens'] += record.reasoning_tokens
        if record.cached_tokens:
            self.model_totals[record.model]['cached_tokens'] += record.cached_tokens
        self.model_totals[record.model]['call_count'] += 1
        
        # Update operation totals
        if record.operation not in self.operation_totals:
            self.operation_totals[record.operation] = {
                'input_tokens': 0,
                'output_tokens': 0,
                'total_tokens': 0,
                'reasoning_tokens': 0,
                'cached_tokens': 0,
                'call_count': 0
            }
        
        self.operation_totals[record.operation]['input_tokens'] += record.input_tokens
        self.operation_totals[record.operation]['output_tokens'] += record.output_tokens
        self.operation_totals[record.operation]['total_tokens'] += record.total_tokens
        if record.reasoning_tokens:
            self.operation_totals[record.operation]['reasoning_tokens'] += record.reasoning_tokens
        if record.cached_tokens:
            self.operation_totals[record.operation]['cached_tokens'] += record.cached_tokens
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
        total_cached = sum(r.cached_tokens for r in self.records if r.cached_tokens)
        total_accepted_pred = sum(r.accepted_prediction_tokens for r in self.records if r.accepted_prediction_tokens)
        total_rejected_pred = sum(r.rejected_prediction_tokens for r in self.records if r.rejected_prediction_tokens)
        
        session_duration = (datetime.now() - self.session_start).total_seconds()
        
        summary = {
            'session_duration_seconds': session_duration,
            'total_calls': len(self.records),
            'total_input_tokens': total_input,
            'total_output_tokens': total_output,
            'total_tokens': total_tokens,
            'total_reasoning_tokens': total_reasoning,
            'total_cached_tokens': total_cached,
            'average_tokens_per_call': total_tokens / len(self.records) if self.records else 0
        }
        
        # Add prediction tokens if any were used
        if total_accepted_pred > 0:
            summary['total_accepted_prediction_tokens'] = total_accepted_pred
        if total_rejected_pred > 0:
            summary['total_rejected_prediction_tokens'] = total_rejected_pred
        
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
            if summary.get('total_cached_tokens', 0) > 0:
                print(f"    - Cached:       {summary['total_cached_tokens']:,}")
                print(f"    - Regular:      {summary['total_input_tokens'] - summary['total_cached_tokens']:,}")
            else:
                # Always show cached status even if 0
                print(f"    - Cached:       0")
            print(f"  Total Output:     {summary['total_output_tokens']:,}")
            print(f"  Total Combined:   {summary['total_tokens']:,}")
            if summary.get('total_reasoning_tokens', 0) > 0:
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
        total_cached_cost = 0.0
        total_output_cost = 0.0
        
        for model, stats in self.model_totals.items():
            if model in pricing:
                # Calculate regular input cost (non-cached tokens)
                cached_tokens = stats.get('cached_tokens', 0)
                regular_input_tokens = stats['input_tokens'] - cached_tokens
                
                # Regular input cost
                input_cost = (regular_input_tokens / 1000) * pricing[model]['input']
                
                # Cached input cost (if pricing available)
                cached_input_cost = 0.0
                if cached_tokens > 0:
                    # Try to get cached pricing from config
                    try:
                        from .api_provider import ModelConfig
                        config = ModelConfig()
                        model_info = None
                        for provider in ['openai', 'anthropic', 'gemini']:
                            model_info = config.get_model_info(provider, model)
                            if model_info and 'pricing' in model_info:
                                break
                        
                        if model_info and 'cached_input' in model_info['pricing']:
                            # Use cached_input pricing from config (per 1M tokens)
                            cached_rate = model_info['pricing']['cached_input'] / 1000  # Convert to per 1K
                            cached_input_cost = (cached_tokens / 1000) * cached_rate
                        else:
                            # Fallback: assume cached is 25% of regular price
                            cached_input_cost = (cached_tokens / 1000) * (pricing[model]['input'] * 0.25)
                    except:
                        # Fallback if config not available
                        cached_input_cost = (cached_tokens / 1000) * (pricing[model]['input'] * 0.25)
                
                # Output cost
                output_cost = (stats['output_tokens'] / 1000) * pricing[model]['output']
                
                total_input_cost += input_cost
                total_cached_cost += cached_input_cost
                total_output_cost += output_cost
        
        total_cost = total_input_cost + total_cached_cost + total_output_cost
        
        # Return detailed cost structure
        result = {
            'total_cost': total_cost,
            'input_cost': total_input_cost,
            'output_cost': total_output_cost
        }
        
        # Add cached cost if any
        if total_cached_cost > 0:
            result['cached_cost'] = total_cached_cost
            # Calculate savings: what would have been paid without caching
            total_cached_tokens = sum(self.model_totals[m].get('cached_tokens', 0) for m in self.model_totals)
            if total_cached_tokens > 0:
                # Calculate what the cached tokens would have cost at full price
                full_price_cost = 0
                for model, stats in self.model_totals.items():
                    if model in pricing and stats.get('cached_tokens', 0) > 0:
                        full_price_cost += (stats['cached_tokens'] / 1000) * pricing[model]['input']
                result['cache_savings'] = full_price_cost - total_cached_cost
        
        return result
    
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
            
            # Build breakdown with cached cost if present
            breakdown = f"Breakdown:      ${costs['input_cost']:.4f} (input)"
            if costs.get('cached_cost', 0) > 0:
                breakdown += f" + ${costs['cached_cost']:.4f} (cached)"
            breakdown += f" + ${costs['output_cost']:.4f} (output)"
            print(breakdown)
            
            # Show cache savings if any
            if costs.get('cache_savings', 0) > 0:
                print(f"Cache Savings:  ${costs['cache_savings']:.4f}")
            
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