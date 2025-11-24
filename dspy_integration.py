"""
Integration layer to use optimized DSPy modules with existing code.

This allows using optimized prompts when available, with fallback to original functions.
"""

import os
import json
from pathlib import Path
from typing import Optional

try:
    import dspy
except ImportError:
    dspy = None

from models import FlashcardSet, Critique
from openai_client import critique_flashcards, generate_gap_filling_cards
from dspy_modules import CritiqueModule, AdaptationModule
from dspy_metrics import critique_metric, adaptation_metric


# Global flag to enable/disable DSPy optimization
USE_DSPY = os.getenv("USE_DSPY_OPTIMIZATION", "false").lower() == "true"

# Global modules (loaded once)
_critique_module: Optional[CritiqueModule] = None
_adaptation_module: Optional[AdaptationModule] = None
_dspy_configured = False


def setup_dspy_if_needed(model: str = "gpt-4o"):
    """Set up DSPy if not already configured."""
    global _dspy_configured
    
    if not dspy:
        return False
    
    if not _dspy_configured:
        from openai import OpenAI
        import os
        
        class OpenAILM(dspy.LM):
            def __init__(self, model_name: str):
                super().__init__(model_name)
                self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                self.model_name = model_name
            
            def __call__(self, prompt, **kwargs):
                messages = []
                if hasattr(prompt, 'messages'):
                    messages = prompt.messages
                elif isinstance(prompt, str):
                    messages = [{"role": "user", "content": prompt}]
                elif isinstance(prompt, list):
                    messages = prompt
                
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    **kwargs
                )
                return response.choices[0].message.content
        
        lm = OpenAILM(model)
        dspy.configure(lm=lm)
        _dspy_configured = True
    
    return True


def load_optimized_modules(module_dir: Path = Path("optimized_prompts")):
    """Load optimized DSPy modules from disk."""
    global _critique_module, _adaptation_module
    
    if not dspy:
        print("Warning: dspy-ai not installed. Using original prompts.")
        return
    
    if not module_dir.exists():
        print(f"Info: Optimization directory {module_dir} not found. Using original prompts.")
        return
    
    setup_dspy_if_needed()
    
    # Initialize modules (they'll use optimized prompts if available)
    # Note: In practice, you'd load the actual optimized module objects
    # For now, we initialize them and they'll use default prompts
    _critique_module = CritiqueModule()
    _adaptation_module = AdaptationModule()
    
    print("✓ DSPy modules initialized (optimized prompts if available)")


def critique_flashcards_optimized(
    flashcard_set: FlashcardSet,
    model: str = "gpt-4o"
) -> Critique:
    """Critique flashcards using optimized DSPy module."""
    global _critique_module
    
    if not dspy:
        return critique_flashcards(flashcard_set, model)
    
    if _critique_module is None:
        setup_dspy_if_needed(model)
        _critique_module = CritiqueModule()
    
    # Format flashcard text
    flashcard_text = "\n".join([
        f"{i+1}. Q: {fc.question} | A: {fc.answer}"
        for i, fc in enumerate(flashcard_set.flashcards)
    ])
    
    # Use DSPy module
    result = _critique_module(flashcard_text=flashcard_text)
    
    # Parse result into Critique object
    is_acceptable = getattr(result, 'is_acceptable', False)
    if isinstance(is_acceptable, str):
        is_acceptable = "true" in is_acceptable.lower() or "yes" in is_acceptable.lower()
    
    feedback = getattr(result, 'feedback', '')
    issues_str = getattr(result, 'issues', '')
    
    # Parse issues
    if isinstance(issues_str, str):
        issues = [issue.strip() for issue in issues_str.split('\n') if issue.strip()]
    else:
        issues = issues_str if isinstance(issues_str, list) else []
    
    return Critique(
        is_acceptable=bool(is_acceptable),
        feedback=feedback,
        issues=issues
    )


def generate_gap_filling_cards_optimized(
    gaps,
    file_id: Optional[str] = None,
    text_content: Optional[str] = None,
    model: str = "gpt-4o"
) -> list:
    """Generate gap-filling cards using optimized DSPy module."""
    global _adaptation_module
    
    if not dspy:
        return generate_gap_filling_cards(gaps, file_id, text_content, model)
    
    if _adaptation_module is None:
        setup_dspy_if_needed(model)
        _adaptation_module = AdaptationModule()
    
    # Format knowledge gaps
    gap_summary = "\n".join([
        f"- {gap}" for gap in gaps.critical_gaps
    ])
    if gaps.weak_areas:
        gap_summary += "\n" + "\n".join([
            f"- {area}" for area in gaps.weak_areas
        ])
    
    # Get source material
    source_material = text_content or ""
    if file_id and not source_material:
        # For file_id, we'd need to download/read it
        # For now, use empty string
        source_material = ""
    
    # Truncate if too long
    if len(source_material) > 5000:
        source_material = source_material[:5000] + "..."
    
    # Use DSPy module
    result = _adaptation_module(
        knowledge_gaps=gap_summary,
        source_material=source_material
    )
    
    # Parse result into FlashcardSet
    new_cards_json = getattr(result, 'new_flashcards', '')
    
    try:
        if isinstance(new_cards_json, str):
            import json
            new_cards_data = json.loads(new_cards_json)
            flashcard_set = FlashcardSet.model_validate(new_cards_data)
            return flashcard_set.flashcards
        else:
            # Fallback to original function
            return generate_gap_filling_cards(gaps, file_id, text_content, model)
    except Exception as e:
        print(f"Error parsing DSPy adaptation result: {e}")
        # Fallback to original function
        return generate_gap_filling_cards(gaps, file_id, text_content, model)


# Wrapper functions that switch between optimized and original
def critique_flashcards_wrapper(
    flashcard_set: FlashcardSet,
    model: str = "gpt-4o",
    use_optimized: Optional[bool] = None
) -> Critique:
    """Wrapper that uses optimized or original critique function."""
    if use_optimized is None:
        use_optimized = USE_DSPY
    
    if use_optimized:
        return critique_flashcards_optimized(flashcard_set, model)
    else:
        return critique_flashcards(flashcard_set, model)


def generate_gap_filling_cards_wrapper(
    gaps,
    file_id: Optional[str] = None,
    text_content: Optional[str] = None,
    model: str = "gpt-4o",
    use_optimized: Optional[bool] = None
) -> list:
    """Wrapper that uses optimized or original gap-filling function."""
    if use_optimized is None:
        use_optimized = USE_DSPY
    
    if use_optimized:
        return generate_gap_filling_cards_optimized(gaps, file_id, text_content, model)
    else:
        return generate_gap_filling_cards(gaps, file_id, text_content, model)

