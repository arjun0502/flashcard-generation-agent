"""Metric functions for DSPy optimization using existing evaluation system."""

import json
from models import FlashcardSet, Critique, AdaptiveUpdate
from openai_client import revise_flashcards
from evaluator import evaluate_flashcard_set, evaluate_adaptation


def critique_metric(example, pred, trace=None):
    """
    Metric for critique quality.
    
    Evaluates: Does a good critique lead to better revised flashcards?
    Uses: evaluate_flashcard_set().overall_deck_score (average of 4 metrics)
    
    Args:
        example: dspy.Example with flashcard_set attribute
        pred: DSPy prediction (CritiqueSignature output)
        trace: Optional trace for debugging
    
    Returns:
        Score 0-1 based on whether critique improves flashcard quality
    """
    flashcard_set = example.flashcard_set
    
    # Parse the predicted critique
    is_acceptable = getattr(pred, 'is_acceptable', False)
    if isinstance(is_acceptable, str):
        is_acceptable = "true" in is_acceptable.lower() or "yes" in is_acceptable.lower()
    
    feedback = getattr(pred, 'feedback', '')
    issues_str = getattr(pred, 'issues', '')
    
    # Parse issues
    if isinstance(issues_str, str):
        issues = [issue.strip() for issue in issues_str.split('\n') if issue.strip()]
    else:
        issues = issues_str if isinstance(issues_str, list) else []
    
    critique = Critique(
        is_acceptable=bool(is_acceptable),
        feedback=feedback,
        issues=issues
    )
    
    # If critique says it's acceptable, verify quality
    if critique.is_acceptable:
        eval_result = evaluate_flashcard_set(flashcard_set)
        # If original is actually good (score >= 7), critique was correct
        if eval_result.overall_deck_score >= 7.0:
            return 1.0
        else:
            # Should have critiqued it
            return 0.5
    
    # If critique says needs revision, revise and check improvement
    try:
        revised = revise_flashcards(flashcard_set, critique)
        
        # Evaluate original vs revised
        original_eval = evaluate_flashcard_set(flashcard_set)
        revised_eval = evaluate_flashcard_set(revised)
        
        # Score based on improvement in overall_deck_score
        improvement = revised_eval.overall_deck_score - original_eval.overall_deck_score
        
        if improvement > 0:
            # Normalize to 0-1 (max improvement could be ~5 points)
            # Base score 0.5 + improvement bonus
            return min(1.0, 0.5 + (improvement / 5.0))
        else:
            # Critique didn't help (or made worse)
            return max(0.0, 0.3 + (improvement / 5.0))
    
    except Exception as e:
        print(f"Error in critique_metric: {e}")
        return 0.0


def adaptation_metric(example, pred, trace=None):
    """
    Metric for adaptation quality.
    
    Evaluates: Do generated cards effectively address knowledge gaps?
    Uses: evaluate_adaptation().overall_personalization (gap-filling + removal)
    
    Args:
        example: dspy.Example with:
            - knowledge_gaps (KnowledgeGaps object)
            - source_material (str)
            - study_session (StudySession object)
            - original_flashcards (FlashcardSet object)
            - file_id (optional str)
        pred: DSPy prediction (AdaptationSignature output)
        trace: Optional trace for debugging
    
    Returns:
        Score 0-1 based on overall_personalization from evaluate_adaptation()
    """
    try:
        # Parse predicted flashcards
        new_cards_json = getattr(pred, 'new_flashcards', '')
        
        if not new_cards_json:
            return 0.0
        
        if isinstance(new_cards_json, str):
            new_cards_data = json.loads(new_cards_json)
        else:
            new_cards_data = new_cards_json
        
        new_flashcards = FlashcardSet.model_validate(new_cards_data)
        
        if len(new_flashcards.flashcards) == 0:
            return 0.0
        
        # Get required data from example
        knowledge_gaps = example.knowledge_gaps
        study_session = example.study_session
        original_flashcards = example.original_flashcards
        file_id = getattr(example, 'file_id', None)
        source_material = getattr(example, 'source_material', None)
        
        # Create AdaptiveUpdate object (needed for evaluate_adaptation)
        # For optimization, we focus on gap-filling, so we'll create a minimal update
        adapted_update = AdaptiveUpdate(
            original_count=len(original_flashcards.flashcards),
            cards_removed=[],  # Focus on gap-filling for optimization
            cards_added=new_flashcards.flashcards,
            final_flashcards=FlashcardSet(
                flashcards=original_flashcards.flashcards + new_flashcards.flashcards
            ),
            gap_report=knowledge_gaps.gap_report
        )
        
        # Use existing evaluation function
        adaptation_eval = evaluate_adaptation(
            original_flashcards=original_flashcards,
            adapted_update=adapted_update,
            knowledge_gaps=knowledge_gaps,
            study_session=study_session,
            file_id=file_id,
            text_content=source_material
        )
        
        # Return overall_personalization score (0-10, normalize to 0-1)
        return adaptation_eval.overall_personalization / 10.0
    
    except Exception as e:
        print(f"Error in adaptation_metric: {e}")
        import traceback
        traceback.print_exc()
        return 0.0

