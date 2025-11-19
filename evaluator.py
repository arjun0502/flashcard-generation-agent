"""LLM-as-a-judge evaluation module for flashcard quality assessment."""

import logging
from openai.lib._pydantic import to_strict_json_schema

from config import client
from models import (
    FlashcardSet,
    FlashcardEvaluation,
    DeckEvaluation,
    GapCardEvaluation,
    RemovalEvaluation,
    AdaptationEvaluation,
    KnowledgeGaps,
    StudySession,
    AdaptiveUpdate,
)


def evaluate_flashcard_set(
    flashcard_set: FlashcardSet,
    file_id: str | None = None,
    text_content: str | None = None,
    stage_name: str = "flashcard set",
    model: str = "gpt-4o"
) -> DeckEvaluation:
    """
    Evaluate a flashcard set using LLM-as-a-judge.
    
    Args:
        flashcard_set: The flashcard set to evaluate
        file_id: OpenAI file ID for PDF source material (if available)
        text_content: Text content of source material (if available)
        stage_name: Name of the evaluation stage (for context)
        model: OpenAI model to use for evaluation
        
    Returns:
        DeckEvaluation with per-card evaluations and aggregate scores
    """
    print(f"Evaluating {stage_name} with {model}...")
    logging.info(f"Evaluating {stage_name}: {len(flashcard_set.flashcards)} flashcards")
    
    # Format flashcards for evaluation
    flashcard_text = "\n".join([
        f"Card {i+1}:\nQ: {fc.question}\nA: {fc.answer}\n"
        for i, fc in enumerate(flashcard_set.flashcards)
    ])
    
    # Build user content based on input type
    if file_id:
        user_content = [
            {
                "type": "file",
                "file": {"file_id": file_id}
            },
            {
                "type": "text",
                "text": f"""Evaluate these flashcards for quality. The flashcards were generated from the attached document.

Flashcards to evaluate:
{flashcard_text}

Evaluate each flashcard individually on the following criteria (1-10 scale):
1. Atomicity: One clear concept per card (not multiple concepts)
2. Clarity: Unambiguous, precise questions and answers
3. Long-term Retention Potential (Active Recall): Promotes active recall and deep understanding vs. surface memorization

Provide detailed feedback for each flashcard."""
            }
        ]
    elif text_content:
        user_content = f"""Evaluate these flashcards for quality. The flashcards were generated from the following source material:

SOURCE MATERIAL:
{text_content[:5000]}...

Flashcards to evaluate:
{flashcard_text}

Evaluate each flashcard individually on the following criteria (1-10 scale):
1. Atomicity: One clear concept per card (not multiple concepts)
2. Clarity: Unambiguous, precise questions and answers
3. Long-term Retention Potential (Active Recall): Promotes active recall and deep understanding vs. surface memorization

Provide detailed feedback for each flashcard."""
    else:
        user_content = f"""Evaluate these flashcards for quality:

{flashcard_text}

Evaluate each flashcard individually on the following criteria (1-10 scale):
1. Atomicity: One clear concept per card (not multiple concepts)
2. Clarity: Unambiguous, precise questions and answers
3. Long-term Retention Potential (Active Recall): Promotes active recall and deep understanding vs. surface memorization

Provide detailed feedback for each flashcard."""
    
    system_content = """You are an expert educational evaluator specializing in flashcard quality assessment for long-term learning and spaced repetition.

Your goal is to evaluate flashcards based on pedagogical principles that support long-term retention. Consider:
- Whether each card focuses on a single, atomic concept
- Whether questions and answers are clear and unambiguous
- Whether the card promotes active recall and deep understanding rather than surface memorization

Rate each criterion on a 1-10 scale where:
- 1-3: Poor (significant issues)
- 4-6: Adequate (some issues, room for improvement)
- 7-8: Good (minor issues)
- 9-10: Excellent (high quality)

Provide detailed, constructive feedback for each flashcard."""
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "deck_evaluation",
                "schema": to_strict_json_schema(DeckEvaluation),
                "strict": True
            }
        }
    )
    
    evaluation = DeckEvaluation.model_validate_json(response.choices[0].message.content)
    
    # Always compute averages from flashcard_evaluations (we never ask LLM for these)
    if evaluation.flashcard_evaluations:
        criteria = ["atomicity", "clarity", "long_term_retention_potential"]
        evaluation.average_scores = {}
        for criterion in criteria:
            scores = [getattr(eval, criterion) for eval in evaluation.flashcard_evaluations]
            evaluation.average_scores[criterion] = sum(scores) / len(scores) if scores else 0.0
        
        # Calculate overall deck score as average of the three metrics
        metric_scores = [
            evaluation.average_scores["atomicity"],
            evaluation.average_scores["clarity"],
            evaluation.average_scores["long_term_retention_potential"]
        ]
        evaluation.overall_deck_score = sum(metric_scores) / len(metric_scores) if metric_scores else 0.0
    else:
        # Ensure defaults if no evaluations
        if evaluation.average_scores is None:
            evaluation.average_scores = {}
        if evaluation.overall_deck_score is None:
            evaluation.overall_deck_score = 0.0
    
    logging.info(f"Evaluation complete: Overall deck score = {evaluation.overall_deck_score:.2f}")
    return evaluation


def evaluate_adaptation(
    original_flashcards: FlashcardSet,
    adapted_update: AdaptiveUpdate,
    knowledge_gaps: KnowledgeGaps,
    study_session: StudySession,
    file_id: str | None = None,
    text_content: str | None = None,
    model: str = "gpt-4o"
) -> AdaptationEvaluation:
    """
    Evaluate the adaptation stage: how well gaps were addressed and removals were appropriate.
    
    Args:
        original_flashcards: Original flashcard set before adaptation
        adapted_update: The adaptive update result
        knowledge_gaps: Identified knowledge gaps
        study_session: Study session with user ratings
        file_id: OpenAI file ID for PDF source material (if available)
        text_content: Text content of source material (if available)
        model: OpenAI model to use for evaluation
        
    Returns:
        AdaptationEvaluation with gap coverage and removal appropriateness scores
    """
    print(f"Evaluating adaptation effectiveness with {model}...")
    logging.info("Evaluating adaptation stage")
    
    # Format new cards
    new_cards_text = "\n".join([
        f"New Card {i+1}:\nQ: {fc.question}\nA: {fc.answer}\n"
        for i, fc in enumerate(adapted_update.cards_added)
    ])
    
    # Format removed cards with user ratings
    removed_cards_text = []
    for removed_card in adapted_update.cards_removed:
        # Find the rating for this card
        rating = None
        for r in study_session.ratings:
            if r.flashcard_index < len(original_flashcards.flashcards):
                if original_flashcards.flashcards[r.flashcard_index].question == removed_card.question:
                    rating = r.difficulty
                    break
        
        removed_cards_text.append(
            f"Removed Card:\nQ: {removed_card.question}\nA: {removed_card.answer}\n"
            f"User Rating: {rating}/5 (1=know well, 5=very difficult)\n"
        )
    
    # Format identified gaps
    all_gaps = knowledge_gaps.critical_gaps + knowledge_gaps.weak_areas
    gaps_text = "\n".join([f"- {gap}" for gap in all_gaps])
    
    # Build user content
    if file_id:
        user_content = [
            {
                "type": "file",
                "file": {"file_id": file_id}
            },
            {
                "type": "text",
                "text": f"""Evaluate the adaptation effectiveness. The system identified knowledge gaps and generated new cards to address them, and removed cards the user rated as "know well" (rating 1).

IDENTIFIED KNOWLEDGE GAPS:
{gaps_text}

NEW CARDS GENERATED TO ADDRESS GAPS:
{new_cards_text}

REMOVED CARDS (user rated 1 = "know well"):
{chr(10).join(removed_cards_text)}

Evaluate:
1. For each identified gap, rate how well the new cards address that specific gap (1-10 scale)
2. For each removed card, rate whether the removal was appropriate given the user's rating (1-10 scale)

Provide detailed feedback explaining your ratings."""
            }
        ]
    elif text_content:
        user_content = f"""Evaluate the adaptation effectiveness. The system identified knowledge gaps and generated new cards to address them, and removed cards the user rated as "know well" (rating 1).

SOURCE MATERIAL:
{text_content[:3000]}...

IDENTIFIED KNOWLEDGE GAPS:
{gaps_text}

NEW CARDS GENERATED TO ADDRESS GAPS:
{new_cards_text}

REMOVED CARDS (user rated 1 = "know well"):
{chr(10).join(removed_cards_text)}

Evaluate:
1. For each identified gap, rate how well the new cards address that specific gap (1-10 scale)
2. For each removed card, rate whether the removal was appropriate given the user's rating (1-10 scale)

Provide detailed feedback explaining your ratings."""
    else:
        user_content = f"""Evaluate the adaptation effectiveness. The system identified knowledge gaps and generated new cards to address them, and removed cards the user rated as "know well" (rating 1).

IDENTIFIED KNOWLEDGE GAPS:
{gaps_text}

NEW CARDS GENERATED TO ADDRESS GAPS:
{new_cards_text}

REMOVED CARDS (user rated 1 = "know well"):
{chr(10).join(removed_cards_text)}

Evaluate:
1. For each identified gap, rate how well the new cards address that specific gap (1-10 scale)
2. For each removed card, rate whether the removal was appropriate given the user's rating (1-10 scale)

Provide detailed feedback explaining your ratings."""
    
    system_content = """You are an expert educational evaluator assessing the effectiveness of adaptive learning interventions.

Your goal is to evaluate:
1. Gap Coverage: How well the newly generated flashcards address each identified knowledge gap
   - Consider semantic alignment, specificity, and whether the cards actually help fill the gap
   - Rate 1-10 where 10 means the gap is fully and effectively addressed
   
2. Removal Appropriateness: Whether cards were correctly removed based on user ratings
   - A card rated 1 ("know well") should generally be removed
   - Rate 1-10 where 10 means removal was completely appropriate
   - Consider if the user's rating accurately reflects mastery

Provide detailed, constructive feedback for each gap and removal."""
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "adaptation_evaluation",
                "schema": to_strict_json_schema(AdaptationEvaluation),
                "strict": True
            }
        }
    )
    
    evaluation = AdaptationEvaluation.model_validate_json(response.choices[0].message.content)
    
    # Calculate averages if not provided
    if evaluation.gap_evaluations:
        gap_scores = [gap.gap_coverage_score for gap in evaluation.gap_evaluations]
        evaluation.average_gap_coverage = sum(gap_scores) / len(gap_scores) if gap_scores else 0.0
    else:
        evaluation.average_gap_coverage = 0.0
    
    if evaluation.removal_evaluations:
        removal_scores = [removal.removal_appropriateness for removal in evaluation.removal_evaluations]
        evaluation.average_removal_appropriateness = sum(removal_scores) / len(removal_scores) if removal_scores else 0.0
    else:
        evaluation.average_removal_appropriateness = 0.0
    
    logging.info(f"Adaptation evaluation complete: Gap coverage = {evaluation.average_gap_coverage:.2f}, "
                 f"Removal appropriateness = {evaluation.average_removal_appropriateness:.2f}")
    return evaluation

