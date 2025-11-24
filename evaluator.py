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


def _build_evaluation_prompt(flashcard_text: str, file_id: bool = False, text_content: str | None = None) -> str:
    """Build the evaluation prompt with 4 metrics."""
    per_card_metrics = """Evaluate each flashcard individually on the following four atomic criteria (1-10 scale). Each criterion is independent and non-overlapping:

1. ATOMICITY: Does the card focus on ONE clear, atomic concept? Cards that combine multiple concepts or require multiple pieces of information should be split or simplified. Rate 1-10 based solely on whether it's one concept or multiple.

2. CLARITY: Are questions and answers unambiguous, precise, and complete? Avoid vague wording, ambiguous phrasing, or questions that could have multiple correct answers. Include necessary context for understanding. The answer should fully address the question without being excessively long. Rate 1-10 based solely on clarity and completeness.

3. LEARNING VALUE: Does the card promote active recall and deep understanding rather than surface memorization? Prefer "why" and "how" questions over "what" questions. Avoid yes/no questions, simple fact recall, or questions that test only memorization. Cards should require the learner to actively construct knowledge. Rate 1-10 based solely on learning value and active recall effectiveness.

4. ACCURACY: Is the information in the flashcard factually correct and free from errors? Verify against the source material that all facts, definitions, and explanations are accurate. Rate 1-10 based solely on factual correctness."""
    
    if file_id:
        return f"""Evaluate these flashcards for quality. The flashcards were generated from the attached document.

OVERARCHING GOAL: These flashcards are designed for long-term understanding and spaced repetition of lecture material. They should help students master important concepts through active recall and deep understanding, not just surface memorization.

Flashcards to evaluate:
{flashcard_text}

{per_card_metrics}

Provide detailed feedback for each flashcard explaining your ratings."""
    elif text_content:
        return f"""Evaluate these flashcards for quality. The flashcards were generated from the following source material:

OVERARCHING GOAL: These flashcards are designed for long-term understanding and spaced repetition of lecture material. They should help students master important concepts through active recall and deep understanding, not just surface memorization.

SOURCE MATERIAL:
{text_content[:5000]}...

Flashcards to evaluate:
{flashcard_text}

{per_card_metrics}

Provide detailed feedback for each flashcard explaining your ratings."""
    else:
        per_card_metrics_no_source = """Evaluate each flashcard individually on the following four atomic criteria (1-10 scale). Each criterion is independent and non-overlapping:

1. ATOMICITY: Does the card focus on ONE clear, atomic concept? Cards that combine multiple concepts or require multiple pieces of information should be split or simplified. Rate 1-10 based solely on whether it's one concept or multiple.

2. CLARITY: Are questions and answers unambiguous, precise, and complete? Avoid vague wording, ambiguous phrasing, or questions that could have multiple correct answers. Include necessary context for understanding. The answer should fully address the question without being excessively long. Rate 1-10 based solely on clarity and completeness.

3. LEARNING VALUE: Does the card promote active recall and deep understanding rather than surface memorization? Prefer "why" and "how" questions over "what" questions. Avoid yes/no questions, simple fact recall, or questions that test only memorization. Cards should require the learner to actively construct knowledge. Rate 1-10 based solely on learning value and active recall effectiveness.

4. ACCURACY: Is the information in the flashcard factually correct and free from errors? Since source material is not available, evaluate based on general knowledge and internal consistency. Rate 1-10 based solely on factual correctness."""
        
        return f"""Evaluate these flashcards for quality:

OVERARCHING GOAL: These flashcards are designed for long-term understanding and spaced repetition of lecture material. They should help students master important concepts through active recall and deep understanding, not just surface memorization.

{flashcard_text}

{per_card_metrics_no_source}

Provide detailed feedback for each flashcard explaining your ratings."""


def _build_system_prompt() -> str:
    """Build the system prompt for evaluation."""
    return """You are an expert educational evaluator specializing in flashcard quality assessment for long-term learning and spaced repetition.

OVERARCHING GOAL: Evaluate flashcards designed for long-term understanding and spaced repetition of lecture material. These flashcards should help students master important concepts through active recall and deep understanding, not just surface memorization.

Evaluate each flashcard on four atomic, independent criteria. Each criterion should be evaluated separately without overlap:

1. ATOMICITY: Is it one clear concept? (Not multiple concepts)
2. CLARITY: Is it unambiguous, precise, and complete? (Not vague or incomplete)
3. LEARNING VALUE: Does it promote active recall and deep understanding? (Not just memorization)
4. ACCURACY: Is it factually correct? (No errors)

Rate each criterion independently on a 1-10 scale where:
- 1-3: Poor (significant issues)
- 4-6: Adequate (some issues, room for improvement)
- 7-8: Good (minor issues)
- 9-10: Excellent (high quality)

Provide detailed, constructive feedback for each flashcard explaining your ratings."""


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
                "text": _build_evaluation_prompt(flashcard_text, file_id=True)
            }
        ]
    elif text_content:
        user_content = _build_evaluation_prompt(flashcard_text, text_content=text_content)
    else:
        user_content = _build_evaluation_prompt(flashcard_text)
    
    system_content = _build_system_prompt()
    
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
        # 4 metrics: atomicity, clarity, learning_value, accuracy
        criteria = ["atomicity", "clarity", "learning_value", "accuracy"]
        
        evaluation.average_scores = {}
        for criterion in criteria:
            scores = [getattr(eval, criterion) for eval in evaluation.flashcard_evaluations]
            evaluation.average_scores[criterion] = sum(scores) / len(scores) if scores else 0.0
        
        # Calculate overall_deck_score as average of 4 metrics
        evaluation.overall_deck_score = sum(evaluation.average_scores.values()) / len(evaluation.average_scores) if evaluation.average_scores else 0.0
    else:
        # Ensure defaults if no evaluations
        if evaluation.average_scores is None:
            evaluation.average_scores = {}
        evaluation.overall_deck_score = 0.0
    
    logging.info(f"Evaluation complete: Average scores computed for {len(evaluation.average_scores)} criteria")
    logging.info(f"Overall deck score: {evaluation.overall_deck_score:.2f}/10")
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

Evaluate ONLY personalization effectiveness (do NOT evaluate general card quality):

1. Gap-Filling Effectiveness: For each identified gap, rate how well the new cards address that specific gap (1-10 scale)
   - Do the cards actually help fill the gap? Are they semantically aligned with what the student struggled with?
   - Focus ONLY on whether the gap is addressed, NOT on card quality (atomicity, clarity, etc.)

2. Removal Appropriateness: For each removed card, rate whether the removal was appropriate given the user's rating (1-10 scale)
   - Was the card correctly removed because the student already knows it well?
   - Focus ONLY on whether removal was appropriate based on student knowledge, NOT on card quality

Provide detailed feedback explaining your personalization ratings."""
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

Evaluate ONLY personalization effectiveness (do NOT evaluate general card quality):

1. Gap-Filling Effectiveness: For each identified gap, rate how well the new cards address that specific gap (1-10 scale)
   - Do the cards actually help fill the gap? Are they semantically aligned with what the student struggled with?
   - Focus ONLY on whether the gap is addressed, NOT on card quality (atomicity, clarity, etc.)

2. Removal Appropriateness: For each removed card, rate whether the removal was appropriate given the user's rating (1-10 scale)
   - Was the card correctly removed because the student already knows it well?
   - Focus ONLY on whether removal was appropriate based on student knowledge, NOT on card quality

Provide detailed feedback explaining your personalization ratings."""
    else:
        user_content = f"""Evaluate the adaptation effectiveness. The system identified knowledge gaps and generated new cards to address them, and removed cards the user rated as "know well" (rating 1).

IDENTIFIED KNOWLEDGE GAPS:
{gaps_text}

NEW CARDS GENERATED TO ADDRESS GAPS:
{new_cards_text}

REMOVED CARDS (user rated 1 = "know well"):
{chr(10).join(removed_cards_text)}

Evaluate ONLY personalization effectiveness (do NOT evaluate general card quality):

1. Gap-Filling Effectiveness: For each identified gap, rate how well the new cards address that specific gap (1-10 scale)
   - Do the cards actually help fill the gap? Are they semantically aligned with what the student struggled with?
   - Focus ONLY on whether the gap is addressed, NOT on card quality (atomicity, clarity, etc.)

2. Removal Appropriateness: For each removed card, rate whether the removal was appropriate given the user's rating (1-10 scale)
   - Was the card correctly removed because the student already knows it well?
   - Focus ONLY on whether removal was appropriate based on student knowledge, NOT on card quality

Provide detailed feedback explaining your personalization ratings."""
    
    system_content = """You are an expert educational evaluator assessing the effectiveness of adaptive learning interventions.

Your goal is to evaluate PERSONALIZATION effectiveness - how well the system adapted the deck to the student's needs.

Evaluate ONLY these two aspects:

1. Gap-Filling Effectiveness: How well do the newly generated flashcards address each identified knowledge gap?
   - Do the cards actually help fill the gap? Are they semantically aligned with what the student struggled with?
   - Rate 1-10 where 10 means the gap is fully and effectively addressed
   - DO NOT evaluate general card quality (atomicity, clarity, learning_value, accuracy) - only evaluate gap-filling effectiveness
   
2. Removal Appropriateness: Were cards correctly removed based on what the student already knows?
   - A card rated 1 ("know well") should generally be removed
   - Rate 1-10 where 10 means removal was completely appropriate
   - DO NOT evaluate card quality - only evaluate whether removal was appropriate based on student knowledge

Focus ONLY on personalization: gap-filling and removal appropriateness. General quality metrics are evaluated in earlier stages."""
    
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
        gap_scores = [gap.personalization_score for gap in evaluation.gap_evaluations]
        evaluation.average_gap_personalization = sum(gap_scores) / len(gap_scores) if gap_scores else 0.0
    else:
        evaluation.average_gap_personalization = 0.0
    
    if evaluation.removal_evaluations:
        removal_scores = [removal.personalization_score for removal in evaluation.removal_evaluations]
        evaluation.average_removal_personalization = sum(removal_scores) / len(removal_scores) if removal_scores else 0.0
    else:
        evaluation.average_removal_personalization = 0.0
    
    # Calculate overall personalization score
    if evaluation.average_gap_personalization > 0 or evaluation.average_removal_personalization > 0:
        # Weighted average: 60% gap-filling, 40% removal
        evaluation.overall_personalization = (
            0.6 * evaluation.average_gap_personalization +
            0.4 * evaluation.average_removal_personalization
        )
    else:
        evaluation.overall_personalization = 0.0
    
    logging.info(f"Adaptation evaluation complete: Gap personalization = {evaluation.average_gap_personalization:.2f}, "
                 f"Removal personalization = {evaluation.average_removal_personalization:.2f}, "
                 f"Overall personalization = {evaluation.overall_personalization:.2f}")
    return evaluation

