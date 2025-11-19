"""Pydantic models for flashcard data structures."""

from typing import Optional
from pydantic import BaseModel


class Flashcard(BaseModel):
    """Single flashcard with question and answer."""
    question: str
    answer: str


class FlashcardSet(BaseModel):
    """Collection of flashcards."""
    flashcards: list[Flashcard]


class Critique(BaseModel):
    """AI critique of flashcard quality."""
    is_acceptable: bool
    feedback: str
    issues: list[str]


class StudyRating(BaseModel):
    """Individual flashcard rating from user."""
    flashcard_index: int  # Which card (0-based)
    difficulty: int  # 1-5 scale: 1=know well, 2=easy, 3=moderate, 4=difficult, 5=very difficult


class StudySession(BaseModel):
    """Complete study session results."""
    flashcards: list[Flashcard]  # All cards shown
    ratings: list[StudyRating]   # All ratings collected
    timestamp: str


class KnowledgeGaps(BaseModel):
    """AI analysis of student performance."""
    strong_areas: list[str]              # What they know well
    weak_areas: list[str]                # What needs work
    critical_gaps: list[str]             # Major knowledge gaps
    recommended_additions: list[Flashcard]  # New cards to add
    recommended_removals: list[int]      # Card indices to remove
    gap_report: str                      # Human-readable summary


class AdaptiveUpdate(BaseModel):
    """Final result of adaptation."""
    original_count: int
    cards_removed: list[Flashcard]
    cards_added: list[Flashcard]
    final_flashcards: FlashcardSet
    gap_report: str


class FlashcardEvaluation(BaseModel):
    """Evaluation of a single flashcard."""
    atomicity: int  # 1-10: One clear concept per card
    clarity: int  # 1-10: Unambiguous questions and answers
    long_term_retention_potential: int  # 1-10: Promotes active recall and deep understanding
    feedback: str  # Detailed feedback per criterion


class DeckEvaluation(BaseModel):
    """Evaluation of a flashcard deck."""
    flashcard_evaluations: list[FlashcardEvaluation]  # One per card - LLM provides this
    average_scores: Optional[dict[str, float]] = None  # Computed from flashcard_evaluations - not from LLM
    overall_deck_score: Optional[float] = None  # Computed from flashcard_evaluations - not from LLM


class GapCardEvaluation(BaseModel):
    """Evaluation of how well new cards address a specific gap."""
    gap_description: str  # The identified gap
    addressing_cards: list[int]  # List of new card indices that address this gap
    gap_coverage_score: int  # 1-10: How well new cards address this specific gap
    relevance_feedback: str  # Detailed explanation


class RemovalEvaluation(BaseModel):
    """Evaluation of whether a card removal was appropriate."""
    removed_card_index: int  # Index of removed card
    removed_card_question: str  # Question text
    user_rating: int  # User's difficulty rating (1-5)
    removal_appropriateness: int  # 1-10: Whether removal was correct
    removal_feedback: str  # Explanation


class AdaptationEvaluation(BaseModel):
    """Evaluation of the adaptation stage."""
    gap_evaluations: list[GapCardEvaluation]  # One per identified gap
    removal_evaluations: list[RemovalEvaluation]  # One per removed card
    average_gap_coverage: float  # Average gap coverage score
    average_removal_appropriateness: float  # Average removal correctness
    overall_adaptation_effectiveness: int  # 1-10: Overall score