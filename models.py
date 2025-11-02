"""Pydantic models for flashcard data structures."""

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

