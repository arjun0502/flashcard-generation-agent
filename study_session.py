"""Study session and adaptive learning functionality."""

from datetime import datetime
import logging

from models import FlashcardSet, StudySession, StudyRating, KnowledgeGaps, AdaptiveUpdate
from openai_client import analyze_knowledge_gaps, generate_gap_filling_cards


def conduct_study_session(flashcard_set: FlashcardSet) -> StudySession:
    """Interactive study session - collects difficulty ratings from user."""
    print(f"\n{'='*60}")
    print("STUDY SESSION")
    print(f"{'='*60}")
    print("\nYou'll rate each flashcard on difficulty (1-5):")
    print("1 = Already know well")
    print("2 = Easy") 
    print("3 = Moderate")
    print("4 = Difficult")
    print("5 = Very difficult / Don't understand")
    print("\n[Press Enter to continue]")
    input()
    
    ratings = []
    total = len(flashcard_set.flashcards)
    
    for idx, flashcard in enumerate(flashcard_set.flashcards):
        print(f"\n{'─'*60}")
        print(f"Flashcard {idx+1} of {total}")
        print(f"{'─'*60}\n")
        
        print("QUESTION:")
        print(flashcard.question)
        print("\n[Think about your answer... Press Enter when ready]")
        input()
        
        print("\nANSWER:")
        print(flashcard.answer)
        print(f"\n{'─'*60}\n")
        
        while True:
            try:
                rating = int(input("Enter difficulty rating (1-5): "))
                if 1 <= rating <= 5:
                    ratings.append(StudyRating(flashcard_index=idx, difficulty=rating))
                    break
                else:
                    print("Please enter a number between 1 and 5.")
            except ValueError:
                print("Please enter a valid number.")
    
    print(f"\n{'='*60}")
    print("STUDY SESSION COMPLETE")
    print(f"{'='*60}\n")
    
    rating_summary = {
        "1 (know well)": sum(1 for r in ratings if r.difficulty == 1),
        "2 (easy)": sum(1 for r in ratings if r.difficulty == 2),
        "3 (moderate)": sum(1 for r in ratings if r.difficulty == 3),
        "4 (difficult)": sum(1 for r in ratings if r.difficulty == 4),
        "5 (very difficult)": sum(1 for r in ratings if r.difficulty == 5),
    }
    
    print("You rated flashcards:")
    for difficulty, count in rating_summary.items():
        if count > 0:
            print(f"  - {difficulty}: {count} cards")
    
    return StudySession(
        flashcards=flashcard_set.flashcards,
        ratings=ratings,
        timestamp=datetime.now().isoformat()
    )


def adaptive_update_flashcards(
    original: FlashcardSet,
    session: StudySession,
    gaps: KnowledgeGaps,
    file_id: str | None = None,
    text_content: str | None = None,
    model: str = "gpt-4o"
) -> AdaptiveUpdate:
    """Apply adaptive updates: remove mastered cards, add gap-filling cards"""
    
    # Identify mastered cards (rated 1) to remove
    mastered_indices = set()
    for rating in session.ratings:
        if rating.difficulty == 1:
            mastered_indices.add(rating.flashcard_index)
    
    removed_cards = [original.flashcards[i] for i in mastered_indices]
    logging.info(f"Identified {len(removed_cards)} mastered cards for removal")
    
    # Keep cards that need more practice (rated 2-5)
    kept_cards = [
        fc for idx, fc in enumerate(original.flashcards)
        if idx not in mastered_indices
    ]
    logging.info(f"Keeping {len(kept_cards)} cards that need practice")
    
    # Generate new cards for gaps
    new_cards = generate_gap_filling_cards(gaps, file_id=file_id, text_content=text_content, model=model)
    logging.info(f"Generated {len(new_cards)} new gap-filling cards")
    
    # Combine kept + new cards
    final_cards = kept_cards + new_cards
    final_flashcard_set = FlashcardSet(flashcards=final_cards)
    
    return AdaptiveUpdate(
        original_count=len(original.flashcards),
        cards_removed=removed_cards,
        cards_added=new_cards,
        final_flashcards=final_flashcard_set,
        gap_report=gaps.gap_report
    )


def print_adaptive_summary(update: AdaptiveUpdate) -> None:
    """Print summary of adaptive changes."""
    print(f"\n{'='*60}")
    print("KNOWLEDGE GAPS IDENTIFIED")
    print(f"{'='*60}\n")
    
    print(update.gap_report)
    
    print(f"\n{'─'*60}")
    print("DECK CHANGES")
    print(f"{'─'*60}")
    print(f"Original cards: {update.original_count}")
    print(f"Cards removed (mastered): {len(update.cards_removed)}")
    print(f"Cards added (gap-filling): {len(update.cards_added)}")
    print(f"Final cards: {len(update.final_flashcards.flashcards)}")
    print(f"{'─'*60}\n")

