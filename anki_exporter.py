"""Anki export functionality."""

import html
import genanki

from config import FLASHCARD_MODEL, FLASHCARD_DECK_ID
from models import FlashcardSet


def export_to_anki(flashcard_set: FlashcardSet, deck_name: str, output_file: str) -> None:
    """
    Export flashcards to an Anki package (.apkg) file.
    
    Following genanki best practices:
    - Using hardcoded model_id and deck_id for consistency
    - HTML-escaping field content to handle special characters
    """
    print(f"Creating Anki deck: {deck_name}...")
    
    # Create deck with hardcoded ID
    deck = genanki.Deck(
        FLASHCARD_DECK_ID,
        deck_name
    )
    
    # Add flashcards as notes
    for fc in flashcard_set.flashcards:
        # HTML-escape the fields to handle special characters like <, >, &
        # This is important per the genanki README
        question_escaped = html.escape(fc.question)
        answer_escaped = html.escape(fc.answer)
        
        note = genanki.Note(
            model=FLASHCARD_MODEL,
            fields=[question_escaped, answer_escaped]
        )
        deck.add_note(note)
    
    # Create and write package
    genanki.Package(deck).write_to_file(output_file)
    print(f"✓ Created Anki package: {output_file}")


def save_flashcards_text(flashcard_set: FlashcardSet, output_file: str = "flashcards.txt") -> None:
    """Save flashcards to a text file in Question|Answer format."""
    with open(output_file, "w", encoding="utf-8") as f:
        for fc in flashcard_set.flashcards:
            f.write(f"{fc.question}|{fc.answer}\n")
    print(f"✓ Text format saved to: {output_file}")

