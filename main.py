from openai import OpenAI
from pydantic import BaseModel
from pathlib import Path
import genanki
import os
import html
import argparse
import logging
from datetime import datetime

# Import internal OpenAI function for strict JSON schema
from openai.lib._pydantic import to_strict_json_schema

# Set up logging
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

# Create log filename with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = LOG_DIR / f"flashcard_generation_{timestamp}.log"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# IMPORTANT: These IDs should be hardcoded and unique
# Generated once using: python3 -c "import random; print(random.randrange(1 << 30, 1 << 31))"
FLASHCARD_MODEL_ID = 1607392319
FLASHCARD_DECK_ID = 2059400110

# Define structured output models
class Flashcard(BaseModel):
    question: str
    answer: str

class FlashcardSet(BaseModel):
    flashcards: list[Flashcard]

class Critique(BaseModel):
    is_acceptable: bool
    feedback: str
    issues: list[str]

# Study session models
class StudyRating(BaseModel):
    """Individual flashcard rating from user"""
    flashcard_index: int  # Which card (0-based)
    difficulty: int  # 1-5 scale: 1=know well, 2=easy, 3=moderate, 4=difficult, 5=very difficult

class StudySession(BaseModel):
    """Complete study session results"""
    flashcards: list[Flashcard]  # All cards shown
    ratings: list[StudyRating]   # All ratings collected
    timestamp: str

class KnowledgeGaps(BaseModel):
    """AI analysis of student performance"""
    strong_areas: list[str]              # What they know well
    weak_areas: list[str]                # What needs work
    critical_gaps: list[str]             # Major knowledge gaps
    recommended_additions: list[Flashcard]  # New cards to add
    recommended_removals: list[int]      # Card indices to remove
    gap_report: str                      # Human-readable summary

class AdaptiveUpdate(BaseModel):
    """Final result of adaptation"""
    original_count: int
    cards_removed: list[Flashcard]
    cards_added: list[Flashcard]
    final_flashcards: FlashcardSet
    gap_report: str

# Define the Anki model (note type) - this should be consistent
FLASHCARD_MODEL = genanki.Model(
    FLASHCARD_MODEL_ID,
    'AI Generated Flashcard Model',
    fields=[
        {'name': 'Question'},
        {'name': 'Answer'},
    ],
    templates=[
        {
            'name': 'Card 1',
            'qfmt': '{{Question}}',
            'afmt': '{{FrontSide}}<hr id="answer">{{Answer}}',
        },
    ])

# Upload PDF to OpenAI
def upload_pdf(file_path: str):
    """Upload a PDF file to OpenAI using the Files API."""
    file_path_obj = Path(file_path)
    if not file_path_obj.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if file_path_obj.suffix.lower() != '.pdf':
        raise ValueError(f"File must be a PDF. Got: {file_path_obj.suffix}")
    
    print(f"Uploading {file_path_obj.name}...")
    
    with open(file_path, "rb") as file:
        uploaded_file = client.files.create(
            file=file,
            purpose="user_data"
        )
    
    print(f"File uploaded successfully. ID: {uploaded_file.id}")
    logging.info(f"PDF uploaded with ID: {uploaded_file.id}")
    return uploaded_file.id

# Generate flashcards from uploaded file
def generate_flashcards_with_file_id(file_id: str, model: str = "gpt-4o"):
    """Generate flashcards using uploaded file ID."""
    print(f"Calling OpenAI API ({model}) to generate flashcards...")
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": """You are an expert at creating effective flashcards for spaced repetition learning.
                Create flashcards that:
                - Focus on atomic concepts (one concept per card)
                - Use clear, concise questions
                - Avoid yes/no questions
                - Include context when needed
                - Test understanding, not just memorization
                - Extract information from both text and any diagrams/images in the PDF"""
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "file",
                        "file": {
                            "file_id": file_id
                        }
                    },
                    {
                        "type": "text",
                        "text": "Generate comprehensive flashcards from this document. Include information from any diagrams, charts, or images."
                    }
                ]
            }
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "flashcard_set",
                "schema": to_strict_json_schema(FlashcardSet),
                "strict": True
            }
        }
    )
    
    return FlashcardSet.model_validate_json(response.choices[0].message.content)

# Cleanup uploaded file
def cleanup_file(file_id: str):
    """Delete the uploaded file to avoid storage costs."""
    try:
        client.files.delete(file_id)
        print(f"File {file_id} deleted successfully.")
        logging.info(f"Deleted uploaded file: {file_id}")
    except Exception as e:
        print(f"Error deleting file {file_id}: {e}")
        logging.error(f"Error deleting file {file_id}: {e}")

# Critique flashcards
def critique_flashcards(flashcard_set: FlashcardSet, model: str = "gpt-4o"):
    flashcard_text = "\n".join([f"{i+1}. Q: {fc.question} | A: {fc.answer}" 
                                 for i, fc in enumerate(flashcard_set.flashcards)])
    
    print(f"Critiquing flashcards with {model}...")
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": """Evaluate flashcards against these pedagogical principles:
                1. Atomicity: One concept per card
                2. Clarity: Unambiguous questions and answers
                3. Difficulty: Appropriate cognitive load
                4. Avoid: Yes/no questions, overly broad questions
                5. Context: Include necessary context for understanding
                
                Determine if the flashcards are acceptable or need revision."""
            },
            {
                "role": "user",
                "content": f"Critique these flashcards:\n\n{flashcard_text}"
            }
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "critique",
                "schema": to_strict_json_schema(Critique),
                "strict": True
            }
        }
    )
    
    return Critique.model_validate_json(response.choices[0].message.content)

# Revise flashcards
def revise_flashcards(flashcard_set: FlashcardSet, critique: Critique, model: str = "gpt-4o"):
    flashcard_text = "\n".join([f"{i+1}. Q: {fc.question} | A: {fc.answer}" 
                                 for i, fc in enumerate(flashcard_set.flashcards)])
    
    print(f"Revising flashcards with {model}...")
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are an expert at refining flashcards based on pedagogical feedback."
            },
            {
                "role": "user",
                "content": f"""Revise these flashcards based on the feedback:

Original flashcards:
{flashcard_text}

Feedback:
{critique.feedback}

Issues to address:
{chr(10).join(f'- {issue}' for issue in critique.issues)}"""
            }
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "flashcard_set",
                "schema": to_strict_json_schema(FlashcardSet),
                "strict": True
            }
        }
    )
    
    return FlashcardSet.model_validate_json(response.choices[0].message.content)

# Export to Anki
def export_to_anki(flashcard_set: FlashcardSet, deck_name: str, output_file: str):
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

# Conduct interactive study session
def conduct_study_session(flashcard_set: FlashcardSet) -> StudySession:
    """Interactive study session - collects difficulty ratings from user"""
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

# Analyze knowledge gaps
def analyze_knowledge_gaps(
    session: StudySession,
    file_id: str,
    model: str = "gpt-4o"
) -> KnowledgeGaps:
    """Use AI to analyze ratings and identify knowledge gaps"""
    # Format session data for AI
    flashcard_ratings = []
    for rating in session.ratings:
        fc = session.flashcards[rating.flashcard_index]
        flashcard_ratings.append(
            f"Card {rating.flashcard_index+1} (Difficulty: {rating.difficulty}/5): "
            f"Q: {fc.question} | A: {fc.answer}"
        )
    
    ratings_text = "\n".join(flashcard_ratings)
    
    print("Analyzing your knowledge gaps...")
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": """You are an expert learning analyst. Analyze student performance on flashcards
                to identify knowledge gaps and recommend improvements. Be specific and actionable."""
            },
            {
                "role": "user",
                "content": f"""Analyze these flashcard ratings:

{ratings_text}

Identify:
1. Strong areas (concepts rated 1-2, they've mastered)
2. Weak areas (concepts rated 3-4, need improvement) 
3. Critical knowledge gaps (rated 5, major misconceptions or missing prerequisites)
4. Which mastered cards (rated 1) could be safely removed
5. What new flashcards should be generated to fill gaps (be specific about concepts)

Provide actionable recommendations with clear reasoning."""
            }
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "knowledge_gaps",
                "schema": to_strict_json_schema(KnowledgeGaps),
                "strict": True
            }
        }
    )
    
    return KnowledgeGaps.model_validate_json(response.choices[0].message.content)

# Generate gap-filling flashcards
def generate_gap_filling_cards(
    gaps: KnowledgeGaps,
    file_id: str,
    model: str = "gpt-4o"
) -> list[Flashcard]:
    """Generate new flashcards specifically for identified gaps"""
    if not gaps.critical_gaps and not gaps.weak_areas:
        logging.info("No gaps identified, skipping card generation")
        return []
    
    # Build prompt for specific gap-filling cards
    gap_summary = "\n".join([
        f"- {gap}" for gap in gaps.critical_gaps
    ])
    
    if gaps.weak_areas:
        gap_summary += "\n" + "\n".join([
            f"- {area}" for area in gaps.weak_areas
        ])
    
    print("Generating gap-filling flashcards...")
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": """You are an expert at creating targeted flashcards to help students 
                fill specific knowledge gaps. Create focused cards that address the identified gaps."""
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "file",
                        "file": {"file_id": file_id}
                    },
                    {
                        "type": "text",
                        "text": f"""Generate flashcards that specifically address these knowledge gaps:

{gap_summary}

Create focused flashcards that:
- Address the exact gaps identified
- Use simpler language if student struggled
- Provide step-by-step breakdowns for complex concepts
- Include examples where helpful
- Are designed to scaffold learning (build up from basics)

Generate approximately 5-8 flashcards tailored to these gaps."""
                    }
                ]
            }
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "flashcard_set",
                "schema": to_strict_json_schema(FlashcardSet),
                "strict": True
            }
        }
    )
    
    new_flashcards = FlashcardSet.model_validate_json(response.choices[0].message.content)
    return new_flashcards.flashcards

# Adaptive update
def adaptive_update_flashcards(
    original: FlashcardSet,
    session: StudySession,
    gaps: KnowledgeGaps,
    file_id: str,
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
    new_cards = generate_gap_filling_cards(gaps, file_id, model)
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

# Print adaptive summary
def print_adaptive_summary(update: AdaptiveUpdate):
    """Print summary of adaptive changes"""
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

# Main workflow
def create_flashcards(file_path: str, deck_name: str = "Generated Flashcards", 
                     model: str = "gpt-4o", max_iterations: int = 2, 
                     keep_file: bool = False, enable_study_session: bool = False):
    print(f"\n{'='*60}")
    print(f"Starting flashcard generation for: {file_path}")
    print(f"Using model: {model}")
    print(f"Max iterations: {max_iterations}")
    print(f"{'='*60}\n")
    
    logging.info(f"Starting flashcard generation for: {file_path}")
    logging.info(f"Using model: {model}, max_iterations: {max_iterations}")
    
    file_id = None
    try:
        # Generate initial flashcards
        file_id = upload_pdf(file_path)
        flashcards = generate_flashcards_with_file_id(file_id, model)
        
        print(f"✓ Generated {len(flashcards.flashcards)} flashcards\n")
        logging.info(f"Generated {len(flashcards.flashcards)} initial flashcards")
        
        # Log initial flashcards
        logging.debug("Initial flashcards:")
        for i, fc in enumerate(flashcards.flashcards):
            logging.debug(f"  {i+1}. Q: {fc.question} | A: {fc.answer}")
        
        # Critique and revise loop
        for i in range(max_iterations):
            print(f"Iteration {i+1}/{max_iterations}:")
            logging.info(f"Iteration {i+1}/{max_iterations}")
            critique = critique_flashcards(flashcards, model)
            
            if critique.is_acceptable:
                print("✓ Flashcards approved!\n")
                logging.info("Flashcards approved - no revision needed")
                break
            
            print(f"⚠ Issues found: {', '.join(critique.issues)}")
            logging.warning(f"Issues found: {', '.join(critique.issues)}")
            logging.info(f"Critique feedback: {critique.feedback}")
            
            flashcards = revise_flashcards(flashcards, critique, model)
            
            # Log revised flashcards
            logging.info(f"Revised to {len(flashcards.flashcards)} flashcards")
            logging.debug("Revised flashcards:")
            for j, fc in enumerate(flashcards.flashcards):
                logging.debug(f"  {j+1}. Q: {fc.question} | A: {fc.answer}")
            
            print()
        
        # Store original flashcards
        original_flashcards = flashcards
        
        # Only export directly if study session not enabled
        if not enable_study_session:
            output_file = "output.apkg"
            export_to_anki(flashcards, deck_name, output_file)
            
            print(f"\nTo import into Anki:")
            print(f"1. Open Anki")
            print(f"2. File → Import")
            print(f"3. Select {output_file}")
            
            # Also save as text file in Question|Answer format
            text_file = "flashcards.txt"
            with open(text_file, "w", encoding="utf-8") as f:
                for fc in flashcards.flashcards:
                    f.write(f"{fc.question}|{fc.answer}\n")
            print(f"✓ Text format saved to: {text_file}")
            
            print(f"\n{'='*60}")
            print(f"Done! Created {len(flashcards.flashcards)} flashcards")
            print(f"{'='*60}\n")
            
            logging.info(f"Completed! Created {len(flashcards.flashcards)} flashcards")
            logging.info(f"Log file saved to: {log_file}")
        else:
            # Study session mode - ask user
            print("\n" + "="*60)
            response = input("Would you like to start a study session? (y/n): ").lower()
            
            if response == 'y':
                # Conduct session
                session = conduct_study_session(original_flashcards)
                logging.info(f"Study session completed: {len(session.ratings)} ratings collected")
                
                # Analyze gaps
                gaps = analyze_knowledge_gaps(session, file_id, model)
                logging.info(f"Knowledge gaps analyzed: {len(gaps.weak_areas)} weak areas identified")
                
                # Adaptive update
                adaptive_result = adaptive_update_flashcards(
                    original_flashcards, session, gaps, file_id, model
                )
                
                # Export adaptive deck
                export_to_anki(
                    adaptive_result.final_flashcards,
                    deck_name + " (Adaptive)",
                    "output.apkg"
                )
                
                # Save gap report
                with open("knowledge_gaps_report.txt", "w", encoding="utf-8") as f:
                    f.write(adaptive_result.gap_report)
                
                # Display summary
                print_adaptive_summary(adaptive_result)
                
                # Also save adaptive deck as text
                text_file = "flashcards.txt"
                with open(text_file, "w", encoding="utf-8") as f:
                    for fc in adaptive_result.final_flashcards.flashcards:
                        f.write(f"{fc.question}|{fc.answer}\n")
                print(f"✓ Text format saved to: {text_file}")
                print(f"✓ Gap report saved to: knowledge_gaps_report.txt")
                
                print(f"\n{'='*60}")
                print(f"Done! Created adaptive deck with {len(adaptive_result.final_flashcards.flashcards)} flashcards")
                print(f"{'='*60}\n")
                
                logging.info(f"Adaptive deck created: {len(adaptive_result.final_flashcards.flashcards)} cards")
                logging.info(f"Log file saved to: {log_file}")
                
                return adaptive_result.final_flashcards
            else:
                # User declined, export original
                output_file = "output.apkg"
                export_to_anki(original_flashcards, deck_name, output_file)
                
                text_file = "flashcards.txt"
                with open(text_file, "w", encoding="utf-8") as f:
                    for fc in original_flashcards.flashcards:
                        f.write(f"{fc.question}|{fc.answer}\n")
                print(f"✓ Exported {len(original_flashcards.flashcards)} flashcards to {output_file}")
                print(f"✓ Text format saved to: {text_file}")
        
        logging.info(f"Log file saved to: {log_file}")
        return flashcards
    
    finally:
        # Clean up uploaded file unless user wants to keep it
        if file_id and not keep_file:
            cleanup_file(file_id)

# Usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Anki flashcards from PDF documents using AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py lecture_notes.pdf
  python main.py lecture_notes.pdf --deck "Biology 101"
  python main.py lecture_notes.pdf --model gpt-4o-mini --iterations 3
  python main.py lecture_notes.pdf --verbose
  python main.py 03-GNN1.pdf --deck "Graph Neural Networks" --model gpt-4o --iterations 1

        """
    )
    
    parser.add_argument(
        "pdf_file",
        help="Path to the PDF file to generate flashcards from"
    )
    
    parser.add_argument(
        "--deck",
        default="Generated Flashcards",
        help="Name of the Anki deck (default: Generated Flashcards)"
    )
    
    parser.add_argument(
        "--model",
        default="gpt-4o",
        choices=["gpt-4o", "gpt-4o-mini", "o1"],
        help="OpenAI model to use (default: gpt-4o)"
    )
    
    parser.add_argument(
        "--iterations",
        type=int,
        default=2,
        help="Maximum number of critique/revision iterations (default: 2)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging (show all flashcards in log)"
    )
    
    parser.add_argument(
        "--keep-file",
        action="store_true",
        help="Keep uploaded file on OpenAI servers (default: delete after use)"
    )
    
    parser.add_argument(
        "--study-session",
        action="store_true",
        help="Enable interactive study session with adaptive learning"
    )
    
    args = parser.parse_args()
    
    # Set logging level based on verbose flag
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.info("Verbose mode enabled - all flashcards will be logged")
    
    if not os.path.exists(args.pdf_file):
        print(f"Error: File not found: {args.pdf_file}")
        logging.error(f"File not found: {args.pdf_file}")
        exit(1)
    
    print(f"Log file: {log_file}")
    
    create_flashcards(args.pdf_file, args.deck, args.model, args.iterations, args.keep_file, args.study_session)