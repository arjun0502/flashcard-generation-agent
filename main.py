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

# Main workflow
def create_flashcards(file_path: str, deck_name: str = "Generated Flashcards", 
                     model: str = "gpt-4o", max_iterations: int = 2, 
                     keep_file: bool = False):
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
        
        # Export to Anki package file
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
    
    create_flashcards(args.pdf_file, args.deck, args.model, args.iterations, args.keep_file)