"""
Main CLI application for Flashcard Generation Agent.
Orchestrates the workflow and handles command-line argument parsing.
"""

import os
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Import models
from models import FlashcardSet

# Import functions from modular files
from openai_client import (
    prepare_input,
    generate_flashcards,
    critique_flashcards,
    revise_flashcards,
    analyze_knowledge_gaps,
    cleanup_file,
)
from anki_exporter import export_to_anki, save_flashcards_text
from study_session import (
    conduct_study_session,
    adaptive_update_flashcards,
    print_adaptive_summary,
)

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


def create_flashcards(
    file_path: str,
    deck_name: str = "Generated Flashcards",
    model: str = "gpt-4o",
    max_iterations: int = 2,
    keep_file: bool = False,
    enable_study_session: bool = False
):
    """
    Main workflow for creating flashcards from a PDF or text file.
    
    Args:
        file_path: Path to the PDF or text file
        deck_name: Name for the Anki deck
        model: OpenAI model to use
        max_iterations: Maximum critique/revision iterations
        keep_file: Whether to keep uploaded file on OpenAI servers (only for PDFs)
        enable_study_session: Whether to enable interactive study session
    """
    print(f"\n{'='*60}")
    print(f"Starting flashcard generation for: {file_path}")
    print(f"Using model: {model}")
    print(f"Max iterations: {max_iterations}")
    print(f"{'='*60}\n")
    
    logging.info(f"Starting flashcard generation for: {file_path}")
    logging.info(f"Using model: {model}, max_iterations: {max_iterations}")
    
    file_id = None
    text_content = None
    try:
        # Prepare input (upload PDF or read text file)
        file_id, text_content = prepare_input(file_path)
        flashcards = generate_flashcards(file_id=file_id, text_content=text_content, model=model)
        
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
            
            # Also save as text file
            save_flashcards_text(flashcards, "flashcards.txt")
            
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
                gaps = analyze_knowledge_gaps(session, file_id=file_id, text_content=text_content, model=model)
                logging.info(f"Knowledge gaps analyzed: {len(gaps.weak_areas)} weak areas identified")
                
                # Adaptive update
                adaptive_result = adaptive_update_flashcards(
                    original_flashcards, session, gaps, file_id, text_content, model
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
                save_flashcards_text(adaptive_result.final_flashcards, "flashcards.txt")
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
                save_flashcards_text(original_flashcards, "flashcards.txt")
                print(f"✓ Exported {len(original_flashcards.flashcards)} flashcards to {output_file}")
        
        logging.info(f"Log file saved to: {log_file}")
        return flashcards
    
    finally:
        # Clean up uploaded file unless user wants to keep it
        if file_id and not keep_file:
            cleanup_file(file_id)


# CLI entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Anki flashcards from PDF or text documents using AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py lecture_notes.pdf
  python main.py lecture_notes.txt
  python main.py lecture_notes.pdf --deck "Biology 101"
  python main.py transcript.txt --model gpt-4o-mini --iterations 3
  python main.py lecture_notes.pdf --verbose
  python main.py 03-GNN1.pdf --deck "Graph Neural Networks" --model gpt-4o --iterations 1
        """
    )
    
    parser.add_argument(
        "input_file",
        help="Path to the PDF or text file (.pdf, .txt, .text) to generate flashcards from"
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
    
    if not os.path.exists(args.input_file):
        print(f"Error: File not found: {args.input_file}")
        logging.error(f"File not found: {args.input_file}")
        exit(1)
    
    print(f"Log file: {log_file}")
    
    create_flashcards(
        args.input_file,
        args.deck,
        args.model,
        args.iterations,
        args.keep_file,
        args.study_session
    )