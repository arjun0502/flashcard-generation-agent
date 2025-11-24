"""
Standalone evaluation script for flashcard quality assessment.
Loads saved flashcard sets and runs LLM-as-a-judge evaluations.
"""

import argparse
import json
import logging
from pathlib import Path

from models import (
    FlashcardSet,
    StudySession,
    KnowledgeGaps,
    AdaptiveUpdate,
    DeckEvaluation,
    AdaptationEvaluation,
)
from evaluator import evaluate_flashcard_set, evaluate_adaptation


def load_json_file(file_path: Path, model_class):
    """Load and parse a JSON file into a Pydantic model."""
    if not file_path.exists():
        return None
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return model_class.model_validate(data)


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(
        description="Evaluate flashcard sets using LLM-as-a-judge",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluate.py evaluation_data/20241101_120000
  python evaluate.py evaluation_data/20241101_120000 --model gpt-4o-mini
  python evaluate.py evaluation_data/20241101_120000 --stages initial revised adapted
        """
    )
    
    parser.add_argument(
        "eval_data_dir",
        type=str,
        help="Path to evaluation data directory (timestamped subdirectory in evaluation_data/)"
    )
    
    parser.add_argument(
        "--model",
        default="gpt-4o",
        choices=["gpt-4o", "gpt-4o-mini", "o1"],
        help="OpenAI model to use for evaluation (default: gpt-4o)"
    )
    
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=["initial", "revised", "adapted"],
        default=["initial", "revised", "adapted"],
        help="Which stages to evaluate (default: all stages)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluations",
        help="Directory to save evaluation results (default: evaluations/)"
    )
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Resolve evaluation data directory
    eval_data_dir = Path(args.eval_data_dir)
    if not eval_data_dir.exists():
        print(f"Error: Evaluation data directory not found: {eval_data_dir}")
        return 1
    
    # Load metadata
    metadata_path = eval_data_dir / "evaluation_metadata.json"
    if not metadata_path.exists():
        print(f"Error: Metadata file not found: {metadata_path}")
        return 1
    
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    
    print(f"\nEvaluating flashcard sets from: {eval_data_dir}")
    print(f"Source: {metadata.get('source_file', 'Unknown')}")
    print(f"Model: {args.model}\n")
    
    # Load source material
    file_id = metadata.get("file_id")
    text_content = None
    if metadata.get("has_text_content"):
        text_content_path = eval_data_dir / "source_text.txt"
        if text_content_path.exists():
            with open(text_content_path, "r", encoding="utf-8") as f:
                text_content = f.read()
    
    if file_id and not text_content:
        print("Warning: PDF file_id found but cannot be used for evaluation.")
        print("Please ensure source_text.txt exists in the evaluation data directory.")
        file_id = None
    
    # Evaluate each stage
    initial_eval = None
    revised_eval = None
    adapted_eval = None
    adaptation_eval = None
    
    if "initial" in args.stages:
        initial_path = eval_data_dir / "flashcards_initial.json"
        initial_flashcards = load_json_file(initial_path, FlashcardSet)
        if initial_flashcards:
            print("Evaluating Stage 1: Initial Generation...")
            initial_eval = evaluate_flashcard_set(
                initial_flashcards,
                file_id=file_id,
                text_content=text_content,
                stage_name="Initial Generation",
                model=args.model
            )
        else:
            print(f"Warning: {initial_path} not found, skipping initial evaluation")
    
    if "revised" in args.stages:
        revised_path = eval_data_dir / "flashcards_revised.json"
        revised_flashcards = load_json_file(revised_path, FlashcardSet)
        if revised_flashcards:
            print("Evaluating Stage 2: After Critique + Revision...")
            revised_eval = evaluate_flashcard_set(
                revised_flashcards,
                file_id=file_id,
                text_content=text_content,
                stage_name="After Critique + Revision",
                model=args.model
            )
        else:
            print(f"Warning: {revised_path} not found, skipping revised evaluation")
    
    if "adapted" in args.stages:
        adapted_path = eval_data_dir / "flashcards_adapted.json"
        adapted_flashcards = load_json_file(adapted_path, FlashcardSet)
        if adapted_flashcards:
            print("Evaluating Stage 3: After Adaptation...")
            print("Note: Stage 3 focuses on personalization, not quality improvement.")
            print("Evaluating personalization effectiveness only (not quality metrics)...")
            
            # Stage 3: Only evaluate personalization (not quality metrics)
            # Load required data for adaptation evaluation
            study_session_path = eval_data_dir / "study_session.json"
            knowledge_gaps_path = eval_data_dir / "knowledge_gaps.json"
            adaptive_update_path = eval_data_dir / "adaptive_update.json"
            revised_path = eval_data_dir / "flashcards_revised.json"
            
            study_session = load_json_file(study_session_path, StudySession)
            knowledge_gaps = load_json_file(knowledge_gaps_path, KnowledgeGaps)
            adaptive_update = load_json_file(adaptive_update_path, AdaptiveUpdate)
            original_revised = load_json_file(revised_path, FlashcardSet)
            
            # Don't evaluate quality metrics for adapted stage - it's about personalization
            adapted_eval = None
            
            if study_session and knowledge_gaps and adaptive_update and original_revised:
                print("Evaluating Adaptation Effectiveness (Personalization)...")
                adaptation_eval = evaluate_adaptation(
                    original_revised,
                    adaptive_update,
                    knowledge_gaps,
                    study_session,
                    file_id=file_id,
                    text_content=text_content,
                    model=args.model
                )
            else:
                print("Warning: Missing data for adaptation effectiveness evaluation")
                adaptation_eval = None
        else:
            print(f"Warning: {adapted_path} not found, skipping adapted evaluation")
            adapted_eval = None
            adaptation_eval = None
    
    # Save evaluation results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    timestamp = eval_data_dir.name
    eval_output_dir = output_dir / f"{timestamp}_eval"
    eval_output_dir.mkdir(exist_ok=True)
    
    # Save individual evaluations as JSON
    if initial_eval:
        with open(eval_output_dir / "evaluation_initial.json", "w", encoding="utf-8") as f:
            json.dump(initial_eval.model_dump(), f, indent=2)
        print(f"\nStage 1 Results:")
        if initial_eval.average_scores:
            for criterion, score in initial_eval.average_scores.items():
                print(f"  {criterion.replace('_', ' ').title()}: {score:.2f}/10")
    
    if revised_eval:
        with open(eval_output_dir / "evaluation_revised.json", "w", encoding="utf-8") as f:
            json.dump(revised_eval.model_dump(), f, indent=2)
        print(f"\nStage 2 Results:")
        if revised_eval.average_scores:
            for criterion, score in revised_eval.average_scores.items():
                print(f"  {criterion.replace('_', ' ').title()}: {score:.2f}/10")
        if revised_eval.overall_deck_score is not None:
            print(f"  Overall Deck Score: {revised_eval.overall_deck_score:.2f}/10")
    
    # Stage 3: Only report personalization metrics (not quality metrics)
    if adaptation_eval:
        with open(eval_output_dir / "evaluation_adaptation.json", "w", encoding="utf-8") as f:
            json.dump(adaptation_eval.model_dump(), f, indent=2)
        
        print(f"\nStage 3 Results (Personalization):")
        print(f"  Overall Personalization Score: {adaptation_eval.overall_personalization:.2f}/10")
        print(f"  Gap-Filling Effectiveness: {adaptation_eval.average_gap_personalization:.2f}/10 (how well new cards address student gaps)")
        print(f"  Removal Appropriateness: {adaptation_eval.average_removal_personalization:.2f}/10 (whether removals were correct)")
        print(f"\n  Note: Stage 3 evaluates ONLY personalization (gap-filling and removal appropriateness).")
        print(f"        Quality metrics (atomicity, clarity, learning_value, accuracy) are NOT evaluated for this stage.")
    
    print(f"\n✓ Evaluation complete! Results saved to: {eval_output_dir}")
    
    return 0


if __name__ == "__main__":
    exit(main())
