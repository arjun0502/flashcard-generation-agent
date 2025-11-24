"""
Script to optimize prompts using DSPy with metric functions.

This script:
1. Loads test examples from evaluation_data
2. Optimizes critique module using critique_metric
3. Optimizes adaptation module using adaptation_metric
4. Saves optimized prompts
"""

import argparse
import json
from pathlib import Path
from typing import List

try:
    import dspy
except ImportError:
    print("Error: dspy-ai not installed. Install with: pip install dspy-ai")
    exit(1)

from models import FlashcardSet, KnowledgeGaps, StudySession
from dspy_modules import CritiqueModule, AdaptationModule
from dspy_metrics import critique_metric, adaptation_metric


def load_critique_examples(eval_data_dir: Path, max_examples: int = 20) -> List[dspy.Example]:
    """Load flashcard sets from evaluation_data for critique optimization."""
    examples = []
    eval_dir = Path(eval_data_dir)
    
    for subdir in sorted(eval_dir.iterdir()):
        if not subdir.is_dir():
            continue
        
        initial_path = subdir / "flashcards_initial.json"
        if initial_path.exists():
            try:
                with open(initial_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    flashcard_set = FlashcardSet.model_validate(data)
                    
                    # Format flashcard text for DSPy
                    flashcard_text = "\n".join([
                        f"{i+1}. Q: {fc.question} | A: {fc.answer}"
                        for i, fc in enumerate(flashcard_set.flashcards)
                    ])
                    
                    example = dspy.Example(
                        flashcard_set=flashcard_set,
                        flashcard_text=flashcard_text
                    ).with_inputs('flashcard_text')
                    
                    examples.append(example)
                    
                    if len(examples) >= max_examples:
                        break
            except Exception as e:
                print(f"Warning: Could not load {initial_path}: {e}")
                continue
    
    return examples


def load_adaptation_examples(eval_data_dir: Path, max_examples: int = 10) -> List[dspy.Example]:
    """Load adaptation examples from evaluation_data."""
    examples = []
    eval_dir = Path(eval_data_dir)
    
    for subdir in sorted(eval_dir.iterdir()):
        if not subdir.is_dir():
            continue
        
        # Need: knowledge_gaps, study_session, flashcards_revised (original), source_text
        gaps_path = subdir / "knowledge_gaps.json"
        session_path = subdir / "study_session.json"
        original_path = subdir / "flashcards_revised.json"  # Original before adaptation
        source_path = subdir / "source_text.txt"
        
        if not (gaps_path.exists() and session_path.exists() and original_path.exists()):
            continue
        
        try:
            # Load knowledge gaps
            with open(gaps_path, 'r', encoding='utf-8') as f:
                gaps_data = json.load(f)
                knowledge_gaps = KnowledgeGaps.model_validate(gaps_data)
            
            # Load study session
            with open(session_path, 'r', encoding='utf-8') as f:
                session_data = json.load(f)
                study_session = StudySession.model_validate(session_data)
            
            # Load original flashcards
            with open(original_path, 'r', encoding='utf-8') as f:
                original_data = json.load(f)
                original_flashcards = FlashcardSet.model_validate(original_data)
            
            # Load source material (optional)
            source_material = None
            if source_path.exists():
                with open(source_path, 'r', encoding='utf-8') as f:
                    source_material = f.read()
            
            # Format knowledge gaps for DSPy
            gap_summary = "\n".join([
                f"- {gap}" for gap in knowledge_gaps.critical_gaps
            ])
            if knowledge_gaps.weak_areas:
                gap_summary += "\n" + "\n".join([
                    f"- {area}" for area in knowledge_gaps.weak_areas
                ])
            
            # Truncate source material if too long
            if source_material and len(source_material) > 5000:
                source_material = source_material[:5000] + "..."
            
            example = dspy.Example(
                knowledge_gaps=knowledge_gaps,
                source_material=source_material or "",
                study_session=study_session,
                original_flashcards=original_flashcards,
                file_id=None  # For optimization, we'll use text_content
            ).with_inputs('knowledge_gaps', 'source_material')
            
            examples.append(example)
            
            if len(examples) >= max_examples:
                break
                
        except Exception as e:
            print(f"Warning: Could not load adaptation example from {subdir}: {e}")
            continue
    
    return examples


def setup_dspy_lm(model: str = "gpt-4o"):
    """Set up DSPy with OpenAI LM."""
    import os
    from openai import OpenAI
    
    # Create a simple OpenAI LM wrapper for DSPy
    class OpenAILM(dspy.LM):
        def __init__(self, model_name: str):
            super().__init__(model_name)
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.model_name = model_name
        
        def __call__(self, prompt, **kwargs):
            messages = []
            if hasattr(prompt, 'messages'):
                messages = prompt.messages
            elif isinstance(prompt, str):
                messages = [{"role": "user", "content": prompt}]
            elif isinstance(prompt, list):
                messages = prompt
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                **kwargs
            )
            return response.choices[0].message.content
    
    lm = OpenAILM(model)
    dspy.configure(lm=lm)
    return lm


def optimize_critique_module(valset: List[dspy.Example], num_candidates: int = 10):
    """Optimize the critique module."""
    print(f"\n{'='*60}")
    print("Optimizing Critique Module")
    print(f"{'='*60}")
    print(f"Test examples: {len(valset)}")
    
    critique_module = CritiqueModule()
    
    # Use MIPRO for optimization
    try:
        optimizer = dspy.teleprompt.MIPRO(
            metric=critique_metric,
            num_candidates=num_candidates,
            init_temperature=1.0,
        )
        
        print("Running optimization (this may take a while)...")
        optimized_critique = optimizer.compile(
            critique_module,
            trainset=valset,
            valset=valset,
            num_threads=4
        )
        
        print("✓ Critique optimization complete!")
        return optimized_critique
        
    except AttributeError:
        # Fallback to BootstrapFewShot if MIPRO not available
        print("MIPRO not available, using BootstrapFewShot...")
        optimizer = dspy.BootstrapFewShot(
            max_bootstrapped_demos=4,
            max_labeled_demos=16
        )
        
        optimized_critique = optimizer.compile(
            critique_module,
            trainset=valset,
            valset=valset
        )
        
        print("✓ Critique optimization complete!")
        return optimized_critique


def optimize_adaptation_module(valset: List[dspy.Example], num_candidates: int = 10):
    """Optimize the adaptation module."""
    print(f"\n{'='*60}")
    print("Optimizing Adaptation Module")
    print(f"{'='*60}")
    print(f"Test examples: {len(valset)}")
    
    adaptation_module = AdaptationModule()
    
    # Use MIPRO for optimization
    try:
        optimizer = dspy.teleprompt.MIPRO(
            metric=adaptation_metric,
            num_candidates=num_candidates,
            init_temperature=1.0,
        )
        
        print("Running optimization (this may take a while)...")
        optimized_adaptive = optimizer.compile(
            adaptation_module,
            trainset=valset,
            valset=valset,
            num_threads=4
        )
        
        print("✓ Adaptation optimization complete!")
        return optimized_adaptive
        
    except AttributeError:
        # Fallback to BootstrapFewShot if MIPRO not available
        print("MIPRO not available, using BootstrapFewShot...")
        optimizer = dspy.BootstrapFewShot(
            max_bootstrapped_demos=4,
            max_labeled_demos=16
        )
        
        optimized_adaptive = optimizer.compile(
            adaptation_module,
            trainset=valset,
            valset=valset
        )
        
        print("✓ Adaptation optimization complete!")
        return optimized_adaptive


def save_optimized_prompts(module, output_path: Path, module_name: str):
    """Save optimized prompts to file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Extract prompt information from optimized module
    # Note: This is a simplified version - actual DSPy modules store prompts differently
    prompt_info = {
        "module_name": module_name,
        "optimized": True,
        "note": "This module has been optimized by DSPy. Use dspy_integration.py to load and use it."
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(prompt_info, f, indent=2)
    
    print(f"Saved optimization info to: {output_path}")
    print("Note: DSPy modules are Python objects. To use optimized prompts, integrate via dspy_integration.py")


def main():
    parser = argparse.ArgumentParser(description="Optimize prompts using DSPy with metric functions")
    parser.add_argument(
        "--eval-data-dir",
        type=str,
        default="evaluation_data",
        help="Directory containing evaluation data"
    )
    parser.add_argument(
        "--module",
        type=str,
        choices=["critique", "adaptation", "both"],
        default="both",
        help="Which module(s) to optimize"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        choices=["gpt-4o", "gpt-4o-mini"],
        help="OpenAI model to use"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="optimized_prompts",
        help="Directory to save optimized prompts"
    )
    parser.add_argument(
        "--num-candidates",
        type=int,
        default=10,
        help="Number of prompt candidates to try (for MIPRO)"
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=20,
        help="Maximum number of examples to use for optimization"
    )
    
    args = parser.parse_args()
    
    eval_data_dir = Path(args.eval_data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    if not eval_data_dir.exists():
        print(f"Error: Evaluation data directory not found: {eval_data_dir}")
        print("Please run your flashcard generation first to create evaluation data.")
        return 1
    
    # Set up DSPy
    print(f"Setting up DSPy with {args.model}...")
    setup_dspy_lm(model=args.model)
    
    # Optimize critique module
    if args.module in ["critique", "both"]:
        print("\nLoading critique examples...")
        critique_examples = load_critique_examples(eval_data_dir, max_examples=args.max_examples)
        
        if len(critique_examples) == 0:
            print("Warning: No critique examples found. Skipping critique optimization.")
        else:
            optimized_critique = optimize_critique_module(critique_examples, args.num_candidates)
            save_optimized_prompts(
                optimized_critique,
                output_dir / "critique_optimized.json",
                "critique"
            )
    
    # Optimize adaptation module
    if args.module in ["adaptation", "both"]:
        print("\nLoading adaptation examples...")
        adaptation_examples = load_adaptation_examples(eval_data_dir, max_examples=args.max_examples)
        
        if len(adaptation_examples) == 0:
            print("Warning: No adaptation examples found. Skipping adaptation optimization.")
            print("Note: Adaptation examples require knowledge_gaps.json, study_session.json, and flashcards_revised.json")
        else:
            optimized_adaptive = optimize_adaptation_module(adaptation_examples, args.num_candidates)
            save_optimized_prompts(
                optimized_adaptive,
                output_dir / "adaptation_optimized.json",
                "adaptation"
            )
    
    print(f"\n{'='*60}")
    print("✓ Optimization complete!")
    print(f"{'='*60}")
    print(f"\nNext steps:")
    print("1. Review the optimized modules")
    print("2. Use dspy_integration.py to integrate optimized prompts into production")
    print("3. Test the optimized prompts on new data")
    
    return 0


if __name__ == "__main__":
    exit(main())

