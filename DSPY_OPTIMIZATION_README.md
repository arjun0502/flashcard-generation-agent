also what # DSPy Prompt Optimization Guide

This guide explains how to use DSPy to optimize prompts for the flashcard generation system using **metric functions** (no training labels needed!).

## Overview

DSPy optimization is implemented for two main areas:
1. **Critique/Revision**: Optimizing the critique step (revise depends on critique)
2. **Adaptation**: Optimizing the generation of gap-filling flashcards

Both use **metric functions** that evaluate quality using your existing evaluation system.

## Setup

1. Install DSPy:
```bash
pip install dspy-ai>=2.4.0
```

2. (Optional) Set environment variable to enable optimized prompts:
```bash
export USE_DSPY_OPTIMIZATION=true
```

## Files

- `dspy_modules.py`: DSPy module definitions (CritiqueModule, AdaptationModule)
- `dspy_metrics.py`: Metric functions that use your evaluation system
- `optimize_prompts.py`: Main script to run optimization
- `dspy_integration.py`: Integration layer to use optimized prompts

## How It Works

### Metric-Based Optimization (No Training Labels!)

Instead of requiring labeled training data, we use **metric functions** that:

1. **For Critique**: 
   - Run critique → revise → `evaluate_flashcard_set()`
   - Score based on `overall_deck_score` improvement (average of 4 metrics)
   
2. **For Adaptation**:
   - Generate gap-filling cards → create AdaptiveUpdate → `evaluate_adaptation()`
   - Score based on `overall_personalization` (gap-filling + removal effectiveness)

### Optimization Process

```
Test Examples (no labels needed!)
    │
    ▼
DSPy Module (current prompt)
    │
    ▼
Run Module → Get Output
    │
    ▼
Execute Full Pipeline (revise/evaluate)
    │
    ▼
Compute Metric Score
    │
    ▼
MIPRO: Try different prompts → Repeat → Keep best
```

## Usage

### Step 1: Run Optimization

First, make sure you have evaluation data from running your flashcard generation:

```bash
# Optimize both modules
python optimize_prompts.py --module both --eval-data-dir evaluation_data

# Or optimize separately
python optimize_prompts.py --module critique --eval-data-dir evaluation_data
python optimize_prompts.py --module adaptation --eval-data-dir evaluation_data
```

**Options:**
- `--module`: `critique`, `adaptation`, or `both` (default: `both`)
- `--eval-data-dir`: Directory with evaluation data (default: `evaluation_data`)
- `--model`: Model to use (default: `gpt-4o`)
- `--num-candidates`: Number of prompt variations to try (default: 10)
- `--max-examples`: Max examples to use (default: 20)

### Step 2: Use Optimized Prompts

**Option A: Use wrapper functions**

```python
from dspy_integration import (
    critique_flashcards_wrapper,
    generate_gap_filling_cards_wrapper
)

# Use optimized critique
critique = critique_flashcards_wrapper(flashcard_set, use_optimized=True)

# Use optimized adaptation
new_cards = generate_gap_filling_cards_wrapper(
    knowledge_gaps, 
    file_id=file_id, 
    use_optimized=True
)
```

**Option B: Set environment variable**

```bash
export USE_DSPY_OPTIMIZATION=true
```

Then the wrapper functions will automatically use optimized prompts.

**Option C: Integrate directly**

Modify `openai_client.py` to use optimized prompts from the DSPy modules.

## Metric Functions

### Critique Metric

Uses `evaluate_flashcard_set().overall_deck_score`:
- If critique says "acceptable": Check if original score >= 7.0
- If critique says "needs revision": Score based on improvement after revision
- Returns: 0-1 score (normalized from 0-10 scale)

### Adaptation Metric

Uses `evaluate_adaptation().overall_personalization`:
- Generates gap-filling cards
- Creates AdaptiveUpdate object
- Evaluates using `evaluate_adaptation()`
- Returns: 0-1 score (normalized from 0-10 scale)

## Data Requirements

### For Critique Optimization

Needs: `flashcards_initial.json` files in `evaluation_data/` subdirectories

### For Adaptation Optimization

Needs (all in same subdirectory):
- `knowledge_gaps.json`
- `study_session.json`
- `flashcards_revised.json` (original before adaptation)
- `source_text.txt` (optional)

## Example Workflow

```bash
# 1. Generate some flashcards (creates evaluation_data)
python main.py lecture.pdf --iterations 2

# 2. Run optimization
python optimize_prompts.py --module both

# 3. Test optimized prompts
export USE_DSPY_OPTIMIZATION=true
python main.py lecture2.pdf  # Uses optimized prompts
```

## How Metrics Work

### Critique Metric Flow

```
Input: FlashcardSet
    │
    ▼
DSPy Critique Module (optimized prompt)
    │
    ▼
Critique Output
    │
    ▼
revise_flashcards() [existing function]
    │
    ▼
Revised Flashcards
    │
    ▼
evaluate_flashcard_set() [existing function]
    │
    ▼
overall_deck_score (avg of 4 metrics: atomicity, clarity, learning_value, accuracy)
    │
    ▼
Metric Score: improvement in overall_deck_score
```

### Adaptation Metric Flow

```
Input: KnowledgeGaps + Source Material
    │
    ▼
DSPy Adaptation Module (optimized prompt)
    │
    ▼
New Flashcards (JSON)
    │
    ▼
Create AdaptiveUpdate [existing logic]
    │
    ▼
evaluate_adaptation() [existing function]
    │
    ▼
overall_personalization (gap-filling + removal)
    │
    ▼
Metric Score: overall_personalization / 10.0
```

## Advantages

✅ **No training labels needed** - Metrics compute scores automatically  
✅ **Uses your existing evaluation** - Leverages `evaluate_flashcard_set()` and `evaluate_adaptation()`  
✅ **Optimizes for actual quality** - Not just matching labels, but improving real metrics  
✅ **Can improve over time** - Add more test examples as you generate more flashcards  

## Troubleshooting

**"No examples found"**
- Make sure you've run flashcard generation first to create `evaluation_data/`
- For adaptation, you need study sessions (run with `--study-session` flag)

**"MIPRO not available"**
- Falls back to BootstrapFewShot automatically
- Install latest dspy-ai for MIPRO support

**"Error parsing DSPy result"**
- Falls back to original functions automatically
- Check that DSPy module outputs match expected format

## Next Steps

1. Run optimization on your evaluation data
2. Compare optimized vs original prompts
3. Integrate optimized prompts into production
4. Periodically re-optimize as you collect more data

