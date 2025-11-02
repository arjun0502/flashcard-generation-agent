# AI Flashcard Generator

Generate Anki flashcards from PDF documents using OpenAI's API with automatic critique and revision.

## Features

- 📄 Upload PDF lecture notes, slides, or transcripts
- 🤖 AI-powered flashcard generation using GPT-4o
- 🔍 Automatic critique based on pedagogical principles
- ✏️ Iterative revision for quality improvement
- 🎯 Interactive study session with adaptive learning
- 📊 Knowledge gap analysis and personalized deck creation
- 📦 Export to Anki (.apkg format)
- 💾 Also exports as text file (Question|Answer format)
- 📝 Detailed logging of generation and study session

## Project Structure

```
flashcard-generator/
├── main.py              # Main application
├── requirements.txt     # Python dependencies
├── .env                 # API keys (create this)
├── lecture_notes.pdf    # Your input PDF
├── output.apkg          # Generated Anki deck
└── flashcards.txt       # Text format output
```

## Setup

### 1. Clone or create project directory

```bash
cd flashcard-generator
```

### 2. Create virtual environment (recommended)

```bash
python -m venv venv

# Activate it
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up OpenAI API key

Get your API key from: https://platform.openai.com/api-keys

```bash
# Linux
export OPENAI_API_KEY='your-api-key-here'

# Windows (Powershell)
$env:OPENAI_API_KEY="your-api-key-here"
```

### 5. Add your PDF file

Place your PDF file in the project directory (e.g., `lecture_notes.pdf`)

## Usage

```bash
# Basic usage
python main.py lecture_notes.pdf

# With custom deck name
python main.py lecture_notes.pdf --deck "Biology 101 - Cell Structure"

# With verbose logging (includes all flashcards in log)
python main.py lecture_notes.pdf --verbose

# Use a cheaper/faster model
python main.py lecture_notes.pdf --model gpt-4o-mini

# Customize number of iterations
python main.py lecture_notes.pdf --iterations 3

# Keep uploaded file on OpenAI servers
python main.py lecture_notes.pdf --keep-file

# Interactive study session with adaptive learning
python main.py lecture_notes.pdf --study-session
```

### Command Line Options

- `pdf_file` - Path to the PDF file (required)
- `--deck` - Name of the Anki deck (default: "Generated Flashcards")
- `--model` - OpenAI model to use: gpt-4o, gpt-4o-mini, or o1 (default: gpt-4o)
- `--iterations` - Maximum number of critique/revision iterations (default: 2)
- `--verbose` - Enable verbose logging with all flashcards shown in log
- `--keep-file` - Keep uploaded file on OpenAI servers (default: delete after use)
- `--study-session` - Enable interactive study session with adaptive learning

## Output Files

After running, you'll get:

- `output.apkg` - Anki package file (import into Anki)
- `flashcards.txt` - Text format with Question|Answer pairs
- `logs/flashcard_generation_TIMESTAMP.log` - Detailed log of the generation process including critiques and revisions
- `knowledge_gaps_report.txt` - Gap analysis report (only when using `--study-session`)

## Import into Anki

1. Open Anki desktop application
2. File → Import
3. Select `output.apkg`
4. Your flashcards will appear in the specified deck!

## How It Works

1. **PDF Upload**: Uploads PDF to OpenAI's Files API and gets a file ID
2. **Generation**: GPT-4o extracts key concepts and creates flashcards
3. **Critique**: Evaluates flashcards against pedagogical principles:
   - Atomicity (one concept per card)
   - Clarity (unambiguous questions/answers)
   - Appropriate difficulty
   - No yes/no questions
   - Necessary context included
4. **Revision**: If issues found, revises flashcards (up to 2 iterations)
5. **Export**: Creates Anki package with all flashcards
6. **Cleanup**: Deletes uploaded file (unless `--keep-file` is used)

## Interactive Study Session (Optional)

When using `--study-session`, you can interactively rate flashcards and get a personalized adaptive deck:

1. **Study Session**: Rate each flashcard on difficulty (1-5)
2. **Gap Analysis**: AI identifies what you know well, weak areas, and critical gaps
3. **Adaptive Deck**: 
   - Removes cards you've mastered (rated 1)
   - Adds targeted gap-filling cards for concepts you struggled with
   - Creates personalized deck based on your knowledge level

### Example Study Session Flow

```
Would you like to start a study session? (y/n): y

============================================================
STUDY SESSION
============================================================

Flashcard 1 of 15
────────────────────────────────────────────────────────────

QUESTION:
What is the intuition behind node embeddings?

[Press Enter to see answer]

ANSWER:
Map nodes to d-dimensional embeddings so that similar 
nodes are embedded close together.

Enter difficulty rating (1-5): 2

[Continues through all flashcards...]

============================================================
STUDY SESSION COMPLETE
============================================================

Analyzing your knowledge gaps...

============================================================
KNOWLEDGE GAPS IDENTIFIED
============================================================

Your Strong Areas:
✓ Node embeddings basics
✓ Graph Neural Network fundamentals

Areas Needing Improvement:
⚠ Graph Convolutional Networks mechanics

Critical Gap Found:
🔴 Inductive capabilities (rated 5/5)

DECK CHANGES
────────────────────────────────────────────────────────────
Original cards: 15
Cards removed (mastered): 3
Cards added (gap-filling): 5
Final cards: 17
```

## genanki Best Practices (from README)

This implementation follows genanki's official guidelines:

### 1. Hardcoded IDs

**✅ Correct (what we do):**
```python
FLASHCARD_MODEL_ID = 1607392319  # Generated once, hardcoded
FLASHCARD_DECK_ID = 2059400110   # Generated once, hardcoded

deck = genanki.Deck(FLASHCARD_DECK_ID, deck_name)
```

**❌ Incorrect (what NOT to do):**
```python
deck_id = random.randrange(1 << 30, 1 << 31)  # DON'T generate each time!
deck = genanki.Deck(deck_id, deck_name)
```

**Why?** Stable IDs allow Anki to track your model/deck across re-imports. Without stable IDs, each import creates a new deck instead of updating the existing one.

### 2. HTML Escaping

**✅ Correct (what we do):**
```python
question_escaped = html.escape(fc.question)
answer_escaped = html.escape(fc.answer)
note = genanki.Note(model=FLASHCARD_MODEL, fields=[question_escaped, answer_escaped])
```

**Why?** Field content is HTML, not plain text. Special characters like `<`, `>`, and `&` need to be escaped to display properly.

### 3. Model Definition

Our model is defined once at the module level and reused:
```python
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
```

## Cost Estimation

Typical costs per PDF (using GPT-4o):
- Small PDF (10 pages): ~$0.05-0.15
- Medium PDF (30 pages): ~$0.15-0.40
- Large PDF (100 pages): ~$0.50-1.50

Note: PDFs with many images cost more due to vision processing.

## Logging

All generation runs are automatically logged to timestamped files in the `logs/` directory.

### Log Levels

**INFO (default):** Shows workflow steps, critiques, and issue summaries
- When generation starts/completes
- Number of flashcards generated/revised
- Critique feedback and issues found
- Log file location

**DEBUG (--verbose flag):** Includes everything above plus:
- Complete Q&A text for all flashcards (initial and revised)
- Full conversation flow

### Example Log Output

```
2025-01-XX XX:XX:XX - INFO - Starting flashcard generation for: lecture.pdf
2025-01-XX XX:XX:XX - INFO - Using model: gpt-4o, max_iterations: 2
2025-01-XX XX:XX:XX - INFO - Generated 15 initial flashcards
2025-01-XX XX:XX:XX - INFO - Iteration 1/2
2025-01-XX XX:XX:XX - WARNING - Issues found: Yes/No questions, Missing context
2025-01-XX XX:XX:XX - INFO - Critique feedback: Several flashcards use yes/no format...
2025-01-XX XX:XX:XX - INFO - Revised to 15 flashcards
2025-01-XX XX:XX:XX - INFO - Completed! Created 15 flashcards
```

Use `--verbose` to see all the intermediate flashcards and detailed critique feedback.
```

## Acknowledgments

- [genanki](https://github.com/kerrickstaley/genanki) - Anki deck generation library
- [OpenAI](https://openai.com) - GPT-4o API
- [Anki](https://apps.ankiweb.net/) - Spaced repetition software