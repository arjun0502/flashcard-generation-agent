# Flashcard Generation Agent

The Flashcard Generation Agent automatically generates flashcards from lecture materials, critiques and revises it based on pedagogical principles for long-term learning, and personalizes flashcards based on your knowledge gaps. 

## User Workflow

1. **Provide Lecture Materials**: The user provides a PDF or text file containing lecture notes, slides, transcripts, or study materials
2. **Automatic Flashcard Generation**: The agent analyzes the lecture materials and automatically generates flashcards covering key concepts
3. **Quality Improvement**: The agent critiques the flashcards based on pedagogical principles for effective learning and then revises them accordingly
4. **Optional Personalization**: The user optionally engages in an interactive study session and, as part of this study session, they will rate the difficulty of each flashcard. Based on the user ratings, the agent identifies what the user knows well, areas needing improvement, and critical knowledge gaps. It then adapts the deck by removing cards the user mastered and adding targeted flashcards to fill the user's knowledge gaps
5. **Ready-to-Study Flashcards**: The agent provides the user with a complete flashcard deck ready for import into your study tools


## Technical Design

### OpenAI API Integration

The agent is built on a series of OpenAI API calls that handle different aspects of flashcard generation, quality 
improvement, and personalization. The system supports both PDF and text file inputs:

- **PDF Files**: Uploaded to OpenAI's Files API for processing with vision-capable models (`gpt-4o`, `gpt-4o-mini`, or `o1`). This allows the agent to extract information from both text and diagrams/images in PDFs.
- **Text Files**: Read directly and included in the API call as text content. Perfect for lecture transcripts, plain text notes, or any `.txt` files.

The core API operations include:

- **Flashcard Generation**: Initial flashcard creation from the document content (PDF or text)
- **Critique**: Evaluation of flashcard quality against pedagogical principles for long-term learning (atomicity, clarity, appropriate difficulty, avoiding yes/no questions, ensuring context)
- **Revision**: Improvement of flashcards based on pedagogical feedback
- **Knowledge Gap Analysis**: Analysis of study session ratings to identify learning gaps and generate targeted gap-filling cards

All API calls use OpenAI's structured outputs feature with Pydantic models to ensure consistent response formatting.

### Pydantic

The project uses Pydantic models for structured data validation and OpenAI's structured outputs:

- **Flashcard**: Single flashcard with question and answer fields
- **FlashcardSet**: Collection of flashcards
- **Critique**: AI evaluation of flashcard quality (is_acceptable, feedback, issues list)
- **StudyRating**: User difficulty rating for a flashcard (1-5 scale)
- **StudySession**: Complete study session results (flashcards + ratings)
- **KnowledgeGaps**: AI analysis of student performance with strong areas, weak areas, critical gaps, and recommendations
- **AdaptiveUpdate**: Final result of adaptive learning updates

These models enable type-safe data handling and ensure OpenAI API responses conform to expected schemas using `to_strict_json_schema()` for structured outputs.

### genanki

genanki is a Python 3 library for generating Anki decks programmatically. The agent uses genanki to create `.apkg` package files that can be imported directly into Anki.

Key implementation details:

- **Hardcoded IDs**: Model ID (`FLASHCARD_MODEL_ID`) and deck ID (`FLASHCARD_DECK_ID`) are hardcoded constants rather than randomly generated. This ensures that re-imports update the existing deck rather than creating duplicates.

- **HTML Escaping**: All flashcard field content is HTML-escaped using `html.escape()` before adding to genanki Notes. This is required because genanki fields are HTML, and special characters like `<`, `>`, and `&` must be escaped to display properly.

- **Consistent Model Definition**: The Anki model (note type) is defined once at module level in `config.py` and reused across all exports, ensuring consistent card formatting.

The model uses a simple two-field template (Question/Answer) with a horizontal rule separator in the answer template.

### Codebase Structure

The codebase follows a modular design:

```
flashcard-generation-agent/
├── main.py              # CLI application entry point - orchestrates workflow, command-line argument parsing
├── streamlit_app.py     # Web application entry point - Streamlit-based interactive web interface
├── models.py            # Pydantic models for flashcard data structures (Flashcard, FlashcardSet, Critique, StudySession, KnowledgeGaps, etc.)
├── openai_client.py     # OpenAI API interactions - file upload/reading (PDF/text), flashcard generation, critique, knowledge gap analysis
├── study_session.py     # Interactive study session and adaptive learning functionality
├── anki_exporter.py     # Anki deck export functionality (.apkg generation and text format export)
├── config.py            # Configuration constants - OpenAI client, genanki model/deck IDs, logging setup
├── requirements.txt     # Python dependencies (includes streamlit, plotly, pandas)
├── logs/                # Timestamped log files for each generation run
├── output.apkg          # Generated Anki deck (created on run)
└── flashcards.txt       # Text format output (created on run)
```

The application provides two interfaces: `main.py` for CLI usage and `streamlit_app.py` for web-based interaction. Both interfaces use the same modular backend functions, ensuring consistent functionality across interfaces. 

## Setup

### 1. Clone GitHub repo

```bash
git clone https://github.com/arjun0502/flashcard-generation-agent.git
cd flashcard-generation-agent
```

### 2. Create virtual environment 

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

### 5. Add your lecture materials

Place your PDF or text file in the project directory (e.g., `lecture_notes.pdf` or `transcript.txt`)

Supported file formats:
- PDF files (`.pdf`) - Supports text and images/diagrams
- Text files (`.txt`, `.text`) - Plain text lecture transcripts or notes

## Running Agent on CLI

The command-line interface provides a terminal-based workflow for generating flashcards.

### Commands

```bash
# Basic usage with PDF
python main.py lecture_notes.pdf

# Basic usage with text file
python main.py transcript.txt

# With custom deck name
python main.py lecture_notes.pdf --deck "Biology 101 - Cell Structure"
python main.py transcript.txt --deck "CS 229 - Machine Learning"

# Use a cheaper/faster model
python main.py lecture_notes.pdf --model gpt-4o-mini
python main.py transcript.txt --model gpt-4o-mini

# Customize number of iterations
python main.py lecture_notes.pdf --iterations 3
python main.py transcript.txt --iterations 3

# Interactive study session with adaptive learning
python main.py lecture_notes.pdf --study-session
python main.py transcript.txt --study-session
```

### Command Line Options

- `input_file` - Path to the PDF or text file (`.pdf`, `.txt`, `.text`) (required)
- `--deck` - Name of the Anki deck (default: "Generated Flashcards")
- `--model` - OpenAI model to use: gpt-4o, gpt-4o-mini, or o1 (default: gpt-4o)
- `--iterations` - Maximum number of critique/revision iterations (default: 2)
- `--verbose` - Enable verbose logging with all flashcards shown in log
- `--keep-file` - Keep uploaded file on OpenAI servers (only applies to PDFs; default: delete after use)
- `--study-session` - Enable interactive study session with adaptive learning

### Output Files

After running, you'll get:

- `output.apkg` - Anki package file (import into Anki)
- `flashcards.txt` - Text format with Question|Answer pairs
- `logs/flashcard_generation_TIMESTAMP.log` - Detailed log of the generation process including critiques and revisions
- `knowledge_gaps_report.txt` - Gap analysis report (only when using `--study-session`)

## Running Agent on Web App

The web interface provides an interactive browser-based experience with visual feedback and easy navigation.

### Starting the Web App

```bash
streamlit run streamlit_app.py
```

This will start a local web server and automatically open your default web browser to the application interface.

### Features

The web app provides:

- **Upload & Generate Tab**: Upload PDF or text files with drag-and-drop, visualize generation progress
- **Review Tab**: Browse and search through generated flashcards
- **Study Tab**: Interactive study session where you can type your answers and rate difficulty
- **Analysis Tab**: View knowledge gap analysis and generate adaptive decks
- **Export Tab**: Download flashcards as Anki packages or text files

### Settings

Configure the following in the sidebar:

- **OpenAI Model**: Choose between gpt-4o, gpt-4o-mini, or o1
- **Max Critique/Revision Iterations**: Set the number of critique-revision cycles (1-5)

## Importing into Anki

1. Open Anki desktop application
2. File → Import
3. Select `output.apkg`
4. Your flashcards will appear in the specified deck!

## Cost Estimation

Typical costs per document (using GPT-4o):

**PDF Files:**
- Small PDF (10 pages): ~$0.05-0.15
- Medium PDF (30 pages): ~$0.15-0.40
- Large PDF (100 pages): ~$0.50-1.50
- Note: PDFs with many images cost more due to vision processing

**Text Files:**
- Small text file (1,000 words): ~$0.01-0.03
- Medium text file (5,000 words): ~$0.03-0.10
- Large text file (20,000 words): ~$0.10-0.30
- Text files are generally more cost-effective as they don't require vision processing

## Interactive Study Session

The interactive study session allows users to rate flashcards during their study and get personalized adaptive decks. This feature is available in both the CLI and web app interfaces.

### How It Works

1. **Study Session**: Rate each flashcard on difficulty (1-5)
   - **CLI**: Flashcards are displayed one at a time in the terminal. Press Enter to reveal answers, then enter a difficulty rating.
   - **Web App**: Use the Study tab to go through flashcards. Type your answer, reveal the correct answer, then rate difficulty using the numbered buttons.
   - The system collects difficulty ratings (1-5 scale) and packages them into a `StudySession` object for analysis
   - Rating scale: 1 = Already know well, 2 = Easy, 3 = Moderate, 4 = Difficult, 5 = Very difficult

2. **Gap Analysis**: AI identifies what you know well, weak areas, and critical gaps
   - Analyzes your performance patterns from the study session ratings
   - Identifies strong areas, areas needing improvement, and critical knowledge gaps
   - Uses `analyze_knowledge_gaps()` from `openai_client.py` to analyze performance

3. **Adaptive Deck**: 
   - Removes cards you've mastered (rated 1)
   - Adds targeted gap-filling cards for concepts you struggled with
   - Creates a personalized deck based on your knowledge level

### Starting a Study Session

- **CLI**: Use the `--study-session` flag: `python main.py lecture_notes.pdf --study-session` or `python main.py transcript.txt --study-session`
- **Web App**: Navigate to the Study tab after generating flashcards and click "Start Study Session"

### Example Study Session Flow (CLI)

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

## Logging

All generation runs are automatically logged to timestamped files in the `logs/` directory using Python's `logging` module. The default log level is `INFO`, which shows workflow steps, critiques, and issue summaries. Use the `--verbose` flag to enable `DEBUG` level logging, which includes complete Q&A text for all flashcards and full conversation flow.

## Acknowledgments

- [genanki](https://github.com/kerrickstaley/genanki) - Anki deck generation library
- [OpenAI](https://openai.com) - GPT-4o API
- [Anki](https://apps.ankiweb.net/) - Spaced repetition software