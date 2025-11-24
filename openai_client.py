"""OpenAI API interaction functions."""

from pathlib import Path
from openai.lib._pydantic import to_strict_json_schema

from config import client
from models import FlashcardSet, Flashcard, Critique, KnowledgeGaps


def upload_pdf(file_path: str) -> str:
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
    return uploaded_file.id


def prepare_input(file_path: str) -> tuple[str | None, str | None]:
    """
    Prepare input file for processing. Returns (file_id, text_content).
    For PDFs: returns (file_id, None)
    For text files: returns (None, text_content)
    """
    file_path_obj = Path(file_path)
    if not file_path_obj.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    suffix = file_path_obj.suffix.lower()
    
    if suffix == '.pdf':
        # Upload PDF and return file_id
        file_id = upload_pdf(file_path)
        return (file_id, None)
    elif suffix in ['.txt', '.text']:
        # Read text file content
        print(f"Reading text file: {file_path_obj.name}...")
        with open(file_path, "r", encoding="utf-8") as f:
            text_content = f.read()
        print(f"Text file read successfully ({len(text_content)} characters)")
        return (None, text_content)
    else:
        raise ValueError(f"Unsupported file type: {suffix}. Supported types: .pdf, .txt, .text")


def cleanup_file(file_id: str) -> None:
    """Delete the uploaded file to avoid storage costs."""
    import logging
    try:
        client.files.delete(file_id)
        print(f"File {file_id} deleted successfully.")
        logging.info(f"Deleted uploaded file: {file_id}")
    except Exception as e:
        print(f"Error deleting file {file_id}: {e}")
        logging.error(f"Error deleting file {file_id}: {e}")


def generate_flashcards(file_id: str | None = None, text_content: str | None = None, model: str = "gpt-4o") -> FlashcardSet:
    """
    Generate flashcards using either an uploaded file ID (for PDFs) or text content (for text files).
    Must provide exactly one of file_id or text_content.
    """
    if (file_id is None) == (text_content is None):
        raise ValueError("Must provide exactly one of file_id or text_content")
    
    print(f"Calling OpenAI API ({model}) to generate flashcards...")
    
    # Build user content based on input type
    if file_id:
        # PDF file - use file attachment
        user_content = [
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
        system_content = """You are an expert at creating flashcards for spaced repetition learning.

Generate comprehensive flashcards from this document. Include information from any diagrams, charts, or images."""
    else:
        # Text file - include content directly
        user_content = f"""Generate comprehensive flashcards from this lecture transcript/text:

{text_content}

Create flashcards that cover the key concepts, definitions, and important information from this text."""
        system_content = """You are an expert at creating flashcards for spaced repetition learning.

Generate comprehensive flashcards from this lecture transcript/text. Create flashcards that cover the key concepts, definitions, and important information from this text."""
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": system_content
            },
            {
                "role": "user",
                "content": user_content
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


def critique_flashcards(flashcard_set: FlashcardSet, model: str = "gpt-4o") -> Critique:
    """Critique flashcards for quality."""
    from config import client
    from openai.lib._pydantic import to_strict_json_schema
    
    flashcard_text = "\n".join([f"{i+1}. Q: {fc.question} | A: {fc.answer}" 
                                 for i, fc in enumerate(flashcard_set.flashcards)])
    
    print(f"Critiquing flashcards with {model}...")
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": """You are an expert educational evaluator specializing in flashcard quality assessment for long-term learning and spaced repetition.

OVERARCHING GOAL: These flashcards are designed for long-term understanding and spaced repetition of lecture material. They should help students master important concepts through active recall and deep understanding, not just surface memorization.

Evaluate these flashcards against four atomic, independent quality metrics. Each metric is separate and non-overlapping:

1. ATOMICITY: Each card should focus on ONE clear, atomic concept. Cards that combine multiple concepts or require multiple pieces of information should be split or simplified.

2. CLARITY: Questions and answers must be unambiguous, precise, and complete. Avoid vague wording, ambiguous phrasing, or questions that could have multiple correct answers. Include necessary context for understanding. The answer should fully address the question without being excessively long.

3. LEARNING VALUE: Cards should promote active recall and deep understanding rather than surface memorization. Prefer "why" and "how" questions over "what" questions. Avoid yes/no questions, simple fact recall, or questions that test only memorization. Cards should require the learner to actively construct knowledge.

4. ACCURACY: The information in the flashcard must be factually correct and free from errors. Verify that all facts, definitions, and explanations are accurate.

For each flashcard, identify specific issues related to these four metrics. Determine if the flashcards are acceptable or need revision."""
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


def revise_flashcards(flashcard_set: FlashcardSet, critique: Critique, model: str = "gpt-4o") -> FlashcardSet:
    """Revise flashcards based on critique."""
    from config import client
    from openai.lib._pydantic import to_strict_json_schema
    
    flashcard_text = "\n".join([f"{i+1}. Q: {fc.question} | A: {fc.answer}" 
                                 for i, fc in enumerate(flashcard_set.flashcards)])
    
    print(f"Revising flashcards with {model}...")
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are an expert at refining flashcards based on pedagogical feedback. \n\nOVERARCHING GOAL: These flashcards are designed for long-term understanding and spaced repetition of lecture material. They should help students master important concepts through active recall and deep understanding, not just surface memorization.\n\nRevise the flashcards to address the feedback below. When making revisions, ensure each flashcard meets these four quality criteria:\n1. ATOMICITY: One clear concept per card\n2. CLARITY: Unambiguous, precise, and complete questions/answers\n3. LEARNING VALUE: Promotes active recall and deep understanding\n4. ACCURACY: Factually correct and free from errors\n\nFocus on addressing the specific feedback and issues identified."
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


def analyze_knowledge_gaps(session, file_id: str | None = None, text_content: str | None = None, model: str = "gpt-4o") -> KnowledgeGaps:
    """Use AI to analyze ratings and identify knowledge gaps."""
    from config import client
    from openai.lib._pydantic import to_strict_json_schema
    
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


def generate_gap_filling_cards(gaps: KnowledgeGaps, file_id: str | None = None, text_content: str | None = None, model: str = "gpt-4o") -> list[Flashcard]:
    """
    Generate new flashcards specifically for identified gaps.
    Must provide exactly one of file_id or text_content.
    """
    from config import client
    from openai.lib._pydantic import to_strict_json_schema
    import logging
    
    if not gaps.critical_gaps and not gaps.weak_areas:
        logging.info("No gaps identified, skipping card generation")
        return []
    
    if (file_id is None) == (text_content is None):
        raise ValueError("Must provide exactly one of file_id or text_content")
    
    # Build prompt for specific gap-filling cards
    gap_summary = "\n".join([
        f"- {gap}" for gap in gaps.critical_gaps
    ])
    
    if gaps.weak_areas:
        gap_summary += "\n" + "\n".join([
            f"- {area}" for area in gaps.weak_areas
        ])
    
    print("Generating gap-filling flashcards...")
    
    # Build user content based on input type
    if file_id:
        user_content = [
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
    else:
        user_content = f"""Generate flashcards that specifically address these knowledge gaps from the original lecture transcript:

{gap_summary}

Original lecture transcript:
{text_content}

Create focused flashcards that:
- Address the exact gaps identified
- Use simpler language if student struggled
- Provide step-by-step breakdowns for complex concepts
- Include examples where helpful
- Are designed to scaffold learning (build up from basics)

Generate approximately 5-8 flashcards tailored to these gaps."""
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": """You are an expert at creating targeted flashcards to help students fill specific knowledge gaps.

OVERARCHING GOAL: These flashcards are designed for long-term understanding and spaced repetition of lecture material. They should help students master important concepts through active recall and deep understanding, not just surface memorization.

Create flashcards that directly and effectively address the identified knowledge gaps. Each card should target a specific gap and help the student understand concepts they struggled with. While addressing gaps, ensure each card also meets these four quality criteria:

1. ATOMICITY: Each card should focus on ONE clear, atomic concept. Break down complex gaps into simpler, focused cards.

2. CLARITY: Use clear, unambiguous, and complete questions and answers. Since the student struggled with these concepts, provide necessary context. The answer should fully address the question.

3. LEARNING VALUE: Design cards that promote active recall and deep understanding. Use "why" and "how" questions rather than simple fact recall.

4. ACCURACY: The information in the flashcard must be factually correct and free from errors. Verify that all facts, definitions, and explanations are accurate.

Create focused flashcards that effectively address the exact gaps identified while maintaining quality on these four metrics."""
            },
            {
                "role": "user",
                "content": user_content
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

