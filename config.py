"""Configuration constants and logging setup."""

from pathlib import Path
from datetime import datetime
import logging
import os
from openai import OpenAI
import genanki

# Set up logging - will be reconfigured in main.py when output directory is created
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# IMPORTANT: These IDs should be hardcoded and unique
# Generated once using: python3 -c "import random; print(random.randrange(1 << 30, 1 << 31))"
FLASHCARD_MODEL_ID = 1607392319
FLASHCARD_DECK_ID = 2059400110

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

