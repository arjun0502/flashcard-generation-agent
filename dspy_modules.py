"""DSPy modules for flashcard generation optimization."""

try:
    import dspy
except ImportError:
    print("Warning: dspy-ai not installed. Install with: pip install dspy-ai")
    dspy = None

if dspy:
    # Define DSPy signatures
    class CritiqueSignature(dspy.Signature):
        """Signature for critiquing flashcards."""
        flashcard_text: str = dspy.InputField(desc="The flashcards to critique, formatted as questions and answers")
        is_acceptable: bool = dspy.OutputField(desc="Whether the flashcards are acceptable or need revision")
        feedback: str = dspy.OutputField(desc="Detailed feedback on the flashcards")
        issues: str = dspy.OutputField(desc="List of specific issues found, one per line")


    class AdaptationSignature(dspy.Signature):
        """Signature for generating adaptive flashcards."""
        knowledge_gaps: str = dspy.InputField(desc="Description of knowledge gaps to address")
        source_material: str = dspy.InputField(desc="Source material to reference (truncated if long)")
        new_flashcards: str = dspy.OutputField(desc="New flashcards in JSON format matching FlashcardSet schema")


    # DSPy modules
    class CritiqueModule(dspy.Module):
        """Optimizable module for critiquing flashcards."""
        
        def __init__(self):
            super().__init__()
            self.critique = dspy.ChainOfThought(CritiqueSignature)
        
        def forward(self, flashcard_text: str):
            """Critique flashcards."""
            result = self.critique(flashcard_text=flashcard_text)
            return result


    class AdaptationModule(dspy.Module):
        """Optimizable module for generating adaptive flashcards."""
        
        def __init__(self):
            super().__init__()
            self.generate = dspy.ChainOfThought(AdaptationSignature)
        
        def forward(self, knowledge_gaps: str, source_material: str):
            """Generate flashcards to address knowledge gaps."""
            result = self.generate(
                knowledge_gaps=knowledge_gaps,
                source_material=source_material
            )
            return result

