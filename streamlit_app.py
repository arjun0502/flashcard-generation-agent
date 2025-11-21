"""
Streamlit web application for Flashcard Generation Agent.
Provides an interactive web interface for PDF upload, flashcard generation, 
study sessions, and knowledge gap analysis.
"""

import streamlit as st
import tempfile
import os
from pathlib import Path
from datetime import datetime

# Import models
from models import FlashcardSet, StudySession, StudyRating, KnowledgeGaps, AdaptiveUpdate

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
from study_session import adaptive_update_flashcards, print_adaptive_summary


# Page config
st.set_page_config(
    page_title="Flashcard Generation Agent",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "flashcard_decks" not in st.session_state:
    st.session_state.flashcard_decks = {}  # {deck_name: {flashcards, file_id, text_content, study_sessions, knowledge_gaps, adaptive_update}}
if "selected_deck" not in st.session_state:
    st.session_state.selected_deck = None
if "current_flashcard_index" not in st.session_state:
    st.session_state.current_flashcard_index = 0
if "ratings" not in st.session_state:
    st.session_state.ratings = []
if "show_answer" not in st.session_state:
    st.session_state.show_answer = False
if "user_answers" not in st.session_state:
    st.session_state.user_answers = {}
if "model" not in st.session_state:
    st.session_state.model = "gpt-4o"


def reset_study_session():
    """Reset study session state."""
    st.session_state.current_flashcard_index = 0
    st.session_state.ratings = []
    st.session_state.show_answer = False
    st.session_state.user_answers = {}


def get_current_deck():
    """Get the currently selected deck data."""
    if st.session_state.selected_deck and st.session_state.selected_deck in st.session_state.flashcard_decks:
        return st.session_state.flashcard_decks[st.session_state.selected_deck]
    return None


def main():
    """Main Streamlit application."""
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        model = st.selectbox(
            "OpenAI Model",
            ["gpt-4o", "gpt-4o-mini", "o1"],
            index=0,
            help="Model to use for flashcard generation and analysis",
            key="model"
        )
        # Fixed number of critique/revision iterations (not user-configurable)
        max_iterations = 1
        
        st.divider()
        
        st.header("📚 Your Decks")
        if st.session_state.flashcard_decks:
            deck_names = list(st.session_state.flashcard_decks.keys())
            if st.session_state.selected_deck not in deck_names:
                st.session_state.selected_deck = deck_names[0] if deck_names else None
            
            selected = st.selectbox(
                "Select Deck",
                deck_names,
                index=deck_names.index(st.session_state.selected_deck) if st.session_state.selected_deck in deck_names else 0,
                key="deck_selector"
            )
            st.session_state.selected_deck = selected
            
            # Show deck info
            deck_data = st.session_state.flashcard_decks[selected]
            if deck_data.get("flashcards"):
                st.info(f"📊 {len(deck_data['flashcards'].flashcards)} cards")
                if deck_data.get("study_sessions"):
                    st.caption(f"📖 {len(deck_data['study_sessions'])} study sessions")
        else:
            st.info("No decks yet")
            st.session_state.selected_deck = None
        
        st.divider()
        
        st.header("📊 Status")
        current_deck = get_current_deck()
        if current_deck and current_deck.get("flashcards"):
            st.success(f"✓ {len(current_deck['flashcards'].flashcards)} flashcards in current deck")
        else:
            st.info("No flashcards yet")
        
        st.divider()
        
        if st.button("🔄 Reset Session", use_container_width=True):
            # Clean up file IDs
            for deck_name, deck_data in st.session_state.flashcard_decks.items():
                if deck_data.get("file_id"):
                    try:
                        cleanup_file(deck_data["file_id"])
                    except:
                        pass
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    
    # Main title
    st.title("📚 SmartFlash")
    st.markdown("An AI-powered platform that automatically generates flaschard decks, helps you study them, identifies your knowledge gaps, and continuously updates your deck as you learn.") 
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📤 Generate",
        "📖 Study",
        "📊 Mastery",
        "💾 Export"
    ])
    
    # Tab 1: Generate
    with tab1:
        st.header("Generate Flashcards")
        
        uploaded_file = st.file_uploader(
            "Upload your lecture slides or transcripts below (PDF or text files)",
            type=["pdf", "txt", "text"]
        )
        
        if uploaded_file is not None:
            # Determine file extension
            file_ext = Path(uploaded_file.name).suffix.lower()
            
            # Save uploaded file temporarily
            if file_ext == '.pdf':
                suffix = ".pdf"
            else:
                suffix = ".txt"
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name
            
            # Deck name input
            deck_name = st.text_input(
                "Deck Name",
                placeholder="e.g., Biology 101, CS 229 Lecture 1",
                help="Give your flashcard deck a name to organize it"
            )
            
            generate_button = st.button("🚀 Generate Flashcards", type="primary", use_container_width=True)
            
            # Show existing deck info if any
            if st.session_state.flashcard_decks:
                st.info(f"You have {len(st.session_state.flashcard_decks)} deck(s). Select a deck in the sidebar to study.")
            
            if generate_button:
                # Generate deck name if not provided
                if not deck_name or deck_name.strip() == "":
                    deck_num = len(st.session_state.flashcard_decks) + 1
                    deck_name = f"Deck {deck_num}"
                
                deck_name = deck_name.strip()
                
                # Check if deck name already exists
                if deck_name in st.session_state.flashcard_decks:
                    st.warning(f"Deck '{deck_name}' already exists. Please choose a different name.")
                else:
                    with st.spinner("Generating flashcards..."):
                        try:
                            # Prepare input (upload PDF or read text file)
                            progress_bar = st.progress(0, text="Processing file...")
                            file_id, text_content = prepare_input(tmp_path)
                            
                            if file_id:
                                progress_bar.progress(20, text="File uploaded. Generating flashcards...")
                            else:
                                progress_bar.progress(20, text="Text file read. Generating flashcards...")
                            
                            # Generate flashcards
                            flashcards = generate_flashcards(file_id=file_id, text_content=text_content, model=st.session_state.model)
                            progress_bar.progress(60, text="Flashcards generated. Critiquing...")
                            
                            # Critique and revise loop
                            current_flashcards = flashcards
                            for i in range(max_iterations):
                                critique = critique_flashcards(current_flashcards, st.session_state.model)
                                
                                if critique.is_acceptable:
                                    progress_bar.progress(100, text="✓ Flashcards approved!")
                                    break
                                
                                progress_bar.progress(60 + (i + 1) * 15, text=f"Revising (iteration {i+1}/{max_iterations})...")
                                current_flashcards = revise_flashcards(current_flashcards, critique, st.session_state.model)
                            
                            # Store deck
                            st.session_state.flashcard_decks[deck_name] = {
                                "flashcards": current_flashcards,
                                "file_id": file_id,
                                "text_content": text_content,
                                "study_sessions": [],
                                "knowledge_gaps": None,
                                "adaptive_update": None
                            }
                            st.session_state.selected_deck = deck_name
                            
                            progress_bar.empty()
                            
                            st.success(f"✓ Generated {len(current_flashcards.flashcards)} flashcards for '{deck_name}'!")
                            st.info(f"Generated {len(current_flashcards.flashcards)} flashcards! To review them, go to the \"Study\" tab.")
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
                        finally:
                            # Clean up temp file
                            if os.path.exists(tmp_path):
                                os.unlink(tmp_path)
        
    # Tab 2: Study
    with tab2:
        st.header("Interactive Study Session")
        
        # Deck selector
        if not st.session_state.flashcard_decks:
            st.info("👆 Please generate flashcards first in the 'Generate' tab")
        else:
            deck_names = list(st.session_state.flashcard_decks.keys())
            if not st.session_state.selected_deck or st.session_state.selected_deck not in deck_names:
                st.session_state.selected_deck = deck_names[0]
            
            selected_study_deck = st.selectbox(
                "Select Deck to Study",
                deck_names,
                index=deck_names.index(st.session_state.selected_deck) if st.session_state.selected_deck in deck_names else 0,
                key="study_deck_selector"
            )
            st.session_state.selected_deck = selected_study_deck
            
            current_deck = get_current_deck()
            if not current_deck or not current_deck.get("flashcards"):
                st.info("👆 Please generate flashcards first in the 'Generate' tab")
            else:
                flashcards = current_deck["flashcards"].flashcards
                
                if len(flashcards) == 0:
                    st.warning("No flashcards to study!")
                else:
                    # Study session in progress
                    if st.session_state.current_flashcard_index < len(flashcards):
                        current_idx = st.session_state.current_flashcard_index
                        current_card = flashcards[current_idx]
                        
                        # Progress
                        progress = (current_idx + 1) / len(flashcards)
                        st.progress(progress, text=f"Card {current_idx + 1} of {len(flashcards)}")
                        
                        # Question
                        st.markdown("### 📝 Question")
                        st.info(current_card.question)
                        
                        # User answer input
                        st.markdown("### ✍️ Your Answer")
                        user_answer_key = f"user_answer_{current_idx}"
                        user_answer = st.text_area(
                            "Type your answer here:",
                            value=st.session_state.user_answers.get(current_idx, ""),
                            key=user_answer_key,
                            height=100,
                            placeholder="Type your answer before viewing the correct answer..."
                        )
                        # Store the answer
                        st.session_state.user_answers[current_idx] = user_answer
                        
                        # Show/Hide answer button
                        show_answer_key = f"show_answer_{current_idx}"
                        if st.button("👁️ Show Correct Answer" if not st.session_state.show_answer else "🙈 Hide Answer", key=show_answer_key):
                            st.session_state.show_answer = not st.session_state.show_answer
                            st.rerun()
                        
                        if st.session_state.show_answer:
                            # Display user's answer if they typed one
                            if user_answer.strip():
                                st.markdown("### 📝 Your Answer:")
                                st.write(user_answer)
                                st.divider()
                            
                            # Correct answer
                            st.markdown("### ✅ Correct Answer")
                            st.success(current_card.answer)
                            
                            # Rating buttons
                            st.markdown("### 📊 Rate Difficulty")
                            st.caption("1 = Already know well | 2 = Easy | 3 = Moderate | 4 = Difficult | 5 = Very difficult")
                            
                            cols = st.columns(5)
                            for i, col in enumerate(cols):
                                with col:
                                    if st.button(f"{i+1}", key=f"rate_{i+1}_{current_idx}", use_container_width=True):
                                        # Save rating and move to next card
                                        st.session_state.ratings.append(
                                            StudyRating(flashcard_index=current_idx, difficulty=i + 1)
                                        )
                                        st.session_state.current_flashcard_index += 1
                                        st.session_state.show_answer = False
                                        st.rerun()
                    else:
                        # Study session complete
                        st.success("🎉 Study Session Complete!")
                        
                        # Summary
                        rating_counts = {i+1: 0 for i in range(5)}
                        for rating in st.session_state.ratings:
                            rating_counts[rating.difficulty] += 1
                        
                        st.markdown("### 📊 Your Ratings Summary")
                        difficulty_labels = {
                            1: "Already know well",
                            2: "Easy",
                            3: "Moderate",
                            4: "Difficult",
                            5: "Very difficult"
                        }
                        
                        for difficulty, count in rating_counts.items():
                            if count > 0:
                                st.write(f"**{difficulty}** ({difficulty_labels[difficulty]}): {count} cards")
                        
                        # Create StudySession object and automatically analyze + adapt
                        if st.button("💾 Save Study Session", type="primary"):
                            session = StudySession(
                                flashcards=flashcards,
                                ratings=st.session_state.ratings,
                                timestamp=datetime.now().isoformat()
                            )
                            
                            # Save study session to deck
                            current_deck["study_sessions"].append(session)
                            
                            # Automatically analyze gaps and generate adaptive deck
                            if current_deck.get("file_id") or current_deck.get("text_content"):
                                try:
                                    with st.spinner("Analyzing knowledge gaps and updating deck..."):
                                        # 1. Analyze gaps
                                        gaps = analyze_knowledge_gaps(
                                            session,
                                            file_id=current_deck.get("file_id"),
                                            text_content=current_deck.get("text_content"),
                                            model=st.session_state.model
                                        )
                                        current_deck["knowledge_gaps"] = gaps
                                        
                                        # 2. Generate adaptive deck
                                        original = FlashcardSet(flashcards=flashcards)
                                        update = adaptive_update_flashcards(
                                            original,
                                            session,
                                            gaps,
                                            file_id=current_deck.get("file_id"),
                                            text_content=current_deck.get("text_content"),
                                            model=st.session_state.model
                                        )
                                        current_deck["adaptive_update"] = update
                                        current_deck["flashcards"] = update.final_flashcards
                                        
                                        # Reset study session state
                                        reset_study_session()
                                        
                                        st.success("✓ Study session saved! Deck automatically updated with new cards.")
                                        st.rerun()
                                except Exception as e:
                                    st.error(f"Error: {str(e)}")
                            else:
                                st.error("File not found. Please regenerate flashcards.")
                        
                        if st.button("🔄 Restart Study Session"):
                            reset_study_session()
                            st.rerun()
                        
                        st.info("See a breakdown of your mastery in the next tab.")
    
    # Tab 3: Mastery
    with tab3:
        st.header("Knowledge Gap Analysis")
        
        if not st.session_state.flashcard_decks:
            st.info("👆 Please generate flashcards first in the 'Generate' tab")
        else:
            deck_names = list(st.session_state.flashcard_decks.keys())
            if not st.session_state.selected_deck or st.session_state.selected_deck not in deck_names:
                st.session_state.selected_deck = deck_names[0]
            
            selected_mastery_deck = st.selectbox(
                "Select Deck to View Mastery",
                deck_names,
                index=deck_names.index(st.session_state.selected_deck) if st.session_state.selected_deck in deck_names else 0,
                key="mastery_deck_selector"
            )
            st.session_state.selected_deck = selected_mastery_deck
            
            current_deck = get_current_deck()
            if not current_deck:
                st.info("👆 Please generate flashcards first")
            else:
                study_sessions = current_deck.get("study_sessions", [])
                
                if not study_sessions:
                    st.info("👆 Complete a study session first in the 'Study' tab")
                else:
                    # Get latest study session
                    latest_session = study_sessions[-1]
                    knowledge_gaps = current_deck.get("knowledge_gaps")
                    adaptive_update = current_deck.get("adaptive_update")
                    
                    # Display knowledge gaps
                    if knowledge_gaps:
                        gaps = knowledge_gaps
                        
                        st.markdown("### 📊 Knowledge Gap Analysis")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Strong Areas", len(gaps.strong_areas))
                        with col2:
                            st.metric("Weak Areas", len(gaps.weak_areas))
                        with col3:
                            st.metric("Critical Gaps", len(gaps.critical_gaps))
                        
                        # Strong areas
                        if gaps.strong_areas:
                            with st.expander("✅ Strong Areas (What you know well)"):
                                for area in gaps.strong_areas:
                                    st.write(f"✓ {area}")
                        
                        # Weak areas
                        if gaps.weak_areas:
                            with st.expander("⚠️ Areas Needing Improvement"):
                                for area in gaps.weak_areas:
                                    st.write(f"⚠ {area}")
                        
                        # Critical gaps
                        if gaps.critical_gaps:
                            with st.expander("🔴 Critical Gaps"):
                                for gap in gaps.critical_gaps:
                                    st.write(f"🔴 {gap}")
                        
                        # Gap report
                        st.markdown("### 📝 Detailed Report")
                        st.text(gaps.gap_report)
                        
                        # Adaptive update summary
                        if adaptive_update:
                            st.divider()
                            st.markdown("### 🔄 Adaptive Deck Update")
                            st.success("✓ Deck automatically updated after your study session!")
                            
                            update = adaptive_update
                            
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Original Cards", update.original_count)
                            with col2:
                                st.metric("Removed", len(update.cards_removed))
                            with col3:
                                st.metric("Added", len(update.cards_added))
                            with col4:
                                st.metric("Final Cards", len(update.final_flashcards.flashcards))
                            
                            if update.cards_removed:
                                with st.expander(f"🗑️ Removed Cards ({len(update.cards_removed)})"):
                                    for card in update.cards_removed:
                                        st.write(f"**Q:** {card.question}")
                                        st.caption(f"**A:** {card.answer}")
                            
                            if update.cards_added:
                                with st.expander(f"➕ Added Cards ({len(update.cards_added)})"):
                                    for card in update.cards_added:
                                        st.write(f"**Q:** {card.question}")
                                        st.caption(f"**A:** {card.answer}")
                    else:
                        st.info("No mastery analysis available yet. Complete a study session to see your knowledge gaps.")
    
    # Tab 4: Export
    with tab4:
        st.header("Export Flashcards")
        
        if not st.session_state.flashcard_decks:
            st.info("👆 Please generate flashcards first")
        else:
            deck_names = list(st.session_state.flashcard_decks.keys())
            if not st.session_state.selected_deck or st.session_state.selected_deck not in deck_names:
                st.session_state.selected_deck = deck_names[0]
            
            selected_export_deck = st.selectbox(
                "Select Deck to Export",
                deck_names,
                index=deck_names.index(st.session_state.selected_deck) if st.session_state.selected_deck in deck_names else 0,
                key="export_deck_selector"
            )
            st.session_state.selected_deck = selected_export_deck
            
            current_deck = get_current_deck()
            if not current_deck or not current_deck.get("flashcards"):
                st.info("👆 Please generate flashcards first")
            else:
                flashcards = current_deck["flashcards"]
                
                st.info(f"Ready to export {len(flashcards.flashcards)} flashcards from '{selected_export_deck}'")
                
                anki_deck_name = st.text_input(
                    "Anki Deck Name",
                    value=selected_export_deck,
                    help="Name for the Anki deck"
                )
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("📦 Export to Anki (.apkg)", type="primary", use_container_width=True):
                        with st.spinner("Creating Anki package..."):
                            try:
                                output_file = "output.apkg"
                                export_to_anki(flashcards, anki_deck_name, output_file)
                                
                                with open(output_file, "rb") as f:
                                    st.download_button(
                                        label="⬇️ Download Anki Package",
                                        data=f.read(),
                                        file_name=output_file,
                                        mime="application/octet-stream"
                                    )
                                st.success("✓ Anki package created!")
                            except Exception as e:
                                st.error(f"Error: {str(e)}")
                
                with col2:
                    if st.button("📄 Export to Text File", use_container_width=True):
                        with st.spinner("Creating text file..."):
                            try:
                                output_file = "flashcards.txt"
                                save_flashcards_text(flashcards, output_file)
                                
                                with open(output_file, "r", encoding="utf-8") as f:
                                    st.download_button(
                                        label="⬇️ Download Text File",
                                        data=f.read(),
                                        file_name=output_file,
                                        mime="text/plain"
                                    )
                                st.success("✓ Text file created!")
                            except Exception as e:
                                st.error(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
