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
if "file_id" not in st.session_state:
    st.session_state.file_id = None
if "text_content" not in st.session_state:
    st.session_state.text_content = None
if "flashcards" not in st.session_state:
    st.session_state.flashcards = None
if "study_session" not in st.session_state:
    st.session_state.study_session = None
if "knowledge_gaps" not in st.session_state:
    st.session_state.knowledge_gaps = None
if "adaptive_update" not in st.session_state:
    st.session_state.adaptive_update = None
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
        max_iterations = st.slider(
            "Max Critique/Revision Iterations",
            min_value=1,
            max_value=5,
            value=1,
            help="Maximum number of critique-revision cycles"
        )
        
        st.divider()
        
        st.header("📊 Status")
        if st.session_state.flashcards:
            st.success(f"✓ {len(st.session_state.flashcards.flashcards)} flashcards generated")
        else:
            st.info("No flashcards yet")
        
        st.divider()
        
        if st.button("🔄 Reset Session", use_container_width=True):
            if st.session_state.file_id:
                try:
                    cleanup_file(st.session_state.file_id)
                except:
                    pass
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    
    # Main title
    st.title("📚 Flashcard Generation Agent")
    st.markdown("Automatically generate, critique, and personalize flashcards from your lecture materials")
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📤 Upload & Generate",
        "📚 Review",
        "📖 Study",
        "📊 Analysis",
        "💾 Export"
    ])
    
    # Tab 1: Upload & Generate
    with tab1:
        st.header("Upload File and Generate Flashcards")
        
        uploaded_file = st.file_uploader(
            "Upload a PDF or text file",
            type=["pdf", "txt", "text"],
            help="Upload your lecture notes, slides, transcripts, or study materials (PDF or text files)"
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
            
            col1, col2 = st.columns([1, 4])
            with col1:
                generate_button = st.button("🚀 Generate Flashcards", type="primary", use_container_width=True)
            with col2:
                if st.session_state.flashcards:
                    st.success(f"✓ Already have {len(st.session_state.flashcards.flashcards)} flashcards")
            
            if generate_button:
                with st.spinner("Generating flashcards..."):
                    try:
                        # Prepare input (upload PDF or read text file)
                        progress_bar = st.progress(0, text="Processing file...")
                        file_id, text_content = prepare_input(tmp_path)
                        st.session_state.file_id = file_id
                        st.session_state.text_content = text_content
                        
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
                        
                        st.session_state.flashcards = current_flashcards
                        progress_bar.empty()
                        
                        st.success(f"✓ Generated {len(current_flashcards.flashcards)} flashcards!")
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                    finally:
                        # Clean up temp file
                        if os.path.exists(tmp_path):
                            os.unlink(tmp_path)
        
        # Display current flashcards summary
        if st.session_state.flashcards:
            st.divider()
            st.subheader("Generated Flashcards Summary")
            st.info(f"Total flashcards: {len(st.session_state.flashcards.flashcards)}")
    
    # Tab 2: Review
    with tab2:
        st.header("Review Flashcards")
        
        if not st.session_state.flashcards:
            st.info("👆 Please generate flashcards first in the 'Upload & Generate' tab")
        else:
            # Search functionality
            search_term = st.text_input("🔍 Search flashcards", placeholder="Search questions or answers...")
            
            # Filter flashcards
            filtered_flashcards = st.session_state.flashcards.flashcards
            if search_term:
                filtered_flashcards = [
                    fc for fc in filtered_flashcards
                    if search_term.lower() in fc.question.lower() or search_term.lower() in fc.answer.lower()
                ]
            
            st.info(f"Showing {len(filtered_flashcards)} of {len(st.session_state.flashcards.flashcards)} flashcards")
            
            # Display flashcards
            for idx, flashcard in enumerate(filtered_flashcards):
                with st.expander(f"Card {idx + 1}: {flashcard.question[:60]}..."):
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        st.markdown("**Question:**")
                        st.write(flashcard.question)
                    with col2:
                        st.markdown("**Answer:**")
                        st.write(flashcard.answer)
    
    # Tab 3: Study
    with tab3:
        st.header("Interactive Study Session")
        
        if not st.session_state.flashcards:
            st.info("👆 Please generate flashcards first in the 'Upload & Generate' tab")
        else:
            flashcards = st.session_state.flashcards.flashcards
            
            if len(flashcards) == 0:
                st.warning("No flashcards to study!")
            else:
                # Start study session button
                if st.session_state.current_flashcard_index == 0 and len(st.session_state.ratings) == 0:
                    if st.button("▶️ Start Study Session", type="primary", use_container_width=True):
                        reset_study_session()
                        st.rerun()
                
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
                    
                    # Create StudySession object
                    if st.button("💾 Save Study Session", type="primary"):
                        st.session_state.study_session = StudySession(
                            flashcards=flashcards,
                            ratings=st.session_state.ratings,
                            timestamp=datetime.now().isoformat()
                        )
                        st.success("✓ Study session saved! Go to the Analysis tab to analyze knowledge gaps.")
                    
                    if st.button("🔄 Restart Study Session"):
                        reset_study_session()
                        st.rerun()
    
    # Tab 4: Analysis
    with tab4:
        st.header("Knowledge Gap Analysis")
        
        if not st.session_state.study_session:
            st.info("👆 Complete a study session first in the 'Study' tab")
        else:
            session = st.session_state.study_session
            
            if st.button("🔍 Analyze Knowledge Gaps", type="primary"):
                if not st.session_state.file_id and not st.session_state.text_content:
                    st.error("File not found. Please regenerate flashcards.")
                else:
                    with st.spinner("Analyzing knowledge gaps..."):
                        try:
                            gaps = analyze_knowledge_gaps(
                                session,
                                file_id=st.session_state.file_id,
                                text_content=st.session_state.text_content,
                                model=st.session_state.model
                            )
                            st.session_state.knowledge_gaps = gaps
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error analyzing gaps: {str(e)}")
            
            # Display knowledge gaps
            if st.session_state.knowledge_gaps:
                gaps = st.session_state.knowledge_gaps
                
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
                
                # Adaptive update
                st.divider()
                st.markdown("### 🔄 Adaptive Deck Update")
                st.info("💡 Based on your study session performance, the adaptive deck will remove cards you've mastered (rated 1) and add new targeted flashcards to fill your knowledge gaps.")
                
                if st.button("✨ Generate Adaptive Deck", type="primary"):
                    if not st.session_state.file_id and not st.session_state.text_content:
                        st.error("File not found. Please regenerate flashcards.")
                    else:
                        with st.spinner("Generating adaptive deck..."):
                            try:
                                original = FlashcardSet(flashcards=session.flashcards)
                                update = adaptive_update_flashcards(
                                    original,
                                    session,
                                    gaps,
                                    file_id=st.session_state.file_id,
                                    text_content=st.session_state.text_content,
                                    model=st.session_state.model
                                )
                                st.session_state.adaptive_update = update
                                st.session_state.flashcards = update.final_flashcards
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error generating adaptive deck: {str(e)}")
                
                # Display adaptive update summary
                if st.session_state.adaptive_update:
                    update = st.session_state.adaptive_update
                    
                    st.success("✓ Adaptive deck generated!")
                    
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
    
    # Tab 5: Export
    with tab5:
        st.header("Export Flashcards")
        
        if not st.session_state.flashcards:
            st.info("👆 Please generate flashcards first")
        else:
            flashcards = st.session_state.flashcards
            
            st.info(f"Ready to export {len(flashcards.flashcards)} flashcards")
            
            deck_name = st.text_input(
                "Deck Name",
                value="Generated Flashcards",
                help="Name for the Anki deck"
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📦 Export to Anki (.apkg)", type="primary", use_container_width=True):
                    with st.spinner("Creating Anki package..."):
                        try:
                            output_file = "output.apkg"
                            export_to_anki(flashcards, deck_name, output_file)
                            
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
