# pip install langchain langgraph faiss-cpu pypdf langchain-google-genai google-generativeai sentence-transformers langchain-huggingface

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langgraph.graph import StateGraph, END, START
from typing_extensions import TypedDict
from langchain_huggingface import HuggingFaceEmbeddings
import os
import pickle
from pathlib import Path

# -------------------
# Step 1: Load & process both PDF files
# -------------------
def load_and_process_pdfs():
    print("Loading curriculum PDFs...")

    pdf_files = [
        ("data/Btech_Curriculum_2023.pdf", "BTech"),
        ("data/PG_Curriculum_2023.pdf", "PG")
    ]

    all_docs = []

    for pdf_path, doc_type in pdf_files:
        if Path(pdf_path).exists():
            print(f"Loading {doc_type} curriculum from {pdf_path}")
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()

            # Add metadata to identify document type
            for doc in documents:
                doc.metadata['curriculum_type'] = doc_type
                doc.metadata['source_file'] = pdf_path

            all_docs.extend(documents)
            print(f"Loaded {len(documents)} pages from {doc_type} curriculum")
        else:
            print(f"Warning: {pdf_path} not found!")

    print("Splitting documents into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(all_docs)

    print(f"Created {len(chunks)} document chunks from both curricula")
    return chunks

# -------------------
# Step 2: Create offline embeddings using Sentence Transformers
# -------------------
def create_or_load_vectorstore(docs, force_recreate=False):
    vectorstore_path = "curriculum_vectorstore.pkl"

    if Path(vectorstore_path).exists() and not force_recreate:
        print("Loading existing curriculum vectorstore...")
        try:
            with open(vectorstore_path, 'rb') as f:
                vectorstore = pickle.load(f)
            print("✅ Vectorstore loaded successfully!")
            return vectorstore
        except Exception as e:
            print(f"Error loading vectorstore: {e}. Creating new one...")

    print("Creating embeddings using Sentence Transformers (offline)...")
    # Use a good sentence transformer model that works offline
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",  # Fast and good quality
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    print("Building FAISS vectorstore from curriculum data...")
    vectorstore = FAISS.from_documents(docs, embeddings)

    # Save vectorstore for future use
    print("Saving vectorstore for future use...")
    try:
        with open(vectorstore_path, 'wb') as f:
            pickle.dump(vectorstore, f)
        print("✅ Vectorstore created and saved!")
    except Exception as e:
        print(f"Warning: Could not save vectorstore: {e}")

    return vectorstore

# -------------------
# Step 3: Initialize Google Flash Model (Free Tier)
# -------------------
def create_llm():
    return ChatGoogleGenerativeAI(
        model="models/gemini-2.5-flash",  # Free tier model
        temperature=0.1,
        convert_system_message_to_human=True
    )

# -------------------
# Step 4: Define state for LangGraph
# -------------------
class State(TypedDict):
    question: str
    answer: str
    sources: list
    curriculum_type: str
    confidence: str

def analyze_question(question: str) -> str:
    """Determine which curriculum the question is likely about"""
    question_lower = question.lower()

    pg_keywords = ['master', 'mtech', 'phd', 'postgraduate', 'pg', 'research', 'thesis']
    btech_keywords = ['bachelor', 'btech', 'undergraduate', 'ug', 'engineering']

    pg_score = sum(1 for keyword in pg_keywords if keyword in question_lower)
    btech_score = sum(1 for keyword in btech_keywords if keyword in question_lower)

    if pg_score > btech_score:
        return "PG"
    elif btech_score > pg_score:
        return "BTech"
    else:
        return "Both"

def curriculum_qa_node(state: State) -> State:
    question = state["question"]
    print(f"🔍 Processing question: {question}")

    # Analyze which curriculum the question is about
    detected_type = analyze_question(question)
    print(f"📚 Detected curriculum focus: {detected_type}")

    try:
        # Create enhanced query with context
        enhanced_query = f"""
        Question about curriculum: {question}

        Please provide a comprehensive answer based on the curriculum documents.
        If the question relates to specific programs (BTech/PG), mention both if relevant.
        Include specific details like course codes, credits, prerequisites, or duration when available.
        """

        # Run the QA chain
        result = qa_chain({"query": enhanced_query})

        # Extract answer and sources
        answer = result["result"]
        source_docs = result.get("source_documents", [])

        # Create detailed source information
        sources = []
        curriculum_types_found = set()

        for i, doc in enumerate(source_docs[:4]):  # Top 4 sources
            doc_type = doc.metadata.get('curriculum_type', 'Unknown')
            curriculum_types_found.add(doc_type)

            source_text = doc.page_content[:150] + "..."
            page_info = f"Page {doc.metadata.get('page', 'N/A')}"
            sources.append(f"📄 {doc_type} Curriculum - {page_info}: {source_text}")

        # Determine confidence based on source relevance and quantity
        if len(source_docs) >= 3:
            confidence = "High"
        elif len(source_docs) >= 2:
            confidence = "Medium"
        else:
            confidence = "Low"

        # Final curriculum type based on sources found
        final_curriculum_type = ", ".join(sorted(curriculum_types_found)) or detected_type

        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "curriculum_type": final_curriculum_type,
            "confidence": confidence
        }

    except Exception as e:
        return {
            "question": question,
            "answer": f"I apologize, but I encountered an error while processing your question: {str(e)}. Please try rephrasing your question or check if the system is properly configured.",
            "sources": [],
            "curriculum_type": "Error",
            "confidence": "Error"
        }

# -------------------
# Step 5: Build LangGraph
# -------------------
def create_curriculum_chatbot():
    graph = StateGraph(State)
    graph.add_node("curriculum_qa", curriculum_qa_node)
    graph.add_edge(START, "curriculum_qa")
    graph.add_edge("curriculum_qa", END)

    return graph.compile()

# -------------------
# Step 6: Interactive curriculum assistant
# -------------------
def chat_with_curriculum():
    app = create_curriculum_chatbot()

    print("\n🎓 Curriculum Assistant Ready!")
    print("📚 I can help you with questions about:")
    print("   • BTech Curriculum 2023")
    print("   • PG (Postgraduate) Curriculum 2023")
    print("   • Course details, prerequisites, credits")
    print("   • Program structure and requirements")
    print("   • Academic policies and procedures")
    print("\n💡 Examples of questions you can ask:")
    print("   - 'What are the core subjects in BTech first year?'")
    print("   - 'How many credits are required for PG programs?'")
    print("   - 'What are the prerequisites for advanced courses?'")
    print("   - 'Tell me about the assessment methods'")
    print("\nType 'quit', 'exit', or 'q' to exit.")
    print("Type 'help' to see this information again.\n")

    conversation_count = 0

    while True:
        question = input("🎓 You: ").strip()

        if question.lower() in ['quit', 'exit', 'q']:
            print("👋 Thank you for using the Curriculum Assistant! Have a great day!")
            break

        if question.lower() == 'help':
            print("\n📋 Available Commands:")
            print("• Ask any question about BTech or PG curriculum")
            print("• 'help' - Show this help message")
            print("• 'stats' - Show system statistics")
            print("• 'quit'/'exit'/'q' - Exit the assistant")
            continue

        if question.lower() == 'stats':
            print(f"\n📊 Session Statistics:")
            print(f"• Questions answered: {conversation_count}")
            print(f"• Curriculum documents: BTech 2023, PG 2023")
            print(f"• AI Model: Google Gemini Flash 1.5 (Free Tier)")
            print(f"• Embeddings: Sentence Transformers (Offline)")
            continue

        if not question:
            continue

        try:
            result = app.invoke({
                "question": question,
                "answer": "",
                "sources": [],
                "curriculum_type": "",
                "confidence": ""
            })

            conversation_count += 1

            print(f"\n🤖 Assistant: {result['answer']}")
            print(f"\n📚 Curriculum: {result['curriculum_type']}")
            print(f"🎯 Confidence: {result['confidence']}")

            if result.get('sources'):
                print("\n📖 Sources:")
                for source in result['sources']:
                    print(f"   {source}")

            print("=" * 80)

        except Exception as e:
            print(f"❌ Sorry, I encountered an error: {e}")
            print("Please try rephrasing your question or type 'help' for assistance.")

# -------------------
# Step 7: Utility functions
# -------------------
def test_curriculum_system():
    """Test the system with curriculum-specific questions"""
    app = create_curriculum_chatbot()

    test_questions = [
        "What is the structure of BTech curriculum?",
        "How many credits are required for PG programs?",
        "What are the core subjects in the first year?",
        "Tell me about the assessment and evaluation methods",
        "What are the prerequisites for advanced courses?",
        "Describe the research component in PG programs"
    ]

    print("🧪 Testing Curriculum Assistant with sample questions...\n")

    for i, question in enumerate(test_questions, 1):
        print(f"Test {i}: {question}")

        try:
            result = app.invoke({
                "question": question,
                "answer": "",
                "sources": [],
                "curriculum_type": "",
                "confidence": ""
            })

            print(f"Answer: {result['answer'][:200]}...")
            print(f"Curriculum: {result['curriculum_type']}")
            print(f"Confidence: {result['confidence']}")
            print("-" * 60)

        except Exception as e:
            print(f"Error: {e}")

def recreate_curriculum_embeddings():
    """Force recreate embeddings from curriculum PDFs"""
    print("🔄 Recreating curriculum embeddings...")
    docs = load_and_process_pdfs()
    if docs:
        vectorstore = create_or_load_vectorstore(docs, force_recreate=True)
        print("✅ Curriculum embeddings recreated successfully!")
        return vectorstore
    else:
        print("❌ No documents found to process!")
        return None

# -------------------
# Step 8: Main execution
# -------------------
if __name__ == "__main__":
    print("🎓 Initializing Curriculum Assistant...")
    print("📚 Loading BTech and PG Curriculum 2023 documents...")

    # Load and process curriculum documents
    docs = load_and_process_pdfs()

    if not docs:
        print("❌ No curriculum documents found! Please ensure:")
        print("- data/Btech_Curriculum_2023.pdf exists")
        print("- data/PG_Curriculum_2023.pdf exists")
        exit(1)

    # Create or load vectorstore with offline embeddings
    vectorstore = create_or_load_vectorstore(docs)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

    # Initialize LLM
    llm = create_llm()

    # Create QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

    print("✅ Curriculum Assistant initialized successfully!")

    # Menu system
    while True:
        print("\n🎯 Curriculum Assistant - Choose an option:")
        print("1. 💬 Start curriculum chat")
        print("2. 🧪 Test system with sample questions")
        print("3. 🔄 Recreate curriculum embeddings")
        print("4. ℹ️  About this assistant")
        print("5. 🚪 Exit")

        choice = input("\nEnter choice (1-5): ").strip()

        if choice == "1":
            chat_with_curriculum()
        elif choice == "2":
            test_curriculum_system()
        elif choice == "3":
            new_vectorstore = recreate_curriculum_embeddings()
            if new_vectorstore:
                vectorstore = new_vectorstore
                retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=retriever,
                    return_source_documents=True
                )
        elif choice == "4":
            print("\n📖 About Curriculum Assistant:")
            print("🎯 Purpose: Answer questions about BTech and PG Curriculum 2023")
            print("🤖 AI Model: Google Gemini Flash 1.5 (Free Tier)")
            print("🔍 Search: Sentence Transformers + FAISS (Offline)")
            print("📚 Documents: BTech & PG Curriculum PDFs")
            print("🔧 Framework: LangChain + LangGraph")
            print("💡 Features:")
            print("   • Automatic curriculum type detection")
            print("   • Source citation with page numbers")
            print("   • Confidence scoring")
            print("   • Offline embeddings for privacy")
        elif choice == "5":
            print("👋 Thank you for using the Curriculum Assistant!")
            break
        else:
            print("❌ Invalid choice, please select 1-5.")
