from langchain_community.llms import Ollama
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from langchain.memory import ConversationBufferMemory
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document
import os
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
import mysql.connector
from langchain.vectorstores import Chroma
from chromadb.config import Settings
import re
from langchain.chains import RetrievalQA


app = Flask(__name__)
app.secret_key = "your_secret_key"

# Phase 1 - Indexing
# Loading and Splitting
def load_merged_documents(folder_path):
    faq_docs = []
    generic_docs = []

    for filename in os.listdir(folder_path):
        if not filename.endswith(".pdf"):
            continue

        path = os.path.join(folder_path, filename)
        loader = PyPDFLoader(path)
        pages = loader.load()

        # decide metadata based on filename
        program = None
        inquiry_type = "general"  # default

        if any(key in filename.upper() for key in ["ATYCB", "CAS", "CCIS", "CEA", "CHS", "HS"]):
            # Program catalogues
            if "ATYCB" in filename.upper():
                program = "ATYCB"
            elif "CAS" in filename.upper():
                program = "CAS"
            elif "CCIS" in filename.upper():
                program = "CCIS"
            elif "CEA" in filename.upper():
                program = "CEA"
            elif "CHS" in filename.upper():
                program = "CHS"
            elif "HS" in filename.upper():
                program = "HS"
            inquiry_type = "catalogue"

        elif "FAQ" in filename.upper() or "HANDBOOK" in filename.upper():
            # admissions documents
            inquiry_type = "admissions"

        # Process FAQ documents separately
        full_text = "\n".join([page.page_content for page in pages])
        if "*Response*" in full_text and "*End*" in full_text:
            sections = full_text.split("# ")
            for section in sections:
                if not section.strip():
                    continue
                lines = section.splitlines()
                title = lines[0].strip()
                content = "\n".join(lines[1:]).strip()

                if "*Response*" in content and "*End*" in content:
                    parts = content.split("*Response*", 1)[1].split("*End*", 1)
                    response = parts[0].strip()
                    questions = content.split("*Response*", 1)[0].strip()
                    full_text = f"{title}\n\n{questions}\n\nResponse:\n{response}"
                else:
                    full_text = content

                faq_docs.append(Document(
                    page_content=full_text,
                    metadata={
                        "topic": title,
                        "inquiry_type": inquiry_type,
                        "program": program or "general", 
                        "source": filename
                    }
                ))
        else:
            # Split handbooks/catalogues into chunks
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = splitter.split_documents(pages)
            for c in chunks:
                c.metadata["inquiry_type"] = inquiry_type
                c.metadata["program"] = program or "general"
                c.metadata["source"] = filename
            generic_docs.extend(chunks)

    combined_chunks = faq_docs + generic_docs
    return combined_chunks

print("Loading pdf files")
documents = load_merged_documents("./data/")
print("Docs loaded:", len(documents))


# Embeddings
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)


db_path = "./chroma_db"

client_settings = Settings(
    anonymized_telemetry=False,
    persist_directory=db_path,
    is_persistent=True
)

print("trying to build vectorstore...")
if not os.path.exists(db_path) or not os.listdir(db_path):
    print("No existing Chroma DB found. Building new one...")
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embedding,
        collection_name="puapal_faqs",
        client_settings=client_settings
    )
    vectorstore.persist()
    print("New Chroma DB has been created.")
else:
    print("Loading existing Chroma DB...")
    vectorstore = Chroma(
        collection_name="puapal_faqs",
        client_settings=client_settings,
        embedding_function=embedding
    )

print("Total docs in Chroma:", vectorstore._collection.count())
# end


llm = Ollama(model="granite3.3:8b")
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

prompt_template = """
You are an AI chatbot for the Admissions Office in Mapua Malayan Colleges Mindanao. Your job is to answer questions using the provided context.  
If the context does not contain the answer, politely say you don't have that information and encourage the user to contact the Admissions Office.

Rules you must follow:
1. Use the provided context to answer the question as accurately as possible.
2. Do not copy exact wording from the context. Paraphrase naturally.
3. If the context contains "*Questions*" and "*Response*":
   - Treat the text after "*Questions*" as examples of possible queries.
   - Treat the text after "*Response*" as the official answer.
   - Base your answer on the "*Response*" section, paraphrased in your own words.
4. If the context contains "*Context*" and "*Response*":
   - Use only the information from the "*Response*" section to answer.
   - Do not include "*Context*" or "*Response*" labels in your reply.
5. If the context has neither of these label pairs, ignore rules 3 and 4 and simply use the information given.
6. Never fabricate details beyond what is in the context.
7. Maintain a professional, friendly tone suitable for an Admissions Office representative.

<context>
{context}
</context>

Question: {question}
"""

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": PROMPT},
    return_source_documents=True,
)

# Retrieval Augmented Generation
# Chatbot Routes
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message", "")
    inquiry_type = request.json.get("inquiryType", None)
    program = request.json.get("program", None)

    print(f"User Message: {user_message}")
    print(f"Inquiry Type: {inquiry_type}")
    print(f"Program: {program}")

    # Build filter dict for Chroma
    filters = {}
    if inquiry_type and inquiry_type != "general":
        filters["inquiry_type"] = {"$eq": inquiry_type}

    if program and program != "general":
        filters["program"] = {"$eq": program}

    # Wrap multiple conditions inside $and
    if len(filters) > 1:
        filters = {"$and": [ {k:v} for k,v in filters.items() ]}
    elif len(filters) == 1:
        # leave as-is, e.g. {"inquiry_type": {"$eq": "admissions"}}
        pass
    else:
        filters = None
        
    print(f"Current filters are: {filters}")

    # Retrieve documents
    if inquiry_type == "general":
        retrieved_docs = vectorstore.similarity_search(
            user_message, k=6
        )
    else:
        retrieved_docs = vectorstore.similarity_search(
            user_message, k=6, filter=filters if filters else None
        )
    
    # Build context
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    # Query LLM
    response = llm.invoke(
        PROMPT.format(context=context, question=user_message)
    )
    print(f"First chatbot response: {response}")

    # Fallback if nothing retrieved
    if not retrieved_docs:
        response = (
            "I don’t have that specific information in the system. "
            "Please contact the Admissions Office directly for further assistance."
        )
    return jsonify({"response": response})


# Login functions    
@app.route("/login", methods=["POST"])
def login():
    username = request.form.get("username")
    password = request.form.get("password")
    
    try:
        conn = mysql.connector.connect(
            host="localhost",
            user="root",  
            password="",  
            database="chatbot_db",
            use_pure=True
        )
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM admin_users WHERE username = %s", (username,))
        user = cursor.fetchone()
        cursor.close()
        conn.close()

        if user and user["password"] == password:
            session["username"] = username
            return redirect("/admin")
        else:
            flash("Invalid username or password. Please try again.", "error")
            return redirect(url_for("index"))

    except mysql.connector.Error as e:
        print("MySQL Error:", e)
        flash("Database connection failed. Please contact admin.", "error")
        return redirect(url_for("index"))

    except Exception as e:
        print("Unexpected Error:", e)
        flash("Internal server error. Please try again later.", "error")
        return redirect(url_for("index"))
    

@app.route("/logout")
def logout():
    session.pop("username", None)
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for("index"))

# Admin route w CRUD Functions
@app.route("/admin")
def admin_dashboard():
    print("/trying to load mysql data")

    if "username" not in session:
        return redirect(url_for("index")) 

    search = request.args.get("search", "").strip()
    nature = request.args.get("nature_of_inquiry", "").strip()
    grade = request.args.get("grade_level", "").strip()
    program = request.args.get("program", "").strip()

    sort_order = request.args.get("sort", "desc")
    order_by = "ASC" if sort_order.lower() == "asc" else "DESC"

    try:
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="",
            database="chatbot_db",
            use_pure=True
        )
        cursor = conn.cursor(dictionary=True)

        # Build query dynamically
        query = "SELECT * FROM inquiries WHERE 1=1"
        params = []

        # Search across multiple fields
        if search:
            query += " AND (first_name LIKE %s OR last_name LIKE %s OR email LIKE %s OR student_number LIKE %s OR contact_number LIKE %s)"
            search_term = f"%{search}%"
            params.extend([search_term] * 5)

        # Filters
        if nature:
            query += " AND nature_of_inquiry = %s"
            params.append(nature)

        if grade:
            query += " AND grade_level = %s"
            params.append(grade)

        if program:
            query += " AND program = %s"
            params.append(program)

        # date sorting
        query += f" ORDER BY date_sent {order_by}"

        cursor.execute(query, tuple(params))
        inquiries = cursor.fetchall()

        cursor.close()
        conn.close()

        return render_template("admin_dashboard.html", inquiries=inquiries)

    except mysql.connector.Error as e:
        print("MySQL Error:", e)
        return "Database error", 500
    
@app.route('/contact', methods=['POST'])
def contact_agent():
    print("/contact route reached")

    print("testing if forms have data")
    # new checks if inputted data is wrong
    fname1 = request.form.get('first_name', '').strip()
    lname1 = request.form.get('last_name', '').strip()
    email1 = request.form.get('email', '').strip()
    student_number1 = request.form.get('student_number', '').strip()
    contact_number1 = request.form.get('contact_number', '').strip()
    nature_of_inquiry1 = request.form.get('nature_of_inquiry', '').strip()
    grade_level1 = request.form.get('grade_level', '').strip()
    program1 = request.form.get('program', '').strip()
    message1 = request.form.get('message', '').strip()

    # Check if any field is empty
    if not all([fname1, lname1, email1, student_number1, contact_number1, nature_of_inquiry1, grade_level1, program1, message1]):
        flash("All fields are required.", "error")
        return redirect(url_for('index'))  

    # Validate email format
    if not re.match(r'^[\w\.-]+@[\w\.-]+\.\w+$', email1):
        flash("Please enter a valid email address.", "error")
        return redirect(url_for('index'))  

    # Ensure student number and contact number are digits only
    if not student_number1.isdigit():
        flash("Student number should only contain numbers. Please try again.", "error")
        return redirect(url_for('index'))  

    if not contact_number1.isdigit():
        flash("Contact number should only contain numbers. Please try again.", "error")
        return redirect(url_for('index'))  
    

    print("data forms are complete, loading data to database")
    
    first_name = request.form.get('first_name')
    last_name = request.form.get('last_name')
    email = request.form.get('email')
    student_number = request.form.get('student_number')
    contact_number = request.form.get('contact_number')
    nature_of_inquiry = request.form.get('nature_of_inquiry')
    grade_level = request.form.get('grade_level')
    program = request.form.get('program')
    message = request.form.get('message')

    try:
        conn = mysql.connector.connect(
            host="localhost",
            user="root",      
            password="",      
            database="chatbot_db",
            use_pure=True
        )
        cursor = conn.cursor()

        # Insert into inquiries table
        cursor.execute("""
            INSERT INTO inquiries 
            (first_name, last_name, email, student_number, contact_number, 
             nature_of_inquiry, grade_level, program, message, date_sent)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,NOW())
        """, (
            first_name, last_name, email, student_number, contact_number,
            nature_of_inquiry, grade_level, program, message
        ))

        conn.commit()
        cursor.close()
        conn.close()

    except mysql.connector.Error as e:
        print("MySQL Error:", e)  
        return "Database connection failed", 500

    except Exception as e:
        print("Unexpected Error:", e) 
        return "Internal server error", 500
    
    flash("Your message has been successfully submitted!", "success")
    return redirect(url_for('index'))

#remove function dashboard
@app.route('/admin/delete/<int:id>', methods=['POST'])
def delete_inquiry(id):
    try:
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="",
            database="chatbot_db",
            use_pure=True
        )
        cursor = conn.cursor()
        cursor.execute("DELETE FROM inquiries WHERE id=%s", (id,))
        conn.commit()
        cursor.close()
        conn.close()
        flash("Inquiry deleted successfully.", "success")
    except mysql.connector.Error as e:
        flash(f"Database error: {e}", "error")

    filters = {
        "search": request.form.get("search", ""),
        "nature_of_inquiry": request.form.get("nature_of_inquiry", ""),
        "grade_level": request.form.get("grade_level", ""),
        "program": request.form.get("program", ""),
        "sort": request.form.get("sort", "desc"),
    }    

    return redirect(url_for('admin_dashboard', **filters))

#view message function
@app.route("/view_message/<int:id>", methods=["POST"])
def view_message(id):
    try:
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="",
            database="chatbot_db",
            use_pure=True
        )
        cursor = conn.cursor(dictionary=True)  

        cursor.execute("SELECT * FROM inquiries WHERE id = %s", (id,))
        inquiry = cursor.fetchone()

        if inquiry:
            flash({
                "first_name": inquiry['first_name'],
                "last_name": inquiry['last_name'],
                "email": inquiry['email'],
                "student_number": inquiry['student_number'],
                "contact_number": inquiry['contact_number'],
                "grade_level": inquiry['grade_level'],
                "program": inquiry['program'],
                "message": inquiry['message'],
                "date_sent": inquiry['date_sent'].strftime("%B %d, %Y %I:%M %p")
            }, "info")
        else:
            flash({"error": "Inquiry not found."}, "error")

        cursor.close()
        conn.close()

    except mysql.connector.Error as e:
        flash(f"Database error: {e}", "error")

    filters = {
    "search": request.form.get("search", ""),
    "nature_of_inquiry": request.form.get("nature_of_inquiry", ""),
    "grade_level": request.form.get("grade_level", ""),
    "program": request.form.get("program", ""),
    "sort": request.form.get("sort", "desc"),
}    

    return redirect(url_for("admin_dashboard", **filters))


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)

