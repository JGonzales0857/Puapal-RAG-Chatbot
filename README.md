# Admissions RAG Chatbot (Flask + Ollama + MySQL)

## Setup Instructions

1. Clone repo:
   ```bash
   git clone https://github.com/YourUser/rag-chatbot-admissions.git
   cd rag-chatbot-admissions

2. Create a virtual environment & install dependencies:
  python -m venv venv
  source venv/bin/activate   # Linux/Mac
  venv\Scripts\activate      # Windows
  pip install -r requirements.txt

3. Pull Ollama (Download Ollama first):
  ollama pull granite3.3:8b

4. Set up database:
  a. Open **phpMyAdmin** in your browser (usually at http://localhost/phpmyadmin).
  b. Create a new database named `chatbot_db`.
     - In phpMyAdmin, click **Databases** → type `chatbot_db` → click **Create**.
  c. Import the database schema:
     - Select the `chatbot_db` database in the left sidebar.
     - Go to the **Import** tab.
     - Click **Choose File** and select `chatbot_db.sql` (I sent you this).
     - Click **Go** to run the import.
  d. Verify:
     - After import, you should see tables like `admin_users`, `messages`, etc. inside `chatbot_db`.
  e. Update your `.env` file (or inside the code where database settings are configured) with your MySQL credentials:
    MYSQL_HOST=localhost (usually this)
    MYSQL_USER=root (usually this)
    MYSQL_PASSWORD=yourpassword or MYSQL_PASSWORD="" (usually password is blank)
    MYSQL_DB=chatbot_db (dont change, this is name of db)
