# Admissions RAG Chatbot (Flask + Ollama + MySQL)

## Setup Instructions

* Clone repo:
   ```bash
   git clone https://github.com/YourUser/rag-chatbot-admissions.git
   cd rag-chatbot-admissions

* Create a virtual environment & install dependencies:
     * python -m venv venv
     * source venv/bin/activate   # Linux/Mac
     * venv\Scripts\activate      # Windows
     * pip install -r requirements.txt

* Pull Ollama (Download Ollama first):
     * ollama pull granite3.3:8b

* Set up database:
  * Open **phpMyAdmin** in your browser (usually at http://localhost/phpmyadmin).
  * Create a new database named `chatbot_db`.
     * In phpMyAdmin, click **Databases** → type `chatbot_db` → click **Create**.
  * Import the database schema:
  * Select the `chatbot_db` database in the left sidebar.
  * Go to the **Import** tab.
  * Click **Choose File** and select `chatbot_db.sql` (I sent you this).
  * Click **Go** to run the import.
* Verify:
  * After import, you should see tables like `admin_users`, `messages`, etc. inside `chatbot_db`.
* Update your `.env` file (or inside the code where database settings are configured) with your MySQL credentials:
    * (MYSQL_HOST=localhost (usually this)
    * MYSQL_USER=root (usually this)
    * MYSQL_PASSWORD=yourpassword or MYSQL_PASSWORD="" (usually password is blank)
    * MYSQL_DB=chatbot_db (dont change, this is name of db)
