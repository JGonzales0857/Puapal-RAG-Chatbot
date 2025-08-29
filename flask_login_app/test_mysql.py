import mysql.connector

print("Starting MySQL connection test...")

try:
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="",  # blank password
        database="chatbot_db",
        use_pure=True
    )
    print("✅ MySQL connection successful!")
    conn.close()
except mysql.connector.Error as err:
    print("❌ MySQL error:", err)
except Exception as e:
    print("❌ Other error:", e)

input("Press Enter to exit...")
