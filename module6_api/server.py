from flask import Flask, request, jsonify
import sqlite3
from datetime import datetime

app = Flask(__name__)

def init_db():
    conn = sqlite3.connect("plates.db")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS plates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plate_number TEXT NOT NULL,
            timestamp TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

@app.route("/api/plates", methods=["POST"])
def receive_plate():
    data = request.get_json()
    plate_number = data.get("plate_number")

    if not plate_number:
        return jsonify({"error": "plate_number manquant"}), 400

    conn = sqlite3.connect("plates.db")
    conn.execute(
        "INSERT INTO plates (plate_number, timestamp) VALUES (?, ?)",
        (plate_number, datetime.now().isoformat())
    )
    conn.commit()
    conn.close()

    print(f"Plaque reçue et stockée : {plate_number}")
    return jsonify({"status": "ok", "plate_number": plate_number}), 201

if __name__ == "__main__":
    init_db()
    app.run(host="0.0.0.0", port=5000)