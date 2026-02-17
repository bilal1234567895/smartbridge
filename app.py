import sqlite3
import os
import cv2
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.xception import preprocess_input

# -----------------------------
# Flask setup
# -----------------------------
app = Flask(__name__)
app.secret_key = "dr_project_secret"
app.config["UPLOAD_FOLDER"] = "uploads"

# -----------------------------
# Database setup (SQLite)
# -----------------------------
DB_PATH = "database/users.db"

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    os.makedirs("database", exist_ok=True)
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

init_db()   # <-- creates database & table

# -----------------------------
# Load trained model
# -----------------------------
model = load_model("model/Updated-Xception-diabetic-retinopathy.h5")

CLASSES = ["No_DR", "Mild", "Moderate", "Severe", "Proliferative_DR"]

# -----------------------------
# Image preprocessing
# -----------------------------
def preprocess_image(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (299, 299))
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    return img

# -----------------------------
# Routes
# -----------------------------
@app.route("/")
def index():
    return render_template("index.html")

# -------- Register --------
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        conn = get_db_connection()
        cursor = conn.cursor()

        try:
            cursor.execute(
                "INSERT INTO users (username, password) VALUES (?, ?)",
                (username, password)
            )
            conn.commit()
            flash("Registration successful! Please login.")
            return redirect(url_for("login"))

        except sqlite3.IntegrityError:
            flash("Username already exists!")

        finally:
            conn.close()

    return render_template("register.html")

# -------- Login --------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM users WHERE username=? AND password=?",
            (username, password)
        )
        user = cursor.fetchone()
        conn.close()

        if user:
            return redirect(url_for("dashboard"))
        else:
            flash("Invalid username or password")

    return render_template("login.html")

# -------- Dashboard --------
@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

# -------- Prediction --------
@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["image"]
    path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(path)

    img = preprocess_image(path)
    pred = model.predict(img)

    result = CLASSES[np.argmax(pred)]
    confidence = round(np.max(pred) * 100, 2)

    return render_template(
        "prediction.html",
        result=result,
        confidence=confidence
    )

# -------- Logout --------
@app.route("/logout")
def logout():
    return render_template("logout.html")

# -----------------------------
# Run app
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)
