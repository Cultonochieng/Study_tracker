import sqlite3
import time
from datetime import datetime, timedelta
import yaml
import os
import click
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image, Flowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import numpy as np
from typing import Dict, List, Optional, Tuple
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
from prompt_toolkit import prompt
from prompt_toolkit.completion import WordCompleter
import random
import json
import requests
from bs4 import BeautifulSoup
from rich.console import Console
from rich.table import Table as RichTable
from rich import print as rprint
from textblob import TextBlob
import qrcode
from io import BytesIO
import base64
from rich.progress import Progress
from rich.panel import Panel
from cryptography.fernet import Fernet
import io
import asyncio
from PIL import Image, ImageDraw
from reportlab.graphics.shapes import Drawing, Line
from reportlab.lib.units import inch, mm
from reportlab.pdfgen import canvas
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.graphics.shapes import String
from reportlab.lib.pagesizes import letter, landscape
from reportlab.graphics.charts.piecharts import Pie
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.linecharts import LineChart
from reportlab.graphics.charts.legends import Legend
from reportlab.graphics.widgets.markers import makeMarker
import threading
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from reportlab.lib import colors  # Import the colors module
from queue import Queue
class UserProfile:
    def __init__(self):
        self.age = None
        self.education = None
        self.career_goals = []
        self.learning_preferences = []
        self.current_skills = []
        self.desired_skills = []
        self.daily_schedule = {}
    def setup_profile(self):
        self.age = click.prompt("Enter your age", type=int)
        self.education = click.prompt("Enter your highest education level")
        self.career_goals = click.prompt("Enter your career goals (comma-separated)").split(',')
        self.learning_preferences = click.prompt("Enter your learning preferences (comma-separated)").split(',')
        self.current_skills = click.prompt("Enter your current skills (comma-separated)").split(',')
        self.desired_skills = click.prompt("Enter skills you want to learn (comma-separated)").split(',')
        self.daily_schedule = {}
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        for day in days:
            self.daily_schedule[day] = click.prompt(f"Enter available study hours for {day} (e.g., 18:00-20:00)")
    def save_to_db(self, db):
        db.save_user_profile(self)
    def load_from_db(self, db):
        profile = db.load_user_profile()
        if profile:
            self.__dict__.update(profile.__dict__)
        else:
            print("No user profile found in the database. You may want to set up a new profile.")
class GoalTracker:
    def __init__(self, db, user_profile):
        self.db = db
        self.user_profile = user_profile
        self.goals = []
    def set_goal(self):
        goal = click.prompt("Enter your study goal")
        target_date = click.prompt("Enter target date (YYYY-MM-DD)")
        self.goals.append({"goal": goal, "target_date": target_date, "progress": 0})
        self.db.save_goal(self.goals[-1])
    def update_progress(self):
        if not self.goals:
            print("No goals set. Please set a goal first.")
            return
        for i, goal in enumerate(self.goals):
            print(f"{i+1}. {goal['goal']} (Target: {goal['target_date']}, Progress: {goal['progress']}%)")
        goal_index = click.prompt("Select a goal to update", type=int) - 1
        new_progress = click.prompt("Enter new progress percentage", type=int)
        self.goals[goal_index]['progress'] = new_progress
        self.db.update_goal_progress(self.goals[goal_index]['goal'], new_progress)
    def visualize_progress(self):
        if not self.goals:
            print("No goals set. Please set a goal first.")
            return
        for goal in self.goals:
            print(f"Goal: {goal['goal']}")
            print(f"Target Date: {goal['target_date']}")
            print(f"Progress: {goal['progress']}%")
            print("█" * int(goal['progress'] / 2) + "░" * (50 - int(goal['progress'] / 2)))
            print()
    def load_goals(self):
        self.goals = self.db.load_goals()
class HealthTracker:
    def __init__(self):
        self.sleep_hours = []
        self.exercise_minutes = []
    def track_sleep(self):
        sleep_hours = click.prompt("Enter hours of sleep last night", type=float)
        self.sleep_hours.append(sleep_hours)
        return sleep_hours
    def track_exercise(self):
        exercise_minutes = click.prompt("Enter minutes of exercise today", type=int)
        self.exercise_minutes.append(exercise_minutes)
        return exercise_minutes
    def remind_break(self):
        print("Remember to take a short break and stretch!")
    def suggest_exercise(self):
        exercises = ["Take a short walk", "Do 10 push-ups", "Perform 2 minutes of jumping jacks", "Do a quick yoga stretch"]
        print(f"Quick exercise suggestion: {random.choice(exercises)}")
    def get_average_sleep(self):
        if not self.sleep_hours:
            return 0
        return sum(self.sleep_hours) / len(self.sleep_hours)
    def get_average_exercise(self):
        if not self.exercise_minutes:
            return 0
        return sum(self.exercise_minutes) / len(self.exercise_minutes)
# Configuration
CONFIG_FILE = 'config.yaml'
class Config:
    def __init__(self):
        self.data: Dict[str, Dict] = {
            'skill_mastery_times': {
                'Erlang': 1250,
                'Rust': 1750,
                'Python': 1250,
                'JavaScript': 1500,
                'Angular TypeScript': 1000,
                'HTML': 400,
                'CSS': 1000,
                'encryption_key': Fernet.generate_key().decode(),
            },
            'quotes': [
                "Code is like humor. When you have to explain it, it's bad.",
                "First, solve the problem. Then, write the code.",
                "Experience is the name everyone gives to their mistakes.",
                "The only way to learn a new programming language is by writing programs in it.",
                "Sometimes it's better to leave something alone, to pause, and that's very true of programming.",
                "Testing leads to failure, and failure leads to understanding.",
                "The most damaging phrase in the language is 'We've always done it this way.'",
                "The best error message is the one that never shows up.",
                "The most important property of a program is whether it accomplishes the intention of its user.",
                "Programming isn't about what you know; it's about what you can figure out.",
            ],
            'pdf_theme': {
                'primary_color': '#007bff',
                'secondary_color': '#6c757d',
                'font': 'Helvetica',
            },
            'pomodoro': {
                'work_duration': 25,
                'break_duration': 5,
                'long_break_duration': 15,
                'long_break_interval': 4
            },
            'gamification': {
                'enabled': True,
                'levels': [
                    {'name': 'Novice', 'xp_required': 0},
                    {'name': 'Apprentice', 'xp_required': 1000},
                    {'name': 'Journeyman', 'xp_required': 5000},
                    {'name': 'Expert', 'xp_required': 15000},
                    {'name': 'Master', 'xp_required': 50000}
                ],
                'xp_per_minute': 1
            },
            'encryption_key': Fernet.generate_key().decode(),
        }
    def load(self) -> None:
        try:
            if os.path.exists(CONFIG_FILE):
                with open(CONFIG_FILE, 'r') as file:
                    loaded_config = yaml.safe_load(file)
                    self.data.update(loaded_config)
            else:
                self.save()
        except yaml.YAMLError as e:
            print(f"Error loading configuration: {e}")
        except IOError as e:
            print(f"Error reading configuration file: {e}")
    def save(self) -> None:
        try:
            with open(CONFIG_FILE, 'w') as file:
                yaml.dump(self.data, file)
        except IOError as e:
            print(f"Error saving configuration: {e}")
    def get_encryption_key(self) -> bytes:
        if 'encryption_key' not in self.data:
            self.data['encryption_key'] = Fernet.generate_key().decode()
            self.save()
        return self.data['encryption_key'].encode()
config = Config()
config.load()
# Database operations
class Database:
    def __init__(self, db_name='study_tracker.db', pool_size=5):
        self.db_name = db_name
        self.pool_size = pool_size
        self.connection_pool = Queue(maxsize=pool_size)
        self.lock = threading.Lock()
        for _ in range(pool_size):
            conn = self.create_connection()
            self.connection_pool.put(conn)
        self.create_tables()
    def create_connection(self):
        try:
            conn = sqlite3.connect(self.db_name, check_same_thread=False)
            return conn
        except sqlite3.Error as e:
            print(f"Error connecting to database: {e}")
            return None
    def get_connection(self):
        return self.connection_pool.get()
    def release_connection(self, conn):
        self.connection_pool.put(conn)
    def execute_query(self, query, params=None):
        conn = self.get_connection()
        try:
            with self.lock:
                if params:
                    result = conn.execute(query, params)
                else:
                    result = conn.execute(query)
                conn.commit()
            return result
        finally:
            self.release_connection(conn)
    def create_tables(self):
        tables = [
            '''
            CREATE TABLE IF NOT EXISTS study_sessions (
                id INTEGER PRIMARY KEY,
                language TEXT NOT NULL,
                duration REAL NOT NULL,
                date TEXT NOT NULL,
                start_time TEXT NOT NULL,
                notes TEXT,
                productivity_score INTEGER,
                mood TEXT,
                energy_level INTEGER
            )
            ''',
            '''
            CREATE TABLE IF NOT EXISTS languages (
                id INTEGER PRIMARY KEY,
                name TEXT UNIQUE NOT NULL,
                target_hours REAL NOT NULL
            )
            ''',
            '''
            CREATE TABLE IF NOT EXISTS resources (
                id INTEGER PRIMARY KEY,
                language TEXT NOT NULL,
                url TEXT NOT NULL,
                title TEXT NOT NULL,
                description TEXT,
                category TEXT,
                tags TEXT
            )
            ''',
            '''
            CREATE TABLE IF NOT EXISTS user_profile (
                id INTEGER PRIMARY KEY,
                xp INTEGER NOT NULL DEFAULT 0,
                level TEXT NOT NULL DEFAULT 'Novice',
                streak INTEGER NOT NULL DEFAULT 0,
                last_study_date TEXT,
                learning_style TEXT,
                total_study_time REAL NOT NULL DEFAULT 0,
                age INTEGER,
                education TEXT,
                career_goals TEXT,
                learning_preferences TEXT,
                current_skills TEXT,
                desired_skills TEXT,
                daily_schedule TEXT
            )
            ''',
            '''
            CREATE TABLE IF NOT EXISTS achievements (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT NOT NULL,
                unlocked BOOLEAN NOT NULL DEFAULT 0,
                unlock_date TEXT
            )
            ''',
            '''
            CREATE TABLE IF NOT EXISTS spaced_repetition (
                id INTEGER PRIMARY KEY,
                language TEXT NOT NULL,
                concept TEXT NOT NULL,
                last_review_date TEXT,
                next_review_date TEXT,
                difficulty INTEGER
            )
            ''',
            '''
            CREATE TABLE IF NOT EXISTS goals (
                id INTEGER PRIMARY KEY,
                user_id INTEGER,
                goal TEXT NOT NULL,
                target_date TEXT NOT NULL,
                progress INTEGER DEFAULT 0,
                FOREIGN KEY (user_id) REFERENCES user_profile (id)
            )
            ''',
            '''
            CREATE TABLE IF NOT EXISTS health_tracking (
                id INTEGER PRIMARY KEY,
                user_id INTEGER,
                date TEXT NOT NULL,
                sleep_hours REAL,
                breaks_taken INTEGER,
                exercises_done INTEGER,
                FOREIGN KEY (user_id) REFERENCES user_profile (id)
            )
            '''
        ]
        for table_query in tables:
            self.execute_query(table_query)
    def log_study_session(self, language, duration, start_time, notes, productivity_score, mood, energy_level):
        query = '''
            INSERT INTO study_sessions (language, duration, date, start_time, notes, productivity_score, mood, energy_level)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        '''
        params = (language, duration, datetime.now().strftime('%Y-%m-%d'), start_time, notes, productivity_score, mood, energy_level)
        self.execute_query(query, params)
    def get_languages(self):
        query = 'SELECT name FROM languages'
        result = self.execute_query(query)
        return [row[0] for row in result.fetchall()]
    def add_language(self, language, target_hours):
        query = 'INSERT INTO languages (name, target_hours) VALUES (?, ?)'
        self.execute_query(query, (language, target_hours))
    def get_study_data(self):
        query = 'SELECT language, SUM(duration) FROM study_sessions GROUP BY language'
        result = self.execute_query(query)
        return {row[0]: row[1] for row in result.fetchall()}
    def get_productive_times(self):
        query = '''
            SELECT
                CASE
                    WHEN CAST(substr(start_time, 1, 2) AS INTEGER) < 6 THEN 'Night'
                    WHEN CAST(substr(start_time, 1, 2) AS INTEGER) < 12 THEN 'Morning'
                    WHEN CAST(substr(start_time, 1, 2) AS INTEGER) < 18 THEN 'Afternoon'
                    ELSE 'Evening'
                END AS time_of_day,
                AVG(duration) as avg_duration
            FROM study_sessions
            GROUP BY time_of_day
        '''
        result = self.execute_query(query)
        return {row[0]: row[1] for row in result.fetchall()}
    def add_resource(self, language, url, title, description, category, tags):
        query = '''
            INSERT INTO resources (language, url, title, description, category, tags)
            VALUES (?, ?, ?, ?, ?, ?)
        '''
        self.execute_query(query, (language, url, title, description, category, tags))
    def get_resources(self, language):
        query = '''
            SELECT url, title, description, category, tags
            FROM resources
            WHERE language = ?
        '''
        result = self.execute_query(query, (language,))
        return result.fetchall()
    def save_user_profile(self, user_profile):
        query = '''
            INSERT OR REPLACE INTO user_profile
            (id, age, education, career_goals, learning_preferences, current_skills, desired_skills, daily_schedule)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        '''
        params = (
            1,  # Assuming single user for personal use
            user_profile.age,
            user_profile.education,
            json.dumps(user_profile.career_goals),
            json.dumps(user_profile.learning_preferences),
            json.dumps(user_profile.current_skills),
            json.dumps(user_profile.desired_skills),
            json.dumps(user_profile.daily_schedule)
        )
        self.execute_query(query, params)
    def load_user_profile(self):
        query = 'SELECT * FROM user_profile WHERE id = 1'
        result = self.execute_query(query)
        row = result.fetchone()
        if row:
            user_profile = UserProfile()
            user_profile.age = row[7] if len(row) > 7 else None
            user_profile.education = row[8] if len(row) > 8 else None
            user_profile.career_goals = json.loads(row[9]) if len(row) > 9 and row[9] else []
            user_profile.learning_preferences = json.loads(row[10]) if len(row) > 10 and row[10] else []
            user_profile.current_skills = json.loads(row[11]) if len(row) > 11 and row[11] else []
            user_profile.desired_skills = json.loads(row[12]) if len(row) > 12 and row[12] else []
            user_profile.daily_schedule = json.loads(row[13]) if len(row) > 13 and row[13] else {}
            return user_profile
        return None
    def save_goal(self, goal):
        query = '''
            INSERT INTO goals (user_id, goal, target_date, progress)
            VALUES (?, ?, ?, ?)
        '''
        self.execute_query(query, (1, goal['goal'], goal['target_date'], goal['progress']))
    def load_goals(self):
        query = 'SELECT goal, target_date, progress FROM goals WHERE user_id = 1'
        result = self.execute_query(query)
        return [{'goal': row[0], 'target_date': row[1], 'progress': row[2]} for row in result.fetchall()]
    def update_goal_progress(self, goal, progress):
        query = '''
            UPDATE goals SET progress = ? WHERE user_id = 1 AND goal = ?
        '''
        self.execute_query(query, (progress, goal))
    def save_health_data(self, date, sleep_hours, breaks_taken, exercises_done):
        query = '''
            INSERT OR REPLACE INTO health_tracking (user_id, date, sleep_hours, breaks_taken, exercises_done)
            VALUES (?, ?, ?, ?, ?)
        '''
        self.execute_query(query, (1, date, sleep_hours, breaks_taken, exercises_done))
    def load_health_data(self, start_date, end_date):
        query = '''
            SELECT date, sleep_hours, breaks_taken, exercises_done
            FROM health_tracking
            WHERE user_id = 1 AND date BETWEEN ? AND ?
            ORDER BY date
        '''
        result = self.execute_query(query, (start_date, end_date))
        return result.fetchall()
    def get_study_duration_for_date(self, date):
        query = '''
            SELECT SUM(duration) FROM study_sessions WHERE date = ?
        '''
        result = self.execute_query(query, (date,))
        return result.fetchone()[0] or 0.0
    def get_spaced_repetition_progress(self):
        query = '''
            SELECT concept, language, last_review_date, next_review_date, difficulty
            FROM spaced_repetition
            ORDER BY next_review_date
        '''
        result = self.execute_query(query)
        return [
            {
                'concept': row[0],
                'language': row[1],
                'last_review': row[2],
                'next_review': row[3],
                'difficulty': row[4]
            }
            for row in result.fetchall()
        ]
    def get_user_profile(self):
        query = 'SELECT xp, level, streak, last_study_date, learning_style, total_study_time FROM user_profile WHERE id = 1'
        result = self.execute_query(query)
        row = result.fetchone()
        if row:
            return row
        return (0, 'Novice', 0, None, None, 0.0)  # Default values if no profile exists
    def update_user_profile(self, xp, level, streak, last_study_date, learning_style, total_study_time):
        query = '''
            UPDATE user_profile
            SET xp = ?, level = ?, streak = ?, last_study_date = ?, learning_style = ?, total_study_time = ?
            WHERE id = 1
        '''
        self.execute_query(query, (xp, level, streak, last_study_date, learning_style, total_study_time))
    def get_due_spaced_repetition_items(self):
        query = '''
            SELECT id, language, concept, last_review_date, difficulty
            FROM spaced_repetition
            WHERE next_review_date <= ?
            ORDER BY next_review_date
        '''
        result = self.execute_query(query, (datetime.now().strftime('%Y-%m-%d'),))
        return result.fetchall()
    def update_spaced_repetition_item(self, item_id, new_difficulty, new_review_date):
        query = '''
            UPDATE spaced_repetition
            SET difficulty = ?, next_review_date = ?, last_review_date = ?
            WHERE id = ?
        '''
        self.execute_query(query, (new_difficulty, new_review_date, datetime.now().strftime('%Y-%m-%d'), item_id))
    def unlock_achievement(self, name):
        query = '''
            INSERT OR REPLACE INTO achievements (name, description, unlocked, unlock_date)
            VALUES (?, (SELECT description FROM achievements WHERE name = ?), 1, ?)
        '''
        self.execute_query(query, (name, name, datetime.now().strftime('%Y-%m-%d')))
    def close(self):
        while not self.connection_pool.empty():
            conn = self.connection_pool.get()
            conn.close()
# Advanced Analysis
class AdvancedAnalysis:
    def __init__(self, db: Database):
        self.db = db
    def get_historical_data(self, language: str) -> List[Tuple[str, float]]:
        conn = self.db.get_connection()
        try:
            cursor = conn.execute('''
                SELECT date, SUM(duration)
                FROM study_sessions
                WHERE language = ?
                GROUP BY date
                ORDER BY date
            ''', (language,))
            return cursor.fetchall()
        finally:
            self.db.release_connection(conn)
    def predict_progress(self, language: str) -> Optional[float]:
        data = self.get_historical_data(language)
        if len(data) < 10:
            return None
        X = np.array(range(len(data))).reshape(-1, 1)
        y = np.array([d[1] for d in data])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        next_day = len(data)
        return model.predict([[next_day]])[0]
    def analyze_productive_times(self) -> Dict[str, float]:
        return self.db.get_productive_times()
    def generate_insights(self) -> List[str]:
        study_data = self.db.get_study_data()
        insights = []
        if study_data:
            most_studied = max(study_data, key=study_data.get)
            least_studied = min(study_data, key=study_data.get)
            insights.append(f"You've spent the most time on {most_studied}.")
            insights.append(f"Consider dedicating more time to {least_studied}.")
        productive_times = self.analyze_productive_times()
        if productive_times:
            best_time = max(productive_times, key=productive_times.get)
            insights.append(f"You're most productive during {best_time}.")
        return insights
    def get_learning_recommendations(self, language: str) -> List[str]:
        study_data = self.db.get_study_data()
        if language not in study_data:
            return ["Start with the basics and fundamentals of the language."]
        hours_studied = study_data[language]
        if hours_studied < 10:
            return [
                "Focus on syntax and basic concepts.",
                "Try simple coding exercises and small projects.",
                "Use interactive tutorials and coding platforms."
            ]
        elif hours_studied < 50:
            return [
                "Start working on more complex projects.",
                "Dive deeper into language-specific features and best practices.",
                "Participate in coding challenges to improve problem-solving skills."
            ]
        else:
            return [
                "Contribute to open-source projects.",
                "Explore advanced topics and design patterns.",
                "Consider teaching or mentoring others to reinforce your knowledge."
            ]
    def detect_burnout_risk(self, language: str) -> str:
        data = self.get_historical_data(language)
        if len(data) < 7:
            return "Insufficient data to assess burnout risk."
        recent_data = data[-7:]
        daily_hours = [d[1] for d in recent_data]
        avg_daily_hours = sum(daily_hours) / len(daily_hours)
        if avg_daily_hours > 6:
            return "High risk of burnout. Consider taking more breaks."
        elif avg_daily_hours > 4:
            return "Moderate risk of burnout. Ensure you're balancing study with rest."
        else:
            return "Low risk of burnout. Keep up the sustainable pace!"
    def analyze_learning_style(self, user_data: Dict[str, any]) -> str:
        night_owl = sum(1 for session in user_data['study_sessions'] if session[0] >= '22:00' or session[0] < '06:00')
        early_bird = sum(1 for session in user_data['study_sessions'] if '06:00' <= session[0] < '10:00')
        long_sessions = sum(1 for session in user_data['study_sessions'] if session[1] > 2)
        short_sessions = sum(1 for session in user_data['study_sessions'] if session[1] <= 1)
        if night_owl > early_bird:
            time_preference = "Night Owl"
        else:
            time_preference = "Early Bird"
        if long_sessions > short_sessions:
            session_preference = "Long Sessions"
        else:
            session_preference = "Short Sessions"
        return f"{time_preference} with preference for {session_preference}"
# NLP for Notes Analysis
class NLP:
    def __init__(self):
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        self.stop_words = set(stopwords.words('english'))

    def analyze_notes(self, notes: str) -> Dict[str, int]:
        words = word_tokenize(notes.lower())
        words = [w for w in words if w.isalnum() and w not in self.stop_words]
        return dict(Counter(words).most_common(10))
    def sentiment_analysis(self, notes: str) -> str:
        blob = TextBlob(notes)
        sentiment = blob.sentiment.polarity
        if sentiment > 0.5:
            return "Very Positive"
        elif sentiment > 0:
            return "Positive"
        elif sentiment == 0:
            return "Neutral"
        elif sentiment > -0.5:
            return "Negative"
        else:
            return "Very Negative"
    def extract_topics(self, notes: str) -> List[str]:
        # Simple topic extraction using noun phrases
        blob = TextBlob(notes)
        return [phrase.string for phrase in blob.noun_phrases]
# Advanced Scheduler
class AdvancedScheduler:
    def __init__(self, db: Database):
        self.db = db
    def suggest_study_time(self) -> str:
        productive_times = self.db.get_productive_times()
        if productive_times:
            best_time = max(productive_times, key=productive_times.get)
            return f"Based on your past sessions, your most productive time is during {best_time}."
        return "Not enough data to suggest a study time yet. Keep logging your sessions!"
    def generate_weekly_schedule(self) -> Dict[str, List[str]]:
        study_data = self.db.get_study_data()
        total_hours = sum(study_data.values())
        schedule = {day: [] for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']}
        for language, hours in study_data.items():
            sessions_per_week = max(1, int((hours / total_hours) * 7))
            days = random.sample(list(schedule.keys()), sessions_per_week)
            for day in days:
                schedule[day].append(language)
        return schedule
    def adaptive_learning_path(self, language: str) -> List[str]:
        study_data = self.db.get_study_data()
        hours_studied = study_data.get(language, 0)
        if hours_studied < 10:
            return [
                "Learn basic syntax and data types",
                "Understand control structures (if, for, while)",
                "Practice with simple coding exercises",
                "Explore built-in functions and methods"
            ]
        elif hours_studied < 50:
            return [
                "Study object-oriented programming concepts",
                "Learn about data structures (lists, dictionaries, etc.)",
                "Explore file I/O operations",
                "Start working on small projects"
            ]
        elif hours_studied < 100:
            return [
                "Dive into advanced language features",
                "Learn about libraries and frameworks",
                "Practice with coding challenges and algorithms",
                "Contribute to open-source projects"
            ]
        else:
            return [
                "Explore design patterns and best practices",
                "Learn about performance optimization",
                "Study advanced topics (e.g., concurrency, networking)",
                "Work on complex, real-world projects"
            ]
# Time tracking
class StudySession:
    def __init__(self, db: Database, language: str):
        self.db = db
        self.language = language
        self.start_time = time.time()
        self.session_start = datetime.now()
        self.duration = 0
        self.breaks = []
        self.notes = ""
        self.productivity_score = 0
        self.mood = ""
        self.energy_level = 0
    def pause(self):
        self.duration += time.time() - self.start_time
        self.breaks.append(time.time())
    def resume(self):
        self.start_time = time.time()
    def end(self):
        self.duration += time.time() - self.start_time
        self.prompt_for_notes()
        self.prompt_for_mood_and_energy()
        self.calculate_productivity_score()
        self.db.log_study_session(
            self.language,
            self.duration / 3600,
            self.session_start.strftime('%H:%M'),
            self.notes,
            self.productivity_score,
            self.mood,
            self.energy_level
        )
    def prompt_for_notes(self):
        self.notes = click.prompt("Enter any notes for this session (optional)", default="")
    def prompt_for_mood_and_energy(self):
        self.mood = click.prompt("How would you describe your mood during this session? (e.g., focused, distracted, excited)", default="neutral")
        self.energy_level = click.prompt("Rate your energy level during this session (1-10)", type=int, default=5)
    def calculate_productivity_score(self):
        if len(self.breaks) >= 2:
            total_break_time = sum(self.breaks[i+1] - self.breaks[i] for i in range(0, len(self.breaks), 2))
        else:
            total_break_time = 0
        effective_study_time = self.duration - total_break_time
        self.productivity_score = int((effective_study_time / self.duration) * 100)
def start_session(db: Database) -> None:
    try:
        languages = db.get_languages()
        console = Console()
        console.print("Available languages:", style="bold green")
        for lang in languages:
            console.print(f"- {lang}")
        language_completer = WordCompleter(languages)
        language = prompt("Enter language to study: ", completer=language_completer).capitalize()
        if language not in languages:
            add = click.confirm(f"{language} is not in the database. Add it?")
            if add:
                target_hours = click.prompt(f"Enter target hours for {language}", type=float)
                db.add_language(language, target_hours)
            else:
                return
        scheduler = AdvancedScheduler(db)
        suggestion = scheduler.suggest_study_time()
        console.print(Panel(suggestion, title="Study Time Suggestion", border_style="cyan"))
        advanced_analysis = AdvancedAnalysis(db)
        recommendations = advanced_analysis.get_learning_recommendations(language)
        console.print("Learning recommendations:", style="bold blue")
        for rec in recommendations:
            console.print(f"- {rec}")
        burnout_risk = advanced_analysis.detect_burnout_risk(language)
        console.print(Panel(burnout_risk, title="Burnout Risk Assessment", border_style="yellow"))
        resources = db.get_resources(language)
        if resources:
            console.print("Helpful resources:", style="bold magenta")
            table = RichTable(title=f"Top Resources for {language}")
            table.add_column("Title", style="cyan")
            table.add_column("Category", style="green")
            table.add_column("URL", style="blue")
            for url, title, description, category, _ in resources[:3]:  # Show top 3 resources
                table.add_row(title, category, url)
            console.print(table)
        pomodoro = click.confirm("Do you want to use the Pomodoro technique?", default=True)
        if pomodoro:
            pomodoro_session(db, language)
        else:
            regular_session(db, language)
        print("\nQuote of the day:")
        print(random.choice(config.data['quotes']))
        update_gamification(db)
    except Exception as e:
        console.print(f"An error occurred: {e}", style="bold red")
def pomodoro_session(db: Database, language: str) -> None:
    work_duration = config.data['pomodoro']['work_duration']
    break_duration = config.data['pomodoro']['break_duration']
    long_break_duration = config.data['pomodoro']['long_break_duration']
    long_break_interval = config.data['pomodoro']['long_break_interval']
    session = StudySession(db, language)
    pomodoro_count = 0
    health_tracker = HealthTracker()
    console = Console()
    try:
        while True:
            pomodoro_count += 1
            console.print(f"\nPomodoro #{pomodoro_count}", style="bold green")
            console.print(f"Work for {work_duration} minutes. Press Enter to start...")
            input()
            with Progress() as progress:
                work_task = progress.add_task("[red]Working", total=work_duration * 60)
                for _ in range(work_duration * 60):
                    time.sleep(1)
                    progress.update(work_task, advance=1)
            session.pause()
            health_tracker.remind_break()
            health_tracker.suggest_exercise()
            if pomodoro_count % long_break_interval == 0:
                console.print(f"\nLong break for {long_break_duration} minutes. Press Enter when you're back...")
                input()
                with Progress() as progress:
                    break_task = progress.add_task("[green]Long Break", total=long_break_duration * 60)
                    for _ in range(long_break_duration * 60):
                        time.sleep(1)
                        progress.update(break_task, advance=1)
            else:
                console.print(f"\nShort break for {break_duration} minutes. Press Enter when you're back...")
                input()
                with Progress() as progress:
                    break_task = progress.add_task("[cyan]Short Break", total=break_duration * 60)
                    for _ in range(break_duration * 60):
                        time.sleep(1)
                        progress.update(break_task, advance=1)
            session.resume()
            continue_pomodoro = click.confirm("\nDo you want to continue with another Pomodoro?")
            if not continue_pomodoro:
                break
    except KeyboardInterrupt:
        console.print("\nPomodoro session interrupted. Ending session.", style="bold yellow")
    finally:
        session.end()
        nlp = NLP()
        key_concepts = nlp.analyze_notes(session.notes)
        console.print("Key concepts from your notes:", style="bold yellow")
        for concept, count in key_concepts.items():
            console.print(f"  {concept}: {count}")
        sentiment = nlp.sentiment_analysis(session.notes)
        console.print(f"Session sentiment: {sentiment}", style="bold blue")
        topics = nlp.extract_topics(session.notes)
        if topics:
            console.print("Extracted topics:", style="bold magenta")
            for topic in topics:
                console.print(f"- {topic}")
def regular_session(db: Database, language: str) -> None:
    console = Console()
    console.print(f"Starting study session for {language}. Press Ctrl+C to stop.", style="bold green")
    session = StudySession(db, language)
    try:
        with Progress() as progress:
            study_task = progress.add_task("[cyan]Studying", total=None)
            while True:
                time.sleep(1)
                progress.update(study_task, advance=1)
    except KeyboardInterrupt:
        session.end()
        nlp = NLP()
        key_concepts = nlp.analyze_notes(session.notes)
        console.print("Key concepts from your notes:", style="bold yellow")
        for concept, count in key_concepts.items():
            console.print(f"  {concept}: {count}")

        sentiment = nlp.sentiment_analysis(session.notes)
        console.print(f"Session sentiment: {sentiment}", style="bold blue")

        topics = nlp.extract_topics(session.notes)
        if topics:
            console.print("Extracted topics:", style="bold magenta")
            for topic in topics:
                console.print(f"- {topic}")
# Gamification
def update_gamification(db: Database):
    xp, current_level, streak, last_study_date, learning_style, total_study_time = db.get_user_profile()
    # Update XP
    conn = db.get_connection()
    try:
        cursor = conn.execute(
            'SELECT duration FROM study_sessions ORDER BY id DESC LIMIT 1'
        )
        session_duration = cursor.fetchone()[0]
    finally:
        db.release_connection(conn)
    xp_gained = int(session_duration * config.data['gamification']['xp_per_minute'])
    xp += xp_gained
    # Update level
    for level in config.data['gamification']['levels']:
        if xp >= level['xp_required']:
            current_level = level['name']
    # Update streak
    today = datetime.now().date()
    if last_study_date:
        last_study_date = datetime.strptime(last_study_date, '%Y-%m-%d').date()
        if today - last_study_date == timedelta(days=1):
            streak += 1
        elif today - last_study_date > timedelta(days=1):
            streak = 1
    else:
        streak = 1
    # Update total study time
    total_study_time += session_duration
    # Update learning style if not set
    if not learning_style:
        conn = db.get_connection()
        try:
            cursor = conn.execute('SELECT start_time, duration FROM study_sessions')
            user_data = {'study_sessions': cursor.fetchall()}
        finally:
            db.release_connection(conn)
        learning_style = AdvancedAnalysis(db).analyze_learning_style(user_data)
    db.update_user_profile(xp, current_level, streak, today.strftime('%Y-%m-%d'), learning_style, total_study_time)
    console = Console()
    console.print(Panel(f"You gained {xp_gained} XP!", title="XP Update", border_style="green"))
    console.print(f"Current level: {current_level}", style="bold blue")
    console.print(f"Study streak: {streak} days", style="bold yellow")
    # Check for achievements
    def check_achievements(db: Database, xp: int, level: str, streak: int, total_study_time: float):
        achievements = [
            ("First Session", "Complete your first study session", xp > 0),
            ("Level Up", "Reach level 'Apprentice'", level == "Apprentice"),
            ("Streak Master", "Maintain a 7-day study streak", streak >= 7),
            ("Century Club", "Study for a total of 100 hours", total_study_time >= 100),
        ]
        for name, description, condition in achievements:
            if condition:
                db.unlock_achievement(name)
                console = Console()
                console.print(f"Achievement unlocked: {name}", style="bold green")
                console.print(description)
    # Call check_achievements function
    check_achievements(db, xp, current_level, streak, total_study_time)
# Reporting
def create_progress_bar(percentage):
    width, height = 400, 20
    img = Image.new('RGB', (width, height), color='lightgrey')
    draw = ImageDraw.Draw(img)
    fill_width = int(width * percentage / 100)
    draw.rectangle([0, 0, fill_width, height], fill='green')
    img_buffer = io.BytesIO()
    img.save(img_buffer, format='PNG')
    img_buffer.seek(0)
    return img_buffer
def generate_pdf_report(db: Database):
    try:
        doc = SimpleDocTemplate("enhanced_study_report.pdf", pagesize=landscape(letter),
                                rightMargin=20, leftMargin=20, topMargin=20, bottomMargin=20)
        styles = getSampleStyleSheet()
        elements = []
        # Enhanced color palette with gradients
        color_palette = [
            (colors.Color(0.9, 0.2, 0.2), colors.Color(1, 0.4, 0.4)),
            (colors.Color(0.2, 0.8, 0.8), colors.Color(0.4, 1, 1)),
            (colors.Color(0.2, 0.6, 0.8), colors.Color(0.4, 0.8, 1)),
            (colors.Color(1, 0.6, 0.4), colors.Color(1, 0.8, 0.6)),
            (colors.Color(0.6, 0.8, 0.6), colors.Color(0.8, 1, 0.8)),
            (colors.Color(0.9, 0.8, 0.2), colors.Color(1, 1, 0.4))
        ]
        # Title with gradient
        title_style = ParagraphStyle('Title', parent=styles['Title'], fontSize=32,
                                     textColor=colors.HexColor('#2C3E50'), spaceAfter=25)
        elements.append(Paragraph("Comprehensive Language Learning Analytics", title_style))
        # Multithreaded data loading
        with ThreadPoolExecutor() as executor:
            user_profile_future = executor.submit(db.load_user_profile)
            study_data_future = executor.submit(db.get_study_data)
            user_profile = user_profile_future.result()
            study_data = study_data_future.result()
        # User Profile Analysis (Enhanced)
        elements.append(Paragraph("Learner Profile Insights", styles['Heading1']))
        profile_data = [
            ['Attribute', 'Value', 'Impact on Learning'],
            ['Age', str(user_profile.age), age_impact_analysis(user_profile.age)],
            ['Education', user_profile.education, education_impact_analysis(user_profile.education)],
            ['Career Goals', ', '.join(user_profile.career_goals), career_goals_impact_analysis(user_profile.career_goals)],
            ['Learning Preferences', ', '.join(user_profile.learning_preferences), learning_preferences_impact_analysis(user_profile.learning_preferences)]
        ]
        profile_table = Table(profile_data, colWidths=[100, 180, 320])
        profile_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#34495E')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 13),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#ECF0F1')),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#34495E')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#F0F3F4'), colors.HexColor('#E8EAED')])
        ]))
        elements.append(profile_table)
        elements.append(Spacer(1, 20))
        # Study Data Analysis
        total_hours = sum(study_data.values())
        # Pie Chart for Language Distribution (Enhanced with 3D effect)
        elements.append(Paragraph("Language Study Distribution", styles['Heading2']))
        pie = Pie()
        pie.x = 50
        pie.y = 50
        pie.width = 250
        pie.height = 250
        pie.data = list(study_data.values())
        pie.labels = list(study_data.keys())
        pie.slices.strokeWidth = 0.5
        pie.slices[0].popout = 10
        pie.slices[0].labelRadius = 1.2
        pie.slices[0].fontColor = colors.HexColor('#FFFFFF')
        for i, slice in enumerate(pie.slices):
            slice.fillColor = color_palette[i % len(color_palette)][0]
        legend = Legend()
        legend.alignment = 'right'
        legend.x = 310
        legend.y = 150
        legend.colorNamePairs = list(zip([color_palette[i % len(color_palette)][0] for i in range(len(study_data))],
                                         [f"{lang}: {hours:.1f}h" for lang, hours in study_data.items()]))
        d = Drawing(460, 300)
        d.add(pie)
        d.add(legend)
        elements.append(d)
        elements.append(Spacer(1, 20))
        # Bar Chart for Study Hours (Enhanced with gradient)
        elements.append(Paragraph("Study Hours per Language", styles['Heading2']))
        bc = VerticalBarChart()
        bc.x = 50
        bc.y = 50
        bc.height = 200
        bc.width = 400
        bc.data = [list(study_data.values())]
        bc.strokeColor = colors.HexColor('#34495E')
        bc.valueAxis.valueMin = 0
        bc.valueAxis.valueMax = max(study_data.values()) * 1.1
        bc.valueAxis.valueStep = max(study_data.values()) / 5
        bc.categoryAxis.labels.boxAnchor = 'ne'
        bc.categoryAxis.labels.dx = 8
        bc.categoryAxis.labels.dy = -2
        bc.categoryAxis.labels.angle = 30
        bc.categoryAxis.categoryNames = list(study_data.keys())
        for i, (start_color, end_color) in enumerate(color_palette):
            bc.bars[0][i].fillColor = colors.linearlyInterpolatedColor(start_color, end_color, 0, bc.height, bc.data[0][i])
        d = Drawing(500, 250)
        d.add(bc)
        elements.append(d)
        elements.append(Spacer(1, 20))
        # Study Efficiency Analysis (Enhanced with conditional formatting)
        elements.append(Paragraph("Study Efficiency Analysis", styles['Heading2']))
        efficiency_data = [['Language', 'Hours', 'Efficiency Score', 'Recommendation']]
        max_hours = max(study_data.values())
        for lang, hours in study_data.items():
            efficiency_score = (hours / max_hours) * 10
            recommendation = generate_recommendation(efficiency_score)
            efficiency_data.append([lang, f"{hours:.2f}", f"{efficiency_score:.2f}", recommendation])
        efficiency_table = Table(efficiency_data, colWidths=[90, 70, 70, 370])
        efficiency_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), color_palette[2][0]),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#ECF0F1')),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#34495E'))
        ]))
        for i in range(1, len(efficiency_data)):
            score = float(efficiency_data[i][2])
            if score < 3:
                bg_color = colors.Color(1, 0.7, 0.7)
            elif 3 <= score < 7:
                bg_color = colors.Color(1, 1, 0.7)
            else:
                bg_color = colors.Color(0.7, 1, 0.7)
            efficiency_table.setStyle(TableStyle([('BACKGROUND', (2, i), (2, i), bg_color)]))
        elements.append(efficiency_table)
        elements.append(Spacer(1, 20))
        # Learning Progress Over Time (Enhanced with smooth curves)
        elements.append(Paragraph("Learning Progress Over Time", styles['Heading2']))
        progress_data = generate_progress_data(study_data)
        lc = LineChart()
        lc.x = 50
        lc.y = 50
        lc.height = 250
        lc.width = 500
        lc.data = progress_data
        lc.joinedLines = 1
        for i, color in enumerate(color_palette):
            lc.lines[i].strokeColor = color[0]
            lc.lines[i].strokeWidth = 2
            lc.lines[i].symbol = makeMarker('FilledCircle', size=5, fillColor=color[1], strokeColor=None)
        lc.valueAxis.valueMin = 0
        lc.valueAxis.valueMax = max([max(data) for data in progress_data]) * 1.1
        lc.valueAxis.valueStep = max([max(data) for data in progress_data]) / 5
        lc.categoryAxis.categoryNames = ['Week 1', 'Week 2', 'Week 3', 'Week 4']
        lc.categoryAxis.labels.boxAnchor = 's'
        lc.categoryAxis.labels.angle = 0
        legend = Legend()
        legend.alignment = 'right'
        legend.x = 600
        legend.y = 150
        legend.colorNamePairs = list(zip([color[0] for color in color_palette[:len(study_data)]],
                                         list(study_data.keys())))
        d = Drawing(650, 300)
        d.add(lc)
        d.add(legend)
        elements.append(d)
        elements.append(Spacer(1, 20))
        # Summary Statistics (Enhanced with icons)
        elements.append(Paragraph("Study Summary Statistics", styles['Heading2']))
        stats_data = [
            ["📚 Total Study Hours", f"{total_hours:.2f}"],
            ["⌛ Average Hours per Language", f"{total_hours / len(study_data):.2f}"],
            ["🏆 Most Studied Language", max(study_data, key=study_data.get)],
            ["🔍 Least Studied Language", min(study_data, key=study_data.get)],
            ["🕰️ Study Time Range", f"{min(study_data.values()):.1f} - {max(study_data.values()):.1f} hours"],
            ["💯 Overall Efficiency Score", f"{sum(study_data.values()) / (len(study_data) * max(study_data.values())) * 10:.2f}/10"]
        ]
        stats_table = Table(stats_data, colWidths=[200, 200])
        stats_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F0F3F4')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#2C3E50')),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 12),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#BDC3C7'))
        ]))
        elements.append(stats_table)
        # Personalized Recommendations (Enhanced with custom formatting)
        elements.append(Spacer(1, 20))
        elements.append(Paragraph("Personalized Learning Recommendations", styles['Heading2']))
        recommendations = generate_personalized_recommendations(user_profile, study_data)
        for i, recommendation in enumerate(recommendations, 1):
            p = Paragraph(f"{i}. {recommendation}",
                          ParagraphStyle('Recommendation',
                                         parent=styles['BodyText'],
                                         textColor=colors.HexColor('#2C3E50'),
                                         borderColor=color_palette[i % len(color_palette)][0],
                                         borderWidth=1,
                                         borderPadding=5,
                                         borderRadius=5))
            elements.append(p)
            elements.append(Spacer(1, 10))
        # Build the PDF
        doc.build(elements)
        print("Enhanced visual PDF report generated: enhanced_study_report.pdf")
    except ImportError as e:
        print(f"Error: Unable to generate enhanced PDF report. {str(e)}")
        print("Please ensure the reportlab library is installed.")
        print("You can install it using: pip install reportlab")
# Helper functions (optimized with caching)
@lru_cache(maxsize=None)
def age_impact_analysis(age):
    if age < 18:
        return "Young learners often have high neuroplasticity, which can aid in language acquisition."
    elif 18 <= age < 30:
        return "Adults in this age range often have a good balance of cognitive abilities and life experiences."
    else:
        return "Older learners can leverage their life experiences and may have more structured learning approaches."
@lru_cache(maxsize=None)
def education_impact_analysis(education):
    education_levels = {
        'High School': "Basic academic skills are present. Focus on building study habits.",
        'Bachelor': "Strong foundation in academic skills. Can handle more advanced language concepts.",
        'Master': "Advanced academic skills. Can engage in more complex language learning strategies.",
        'PhD': "Highly developed research skills. Can approach language learning with academic rigor."
    }
    return education_levels.get(education, "Education level impact not specified.")
@lru_cache(maxsize=None)
def career_goals_impact_analysis(goals):
    goals = tuple(goals) if goals else ()  # Convert list to tuple
    if not goals:
        return "No specific career goals mentioned. Consider defining career objectives to guide language learning."
    elif any('international' in goal.lower() for goal in goals):
        return "International career goals align well with language learning. Focus on professional vocabulary."
    else:
        return "Consider how language skills can enhance your career prospects in your chosen field."
@lru_cache(maxsize=None)
def learning_preferences_impact_analysis(preferences):
    preferences = tuple(preferences) if preferences else ()  # Convert list to tuple
    if not preferences:
        return "No specific learning preferences mentioned. Experiment with different learning styles."
    elif 'visual' in preferences:
        return "Visual learners benefit from diagrams, charts, and video content. Incorporate these in your study."
    elif 'auditory' in preferences:
        return "Auditory learners benefit from listening exercises and spoken practice. Focus on these areas."
    else:
        return "Your learning preferences suggest a mixed approach. Utilize varied study methods."
@lru_cache(maxsize=None)
def generate_recommendation(efficiency_score):
    if efficiency_score < 3:
        return "Consider increasing study time and trying new learning methods."
    elif 3 <= efficiency_score < 7:
        return "Good progress. Try to identify and focus on areas where you can improve efficiency."
    else:
        return "Excellent efficiency. Maintain your current study habits and consider helping others."

def generate_progress_data(study_data):
    # This is a placeholder. In a real scenario, you'd use actual historical data.
    return [[random.randint(0, 10) for _ in range(4)] for _ in range(len(study_data))]
def generate_personalized_recommendations(user_profile, study_data):
    recommendations = [
        f"Based on your age of {user_profile.age}, consider {age_impact_analysis(user_profile.age)}",
        f"Your education level suggests: {education_impact_analysis(user_profile.education)}",
        f"Career impact: {career_goals_impact_analysis(tuple(user_profile.career_goals))}",
        f"Learning style recommendation: {learning_preferences_impact_analysis(tuple(user_profile.learning_preferences))}"
    ]
    least_studied = min(study_data, key=study_data.get)
    recommendations.append(f"Consider dedicating more time to {least_studied} to balance your studies.")
    return recommendations
# New helper functions for enhanced visualizations
def create_gradient_fill(color1, color2):
    """Create a gradient fill between two colors."""
    from reportlab.graphics.shapes import LinearGradient
    grad = LinearGradient(x1=0, y1=0, x2=0, y2=100, colors=[color1, color2])
    return grad
def add_shadow(drawing, shape):
    """Add a shadow effect to a shape in the drawing."""
    shadow = shape.copy()
    shadow.fillColor = colors.black
    shadow.strokeColor = None
    shadow.opacity = 0.3
    shadow.dx = 3
    shadow.dy = -3
    drawing.insert(0, shadow)
def create_3d_effect(drawing, shape):
    """Create a pseudo-3D effect for a shape."""
    for i in range(5):
        shadow = shape.copy()
        shadow.fillColor = colors.black
        shadow.strokeColor = None
        shadow.opacity = 0.1 - (i * 0.02)
        shadow.dx = i
        shadow.dy = -i
        drawing.insert(0, shadow)
def generate_text_report(db: Database):
    report = "Comprehensive Study Progress Report\n"
    report += "===================================\n\n"
    # User Profile Summary
    user_profile = db.load_user_profile()
    report += "User Profile:\n"
    report += f"Age: {user_profile.age}\n"
    report += f"Education: {user_profile.education}\n"
    report += f"Career Goals: {', '.join(user_profile.career_goals)}\n"
    report += f"Learning Preferences: {', '.join(user_profile.learning_preferences)}\n\n"
    # Summary statistics
    study_data = db.get_study_data()
    total_hours = sum(study_data.values())
    report += f"Total Study Time: {total_hours:.2f} hours\n\n"
    # Language breakdown
    report += "Language Breakdown:\n"
    for lang, hours in study_data.items():
        percentage = (hours / total_hours) * 100 if total_hours > 0 else 0
        report += f"{lang}: {hours:.2f} hours ({percentage:.1f}%)\n"
    report += "\n"
    # Goal Progress
    report += "Goal Progress:\n"
    goals = db.load_goals()
    for goal in goals:
        report += f"Goal: {goal['goal']}\n"
        report += f"Target Date: {goal['target_date']}\n"
        report += f"Progress: {goal['progress']}%\n"
        report += "\n"
    # Productivity Analysis
    report += "Productivity Analysis:\n"
    productive_times = db.get_productive_times()
    for time, duration in productive_times.items():
        report += f"{time}: {duration:.2f} hours\n"
    report += "\n"
    return report
# Main program loop
def main_loop():
    db = Database()
    console = Console()
    user_profile = UserProfile()
    user_profile.load_from_db(db)
    goal_tracker = GoalTracker(db, user_profile)
    goal_tracker.load_goals()
    health_tracker = HealthTracker()
    while True:
        try:
            console.print("\nStudy Tracker Menu:", style="bold blue")
            console.print("1. Start a study session")
            console.print("2. View study statistics")
            console.print("3. Add a learning resource")
            console.print("4. View learning resources")
            console.print("5. Generate PDF report")
            console.print("6. Review spaced repetition items")
            console.print("7. Set up user profile")
            console.print("8. Set a new goal")
            console.print("9. Update goal progress")
            console.print("10. View goal progress")
            console.print("11. Track sleep")
            console.print("12. Exit")
            try:
                choice = click.prompt("Enter your choice", type=int)
            except click.exceptions.BadParameter:
                console.print("Invalid input. Please enter a number.", style="bold red")
                continue
            if choice == 1:
                if not user_profile.age:
                    console.print("Please set up your user profile first (option 7).", style="bold yellow")
                    continue
                start_session(db)
            elif choice == 2:
                study_data = db.get_study_data()
                console.print("\nStudy Statistics:", style="bold green")
                table = RichTable(title="Language Study Time")
                table.add_column("Language", style="cyan")
                table.add_column("Hours", style="magenta")
                for language, hours in study_data.items():
                    table.add_row(language, f"{hours:.2f}")
                console.print(table)
                advanced_analysis = AdvancedAnalysis(db)
                insights = advanced_analysis.generate_insights()
                console.print("\nInsights:", style="bold yellow")
                for insight in insights:
                    console.print(f"- {insight}")
            elif choice == 3:
                language = click.prompt("Enter language")
                url = click.prompt("Enter resource URL")
                title = click.prompt("Enter resource title")
                description = click.prompt("Enter resource description")
                category = click.prompt("Enter resource category")
                tags = click.prompt("Enter tags (comma-separated)")
                db.add_resource(language, url, title, description, category, tags)
                console.print("Resource added successfully!", style="bold green")
            elif choice == 4:
                language = click.prompt("Enter language to view resources")
                resources = db.get_resources(language)
                if resources:
                    console.print(f"\nResources for {language}:", style="bold blue")
                    table = RichTable(title=f"Learning Resources for {language}")
                    table.add_column("Title", style="cyan")
                    table.add_column("Category", style="green")
                    table.add_column("URL", style="blue")
                    for url, title, description, category, _ in resources:
                        table.add_row(title, category, url)
                    console.print(table)
                else:
                    console.print(f"No resources found for {language}", style="yellow")
            elif choice == 5:
                generate_pdf_report(db)
            elif choice == 6:
                due_items = db.get_due_spaced_repetition_items()
                if due_items:
                    console.print("\nItems due for review:", style="bold green")
                    for item_id, language, concept, last_review_date, difficulty in due_items:
                        console.print(f"Concept: {concept} (Language: {language})")
                        recall_quality = click.prompt("Rate your recall (0-5)", type=int)
                        new_difficulty = max(1, min(5, difficulty + (3 - recall_quality)))
                        new_interval = 1 if recall_quality <= 3 else 2 ** (new_difficulty - 1)
                        new_review_date = (datetime.now() + timedelta(days=new_interval)).strftime('%Y-%m-%d')
                        db.update_spaced_repetition_item(item_id, new_difficulty, new_review_date)
                    console.print("Review completed!", style="bold blue")
                else:
                    console.print("No items due for review.", style="yellow")
            elif choice == 7:
                user_profile.setup_profile()
                user_profile.save_to_db(db)
                console.print("User profile updated successfully!", style="bold green")
            elif choice == 8:
                goal_tracker.set_goal()
                console.print("New goal set successfully!", style="bold green")
            elif choice == 9:
                goal_tracker.update_progress()
                console.print("Goal progress updated successfully!", style="bold green")
            elif choice == 10:
                goal_tracker.visualize_progress()
            elif choice == 11:
                sleep_hours = health_tracker.track_sleep()
                console.print(f"Recorded sleep duration: {sleep_hours} hours", style="bold blue")
                exercise_minutes = health_tracker.track_exercise()
                console.print(f"Recorded exercise duration: {exercise_minutes} minutes", style="bold blue")
                db.save_health_data(datetime.now().strftime('%Y-%m-%d'), sleep_hours, 0, exercise_minutes)
                console.print("Health data saved successfully!", style="bold green")
            elif choice == 12:
                console.print("Thank you for using Study Tracker. Goodbye!", style="bold green")
                break
            else:
                console.print("Invalid choice. Please try again.", style="bold red")
        except Exception as e:
            console.print(f"An error occurred: {e}", style="bold red")
            if click.confirm("Do you want to continue?", default=True):
                continue
            else:
                break
    db.close()
def main():
    main_loop()
if __name__ == "__main__":
    main()

