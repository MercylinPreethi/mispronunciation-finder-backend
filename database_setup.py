#!/usr/bin/env python3
"""
Database Setup Script for Pronunciation Coach (Cluster Safe)
"""

import mysql.connector
from mysql.connector import Error
import os
from werkzeug.security import generate_password_hash

# Database configuration from environment variables
DATABASE_CONFIG = {
    'host': os.environ.get('DB_HOST', 'localhost'),
    'user': os.environ.get('DB_USER', 'root'),
    'password': os.environ.get('DB_PASSWORD', 'mcw34.pass'),
    'port': int(os.environ.get('DB_PORT', '3306')),
}
DATABASE_NAME = os.environ.get('DB_NAME', 'pronunciation_coach')


def connect_to_server(db=None):
    """Helper: connect to MySQL server, optionally to a specific database"""
    cfg = DATABASE_CONFIG.copy()
    if db:
        cfg['database'] = db
    return mysql.connector.connect(**cfg)


def setup_database():
    """Create database and required tables"""
    try:
        print("Connecting to MySQL server...")
        connection = connect_to_server()
        cursor = connection.cursor()

        print(f"Creating database '{DATABASE_NAME}'...")
        cursor.execute(
            f"CREATE DATABASE IF NOT EXISTS {DATABASE_NAME} "
            f"CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
        )
        print(f"✓ Database '{DATABASE_NAME}' ready")

        cursor.close()
        connection.close()

        # Reconnect to the new database
        connection = connect_to_server(DATABASE_NAME)
        cursor = connection.cursor()

        # ---- Create tables ----
        print("Creating tables...")

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INT AUTO_INCREMENT PRIMARY KEY,
            account_id VARCHAR(50) UNIQUE NOT NULL,
            username VARCHAR(100) NOT NULL,
            email VARCHAR(255) UNIQUE NOT NULL,
            password_hash VARCHAR(255) NOT NULL,
            profile_url TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            is_active BOOLEAN DEFAULT TRUE,
            last_login TIMESTAMP NULL,
            INDEX idx_email (email),
            INDEX idx_account_id (account_id)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
        """)
        print("✓ Users table")

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_preferences (
            id INT AUTO_INCREMENT PRIMARY KEY,
            user_id INT NOT NULL,
            audio_feedback_enabled BOOLEAN DEFAULT TRUE,
            feedback_voice_speed DECIMAL(3,1) DEFAULT 0.9,
            target_accuracy INT DEFAULT 85,
            practice_reminders BOOLEAN DEFAULT TRUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
            UNIQUE KEY unique_user_prefs (user_id)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
        """)
        print("✓ User preferences table")

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_statistics (
            id INT AUTO_INCREMENT PRIMARY KEY,
            user_id INT NOT NULL,
            total_sessions INT DEFAULT 0,
            total_attempts INT DEFAULT 0,
            average_accuracy DECIMAL(5,2) DEFAULT 0.00,
            best_accuracy DECIMAL(5,2) DEFAULT 0.00,
            current_streak INT DEFAULT 0,
            last_practice_date DATE NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
            UNIQUE KEY unique_user_stats (user_id)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
        """)
        print("✓ User statistics table")

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS practice_sessions (
            id INT AUTO_INCREMENT PRIMARY KEY,
            session_id VARCHAR(100) UNIQUE NOT NULL,
            user_id INT NOT NULL,
            reference_text TEXT NOT NULL,
            accuracy_score DECIMAL(5,2),
            analysis_result JSON,
            feedback TEXT,
            processing_time DECIMAL(8,3),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
            INDEX idx_user_sessions (user_id, created_at),
            INDEX idx_session_id (session_id)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
        """)
        print("✓ Practice sessions table")

        connection.commit()

        # Create test user
        create_test_user(cursor, connection)

        print("\n✅ Database setup completed successfully")

    except Error as e:
        print(f"❌ Error: {e}")
        print("Check: MySQL running? Credentials correct? Privileges OK?")
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()
            print("MySQL connection closed")


def create_test_user(cursor, connection):
    """Create a test user if not exists"""
    cursor.execute("SELECT id FROM users WHERE email = %s", ('test@example.com',))
    if cursor.fetchone():
        print("Test user already exists")
        return

    print("Creating test user...")
    password_hash = generate_password_hash('mcw34.pass', method='pbkdf2:sha256')

    cursor.execute(
        "INSERT INTO users (account_id, username, email, password_hash) VALUES (%s, %s, %s, %s)",
        ('test_account_001', 'Test User', 'test@example.com', password_hash)
    )
    user_id = cursor.lastrowid

    cursor.execute("INSERT INTO user_preferences (user_id) VALUES (%s)", (user_id,))
    cursor.execute("INSERT INTO user_statistics (user_id) VALUES (%s)", (user_id,))

    connection.commit()
    print("✓ Test user created (email: test@example.com / pass: mcw34.pass)")


if __name__ == "__main__":
    print("=== Pronunciation Coach Database Setup ===")
    setup_database()
