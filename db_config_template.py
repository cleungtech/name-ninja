"""
DB Configuration Template

This file is a template for database configuration. To set up your database connection:
1. Copy this file and rename the copy to 'db_config.py'.
2. Replace the placeholder values (e.g., "localhost", "root", "", "name_ninja") with your actual database settings.

The 'get_db_connection' function will use these settings to establish a connection to your MySQL database.
"""
import mysql.connector

def get_db_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        passwd="",
        database="name_ninja"
    )
