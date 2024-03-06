#!/bin/sh

sqlite3 Chinook.db << 'END_SQL'
.read Chinook_Sqlite.sql
END_SQL