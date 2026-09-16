@echo off
echo ========================================================
echo Starting SmartNews MongoDB Standalone Instance...
echo ========================================================
set MONGO_BIN=C:\Program Files\MongoDB\Server\8.3\bin\mongod.exe
set DB_PATH=%~dp0..\backend\data\db

if not exist "%DB_PATH%" (
    mkdir "%DB_PATH%"
)

if exist "%MONGO_BIN%" (
    echo Using MongoDB at: %MONGO_BIN%
    "%MONGO_BIN%" --dbpath "%DB_PATH%" --port 27017 --bind_ip 127.0.0.1
) else (
    echo MongoDB binary not found at %MONGO_BIN%
    echo Attempting default system mongod...
    mongod --dbpath "%DB_PATH%" --port 27017 --bind_ip 127.0.0.1
)
