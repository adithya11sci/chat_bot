import uvicorn

if __name__ == "__main__":
    print("🚀 Starting BookMind server...")
    print("📖 Open http://localhost:8000 in your browser\n")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
