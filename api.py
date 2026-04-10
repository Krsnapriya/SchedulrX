from schedulrx.seed import set_seed
set_seed(42)

from server.app import app

def main():
    import uvicorn
    # Use the string-based factory to ensure reloader finds the app correctly
    uvicorn.run("api:app", host="0.0.0.0", port=8001, reload=False)

if __name__ == "__main__":
    main()
