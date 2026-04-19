from dotenv import load_dotenv

load_dotenv()

import uvicorn


if __name__ == "__main__":
    uvicorn.run("console_app.app:app", host="127.0.0.1", port=8000, reload=False)
