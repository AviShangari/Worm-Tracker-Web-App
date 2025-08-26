# Worm Tracker Web App

This project lets you upload a worm video, run automated tracking, and download all results (processed video + metadata) as a single ZIP package.

It has two parts:

- **Backend** (Python + FastAPI) — runs the worm-tracking code
- **Frontend** (React + Vite) — website interface for uploading and viewing

---

## Prerequisites

Install these first:

1. **Python 3.9+**

   - Download: <https://www.python.org/downloads/>
   - On Windows: check **“Add Python to PATH”** during installation.

2. **Node.js (v18 or newer)**

   - Download LTS version: <https://nodejs.org>
   - This also installs **npm**.

3. **Git** (to clone the repository)

   - Download: <https://git-scm.com/downloads>
   - Or click **Code → Download ZIP** on GitHub.

4. **FFmpeg** (recommended for video playback)
   - Windows: <https://www.gyan.dev/ffmpeg/builds/> or `choco install ffmpeg` (if using Chocolatey)
   - macOS: `brew install ffmpeg` (via Homebrew)
   - Linux: install via your package manager (`apt`, `dnf`, etc.).
   - Without FFmpeg, the app still works but some videos may not play in-browser.

---

## Setup Instructions

Follow these steps in order.

### 1. Clone the repository

```bash
git clone https://github.com/AviShangari/Worm-Tracker-Web-App.git
cd Worm-Tracker-Web-App
```

(Or unzip if you downloaded the ZIP.)

---

### 2. Install backend (Python)

Create a virtual environment:

```bash
python -m venv venv
```

Activate it:

- **Windows**
  ```bash
  venv\\Scripts\\activate
  ```
- **macOS/Linux**
  ```bash
  source venv/bin/activate
  ```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

### 3. Start the backend server

From the project root (the main folder):

```bash
uvicorn app.main:app --reload --port 8000
```

You should see:

```
Uvicorn running on http://127.0.0.1:8000
```

Leave this terminal running.

---

### 4. Install frontend (React)

Open a **new terminal**, then:

```bash
cd frontend
npm install
```

---

### 5. Start the frontend

Still inside `frontend`:

```bash
npm run dev
```

It will display something like:

```
Local:   http://127.0.0.1:5173/
```

Open that link in your browser.

---

## How to Use

1. Open the site in your browser (default: [http://127.0.0.1:5173](http://127.0.0.1:5173))
2. (Optional) Adjust parameters: **Keypoints**, **Area Threshold**, **Max Age**, **Output Name**
3. Click **Select video** and choose an MP4 or .mov file (recommended).
4. A progress bar will show while processing (The backend terminal shows the estimated time for processing, if needed).
5. Once done:
   - The processed video will appear and play in the browser
   - A **Download All (ZIP)** link will be available, containing:
     - Processed video (`.mp4`, H.264)
     - Metadata (`.yaml`)
     - Keypoints (`.npz`)
6. Use the **Run on another file** button to process a new video.

---

## Where Files Are Saved (Locally)

- Uploaded raw videos → `app/uploads/`
- Processed outputs → `app/outputs/<job_id>/`
  - Contains the processed MP4 (H.264), YAML, NPZ, and the packaged ZIP.
- These folders are **gitignored** and created automatically at runtime.

---

## Shutting Down The Web Application

- Press Ctrl + C (Windows) or Command + C (Mac) on both backend and frontend terminals to terminate the program

---

## Troubleshooting

- **“command not found” (pip, python, node, npm)**  
  Ensure Python/Node.js are installed and added to PATH. Close & reopen the terminal after installing.

- **pip not recognized**  
  Use:

  ```bash
  python -m pip install -r requirements.txt
  ```

- **npm install is slow**  
  First run may take a while; that’s normal.

- **Frontend opens but video won’t play**  
  Install **FFmpeg** (see prerequisites). The backend uses it to transcode to web-friendly H.264.  
  MP4 input is recommended.

- **CORS / network errors in the browser**  
  Make sure the backend is running at `http://127.0.0.1:8000` before starting the frontend.

- **“Address already in use”** when starting the frontend  
  Run with a different port:
  ```bash
  npm run dev -- --port 5174
  ```

---

## What’s Included vs. Ignored in Git

- **Included:** source code (`app/`, `frontend/`), configs, `requirements.txt`, this README.
- **Ignored:** `app/uploads/`, `app/outputs/`, `frontend/node_modules/`, build artifacts, and large media files.

---
