# FlashPhoto-AI

FlashPhoto-AI is a web application built with Flask that allows users to create photo-sharing events, upload event photos, and use facial recognition to instantly find pictures of themselves. The app processes photos in the background using DeepFace to extract and store facial encodings, making photo retrieval fast and accurate.

## Features

- **Event Creation**: Generate unique events with a secure access code.
- **Photo Uploading**: Upload multiple photos to an event. Photos are processed in the background to extract face encodings.
- **Facial Recognition Matching**: Upload a "selfie" to an event and instantly find all photos containing your face.
- **GPU Acceleration**: Utilizes PyTorch and CUDA for fast face processing, with built-in endpoints for managing GPU memory.

## Tech Stack

- **Backend**: Flask (Python)
- **Database**: MySQL via SQLAlchemy (with Flask-Migrate)
- **Face Recognition**: DeepFace, OpenCV, PyTorch
- **Production Server**: Waitress
- **Concurrency**: Python's concurrent.futures for background tasks

## Prerequisites

- Python 3.8+
- MySQL Server
- (Optional but recommended) CUDA-compatible GPU for faster face processing

## Setup & Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Prince1334/FlashPhoto-AI.git
   cd FlashPhoto-AI
   ```

2. **Install dependencies**:
   ```bash
   pip install -r "FlashPhoto AI/app/requirements.txt"
   ```

3. **Database Configuration**:
   The app expects a MySQL database. Create a database (e.g., `flashphoto_db`) and set the following environment variables in a `.env` file (or just rely on the defaults):
   - `DB_HOST`: Database host (default: localhost)
   - `DB_USER`: Database user (default: root)
   - `DB_PASSWORD`: Database password (default: root)
   - `DB_NAME`: Database name (default: flashphoto_db)
   - `SECRET_KEY`: Flask secret key

4. **Run the Application**:
   Navigate into the application folder and run the app:
   ```bash
   cd "FlashPhoto AI"
   python run.py --dev
   ```
   To run in production mode (via Waitress):
   ```bash
   python run.py
   ```

## API Endpoints

- `POST /api/events` - Create a new event and receive an access code.
- `POST /api/events/<access_code>/upload` - Upload a photo to an event (processes face encodings in the background).
- `POST /api/events/<access_code>/match` - Upload a selfie to find matching photos within an event.
- `DELETE /api/events/<access_code>/clear` - Clear all photos and data for an event.
- `POST /api/gpu/clear-cache` - Free unused tensors from the PyTorch CUDA memory cache.

## Project Structure

- `app/` - Core application directory (models, routes, services).
  - `routes/api.py` - API endpoints for event and photo management.
  - `services/face_service.py` - Functions for face extraction and matching.
- `run.py` - Application entry point.
