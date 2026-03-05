import os
import uuid
import torch
from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename
from app.models import Event, Photo, FaceEncoding
from app.extensions import db
from app.services.face_service import extract_faces_from_image, find_matching_photos

from app.tasks import executor

api_bp = Blueprint('api', __name__, url_prefix='/api')

def process_photo_background(filepath, photo_id, app_context):
    """Background task to extract faces and save encodings."""
    # Ensure background thread has access to Flask's context for DB operations
    with app_context:
        try:
            encodings = extract_faces_from_image(filepath)
            for enc in encodings:
                face_encoding = FaceEncoding(photo_id=photo_id, encoding_list=enc)
                db.session.add(face_encoding)
            db.session.commit()
            print(f"Processed photo {photo_id} successfully.")
        except Exception as e:
            print(f"Error processing photo {photo_id} in background: {e}")

@api_bp.route('/events', methods=['POST'])
def create_event():
    data = request.json
    name = data.get('name', 'Unnamed Event')
    access_code = str(uuid.uuid4())[:8]
    
    new_event = Event(name=name, access_code=access_code)
    db.session.add(new_event)
    db.session.commit()
    
    return jsonify({"success": True, "access_code": access_code, "name": name})

@api_bp.route('/events/<access_code>/upload', methods=['POST'])
def upload_photo(access_code):
    event = Event.query.filter_by(access_code=access_code).first()
    if not event:
        return jsonify({"success": False, "error": "Event not found"}), 404

    if 'file' not in request.files:
        return jsonify({"success": False, "error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"success": False, "error": "No selected file"}), 400
        
    unique_prefix = str(uuid.uuid4())[:8]
    filename = f"{unique_prefix}_{secure_filename(file.filename)}"
    filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    photo = Photo(event_id=event.id, file_path=filename)
    db.session.add(photo)
    db.session.commit() # Save the photo entry first
    
    # Dispatch processing to the background thread
    executor.submit(process_photo_background, filepath, photo.id, current_app.app_context())
        
    return jsonify({
        "success": True, 
        "photo_id": photo.id, 
        "filename": filename, 
        "status": "processing_in_background"
    }), 202

@api_bp.route('/events/<access_code>/clear', methods=['DELETE'])
def clear_event_data(access_code):
    """Delete all photos and face encodings for an event, and remove files from disk."""
    event = Event.query.filter_by(access_code=access_code).first()
    if not event:
        return jsonify({"success": False, "error": "Event not found"}), 404

    photos = Photo.query.filter_by(event_id=event.id).all()
    deleted_files = 0
    errors = []

    for photo in photos:
        # Delete physical file from disk
        filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], photo.file_path)
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
                deleted_files += 1
        except Exception as e:
            errors.append(str(e))

        # Cascade delete handles FaceEncoding rows via the relationship
        db.session.delete(photo)

    db.session.commit()

    return jsonify({
        "success": True,
        "deleted_photos": len(photos),
        "deleted_files": deleted_files,
        "errors": errors
    })

@api_bp.route('/gpu/clear-cache', methods=['POST'])
def clear_gpu_cache():
    """Free all unused tensors from the PyTorch CUDA memory cache."""
    if not torch.cuda.is_available():
        return jsonify({"success": False, "error": "No CUDA GPU available"})

    before_reserved = torch.cuda.memory_reserved(0)
    before_alloc    = torch.cuda.memory_allocated(0)

    torch.cuda.empty_cache()          # release cached-but-unused blocks back to OS
    torch.cuda.ipc_collect()          # collect any inter-process CUDA handles

    after_reserved = torch.cuda.memory_reserved(0)
    after_alloc    = torch.cuda.memory_allocated(0)

    freed_mb = (before_reserved - after_reserved) / 1024 ** 2

    return jsonify({
        "success": True,
        "freed_mb": round(freed_mb, 2),
        "before": {
            "reserved_mb": round(before_reserved / 1024 ** 2, 2),
            "allocated_mb": round(before_alloc   / 1024 ** 2, 2),
        },
        "after": {
            "reserved_mb": round(after_reserved / 1024 ** 2, 2),
            "allocated_mb": round(after_alloc   / 1024 ** 2, 2),
        },
    })

@api_bp.route('/events/<access_code>/match', methods=['POST'])
def match_selfie(access_code):
    event = Event.query.filter_by(access_code=access_code).first()
    if not event:
        return jsonify({"success": False, "error": "Event not found"}), 404
        
    if 'selfie' not in request.files:
        return jsonify({"success": False, "error": "No selfie uploaded"}), 400
        
    file = request.files['selfie']
    filename = secure_filename(f"selfie_{access_code}_{file.filename}")
    filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    encodings = extract_faces_from_image(filepath)
    if not encodings:
        return jsonify({"success": False, "error": "No face detected in selfie"}), 400
        
    user_encoding = encodings[0]
    
    # Query all face encodings for this event using a join instead of raw SQL
    all_encodings = FaceEncoding.query.join(Photo).filter(Photo.event_id == event.id).all()
    event_encodings_data = [
        {
            'photo_id': fe.photo_id,
            'file_path': fe.photo.file_path,
            'encoding': fe.encoding_list
        }
        for fe in all_encodings
    ]
    
    matched_filenames = find_matching_photos(user_encoding, event_encodings_data)
    
    try:
        os.remove(filepath)
    except:
        pass
        
    return jsonify({
        "success": True, 
        "matches": matched_filenames
    })
