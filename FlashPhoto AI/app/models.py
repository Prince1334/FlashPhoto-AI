from .extensions import db
from datetime import datetime, timezone
import json

class Event(db.Model):
    __tablename__ = 'events'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    name = db.Column(db.String(255), nullable=False)
    access_code = db.Column(db.String(255), unique=True, nullable=False)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    
    # Relationships
    photos = db.relationship('Photo', backref='event', lazy=True, cascade='all, delete-orphan')

class Photo(db.Model):
    __tablename__ = 'photos'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    event_id = db.Column(db.Integer, db.ForeignKey('events.id', ondelete='CASCADE'), nullable=False)
    file_path = db.Column(db.String(255), nullable=False)
    
    # Relationships
    face_encodings = db.relationship('FaceEncoding', backref='photo', lazy=True, cascade='all, delete-orphan')

class FaceEncoding(db.Model):
    __tablename__ = 'face_encodings'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    photo_id = db.Column(db.Integer, db.ForeignKey('photos.id', ondelete='CASCADE'), nullable=False)
    encoding_json = db.Column(db.Text(length=(2**32)-1), nullable=False)  # LONGTEXT equivalent

    @property
    def encoding_list(self):
        return json.loads(self.encoding_json)

    @encoding_list.setter
    def encoding_list(self, value):
        self.encoding_json = json.dumps(value)
