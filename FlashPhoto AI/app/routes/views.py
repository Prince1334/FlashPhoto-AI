import qrcode
import io
import base64
from flask import Blueprint, render_template, request, url_for, abort
from app.models import Event, Photo
from app.extensions import db

views_bp = Blueprint('views', __name__)

@views_bp.route('/')
def index():
    return render_template('index.html')

@views_bp.route('/event/<access_code>')
def event_dashboard(access_code):
    event = Event.query.filter_by(access_code=access_code).first()
    if not event:
        abort(404, description="Event not found")
    
    scan_url = request.host_url.rstrip('/') + url_for('views.attendee_scan', access_code=access_code)
    qr = qrcode.QRCode(version=1, box_size=10, border=5)
    qr.add_data(scan_url)
    qr.make(fit=True)
    img = qr.make_image(fill='black', back_color='white')
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    qr_b64 = base64.b64encode(buffer.getvalue()).decode()

    photos = Photo.query.filter_by(event_id=event.id).all()
    
    return render_template('event_dashboard.html', event=event, qr_b64=qr_b64, photos=photos)

@views_bp.route('/scan/<access_code>')
def attendee_scan(access_code):
    event = Event.query.filter_by(access_code=access_code).first()
    if not event:
        abort(404, description="Event not found")
    return render_template('attendee_scan.html', event=event)
