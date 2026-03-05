import os
from flask import Flask, request, jsonify, send_from_directory
from .extensions import db, migrate
from .config import Config

def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)

    # Ensure upload folder exists
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    # Initialize extensions
    db.init_app(app)
    migrate.init_app(app, db)

    # Register blueprints
    from .routes.api import api_bp
    from .routes.views import views_bp

    app.register_blueprint(api_bp)
    app.register_blueprint(views_bp)

    # Initialize ThreadPoolExecutor for background tasks
    from .tasks import executor
    
    # Ensure executor shuts down cleanly when the app stops
    import atexit

    def _cleanup_on_shutdown():
        """Delete all uploaded files and clear photo/encoding records when server stops."""
        executor.shutdown(wait=False)
        upload_folder = app.config['UPLOAD_FOLDER']
        # Delete every file in the uploads directory
        for fname in os.listdir(upload_folder):
            fpath = os.path.join(upload_folder, fname)
            try:
                if os.path.isfile(fpath):
                    os.remove(fpath)
            except Exception as e:
                print(f"[cleanup] Could not delete {fpath}: {e}")
        # Clear database records so stale rows don't accumulate
        with app.app_context():
            from .models import FaceEncoding, Photo
            try:
                FaceEncoding.query.delete()
                Photo.query.delete()
                db.session.commit()
                print("[cleanup] Uploads and DB records cleared.")
            except Exception as e:
                db.session.rollback()
                print(f"[cleanup] DB cleanup error: {e}")

    atexit.register(_cleanup_on_shutdown)

    # Setup Logging
    from .logging import setup_logging
    setup_logging(app)

    # Serve favicon
    @app.route('/favicon.ico')
    def favicon():
        return send_from_directory(app.static_folder, 'favicon.ico', mimetype='image/x-icon')
    
    # Global Error Handlers
    @app.errorhandler(404)
    def not_found_error(error):
        if request.path == '/favicon.ico':
            return '', 204
        app.logger.info(f"404 Error: {request.url}")
        return jsonify({"success": False, "error": "Resource not found"}), 404

    @app.errorhandler(500)
    def internal_error(error):
        app.logger.error(f"500 Server Error: {error}")
        db.session.rollback()
        return jsonify({"success": False, "error": "Internal server error"}), 500

    return app
