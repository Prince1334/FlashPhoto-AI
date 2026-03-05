from app import create_app
import sys
import os
import flask
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

app = create_app()

if __name__ == '__main__':
    # During development, we can run this directly with `python run.py --dev`. 
    # Otherwise it defaults to production WSGI via Waitress.
    if '--dev' in sys.argv:
        app.run(host='0.0.0.0', debug=True, port=5000, ssl_context='adhoc')
    else:
        from waitress import serve
        print("Starting production server with Waitress on port 5000...")
        serve(app, host='0.0.0.0', port=5000)
