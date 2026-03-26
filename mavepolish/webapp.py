"""Entry point for the MAVEpolish web app."""

import os
import sys


def main():
    """Start the MAVEpolish Dash web server."""
    # Ensure the project root is on sys.path so `import app` works
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    import app as webapp
    port = int(os.getenv('PORT', 8051))
    webapp.app.run(debug=True, port=port, host='0.0.0.0')


if __name__ == '__main__':
    main()
