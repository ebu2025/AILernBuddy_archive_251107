"""agent_test_harness.py

Lightweight smoke-check for local development and automated agents.
It imports the FastAPI `app` and `tutor` to ensure key modules load and
prints a short summary of API routes and prompt version.

This script is intentionally small and avoids starting servers or requiring
LLM endpoints.
"""
import sys
import importlib

def main():
    errors = []
    try:
        app = importlib.import_module('app')
        title = getattr(app, 'app').title if getattr(app, 'app', None) else 'unknown'
        print(f"Loaded FastAPI app: {title}")
        routes = [r.path for r in getattr(app, 'app').routes]
        print(f"Routes ({len(routes)}):")
        for p in sorted(routes):
            print("  ", p)
    except Exception as e:
        errors.append(('app import', e))

    try:
        tutor = importlib.import_module('tutor')
        pv = getattr(tutor, 'PROMPT_VERSION', None)
        pv_text = pv if pv is not None else 'MISSING'
        print(f"tutor.PROMPT_VERSION: {pv_text}")
    except Exception as e:
        errors.append(('tutor import', e))

    if errors:
        print('\nERRORS:')
        for name, exc in errors:
            print(f" - {name}: {type(exc).__name__}: {exc}")
        sys.exit(2)

    print('\nSMOKE CHECK: OK')

if __name__ == '__main__':
    main()
