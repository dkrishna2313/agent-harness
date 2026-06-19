# Profile Editor

A web UI for editing research agent domain profile YAML files.

## Dev Setup

### Backend (FastAPI)

```bash
cd profile_editor/backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### Frontend (React + Vite + Tailwind)

```bash
cd profile_editor/frontend
npm install
npm run dev
# Open http://localhost:5173
```

The Vite dev server proxies `/api` requests to `http://localhost:8000`, so both servers must be running during development.

## Production Build

```bash
cd profile_editor/frontend
npm run build
# Serves from backend at http://localhost:8000
uvicorn main:app --port 8000
```

## Features

- List, create, edit, and delete domain profile YAML files
- Specialized editors per field type:
  - **SimpleList**: tag chips for `list[str]` fields
  - **DictOfLists**: collapsible sections for `dict[str, list[str]]` fields
  - **ListOfDicts**: table rows for entity patterns and source quality rules
  - **TextAreaList**: plain text inputs for regex pattern lists
  - **Raw JSON**: advanced/complex nested fields
- Unsaved changes indicator and Cmd/Ctrl+S shortcut
- Toast notifications for save success/error
- Create and delete profiles with confirmation
