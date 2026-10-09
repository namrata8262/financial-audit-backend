from pathlib import Path
from uuid import uuid4

import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from audit_engine import run_audit

app = FastAPI(title="AUDITX Financial Audit API")

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "https://financial-audit-frontend-wcho.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

audit_results = {}


@app.get("/")
def home():
    return {"message": "AUDITX API is running"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/audit/run")
async def run_audit_endpoint(
    beneficiaries_file: UploadFile = File(...),
    bank_accounts_file: UploadFile = File(...),
    disbursements_file: UploadFile = File(...),
):
    uploaded_files = {
        "beneficiaries": beneficiaries_file,
        "bank_accounts": bank_accounts_file,
        "disbursements": disbursements_file,
    }

    dataframes = {}

    try:
        for name, upload in uploaded_files.items():
            if not upload.filename or not upload.filename.lower().endswith(".csv"):
                raise HTTPException(
                    status_code=400,
                    detail=f"{name}: please upload a CSV file",
                )

            content = await upload.read()

            if not content:
                raise HTTPException(
                    status_code=400,
                    detail=f"{name}: uploaded file is empty",
                )

            # Keep uploaded data in memory; no permanent raw CSV storage.
            from io import BytesIO
            dataframes[name] = pd.read_csv(BytesIO(content))

        result = run_audit(
            dataframes["beneficiaries"],
            dataframes["bank_accounts"],
            dataframes["disbursements"],
        )

        audit_id = str(uuid4())
        audit_results[audit_id] = result

        return {
            "audit_id": audit_id,
            "status": "completed",
            "results": result,
        }

    except HTTPException:
        raise
    except (ValueError, KeyError, pd.errors.ParserError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Audit failed: {str(exc)}",
        )


@app.get("/audit/{audit_id}/results")
def get_audit_results(audit_id: str):
    if audit_id not in audit_results:
        raise HTTPException(status_code=404, detail="Audit not found")

    return {
        "audit_id": audit_id,
        "status": "completed",
        "results": audit_results[audit_id],
    }