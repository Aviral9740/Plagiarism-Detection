from fastapi import FastAPI, HTTPException, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict
import uvicorn
from datetime import datetime
import logging
from contextlib import asynccontextmanager

# Import your detector (assuming it's in plagiarism_detector.py)
from plagiarism_detector import PlagiarismDetector, Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for FastAPI app"""
    logger.info("=" * 60)
    logger.info("Starting Plagiarism Detection API...")
    logger.info("=" * 60)

    try:
        logger.info("Step 1: Loading configuration from .env...")
        config = Config()
        logger.info("✓ Config loaded")

        logger.info("Step 2: Initializing PlagiarismDetector...")
        app.state.detector = PlagiarismDetector(config)
        logger.info("✓ PlagiarismDetector initialized successfully")

        logger.info("=" * 60)
        logger.info("🚀 Server ready! Visit http://localhost:8000/docs")
        logger.info("=" * 60)

    except Exception as e:
        logger.error("=" * 60)
        logger.error(f"❌ INITIALIZATION FAILED: {e}")
        logger.error("=" * 60)
        logger.exception("Full traceback:")
        raise

    yield

    logger.info("Shutting down Plagiarism Detection API...")


# Initialize FastAPI app
app = FastAPI(
    title="Plagiarism Detection API",
    description=(
        "API for detecting plagiarism in text using TF-IDF, N-gram, and LCS algorithms. "
        "Searches multiple sources including ArXiv, Google Scholar, and web search."
    ),
    version="1.0.1",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DetectionRequest(BaseModel):
    text: str = Field(..., min_length=50, description="Text to check (minimum 50 characters)")
    use_web_search: bool = Field(
        default=False, description="Enable web and Google Scholar search (requires API key)"
    )

    @validator("text")
    def validate_text(cls, v):
        if not v.strip():
            raise ValueError("Text cannot be empty")
        if len(v.strip()) < 50:
            raise ValueError("Text must be at least 50 characters long")
        return v.strip()

    class Config:
        schema_extra = {
            "example": {
                "text": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder.",
                "use_web_search": False,
            }
        }


class MatchDetail(BaseModel):
    title: str
    url: str
    snippet: str
    similarity: float
    source: str
    authors: Optional[str] = None
    year: Optional[int] = None
    venue: Optional[str] = None
    details: Dict[str, float]  # Contains: tfidf, ngram, lcs, combined


class DetectionResponse(BaseModel):
    overall_score: float = Field(..., description="Overall plagiarism score (0-1)")
    verdict: str = Field(..., description="Human-readable verdict")
    matches: List[MatchDetail] = Field(..., description="List of matching sources")
    total_matches: int = Field(..., description="Total number of matches found")
    word_count: int = Field(..., description="Word count of input text")
    timestamp: str
    processing_time_ms: Optional[float] = None


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str
    detector_ready: bool
    web_search_enabled: bool


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    timestamp: str

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Plagiarism Detection API",
        "version": "1.0.1",
        "description": "Detect plagiarism using TF-IDF, N-gram, and LCS algorithms",
        "documentation": "/docs",
        "health_check": "/health",
        "endpoints": {
            "detect": "/api/v1/detect",
            "batch": "/api/v1/detect/batch",
            "config": "/api/v1/config",
            "knowledge_base": "/api/v1/knowledge-base"
        },
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check(request: Request):
    """Health check endpoint"""
    detector = getattr(request.app.state, "detector", None)
    return HealthResponse(
        status="healthy" if detector else "unhealthy",
        timestamp=datetime.utcnow().isoformat(),
        version="1.0.1",
        detector_ready=detector is not None,
        web_search_enabled=bool(detector and detector.searchapi_key),
    )


@app.get("/api/v1/config", tags=["Configuration"])
async def get_config(request: Request):
    """Get current detector configuration (non-sensitive info only)"""
    detector = getattr(request.app.state, "detector", None)
    if not detector:
        raise HTTPException(status_code=503, detail="Detector not initialized")

    return {
        "similarity_methods": ["TF-IDF", "N-gram", "LCS"],
        "similarity_weights": {
            "tfidf": 0.45,
            "ngram": 0.35,
            "lcs": 0.20
        },
        "max_results_web": detector.config.MAX_RESULTS_WEB,
        "max_results_scholar": detector.config.MAX_RESULTS_SCHOLAR,
        "max_results_arxiv": detector.config.MAX_RESULTS_ARXIV,
        "similarity_threshold": detector.config.SIMILARITY_THRESHOLD,
        "api_timeout": detector.config.API_TIMEOUT,
        "web_search_enabled": detector.searchapi_key is not None,
        "knowledge_base_size": len(detector.knowledge_base),
    }


@app.post(
    "/api/v1/detect",
    response_model=DetectionResponse,
    tags=["Detection"],
    status_code=status.HTTP_200_OK,
)
async def detect_plagiarism(request: Request, payload: DetectionRequest):

    detector = getattr(request.app.state, "detector", None)
    if not detector:
        raise HTTPException(status_code=503, detail="Detector not initialized")

    try:
        logger.info(f"🔍 Detection started (length={len(payload.text)} chars, web_search={payload.use_web_search})")
        start = datetime.utcnow()

        results = detector.detect_plagiarism(
            text=payload.text,
            use_web_search=payload.use_web_search
        )

        if "error" in results:
            raise HTTPException(status_code=400, detail=results["error"])

        processing_ms = (datetime.utcnow() - start).total_seconds() * 1000

        response = DetectionResponse(
            overall_score=results["overall_score"],
            verdict=results["verdict"],
            matches=[
                MatchDetail(
                    title=m["title"],
                    url=m["url"],
                    snippet=m["snippet"],
                    similarity=m["similarity"],
                    source=m["source"],
                    authors=m.get("authors"),
                    year=m.get("year"),
                    venue=m.get("venue"),
                    details=m["details"],
                )
                for m in results["matches"]
            ],
            total_matches=results["total_matches"],
            word_count=results["word_count"],
            timestamp=datetime.utcnow().isoformat(),
            processing_time_ms=processing_ms,
        )

        logger.info(
            f"✅ Detection complete in {processing_ms:.2f}ms "
            f"(Score={results['overall_score'] * 100:.1f}%, Matches={results['total_matches']})"
        )
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Unexpected error during detection")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.post("/api/v1/detect/batch", tags=["Detection"])
async def detect_batch(request: Request, payloads: List[DetectionRequest]):
    """
    Batch plagiarism detection (maximum 10 texts per request).

    Returns results for each text, including any errors encountered.
    """
    detector = getattr(request.app.state, "detector", None)
    if not detector:
        raise HTTPException(status_code=503, detail="Detector not initialized")

    if len(payloads) > 10:
        raise HTTPException(
            status_code=400,
            detail="Maximum 10 texts per batch request. Please split into smaller batches."
        )

    logger.info(f"📦 Batch detection started ({len(payloads)} texts)")
    results = []

    for i, item in enumerate(payloads):
        try:
            res = detector.detect_plagiarism(
                text=item.text,
                use_web_search=item.use_web_search
            )
            results.append({
                "index": i,
                "success": True,
                "result": res
            })
        except Exception as e:
            logger.error(f"Error processing text {i}: {e}")
            results.append({
                "index": i,
                "success": False,
                "error": str(e)
            })

    successful = sum(1 for r in results if r["success"])
    failed = sum(1 for r in results if not r["success"])

    logger.info(f"✅ Batch complete: {successful} successful, {failed} failed")

    return {
        "total": len(payloads),
        "successful": successful,
        "failed": failed,
        "results": results,
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/api/v1/knowledge-base", tags=["Knowledge Base"])
async def get_knowledge_base(request: Request):
    """
    Get the internal knowledge base of papers.

    The knowledge base contains famous ML/AI papers that are checked first
    during plagiarism detection.
    """
    detector = getattr(request.app.state, "detector", None)
    if not detector:
        raise HTTPException(status_code=503, detail="Detector not initialized")

    return {
        "count": len(detector.knowledge_base),
        "papers": [
            {
                "title": p["title"],
                "authors": p["authors"],
                "year": p["year"],
                "venue": p["venue"],
                "url": p["url"],
            }
            for p in detector.knowledge_base
        ],
    }


# -------------------------------------------------------
# ⚠️ Error Handlers
# -------------------------------------------------------


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error="HTTP Error",
            detail=detail,
            timestamp=datetime.utcnow().isoformat(),
        ).dict(),
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal Server Error",
            detail=str(exc),
            timestamp=datetime.utcnow().isoformat(),
        ).dict(),
    )
if __name__ == "__main__":
    logger.info("Starting uvicorn server...")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Set to False in production
        log_level="info",
    )