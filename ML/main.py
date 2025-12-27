# main.py
"""
Certificate OCR + Region Authenticity Detection (Version B)

Features:
 - /analyze  -> full pipeline: Gemini OCR (if available) -> region detection -> per-region authenticity -> verdict
 - /extract  -> OCR-only (Gemini preferred, fallback to local pytesseract)
 - /upload_reference -> upload logo images used for template matching
 - /           -> health check
Notes:
 - Configure GEMINI_API_KEY and optional MODEL_NAME in .env
 - Reference logos stored under ./refs (created automatically) for template matching
"""
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
import os
import io
import json
import re
import base64
import traceback
from typing import Dict, Any, List, Tuple

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# optional libs
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except Exception:
    GENAI_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False

try:
    import numpy as np
    import cv2
    OPENCV_AVAILABLE = True
except Exception:
    OPENCV_AVAILABLE = False

# Local OCR fallback (pytesseract)
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except Exception:
    TESSERACT_AVAILABLE = False

# optional SSIM
try:
    from skimage.metrics import structural_similarity as ssim
    SKIMAGE_AVAILABLE = True
except Exception:
    SKIMAGE_AVAILABLE = False

# load env
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-2.0-flash")  # override if needed

if GEMINI_API_KEY and GENAI_AVAILABLE:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
    except Exception:
        # If configure fails, continue and rely on fallback
        pass

# FastAPI app
app = FastAPI(title="Certificate OCR + Region Authenticity Detection (vB)")

# CORS - dev-friendly; restrict in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

# refs dir
REF_DIR = "refs"
os.makedirs(REF_DIR, exist_ok=True)


# ----------------- Utilities -----------------
def compress_image_bytes(image_bytes: bytes, max_side: int = 1200, quality: int = 70) -> bytes:
    """
    Resize and compress image to reduce upload size / speed up model calls.
    Uses OpenCV if available, else returns original bytes.
    """
    if not OPENCV_AVAILABLE:
        return image_bytes
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return image_bytes
    h, w = img.shape[:2]
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    _, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    return enc.tobytes()


def pil_image_from_bytes(b: bytes) -> "Image.Image":
    if not PIL_AVAILABLE:
        raise RuntimeError("PIL not available")
    return Image.open(io.BytesIO(b)).convert("RGB")


# ----------------- Gemini OCR (wrapped) -----------------
def gemini_extract_fields(image_bytes: bytes) -> Dict[str, Any]:
    """Call Gemini to extract fields. Returns dict or raises Exception on failure."""
    if not GEMINI_API_KEY or not GENAI_AVAILABLE:
        raise RuntimeError("Gemini not configured or library not installed")

    # compress image
    compressed = compress_image_bytes(image_bytes, max_side=1200, quality=70)

    prompt = """
You are a precise OCR and information extraction system specialized in educational certificates/marksheets.
Given an image, determine if it's an educational certificate. If not, return exactly:
{"error":"Please upload an educational certificate."}

If it is, return EXACTLY a JSON object with these keys (use null for missing values):
{
  "Name": null,
  "MotherName": null,
  "Programme": null,
  "PRN": null,
  "RollNumber": null,
  "ABC_ID": null,
  "YearSemester": null,
  "AcademicYear": null,
  "SGPA": null,
  "CGPA": null,
  "Institution": null,
  "University": null
}

Return ONLY JSON, no commentary.
"""

    # The google.generativeai API surface varies by version. We'll attempt the common patterns
    # Try using GenerativeModel if available
    try:
        model = genai.GenerativeModel(MODEL_NAME)
        resp = model.generate_content(
            [
                {"role": "user", "content": prompt},
                {"mime_type": "image/jpeg", "data": compressed},
            ],
            # safe timeout
            timeout=60,
        )
        text = getattr(resp, "text", None) or str(resp)
        # parse JSON
        text = text.strip()
        try:
            return json.loads(text)
        except Exception:
            m = re.search(r"\{.*\}", text, re.DOTALL)
            if m:
                return json.loads(m.group())
            raise RuntimeError("Gemini returned non-JSON response")
    except Exception as e:
        # bubble up the original error to be handled by the caller
        raise RuntimeError(f"Gemini call failed: {e}")


# ----------------- Local OCR fallback -----------------
def local_ocr_extract_fields(image_bytes: bytes) -> Dict[str, Any]:
    """
    Use pytesseract to extract raw text and attempt to parse common certificate fields heuristically.
    Returns dictionary with same keys as Gemini format (null when not found).
    """
    if not TESSERACT_AVAILABLE or not PIL_AVAILABLE:
        # return minimal error-like structure
        return {"error": "Local OCR not available (pytesseract/PIL required)"}

    img = pil_image_from_bytes(image_bytes)
    raw = pytesseract.image_to_string(img, lang="eng")
    text = raw.replace("\r", "\n")
    # simple heuristics
    def find_first(patterns, text=text):
        for p in patterns:
            m = re.search(p, text, re.IGNORECASE)
            if m:
                return m.group(1).strip()
        return None

    # candidate regex patterns - may need tuning
    name = find_first([r"Name Of the Student[:\s]*([A-Z\s]+)", r"Candidate['’]?s Full Name[:\s]*([A-Z][A-Za-z\s]+)", r"Name[:\s]*([A-Z][A-Za-z\s]+)"])
    mother = find_first([r"Mother(?:'s)? Name[:\s]*([A-Z][A-Za-z\s]+)", r"Mother Name[:\s]*([A-Z][A-Za-z\s]+)"])
    prn = find_first([r"PRN[:\s]*([0-9]{5,15})", r"PRN\s*([0-9]{5,15})", r"Roll(?: Number| No|No)[:\s]*([0-9]{5,15})"])
    abc = find_first([r"ABC ID[:\s]*([0-9]{5,20})", r"ABC[_\s]*ID[:\s]*([0-9]{5,20})"])
    programme = find_first([r"Programme[:\s]*(.+)", r"Course[:\s]*(.+)"])
    yearsem = find_first([r"Year & Semester[:\s]*(.+)", r"Year Semester[:\s]*(.+)", r"Year[:\s]*(First|Second|Third|Fourth).*(SEM-?II|SEM-II|SEM I)?" ])
    acad = find_first([r"Academic Year[:\s]*([0-9]{4}\s*-\s*[0-9]{4})", r"Academic Year[:\s]*([0-9]{4}\/[0-9]{4})"])
    cgpa = find_first([r"CGPA[:\s]*([0-9]\.\d{1,2})", r"C G P A[:\s]*([0-9]\.\d{1,2})"])
    sgpa = find_first([r"SGPA[:\s]*([0-9]\.\d{1,2})"])

    # institution/university attempts
    inst = find_first([r"INSTITUTE OF TECHNOLOGY[:\s]*(.+)", r"Institute[:\s]*(.+)"])
    uni = find_first([r"Affiliated to (.+)", r"University[:\s]*(.+)"])

    def norm(v):
        return v if v and len(str(v).strip())>0 else None

    return {
        "Name": norm(name),
        "MotherName": norm(mother),
        "Programme": norm(programme),
        "PRN": norm(prn),
        "RollNumber": norm(prn),
        "ABC_ID": norm(abc),
        "YearSemester": norm(yearsem),
        "AcademicYear": norm(acad),
        "SGPA": norm(sgpa),
        "CGPA": norm(cgpa),
        "Institution": norm(inst),
        "University": norm(uni),
    }


# ----------------- Layout region detection -----------------
def detect_regions_layout(image_bytes: bytes) -> Dict[str, List[int]]:
    """
    Return reasonable bounding boxes (x1,y1,x2,y2) for common certificate regions
    relative to image size
    """
    if not OPENCV_AVAILABLE:
        # fallback fixed rectangles
        return {
            "logo_left": [10, 10, 180, 180],
            "logo_right": [600, 10, 780, 180],
            "student_photo": [600, 60, 780, 240],
            "stamp_area": [300, 500, 560, 760],
            "signatures_area": [30, 650, 780, 810]
        }
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid image data")
    h, w = img.shape[:2]

    regions = {
        "logo_left": [int(0.02*w), int(0.02*h), int(0.18*w), int(0.20*h)],
        "logo_right": [int(0.78*w), int(0.02*h), int(0.98*w), int(0.20*h)],
        "student_photo": [int(0.72*w), int(0.08*h), int(0.92*w), int(0.32*h)],
        "stamp_area": [int(0.30*w), int(0.66*h), int(0.66*w), int(0.92*h)],
        "signatures_area": [int(0.04*w), int(0.76*h), int(0.96*w), int(0.96*h)]
    }
    return regions


# ----------------- Region analysis -----------------
def _normalize_score(v: float) -> float:
    v = float(v)
    if v < 0: v = 0.0
    if v > 1: v = 1.0
    return round(v * 100, 1)  # percent


def template_match_score(region_gray: "np.ndarray", template_gray: "np.ndarray") -> float:
    """
    Compute similarity [0..1] between region and template.
    Use SSIM if available, otherwise use cv2.matchTemplate.
    """
    try:
        rh, rw = region_gray.shape[:2]
        th, tw = template_gray.shape[:2]
        if (rh, rw) != (th, tw):
            template_resized = cv2.resize(template_gray, (rw, rh))
        else:
            template_resized = template_gray
        if SKIMAGE_AVAILABLE:
            score = ssim(region_gray, template_resized, data_range=region_gray.max()-region_gray.min() if region_gray.max()!=region_gray.min() else 1)
            return max(0.0, min(1.0, float(score)))
        res = cv2.matchTemplate(region_gray, template_resized, cv2.TM_CCOEFF_NORMED)
        return max(0.0, min(1.0, float(res.max())))
    except Exception:
        return 0.0


def analyze_region_authenticity(image_bytes: bytes, regions: Dict[str, List[int]]) -> Dict[str, Any]:
    """
    For each region, compute an authenticity score and details.
    Uses reference logos if present in ./refs.
    """
    results: Dict[str, Any] = {}
    if not OPENCV_AVAILABLE:
        # simple default: return zero scores
        for name, bbox in regions.items():
            results[name] = {"bbox": bbox, "authenticity_pct": 0.0, "notes": "opencv not available"}
        return results

    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("invalid image data")

    # load reference logos (grayscale)
    ref_files = [os.path.join(REF_DIR, f) for f in os.listdir(REF_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    ref_imgs = []
    for rf in ref_files:
        r = cv2.imread(rf, cv2.IMREAD_GRAYSCALE)
        if r is not None:
            ref_imgs.append(r)

    for name, (x1, y1, x2, y2) in regions.items():
        x1i, y1i, x2i, y2i = int(x1), int(y1), int(x2), int(y2)
        crop = img[y1i:y2i, x1i:x2i]
        if crop is None or crop.size == 0:
            results[name] = {"bbox": [x1i, y1i, x2i, y2i], "authenticity_pct": 0.0, "notes": "empty region"}
            continue

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        h_r, w_r = gray.shape[:2]
        notes = []

        score = 0.0
        # logo heuristics
        if name.startswith("logo"):
            if ref_imgs:
                best = 0.0
                for ref in ref_imgs:
                    s = template_match_score(gray, ref)
                    if s > best:
                        best = s
                score = best
                notes.append(f"matched {len(ref_imgs)} ref(s) best={best:.3f}")
            else:
                # fallback: logos have distinct edge density
                edges = cv2.Canny(gray, 50, 150)
                edge_density = edges.sum() / (255 * h_r * w_r + 1)
                score = min(1.0, edge_density * 12)
                notes.append("no ref logos - using edge-density")
        elif "photo" in name or "student" in name:
            # face detection
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=4, minSize=(24, 24))
            if len(faces) > 0:
                fa = [w * h for (_, _, w, h) in faces]
                f_ratio = max(fa) / (h_r * w_r)
                score = min(1.0, 0.6 + f_ratio * 3)
                notes.append(f"faces_detected={len(faces)}")
            else:
                edges = cv2.Canny(gray, 50, 150)
                line_density = edges.sum() / (255 * h_r * w_r + 1)
                score = 0.15 + min(0.4, line_density * 6)
                notes.append("no face - fallback heuristics")
        elif "stamp" in name or "sign" in name:
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            lower1 = (0, 70, 40); upper1 = (10, 255, 255)
            lower2 = (160, 70, 40); upper2 = (180, 255, 255)
            mask1 = cv2.inRange(hsv, lower1, upper1)
            mask2 = cv2.inRange(hsv, lower2, upper2)
            red_mask = cv2.bitwise_or(mask1, mask2)
            red_ratio = red_mask.sum() / (255 * h_r * w_r + 1)
            blurred = cv2.GaussianBlur(gray, (7, 7), 0)
            try:
                circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=min(h_r, w_r) / 8,
                                           param1=40, param2=24,
                                           minRadius=int(min(h_r, w_r) * 0.04), maxRadius=int(min(h_r, w_r) * 0.45))
            except Exception:
                circles = None
            edges = cv2.Canny(gray, 50, 150)
            line_density = edges.sum() / (255 * h_r * w_r + 1)
            stamp_score = min(1.0, red_ratio * 6 + (0.45 if circles is not None else 0.0))
            sign_score = min(1.0, line_density * 6)
            score = max(stamp_score, sign_score)
            notes.append(f"red={red_ratio:.4f}, circles={'yes' if circles is not None else 'no'}, line_density={line_density:.4f}")
        else:
            score = 0.1
            notes.append("unknown region type")

        results[name] = {
            "bbox": [x1i, y1i, x2i, y2i],
            "authenticity_pct": _normalize_score(score),
            "raw_score": round(float(score), 3),
            "notes": "; ".join(notes)
        }

    return results


# ----------------- Endpoints -----------------
@app.get("/")
def root():
    ref_count = len([f for f in os.listdir(REF_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
    return {"status": "healthy", "ref_logo_count": ref_count, "gemini_configured": bool(GEMINI_API_KEY and GENAI_AVAILABLE)}


@app.post("/upload_reference")
async def upload_reference(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload an image file")
    contents = await file.read()
    fname = file.filename or "ref_logo.jpg"
    safe = f"ref_{len(os.listdir(REF_DIR))+1}_{fname}"
    path = os.path.join(REF_DIR, safe)
    with open(path, "wb") as f:
        f.write(contents)
    return {"status": "ok", "path": path}


@app.post("/extract")
async def extract_only(file: UploadFile = File(...)):
    """
    Simple OCR-only endpoint. Attempts Gemini first (if configured) and falls back to local pytesseract.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "Only image files allowed")
    image_bytes = await file.read()

    # try Gemini
    if GEMINI_API_KEY and GENAI_AVAILABLE:
        try:
            fields = gemini_extract_fields(image_bytes)
            return JSONResponse({"source": "gemini", "fields": fields})
        except Exception as e:
            # don't fail immediately - fallback
            err = str(e)
    # fallback to local OCR
    local = local_ocr_extract_fields(image_bytes)
    return JSONResponse({"source": "local_ocr", "fields": local})


@app.post("/analyze")
async def analyze_certificate(file: UploadFile = File(...)):
    """
    Full analyze: OCR (Gemini preferred) + region detection + per-region authenticity + verdict
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "Only image files allowed")

    image_bytes = await file.read()

    # 1) Extract fields using Gemini if possible, else fallback
    extracted = None
    ocr_source = "none"
    if GEMINI_API_KEY and GENAI_AVAILABLE:
        try:
            extracted = gemini_extract_fields(image_bytes)
            ocr_source = "gemini"
        except Exception as e:
            # log and fallback
            extracted = None
            ocr_source = f"gemini_error: {str(e)}"

    if extracted is None:
        # fallback to local OCR
        extracted = local_ocr_extract_fields(image_bytes)
        ocr_source = "local"

    # 2) layout regions
    try:
        regions = detect_regions_layout(image_bytes)
    except Exception as e:
        regions = {}
        # continue, return error to user at end

    # 3) analyze regions
    try:
        region_results = analyze_region_authenticity(image_bytes, regions)
    except Exception as e:
        region_results = {}
        # keep going

    # 4) combine overall authenticity (weighted)
    # weights tuned: left logo 0.28, right logo 0.28, photo 0.22, stamp 0.22
    weights = {"logo_left": 0.28, "logo_right": 0.28, "student_photo": 0.22, "stamp_area": 0.22}
    total = 0.0
    wsum = 0.0
    for k, w in weights.items():
        r = region_results.get(k, {})
        val = r.get("raw_score", 0.0)
        total += float(val) * w
        wsum += w
    # text boost if PRN or Roll present and looks numeric
    text_boost = 0.0
    try:
        prn = extracted.get("PRN") if isinstance(extracted, dict) else None
        if prn and re.search(r"\d{5,12}", str(prn)):
            text_boost = 0.05
    except Exception:
        pass

    overall_raw = min(1.0, (total / wsum) + text_boost) if wsum > 0 else text_boost
    overall_pct = _normalize_score(overall_raw)
    verdict = "authentic" if overall_raw >= 0.72 else ("review" if overall_raw >= 0.45 else "fake")

    response = {
        "status": "success",
        "ocr_source": ocr_source,
        "extracted_fields": extracted,
        "region_details": region_results,
        "overall_authenticity_pct": overall_pct,
        "verdict": verdict,
    }
    return JSONResponse(response)


# ----------------- run -----------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
