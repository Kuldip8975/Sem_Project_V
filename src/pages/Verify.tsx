import { useState, useCallback } from "react";
import { useDropzone } from "react-dropzone";
import { Navbar } from "@/components/Navbar";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import {
  Upload,
  FileText,
  Scan,
  CheckCircle,
  AlertTriangle,
  XCircle,
  Camera,
  QrCode,
  Download,
  Copy,
  Eye,
  Clock,
  Shield,
  ZoomIn,
  Sparkles,
  ShieldCheck,
  FileCheck,
} from "lucide-react";
import { useToast } from "@/hooks/use-toast";

// Mock user - in real app this would come from auth context
const mockUser = {
  name: "Dr. Sarah Johnson",
  role: "verifier" as const,
};

interface Detection {
  bbox: number[];
  class_name: "fake" | "true";
  confidence: number;
}

interface VerificationApiResponse {
  detections: Detection[];
}

interface VerificationResult {
  status: "valid" | "review" | "invalid";
  detections: Detection[];
  summary: {
    totalDetections: number;
    fakeDetections: number;
    trueDetections: number;
  };
  visualizationUrl?: string;
}

const PredictBBoxWidget = () => {
  const [image, setImage] = useState<File | null>(null);
  const [result, setResult] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleUpload = async () => {
    if (!image) return;
    setLoading(true);
    setError(null);

    try {
      // Simulate API call to /predict endpoint
      await new Promise((resolve) => setTimeout(resolve, 2000));

      // For demo, we'll create a mock result
      const canvas = document.createElement("canvas");
      canvas.width = 800;
      canvas.height = 600;
      const ctx = canvas.getContext("2d");

      if (ctx) {
        // Create mock bounding box visualization
        ctx.fillStyle = "#f0f9ff";
        ctx.fillRect(0, 0, 800, 600);

        // Draw bounding boxes
        ctx.strokeStyle = "#22c55e";
        ctx.lineWidth = 3;
        ctx.strokeRect(50, 50, 200, 100); // Seal
        ctx.strokeRect(300, 200, 250, 50); // Signature
        ctx.strokeRect(100, 400, 300, 80); // Certificate ID

        // Add labels
        ctx.fillStyle = "#22c55e";
        ctx.font = "16px Inter";
        ctx.fillText("Official Seal ✓", 55, 45);
        ctx.fillText("Signature ✓", 305, 195);
        ctx.fillText("Certificate ID ✓", 105, 395);
      }

      canvas.toBlob((blob) => {
        if (blob) {
          setResult(URL.createObjectURL(blob));
        }
      });
    } catch (e: any) {
      setError(e.message ?? "Detection failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-4">
      <div className="border-2 border-dashed border-muted-foreground/25 rounded-lg p-4 transition-all duration-300 hover:border-primary/50 hover:bg-primary/5">
        <input
          type="file"
          accept="image/*"
          onChange={(e) => setImage(e.target.files?.[0] ?? null)}
          className="block w-full text-sm text-muted-foreground file:mr-4 file:py-2 file:px-4 file:rounded-md file:border-0 file:bg-gradient-to-r file:from-primary file:to-primary/80 file:text-primary-foreground hover:file:bg-primary/90 transition-all duration-200"
        />
      </div>

      <Button
        onClick={handleUpload}
        disabled={!image || loading}
        className="w-full bg-gradient-to-r from-primary to-primary/80 hover:from-primary/90 hover:to-primary/70 transition-all duration-300 transform hover:scale-[1.02] shadow-lg hover:shadow-xl"
      >
        {loading ? (
          <div className="flex items-center space-x-2">
            <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
            <span>Detecting Objects...</span>
          </div>
        ) : (
          "Detect Seals & Signatures"
        )}
      </Button>

      {error && (
        <div className="flex items-center space-x-2 p-3 bg-destructive/10 border border-destructive/20 rounded-lg text-destructive text-sm animate-pulse">
          <AlertTriangle className="h-4 w-4" />
          <span>{error}</span>
        </div>
      )}

      {result && (
        <div className="rounded-xl overflow-hidden border shadow-lg transition-all duration-300 hover:shadow-xl">
          <img
            src={result}
            alt="Bounding Box Detection Result"
            className="w-full h-auto transition-transform duration-300 hover:scale-105"
          />
        </div>
      )}
    </div>
  );
};

export default function Verify() {
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [verificationStep, setVerificationStep] = useState(0);
  const [ocrData, setOcrData] = useState<any>(null);
  const [verificationResult, setVerificationResult] =
    useState<VerificationResult | null>(null);
  const [isProcessingOCR, setIsProcessingOCR] = useState(false);
  const [isProcessingVerification, setIsProcessingVerification] =
    useState(false);
  const { toast } = useToast();

  const onDrop = useCallback((acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    if (file) {
      // Clear all existing data and reset to initial state
      setOcrData(null);
      setVerificationResult(null);
      setIsProcessingOCR(false);
      setIsProcessingVerification(false);

      // Set new file and start process
      setUploadedFile(file);
      setVerificationStep(1);
      // Extract data for UI
      processOCR(file);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      "image/*": [".png", ".jpg", ".jpeg"],
      "application/pdf": [".pdf"],
    },
    maxSize: 10 * 1024 * 1024, // 10MB
    multiple: false,
  });

    const processOCR = async (file: File) => {
      setIsProcessingOCR(true);
      try {
        const formData = new FormData();
        formData.append("file", file);
        const response = await fetch(
          "http://localhost:8000/extract",
          {
            method: "POST",
            body: formData,
          }
        );
        if (!response.ok) throw new Error("OCR API error");
        const ocrResult = await response.json();

        // Check if OCR API returned an error (not an educational certificate)
        if (ocrResult.error) {
          setOcrData({ error: ocrResult.error });
          toast({
            variant: "destructive",
            title: "Invalid Document Type",
            description: ocrResult.error,
          });
          return; // Don't proceed to verification
        } 

        setOcrData(ocrResult);
        console.log("OCR Result:", ocrResult.Name, ocrResult.Institution);
        
        // Store results in database only if we have valid data AND user is authenticated
        if (ocrResult.Name && ocrResult.Institution) {
          const token = localStorage.getItem("authToken");
          
          // Check if token exists before making the call
          if (token) {
            try {
              const response = await fetch("http://localhost:3000/api/users/results", {
                method: "POST",
                headers: {
                  "Content-Type": "application/json",
                  "Authorization": `Bearer ${token}`
                },
                body: JSON.stringify({
                  name: ocrResult.Name,
                  institution: ocrResult.Institution,
                }),
              });
              
              if (response.ok) {
                const data = await response.json();
                console.log("Stored in DB:", data);
              } else {
                console.log("Failed to store in DB, but continuing verification");
              }
            } catch (err) {
              console.log("DB store error (non-critical):", err);
              // Don't show error to user, just continue with verification
            }
          } else {
            console.log("No auth token, skipping DB storage");
            // Continue with verification even without token
          }
        }
        
        setVerificationStep(2);

      // Automatically proceed to verification only if OCR was successful
      setTimeout(() => processVerification(file), 1000);
    } catch (error) {
      toast({
        variant: "destructive",
        title: "OCR Processing Failed",
        description: "Could not extract text from document",
      });
    } finally {
      setIsProcessingOCR(false);
    }
  };

  const processVerification = async (file: File) => {
    setIsProcessingVerification(true);

    try {
      const formData = new FormData();
      formData.append("file", file);
      const response = await fetch(
        "http://localhost:8000/predict",
        {
          method: "POST",
          body: formData,
        }
      );

      if (!response.ok) throw new Error("Verification API error");

      const apiResult: VerificationApiResponse = await response.json();
      console.log("Verification API result:", apiResult);

      // Process the API result
      const { detections } = apiResult;
      const fakeDetections = detections.filter((d) => d.class_name === "fake");
      const trueDetections = detections.filter((d) => d.class_name === "true");

      // Determine status - if any fake detections found, mark as invalid/review
      let status: "valid" | "review" | "invalid";
      if (fakeDetections.length > 0) {
        // Check if any high confidence fake detections
        const highConfidenceFakes = fakeDetections.filter(
          (d) => d.confidence > 0.7
        );
        status = highConfidenceFakes.length > 0 ? "invalid" : "review";
      } else {
        status = "valid";
      }

      // Create visualization with bounding boxes
      const visualizationUrl = await createVisualization(file, detections);

      const result: VerificationResult = {
        status,
        detections,
        summary: {
          totalDetections: detections.length,
          fakeDetections: fakeDetections.length,
          trueDetections: trueDetections.length,
        },
        visualizationUrl,
      };

      setVerificationResult(result);
      setVerificationStep(3);

      toast({
        title: "Verification Complete",
        description: `Found ${fakeDetections.length} suspicious regions out of ${detections.length} total`,
        variant: status === "invalid" ? "destructive" : "default",
      });
    } catch (error) {
      toast({
        variant: "destructive",
        title: "Verification Failed",
        description: "Could not verify document against registry",
      });
    } finally {
      setIsProcessingVerification(false);
    }
  };

  // Create visualization with bounding boxes overlaid on original image
  const createVisualization = async (
    file: File,
    detections: Detection[]
  ): Promise<string> => {
    return new Promise((resolve) => {
      const img = new Image();
      const canvas = document.createElement("canvas");
      const ctx = canvas.getContext("2d");

      img.onload = () => {
        if (!ctx) return;

        // Set canvas size to match image
        canvas.width = img.width;
        canvas.height = img.height;

        // Draw original image
        ctx.drawImage(img, 0, 0);

        // Draw bounding boxes
        detections.forEach((detection, index) => {
          const [x1, y1, x2, y2] = detection.bbox;
          const width = x2 - x1;
          const height = y2 - y1;

          // Set color based on detection result
          const isFake = detection.class_name === "fake";
          const color = isFake ? "#ef4444" : "#22c55e"; // Red for fake, green for authentic
          const confidence = Math.round(detection.confidence * 100);

          // Draw bounding box rectangle
          ctx.strokeStyle = color;
          ctx.lineWidth = 3;
          ctx.strokeRect(x1, y1, width, height);

          // Draw semi-transparent fill
          ctx.fillStyle = color + "20"; // Add transparency
          ctx.fillRect(x1, y1, width, height);

          // Prepare label text
          const statusText = isFake ? "FAKE" : "AUTHENTIC";
          const labelText = `${statusText} (${confidence}%)`;

          // Set label styling
          ctx.font = "bold 14px Arial";
          ctx.textAlign = "left";

          // Measure text for background
          const textMetrics = ctx.measureText(labelText);
          const textWidth = textMetrics.width;
          const textHeight = 20;
          const padding = 4;

          // Position label (try to place it above the box, if space available)
          let labelX = x1;
          let labelY = y1 - textHeight - padding;

          // If label would go off top of image, place it inside the box
          if (labelY < 0) {
            labelY = y1 + textHeight + padding;
          }

          // If label would go off right side, adjust x position
          if (labelX + textWidth + padding * 2 > canvas.width) {
            labelX = canvas.width - textWidth - padding * 2;
          }

          // Draw label background
          ctx.fillStyle = color;
          ctx.fillRect(
            labelX - padding,
            labelY - textHeight,
            textWidth + padding * 2,
            textHeight + padding
          );

          // Draw label text
          ctx.fillStyle = "white";
          ctx.fillText(labelText, labelX, labelY - 4);
        });

        // Convert to blob URL
        canvas.toBlob((blob) => {
          if (blob) {
            resolve(URL.createObjectURL(blob));
          }
        });
      };

      img.src = URL.createObjectURL(file);
    });
  };

  const resetVerification = () => {
    setUploadedFile(null);
    setVerificationStep(0);
    setOcrData(null);
    setVerificationResult(null);
    setIsProcessingOCR(false);
    setIsProcessingVerification(false);
  };

  const downloadReport = async () => {
    if (!verificationResult || !uploadedFile) return;

    // Create report content
    const reportData = {
      documentName: uploadedFile.name,
      timestamp: new Date().toLocaleString(),
      verificationStatus: verificationResult.status,
      summary: verificationResult.summary,
      detections: verificationResult.detections.map((detection, index) => ({
        regionId: index + 1,
        status: detection.class_name === "fake" ? "Suspicious" : "Authentic",
        confidence: Math.round(detection.confidence * 100),
        coordinates: detection.bbox.map((coord) => Math.round(coord)),
      })),
      ocrData: ocrData,
    };

    const baseFileName = uploadedFile.name.split(".")[0];
    const dateString = new Date().toISOString().split("T")[0];

    // Convert visualization image to base64 for embedding
    let imageBase64 = "";
    if (verificationResult.visualizationUrl) {
      try {
        const response = await fetch(verificationResult.visualizationUrl);
        const blob = await response.blob();
        const reader = new FileReader();
        imageBase64 = await new Promise((resolve) => {
          reader.onload = () => resolve(reader.result as string);
          reader.readAsDataURL(blob);
        });
      } catch (error) {
        console.error("Failed to convert image to base64:", error);
      }
    }

    // Create comprehensive HTML content for PDF conversion
    const htmlForPdf = `
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Document Verification Report</title>
    <style>
        @page {
            margin: 1in;
            size: A4;
        }
        body {
            font-family: 'Segoe UI', Arial, sans-serif;
            line-height: 1.4;
            color: #333;
            margin: 0;
            padding: 0;
        }
        .header {
            text-align: center;
            border-bottom: 3px solid #2563eb;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }
        .header h1 {
            color: #1f2937;
            margin: 10px 0;
            font-size: 28px;
        }
        .status-badge {
            display: inline-block;
            padding: 8px 16px;
            border-radius: 20px;
            font-weight: bold;
            font-size: 14px;
            color: white;
            margin: 10px 0;
            background-color: ${
              reportData.verificationStatus === "valid"
                ? "#10b981"
                : reportData.verificationStatus === "review"
                ? "#f59e0b"
                : "#ef4444"
            };
        }
        .document-info {
            background: #f8fafc;
            padding: 15px;
            border-radius: 8px;
            margin: 20px 0;
        }
        .document-info p {
            margin: 5px 0;
            font-size: 14px;
        }
        .analysis-image {
            text-align: center;
            margin: 30px 0;
            page-break-inside: avoid;
        }
        .analysis-image img {
            max-width: 100%;
            max-height: 400px;
            border: 2px solid #e5e7eb;
            border-radius: 8px;
        }
        .legend {
            display: flex;
            justify-content: center;
            gap: 20px;
            margin: 15px 0;
            font-size: 12px;
        }
        .legend-item {
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .legend-color {
            width: 16px;
            height: 16px;
            border-radius: 3px;
        }
        .section {
            margin: 25px 0;
            page-break-inside: avoid;
        }
        .section h3 {
            color: #1f2937;
            font-size: 18px;
            margin-bottom: 15px;
            border-bottom: 2px solid #e5e7eb;
            padding-bottom: 5px;
        }
        .summary-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 15px;
            margin: 20px 0;
        }
        .stat-card {
            text-align: center;
            padding: 15px;
            border: 2px solid #e5e7eb;
            border-radius: 8px;
            background: #f9fafb;
        }
        .stat-number {
            font-size: 24px;
            font-weight: bold;
            margin: 0;
        }
        .stat-label {
            color: #6b7280;
            font-size: 12px;
            margin: 5px 0 0 0;
        }
        .detection-list {
            margin: 15px 0;
        }
        .detection-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px;
            margin: 8px 0;
            border-radius: 6px;
            font-size: 13px;
        }
        .detection-authentic {
            background-color: #f0fdf4;
            border: 1px solid #22c55e;
        }
        .detection-suspicious {
            background-color: #fef2f2;
            border: 1px solid #ef4444;
        }
        .ocr-table {
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
            font-size: 13px;
        }
        .ocr-table th,
        .ocr-table td {
            border: 1px solid #e5e7eb;
            padding: 8px;
            text-align: left;
        }
        .ocr-table th {
            background-color: #f3f4f6;
            font-weight: bold;
        }
        .conclusion {
            background: ${
              reportData.verificationStatus === "valid" ? "#f0fdf4" : "#fef2f2"
            };
            border: 2px solid ${
              reportData.verificationStatus === "valid" ? "#22c55e" : "#ef4444"
            };
            padding: 20px;
            border-radius: 8px;
            margin: 25px 0;
            page-break-inside: avoid;
        }
        .conclusion h3 {
            margin-top: 0;
            color: ${
              reportData.verificationStatus === "valid" ? "#15803d" : "#dc2626"
            };
        }
        .footer {
            text-align: center;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #e5e7eb;
            color: #6b7280;
            font-size: 12px;
            page-break-inside: avoid;
        }
        @media print {
            body { -webkit-print-color-adjust: exact; }
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🛡️ Document Verification Report</h1>
        <div class="status-badge">
            ${
              reportData.verificationStatus === "valid"
                ? "✅ AUTHENTIC"
                : reportData.verificationStatus === "review"
                ? "⚠️ NEEDS REVIEW"
                : "❌ SUSPICIOUS"
            }
        </div>
    </div>

    <div class="document-info">
        <p><strong>📄 Document:</strong> ${reportData.documentName}</p>
        <p><strong>📅 Verification Date:</strong> ${reportData.timestamp}</p>
        <p><strong>🔍 Status:</strong> ${reportData.verificationStatus.toUpperCase()}</p>
    </div>

    ${
      imageBase64
        ? `
    <div class="analysis-image">
        <h3>🔬 Detection Analysis Visualization</h3>
        <img src="${imageBase64}" alt="Document Analysis with Detection Regions" />
        <div class="legend">
            <div class="legend-item">
                <div class="legend-color" style="background-color: #ef4444;"></div>
                <span>Suspicious Regions</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background-color: #22c55e;"></div>
                <span>Authentic Regions</span>
            </div>
        </div>
    </div>
    `
        : ""
    }

    <div class="section">
        <h3>📊 Verification Summary</h3>
        <div class="summary-grid">
            <div class="stat-card">
                <p class="stat-number">${reportData.summary.totalDetections}</p>
                <p class="stat-label">Total Regions</p>
            </div>
            <div class="stat-card">
                <p class="stat-number" style="color: #22c55e;">${
                  reportData.summary.trueDetections
                }</p>
                <p class="stat-label">Authentic</p>
            </div>
            <div class="stat-card">
                <p class="stat-number" style="color: #ef4444;">${
                  reportData.summary.fakeDetections
                }</p>
                <p class="stat-label">Suspicious</p>
            </div>
        </div>
    </div>

    <div class="section">
        <h3>🔍 Detection Details</h3>
        <div class="detection-list">
            ${reportData.detections
              .map(
                (detection) => `
                <div class="detection-item detection-${detection.status.toLowerCase()}">
                    <span>
                        ${detection.status === "Suspicious" ? "🚨" : "✅"} 
                        Region ${detection.regionId}: ${detection.status}
                    </span>
                    <span style="font-weight: bold;">${
                      detection.confidence
                    }%</span>
                </div>
            `
              )
              .join("")}
        </div>
    </div>

    ${
      reportData.ocrData && !reportData.ocrData.error
        ? `
    <div class="section">
        <h3>📄 Extracted Document Data</h3>
        <table class="ocr-table">
            <thead>
                <tr>
                    <th>Field</th>
                    <th>Value</th>
                </tr>
            </thead>
            <tbody>
                ${Object.entries(reportData.ocrData)
                  .map(
                    ([key, value]) => `
                    <tr>
                        <td><strong>${key
                          .replace(/([A-Z])/g, " $1")
                          .trim()}</strong></td>
                        <td>${value}</td>
                    </tr>
                `
                  )
                  .join("")}
            </tbody>
        </table>
    </div>
    `
        : ""
    }

    <div class="conclusion">
        <h3>🎯 Final Assessment</h3>
        <p>
            ${
              reportData.verificationStatus === "valid"
                ? "✅ This certificate appears to be <strong>AUTHENTIC</strong>. All detected regions show genuine characteristics and pass verification checks. The document can be considered legitimate based on the forensic analysis."
                : reportData.verificationStatus === "review"
                ? "⚠️ This certificate requires <strong>MANUAL REVIEW</strong>. Some regions show suspicious patterns that need human verification before making a final determination. Additional scrutiny is recommended."
                : "❌ This certificate may be <strong>FORGED</strong>. Suspicious regions detected with high confidence levels indicate potential document tampering or forgery. This document should not be accepted without further investigation."
            }
        </p>
    </div>

    <div class="footer">
        <p><strong>Report generated by Document Verification System</strong></p>
        <p>Generated on: ${new Date().toISOString()}</p>
        <p>This report contains embedded analysis visualization and comprehensive verification data.</p>
        <p>⚠️ This report is for verification purposes only and should be used in conjunction with other authentication methods.</p>
    </div>
</body>
</html>`;
    // Create a new window for PDF generation
    const printWindow = window.open("", "_blank");
    if (printWindow) {
      printWindow.document.write(htmlForPdf);
      printWindow.document.close();

      // Wait for content to load, then trigger print dialog
      printWindow.onload = () => {
        setTimeout(() => {
          printWindow.print();
          // Close the window after printing
          printWindow.onafterprint = () => {
            printWindow.close();
          };
        }, 500);
      };
    }

    toast({
      title: "Report Ready for Download",
      description: "Print dialog opened - save as PDF from the print menu.",
    });
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-background via-blue-50/30 to-accent/10 backdrop-blur-sm">
      <Navbar />

      <div className="container py-8 px-4 sm:px-6">
        <div className="max-w-7xl mx-auto">
          {/* Enhanced Header */}
          <div className="text-center mb-12 relative">
            <div className="absolute -top-4 left-1/2 transform -translate-x-1/2">
              <div className="w-24 h-24 bg-gradient-to-r from-primary/20 to-accent/20 rounded-full blur-2xl opacity-60"></div>
            </div>
            <div className="relative bg-white/80 backdrop-blur-sm rounded-2xl p-8 shadow-2xl border border-white/20">
              <div className="inline-flex items-center justify-center w-20 h-20 bg-gradient-to-r from-primary to-accent rounded-full shadow-lg mb-6">
                <ShieldCheck className="h-10 w-10 text-white" />
              </div>
              <h1 className="text-5xl font-display font-bold bg-gradient-to-r from-primary to-accent bg-clip-text text-transparent mb-4">
                Document Verification Workspace
              </h1>
              <p className="text-xl text-muted-foreground max-w-2xl mx-auto leading-relaxed">
                Upload your academic certificate for instant verification and fraud detection
              </p>
              <div className="flex justify-center space-x-4 mt-6">
                <Badge variant="secondary" className="px-4 py-2 bg-primary/10 text-primary">
                  <Sparkles className="h-3 w-3 mr-1" />
                  AI-Powered
                </Badge>
                <Badge variant="secondary" className="px-4 py-2 bg-green-500/10 text-green-600">
                  <FileCheck className="h-3 w-3 mr-1" />
                  Secure Verification
                </Badge>
              </div>
            </div>
          </div>

          {/* Enhanced Progress Indicator */}
          <div className="mb-12 bg-white/80 backdrop-blur-sm rounded-2xl p-6 shadow-2xl border border-white/20">
            <div className="flex items-center justify-between mb-6">
              {[
                { step: 1, icon: Upload, label: "Upload", color: "from-blue-500 to-cyan-500" },
                { step: 2, icon: Scan, label: "Extract", color: "from-purple-500 to-pink-500" },
                { step: 3, icon: Shield, label: "Verify", color: "from-green-500 to-emerald-500" }
              ].map(({ step, icon: Icon, label, color }, index) => (
                <div key={step} className="flex flex-col items-center space-y-3 flex-1">
                  <div className={`relative flex items-center justify-center w-16 h-16 rounded-full bg-gradient-to-r ${color} text-white shadow-lg transform transition-all duration-300 ${verificationStep >= step ? 'scale-110' : 'scale-100'}`}>
                    {verificationStep > step ? (
                      <CheckCircle className="h-8 w-8" />
                    ) : (
                      <Icon className="h-6 w-6" />
                    )}
                    <div className="absolute -top-2 -right-2 w-6 h-6 bg-white rounded-full border-2 border-current flex items-center justify-center">
                      <span className={`text-xs font-bold ${verificationStep >= step ? 'text-current' : 'text-gray-400'}`}>
                        {step}
                      </span>
                    </div>
                  </div>
                  <span className={`font-semibold transition-all duration-300 ${verificationStep >= step ? 'text-gray-900 scale-105' : 'text-gray-500'}`}>
                    {label}
                  </span>
                </div>
              ))}
            </div>
            <Progress 
              value={(verificationStep / 3) * 100} 
              className="h-3 bg-gradient-to-r from-blue-500 via-purple-500 to-green-500 rounded-full shadow-inner"
            />
          </div>

          {/* Enhanced Cards Grid */}
          <div className="grid lg:grid-cols-3 gap-8">
            {/* Panel 1: Upload & Capture */}
            <Card className="border-0 bg-gradient-to-br from-blue-50/80 to-white backdrop-blur-sm shadow-2xl hover:shadow-3xl transition-all duration-500 transform hover:-translate-y-1">
              <CardHeader className="pb-4">
                <CardTitle className="flex items-center space-x-3 text-2xl">
                  <div className="p-2 bg-gradient-to-r from-blue-500 to-cyan-500 rounded-lg shadow-lg">
                    <Upload className="h-6 w-6 text-white" />
                  </div>
                  <span className="bg-gradient-to-r from-blue-600 to-cyan-600 bg-clip-text text-transparent">
                    Upload Document
                  </span>
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-6">
                {!uploadedFile ? (
                  <div
                    {...getRootProps()}
                    className={`border-3 border-dashed rounded-2xl p-8 text-center cursor-pointer transition-all duration-300 group ${
                      isDragActive
                        ? "border-blue-500 bg-blue-50/50 scale-105"
                        : "border-gray-300/50 hover:border-blue-400 hover:bg-blue-50/30 hover:scale-[1.02]"
                    }`}
                  >
                    <input {...getInputProps()} />
                    <div className="group-hover:scale-110 transition-transform duration-300 mb-4">
                      <Upload className="h-16 w-16 text-blue-400/60 mx-auto group-hover:text-blue-500" />
                    </div>
                    <p className="text-xl font-semibold text-gray-700 mb-2 group-hover:text-gray-900">
                      Drop certificate here
                    </p>
                    <p className="text-sm text-gray-500 mb-6 group-hover:text-gray-600">
                      or click to browse files
                    </p>
                    <Badge variant="outline" className="bg-white/80 backdrop-blur-sm border-blue-200 text-blue-600 px-4 py-2">
                      PDF, JPG, PNG up to 10MB
                    </Badge>
                    <div className="mt-4 text-xs text-gray-400">
                      Secure • Fast • Accurate
                    </div>
                  </div>
                ) : (
                  <div className="space-y-6">
                    <div className="flex items-center space-x-4 p-5 bg-gradient-to-r from-blue-50 to-green-50 rounded-xl border border-blue-200/50 shadow-sm">
                      <div className="p-3 bg-blue-100 rounded-lg">
                        <FileText className="h-6 w-6 text-blue-600" />
                      </div>
                      <div className="flex-1">
                        <p className="font-semibold text-gray-800">{uploadedFile.name}</p>
                        <p className="text-sm text-gray-600">
                          {(uploadedFile.size / 1024 / 1024).toFixed(2)} MB • Ready for analysis
                        </p>
                      </div>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={resetVerification}
                        className="text-gray-400 hover:text-red-500 hover:bg-red-50 transition-colors"
                      >
                        <XCircle className="h-5 w-5" />
                      </Button>
                    </div>

                    {/* Enhanced Choose New Document */}
                    <div
                      {...getRootProps()}
                      className="border-2 border-dashed border-gray-300/50 rounded-xl p-5 text-center cursor-pointer transition-all duration-300 hover:border-blue-400 hover:bg-blue-50/30 hover:scale-[1.02] group"
                    >
                      <input {...getInputProps()} />
                      <Upload className="h-8 w-8 text-blue-400 mx-auto mb-3 group-hover:scale-110 transition-transform" />
                      <p className="text-sm font-medium text-gray-700 mb-1 group-hover:text-gray-900">
                        Choose New Document
                      </p>
                      <p className="text-xs text-gray-500 group-hover:text-gray-600">
                        Click or drop to upload a different file
                      </p>
                    </div>

                    {/* Enhanced Processing Status */}
                    {(isProcessingOCR || isProcessingVerification) && (
                      <div className="space-y-4 bg-gradient-to-r from-purple-50 to-pink-50 rounded-xl p-5 border border-purple-200/50">
                        <div className="flex items-center space-x-3">
                          <div className="relative">
                            <Clock className="h-6 w-6 text-purple-600 animate-spin" />
                            <div className="absolute inset-0 bg-purple-100 rounded-full animate-ping"></div>
                          </div>
                          <div className="flex-1">
                            <span className="font-medium text-gray-700">
                              {isProcessingOCR ? "Extracting text..." : "Verifying document..."}
                            </span>
                            <p className="text-xs text-gray-500 mt-1">
                              Analyzing document contents...
                            </p>
                          </div>
                        </div>
                        <Progress
                          value={isProcessingOCR ? 50 : 90}
                          className="h-2 bg-gray-200 rounded-full overflow-hidden"
                        >
                          <div className={`h-full bg-gradient-to-r from-purple-500 to-pink-500 rounded-full transition-all duration-1000 ease-out ${isProcessingOCR ? 'w-1/2' : 'w-11/12'}`} />
                        </Progress>
                      </div>
                    )}
                  </div>
                )}
              </CardContent>
            </Card>

            {/* Panel 2: OCR & Preview */}
            <Card className="border-0 bg-gradient-to-br from-purple-50/80 to-white backdrop-blur-sm shadow-2xl hover:shadow-3xl transition-all duration-500 transform hover:-translate-y-1">
              <CardHeader className="pb-4">
                <CardTitle className="flex items-center space-x-3 text-2xl">
                  <div className="p-2 bg-gradient-to-r from-purple-500 to-pink-500 rounded-lg shadow-lg">
                    <Scan className="h-6 w-6 text-white" />
                  </div>
                  <span className="bg-gradient-to-r from-purple-600 to-pink-600 bg-clip-text text-transparent">
                    Extracted Data
                  </span>
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-6">
                {isProcessingOCR ? (
                  <div className="text-center py-12">
                    <div className="relative inline-block mb-6">
                      <Scan className="h-16 w-16 text-purple-400/60 animate-pulse" />
                      <div className="absolute inset-0 bg-purple-200 rounded-full animate-ping"></div>
                    </div>
                    <p className="text-gray-600 font-medium">Extracting text from document...</p>
                    <p className="text-sm text-gray-400 mt-2">AI-powered text recognition in progress</p>
                  </div>
                ) : ocrData ? (
                  <div className="space-y-4">
                    <div className="bg-gradient-to-r from-green-50 to-emerald-50 rounded-xl p-4 border border-green-200/50">
                      <div className="flex items-center space-x-2 mb-3">
                        <CheckCircle className="h-5 w-5 text-green-600" />
                        <span className="font-semibold text-green-700">Data Extraction Complete</span>
                      </div>
                      <p className="text-sm text-green-600">Successfully extracted {Object.keys(ocrData).length} data fields</p>
                    </div>
                    
                    <div className="space-y-3 max-h-80 overflow-y-auto pr-2">
                      {Object.entries(ocrData).map(([key, value]) => (
                        <div
                          key={key}
                          className="flex justify-between items-center p-3 bg-white/50 backdrop-blur-sm rounded-lg border border-gray-200/50 hover:border-purple-200 hover:bg-purple-50/30 transition-all duration-200 group"
                        >
                          <span className="text-sm font-medium text-gray-700 capitalize group-hover:text-purple-700">
                            {key.replace(/([A-Z])/g, " $1")}:
                          </span>
                          <span className="text-sm bg-white px-2 py-1 rounded border text-gray-600 group-hover:text-gray-800">
                            {String(value)}
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                ) : (
                  <div className="text-center py-12">
                    <Scan className="h-16 w-16 text-gray-300 mx-auto mb-4" />
                    <p className="text-gray-500 font-medium">Waiting for document upload</p>
                    <p className="text-sm text-gray-400 mt-2">Extracted data will appear here</p>
                  </div>
                )}
              </CardContent>
            </Card>

            {/* Panel 3: Verification Results */}
            <Card className="border-0 bg-gradient-to-br from-green-50/80 to-white backdrop-blur-sm shadow-2xl hover:shadow-3xl transition-all duration-500 transform hover:-translate-y-1">
              <CardHeader className="pb-4">
                <CardTitle className="flex items-center space-x-3 text-2xl">
                  <div className="p-2 bg-gradient-to-r from-green-500 to-emerald-500 rounded-lg shadow-lg">
                    <Shield className="h-6 w-6 text-white" />
                  </div>
                  <span className="bg-gradient-to-r from-green-600 to-emerald-600 bg-clip-text text-transparent">
                    Verification Result
                  </span>
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-6">
                {isProcessingVerification ? (
                  <div className="text-center py-12">
                    <div className="relative inline-block mb-6">
                      <Shield className="h-16 w-16 text-green-400/60 animate-pulse" />
                      <div className="absolute inset-0 bg-green-200 rounded-full animate-ping"></div>
                    </div>
                    <p className="text-gray-600 font-medium">Analyzing document authenticity...</p>
                    <p className="text-sm text-gray-400 mt-2">Advanced fraud detection in progress</p>
                  </div>
                ) : verificationResult ? (
                  <div className="space-y-6">
                    {/* Enhanced Status Badge */}
                    <div className="text-center">
                      <Badge
                        variant="default"
                        className={`text-lg px-6 py-3 rounded-full shadow-lg border-0 ${
                          verificationResult.status === "valid"
                            ? "bg-gradient-to-r from-green-500 to-emerald-500 hover:from-green-600 hover:to-emerald-600"
                            : verificationResult.status === "review"
                            ? "bg-gradient-to-r from-yellow-500 to-orange-500 hover:from-yellow-600 hover:to-orange-600"
                            : "bg-gradient-to-r from-red-500 to-rose-500 hover:from-red-600 hover:to-rose-600"
                        } text-white font-bold transform hover:scale-105 transition-all duration-300`}
                      >
                        {verificationResult.status === "valid" && (
                          <CheckCircle className="h-5 w-5 mr-2 inline" />
                        )}
                        {verificationResult.status === "review" && (
                          <AlertTriangle className="h-5 w-5 mr-2 inline" />
                        )}
                        {verificationResult.status === "invalid" && (
                          <XCircle className="h-5 w-5 mr-2 inline" />
                        )}
                        {verificationResult.status === "valid"
                          ? "AUTHENTIC DOCUMENT"
                          : verificationResult.status === "review"
                          ? "NEEDS REVIEW"
                          : "SUSPICIOUS DOCUMENT"}
                      </Badge>
                    </div>

                    {/* Enhanced Visualization */}
                    {verificationResult.visualizationUrl && (
                      <div className="space-y-3">
                        <h4 className="font-semibold text-gray-700 flex items-center space-x-2">
                          <ZoomIn className="h-4 w-4 text-blue-500" />
                          <span>Detection Visualization</span>
                        </h4>
                        <Dialog>
                          <DialogTrigger asChild>
                            <div className="rounded-xl overflow-hidden border-2 border-gray-200/50 cursor-pointer hover:border-blue-300 transition-all duration-300 group relative shadow-lg hover:shadow-xl">
                              <img
                                src={verificationResult.visualizationUrl}
                                alt="Detection Results"
                                className="w-full h-auto group-hover:scale-105 transition-transform duration-500"
                              />
                              <div className="absolute inset-0 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-all duration-300 bg-black/20 backdrop-blur-sm">
                                <div className="bg-white/90 rounded-full p-3 transform group-hover:scale-110 transition-transform">
                                  <ZoomIn className="h-6 w-6 text-gray-700" />
                                </div>
                              </div>
                            </div>
                          </DialogTrigger>
                          <DialogContent className="max-w-6xl w-full max-h-[90vh] overflow-auto bg-white/95 backdrop-blur-sm border-0 shadow-2xl rounded-2xl">
                            <DialogHeader className="bg-gradient-to-r from-gray-50 to-white p-6 rounded-t-2xl border-b">
                              <DialogTitle className="flex items-center space-x-3 text-2xl">
                                <Shield className="h-6 w-6 text-green-600" />
                                <span className="bg-gradient-to-r from-gray-900 to-gray-700 bg-clip-text text-transparent">
                                  Document Verification Analysis
                                </span>
                              </DialogTitle>
                            </DialogHeader>
                            <div className="space-y-6 p-6">
                              {/* Large Image Display */}
                              <div className="rounded-xl overflow-hidden border-2 border-gray-200 shadow-lg">
                                <img
                                  src={verificationResult.visualizationUrl}
                                  alt="Detailed Detection Results"
                                  className="w-full h-auto"
                                />
                              </div>

                              {/* Enhanced Legend and Info */}
                              <div className="grid md:grid-cols-2 gap-6">
                                <div className="space-y-4 bg-gradient-to-br from-blue-50 to-cyan-50 rounded-xl p-5 border border-blue-200/50">
                                  <h4 className="font-semibold text-gray-700 flex items-center space-x-2">
                                    <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
                                    <span>Color Legend</span>
                                  </h4>
                                  <div className="space-y-3">
                                    <div className="flex items-center space-x-3 p-3 bg-white/50 rounded-lg border">
                                      <div className="w-6 h-6 bg-red-500 rounded border-2 border-white shadow"></div>
                                      <div>
                                        <span className="font-medium text-gray-700">Suspicious/Fake Regions</span>
                                        <p className="text-xs text-gray-500">Potential forgery detected</p>
                                      </div>
                                    </div>
                                    <div className="flex items-center space-x-3 p-3 bg-white/50 rounded-lg border">
                                      <div className="w-6 h-6 bg-green-500 rounded border-2 border-white shadow"></div>
                                      <div>
                                        <span className="font-medium text-gray-700">Authentic Regions</span>
                                        <p className="text-xs text-gray-500">Verified as genuine</p>
                                      </div>
                                    </div>
                                  </div>
                                </div>

                                <div className="space-y-4 bg-gradient-to-br from-green-50 to-emerald-50 rounded-xl p-5 border border-green-200/50">
                                  <h4 className="font-semibold text-gray-700 flex items-center space-x-2">
                                    <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                                    <span>Detection Summary</span>
                                  </h4>
                                  <div className="space-y-3">
                                    <div className="flex justify-between items-center p-3 bg-white/50 rounded-lg border">
                                      <span className="text-sm font-medium text-gray-600">Total Regions:</span>
                                      <span className="font-bold text-gray-800 text-lg">
                                        {verificationResult.summary.totalDetections}
                                      </span>
                                    </div>
                                    <div className="flex justify-between items-center p-3 bg-white/50 rounded-lg border">
                                      <span className="text-sm font-medium text-green-600">Authentic:</span>
                                      <span className="font-bold text-green-600 text-lg">
                                        {verificationResult.summary.trueDetections}
                                      </span>
                                    </div>
                                    <div className="flex justify-between items-center p-3 bg-white/50 rounded-lg border">
                                      <span className="text-sm font-medium text-red-600">Suspicious:</span>
                                      <span className="font-bold text-red-600 text-lg">
                                        {verificationResult.summary.fakeDetections}
                                      </span>
                                    </div>
                                  </div>
                                </div>
                              </div>

                              {/* Enhanced Detailed Detection List */}
                              <div className="space-y-4 bg-gradient-to-br from-purple-50 to-pink-50 rounded-xl p-5 border border-purple-200/50">
                                <h4 className="font-semibold text-gray-700 flex items-center space-x-2">
                                  <Scan className="h-4 w-4 text-purple-500" />
                                  <span>Detailed Analysis</span>
                                </h4>
                                <div className="max-h-60 overflow-y-auto space-y-2 pr-2">
                                  {verificationResult.detections.map(
                                    (detection, index) => (
                                      <div
                                        key={index}
                                        className={`p-4 rounded-xl border-2 transition-all duration-200 hover:scale-[1.02] ${
                                          detection.class_name === "fake"
                                            ? "bg-red-50/80 border-red-200 hover:border-red-300"
                                            : "bg-green-50/80 border-green-200 hover:border-green-300"
                                        }`}
                                      >
                                        <div className="flex items-center justify-between mb-2">
                                          <div className="flex items-center space-x-3">
                                            {detection.class_name === "fake" ? (
                                              <XCircle className="h-5 w-5 text-red-600" />
                                            ) : (
                                              <CheckCircle className="h-5 w-5 text-green-600" />
                                            )}
                                            <span
                                              className={`font-semibold ${
                                                detection.class_name === "fake"
                                                  ? "text-red-700"
                                                  : "text-green-700"
                                              }`}
                                            >
                                              Region {index + 1}:{" "}
                                              {detection.class_name === "fake"
                                                ? "Suspicious"
                                                : "Authentic"}
                                            </span>
                                          </div>
                                          <span
                                            className={`font-bold text-lg ${
                                              detection.class_name === "fake"
                                                ? "text-red-600"
                                                : "text-green-600"
                                            }`}
                                          >
                                            {Math.round(
                                              detection.confidence * 100
                                            )}
                                            %
                                          </span>
                                        </div>
                                        <div className="text-xs text-gray-500 bg-white/50 rounded px-2 py-1 inline-block">
                                          Coordinates: [
                                          {detection.bbox
                                            .map((coord) => Math.round(coord))
                                            .join(", ")}
                                          ]
                                        </div>
                                      </div>
                                    )
                                  )}
                                </div>
                              </div>
                            </div>
                          </DialogContent>
                        </Dialog>
                        <div className="flex items-center justify-center space-x-6 text-sm">
                          <div className="flex items-center space-x-2 bg-white/80 backdrop-blur-sm px-3 py-2 rounded-full border">
                            <div className="w-3 h-3 bg-red-500 rounded shadow"></div>
                            <span className="text-gray-700">Suspicious</span>
                          </div>
                          <div className="flex items-center space-x-2 bg-white/80 backdrop-blur-sm px-3 py-2 rounded-full border">
                            <div className="w-3 h-3 bg-green-500 rounded shadow"></div>
                            <span className="text-gray-700">Authentic</span>
                          </div>
                          <div className="flex items-center space-x-2 bg-blue-50/80 backdrop-blur-sm px-3 py-2 rounded-full border border-blue-200">
                            <ZoomIn className="w-3 h-3 text-blue-500" />
                            <span className="text-blue-600">Click to enlarge</span>
                          </div>
                        </div>
                      </div>
                    )}

                    {/* Enhanced Detection Summary */}
                    <div className="space-y-3">
                      <h4 className="font-semibold text-gray-700">Detection Summary</h4>
                      <div className="grid grid-cols-2 gap-4">
                        <div className="p-4 bg-gradient-to-br from-green-50 to-emerald-50 border-2 border-green-200 rounded-xl text-center shadow-sm hover:shadow-md transition-shadow">
                          <div className="font-bold text-3xl text-green-600 mb-1">
                            {verificationResult.summary.trueDetections}
                          </div>
                          <div className="text-sm text-green-700 font-medium">
                            Authentic Regions
                          </div>
                        </div>
                        <div className="p-4 bg-gradient-to-br from-red-50 to-rose-50 border-2 border-red-200 rounded-xl text-center shadow-sm hover:shadow-md transition-shadow">
                          <div className="font-bold text-3xl text-red-600 mb-1">
                            {verificationResult.summary.fakeDetections}
                          </div>
                          <div className="text-sm text-red-700 font-medium">
                            Suspicious Regions
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* Enhanced Detailed Detection List */}
                    <div className="space-y-3">
                      <h4 className="font-semibold text-gray-700">Detection Details</h4>
                      <div className="max-h-40 overflow-y-auto space-y-2 pr-2">
                        {verificationResult.detections.map(
                          (detection, index) => (
                            <div
                              key={index}
                              className={`flex items-center justify-between p-3 rounded-lg border-2 transition-all duration-200 hover:scale-[1.02] ${
                                detection.class_name === "fake"
                                  ? "bg-red-50/80 border-red-200 hover:border-red-300"
                                  : "bg-green-50/80 border-green-200 hover:border-green-300"
                              }`}
                            >
                              <div className="flex items-center space-x-3">
                                {detection.class_name === "fake" ? (
                                  <XCircle className="h-5 w-5 text-red-600" />
                                ) : (
                                  <CheckCircle className="h-5 w-5 text-green-600" />
                                )}
                                <span
                                  className={`font-medium ${
                                    detection.class_name === "fake"
                                      ? "text-red-700"
                                      : "text-green-700"
                                  }`}
                                >
                                  Region {index + 1}:{" "}
                                  {detection.class_name === "fake"
                                    ? "Suspicious"
                                    : "Authentic"}
                                </span>
                              </div>
                              <span
                                className={`font-bold text-lg ${
                                  detection.class_name === "fake"
                                    ? "text-red-600"
                                    : "text-green-600"
                                }`}
                              >
                                {Math.round(detection.confidence * 100)}%
                              </span>
                            </div>
                          )
                        )}
                      </div>
                    </div>

                    {/* Enhanced Conclusion */}
                    <div className="space-y-3">
                      <h4 className="font-semibold text-gray-700">Analysis Result</h4>
                      <div
                        className={`p-4 rounded-xl border-2 backdrop-blur-sm ${
                          verificationResult.status === "valid"
                            ? "bg-gradient-to-r from-green-50 to-emerald-50 border-green-200"
                            : verificationResult.status === "review"
                            ? "bg-gradient-to-r from-yellow-50 to-orange-50 border-yellow-200"
                            : "bg-gradient-to-r from-red-50 to-rose-50 border-red-200"
                        }`}
                      >
                        <p
                          className={`text-sm font-medium ${
                            verificationResult.status === "valid"
                              ? "text-green-700"
                              : verificationResult.status === "review"
                              ? "text-yellow-700"
                              : "text-red-700"
                          }`}
                        >
                          {verificationResult.status === "valid"
                            ? "✅ This certificate appears to be authentic. All detected regions show genuine characteristics and pass our advanced verification checks."
                            : verificationResult.status === "review"
                            ? "⚠️ This certificate requires manual review. Some regions show suspicious patterns that need expert verification before final determination."
                            : "❌ This certificate exhibits signs of potential forgery. Suspicious regions were detected with high confidence levels, indicating possible document tampering."}
                        </p>
                      </div>
                    </div>

                    {/* Enhanced Actions */}
                    <div className="w-full">
                      <Button
                        onClick={downloadReport}
                        className="w-full bg-gradient-to-r from-blue-500 to-cyan-500 hover:from-blue-600 hover:to-cyan-600 text-white font-semibold py-3 rounded-xl shadow-lg hover:shadow-xl transform hover:scale-[1.02] transition-all duration-300"
                      >
                        <Download className="h-5 w-5 mr-2" />
                        Download Comprehensive Report
                      </Button>
                    </div>
                  </div>
                ) : (
                  <div className="text-center py-12">
                    <Shield className="h-16 w-16 text-gray-300 mx-auto mb-4" />
                    <p className="text-gray-500 font-medium">Verification results will appear here</p>
                    <p className="text-sm text-gray-400 mt-2">Complete the upload process to see analysis</p>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
}