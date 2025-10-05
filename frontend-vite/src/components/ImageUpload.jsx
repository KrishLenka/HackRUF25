import React, { useState, useRef, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Progress } from './ui/progress';
import { Alert, AlertDescription } from './ui/alert';
import { Upload, FileText, AlertTriangle, CheckCircle } from 'lucide-react';

// map backend 'urgency' to a severity label for your UI
function mapUrgencyToSeverity(urgency) {
  switch ((urgency || '').toLowerCase()) {
    case 'urgent-dermatologist':
    case 'high':
      return 'High';
    case 'book-dermatologist':
    case 'telederm':
    case 'medium':
      return 'Moderate';
    case 'self-monitor':
    case 'low':
    default:
      return 'Low';
  }
}

const ImageUpload = () => {
  const [uploadedImage, setUploadedImage] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisProgress, setAnalysisProgress] = useState(0);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [error, setError] = useState('');
  const fileInputRef = useRef(null);
  const analysisRef = useRef(null);
  const [previewExpanded, setPreviewExpanded] = useState(true);

  const handleImageUpload = (event) => {
    setError('');
    const file = event.target.files[0];
    if (file && file.type.startsWith('image/')) {
      const reader = new FileReader();
      reader.onload = (e) => {
        setUploadedImage({
          file: file,
          preview: e.target.result,
          name: file.name,
          size: file.size
        });
        setAnalysisResult(null);
      };
      reader.readAsDataURL(file);
    } else {
      setUploadedImage(null);
      setError('Please choose a valid image file (JPEG/PNG/TIFF).');
    }
  };

  // Real backend call
  const handleAnalyze = async () => {
    if (!uploadedImage?.file) return;

    setIsAnalyzing(true);
    setError('');
    setAnalysisResult(null);
    setAnalysisProgress(5);

    const interval = setInterval(() => {
      setAnalysisProgress((p) => (p < 90 ? p + Math.random() * 10 : p));
    }, 250);

    try {
      const fd = new FormData();
      fd.append('file', uploadedImage.file); // Flask expects 'file'

      const url =
        (import.meta.env?.VITE_API_BASE
          ? `${import.meta.env.VITE_API_BASE}/predict`
          : '/api/predict');

      const res = await fetch(`/api/predict`, { method: 'POST', body: fd });

      const ct = res.headers.get('content-type') || '';
      console.log('predict status:', res.status, 'content-type:', ct);

      let data;
      if (ct.includes('application/json')) {
        data = await res.json();
      } else {
        const text = await res.text();
        console.warn('Non-JSON response:', text);
        if (!res.ok) throw new Error(text || `HTTP ${res.status}`);
        throw new Error('Server returned non-JSON response');
      }

      if (!res.ok) {
        throw new Error(data?.error || `HTTP ${res.status}`);
      }

      console.log('predict payload:', data);

      const severity = mapUrgencyToSeverity(data?.advice?.urgency);
      const confidencePct = Math.round((data?.final_confidence ?? 0) * 100);
      const alts = Array.isArray(data?.per_model_predictions)
        ? data.per_model_predictions
            .map((p) => p?.class)
            .filter((name) => !!name && name !== data?.prediction)
            .slice(0, 3)
        : [];

      setAnalysisResult({
        condition: data?.prediction || '—',
        icd10: undefined,
        confidence: Number.isFinite(confidencePct) ? confidencePct : undefined,
        severity,
        description: data?.advice?.short || '',
        recommendations: [data?.advice?.recommendation].filter(Boolean),
        areas: [],
        differentialDx: alts,
        _raw: data,
      });

      setAnalysisProgress(100);
    } catch (e) {
      console.error('analyze error:', e);
      setError(e?.message || 'Something went wrong during analysis.');
    } finally {
      clearInterval(interval);
      setIsAnalyzing(false);
    }
  };

  // scroll to results when ready
  useEffect(() => {
    if (analysisResult && analysisRef.current) {
      setTimeout(() => {
        try {
          analysisRef.current.scrollIntoView({ behavior: 'smooth', block: 'start' });
        } catch {
          const top = analysisRef.current.getBoundingClientRect().top + window.scrollY - 20;
          window.scrollTo({ top, behavior: 'smooth' });
        }
      }, 150);
    }
  }, [analysisResult]);

  const removeImage = () => {
    setUploadedImage(null);
    setAnalysisResult(null);
    setAnalysisProgress(0);
    setError('');
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  return (
    <div className="min-h-screen relative">
      {/* same warm skin-tone gradient as Home */}
      <div className="absolute inset-0 -z-10">
        <div className="h-full w-full bg-gradient-to-b from-rose-100 via-orange-50 to-amber-100" />
        {/* soft radial glow */}
        <div className="pointer-events-none absolute -top-24 left-1/2 -translate-x-1/2 h-[40rem] w-[40rem] rounded-full bg-white/40 blur-3xl opacity-50" />
      </div>

      {/* Top-right mini preview (only when results exist) */}
      {uploadedImage && analysisResult && (
        <div className="hidden md:block fixed top-16 right-6 z-50">
          <div
            role="button"
            tabIndex={0}
            onClick={() => setPreviewExpanded(prev => !prev)}
            onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); setPreviewExpanded(prev => !prev); } }}
            style={{ width: previewExpanded ? '368px' : '160px', height: previewExpanded ? '368px' : '160px', transition: 'width 200ms ease, height 200ms ease' }}
            className="border border-gray-300 overflow-hidden rounded shadow-lg bg-white cursor-pointer"
            title={previewExpanded ? 'Click to shrink preview' : 'Click to enlarge preview'}
            aria-pressed={!previewExpanded}
          >
            <img src={uploadedImage.preview} alt="mini-preview" className="w-full h-full object-cover" />
          </div>
          <div className="text-xs text-gray-700 text-center mt-2">
            <button
              onClick={() => setPreviewExpanded(prev => !prev)}
              className="underline focus:outline-none"
              aria-label={previewExpanded ? 'Shrink preview' : 'Enlarge preview'}
            >
              {previewExpanded ? 'Shrink' : 'Expand'}
            </button>
          </div>
        </div>
      )}

      <div>
        <main className="w-full p-6">
          <div className="w-full">      
            <h1 className="text-2xl font-bold text-gray-900 mb-4">
              Skindex Skin Consultation
            </h1>

            {/* Error banner */}
            {error && (
              <Alert className="mb-4 border-red-300 bg-red-50">
                <AlertDescription className="text-red-800">
                  {error}
                </AlertDescription>
              </Alert>
            )}

            {/* Softer instruction box to match palette (optional) */}
            <div className="bg-rose-50 border border-rose-200 p-4 mb-6">
              <p className="text-rose-900 text-sm">
                <strong>Instructions:</strong> Upload images of skin for AI-powered analysis. 
                Ensure images are well-lit, in focus, and show the entire area of concern. 
                System supports JPEG, PNG, and TIFF formats up to 10MB.
              </p>
            </div>

            <div className="flex flex-col gap-6">
              {/* Upload Section */}
              <Card className="border-gray-300">
                <CardHeader className="bg-gray-50 border-b border-gray-300">
                  <CardTitle className="text-base text-gray-900 flex items-center">
                    <Upload className="w-5 h-5 mr-2 text-rose-700" />
                    Image Upload Interface
                  </CardTitle>
                </CardHeader>
                <CardContent className="p-4">
                  {!uploadedImage ? (
                    <div className="border-2 border-dashed border-gray-400 p-12 text-center bg-gray-50">
                      <Upload className="w-20 h-20 text-gray-500 mx-auto mb-6" />
                      <p className="text-gray-700 mb-6 text-base">
                        Select a dermatological image to analyze. This is the main tool — upload a single image to begin.
                      </p>
                      <Button 
                        onClick={() => fileInputRef.current?.click()}
                        className="bg-rose-600 hover:bg-rose-700 mb-3 px-6 py-3"
                      >
                        <FileText className="w-4 h-4 mr-2" />
                        Browse Files
                      </Button>
                      <input
                        ref={fileInputRef}
                        type="file"
                        accept="image/*"
                        onChange={handleImageUpload}
                        className="hidden"
                      />
                      <p className="text-xs text-gray-600 mt-4">
                        <strong>Supported formats:</strong> JPEG, PNG, TIFF — up to 10MB
                      </p>
                    </div>
                  ) : (
                    <div className="space-y-4 flex flex-col items-center">
                      <div className="w-64 h-64 border border-gray-300 overflow-hidden">
                        <img src={uploadedImage.preview} alt="preview" className="w-full h-full object-cover" />
                      </div>
                      <div className="text-sm text-gray-700">
                        <div className="font-medium">{uploadedImage.name}</div>
                        <div>Size: {formatFileSize(uploadedImage.size)}</div>
                      </div>

                      {/* Progress while analyzing */}
                      {isAnalyzing && (
                        <div className="w-full">
                          <Progress value={analysisProgress} className="h-2" />
                          <p className="text-xs text-gray-600 mt-1">Analyzing…</p>
                        </div>
                      )}

                      <div className="flex gap-2 w-full">
                        <Button 
                          onClick={handleAnalyze} 
                          disabled={isAnalyzing}
                          className="flex-1 bg-green-700 hover:bg-green-800"
                        >
                          {isAnalyzing ? 'Processing...' : 'Analyze Image'}
                        </Button>
                        <Button variant="outline" onClick={removeImage} className="flex-none">
                          Remove
                        </Button>
                      </div>
                    </div>
                  )}
                </CardContent>
              </Card>

              {/* Results */}
              {analysisResult && (
                <div ref={analysisRef}>
                  <Card className="border-gray-300">
                    <CardHeader className="bg-gray-50 border-b border-gray-300">
                      <CardTitle className="text-base text-gray-900 flex items-center">
                        <FileText className="w-5 h-5 mr-2 text-rose-700" />
                        Analysis Results & Diagnostic Report
                      </CardTitle>
                    </CardHeader>
                    <CardContent className="p-4">
                      <div className="space-y-6 text-sm">
                        <Alert className="border-green-300 bg-green-50">
                          <CheckCircle className="h-4 w-4 text-green-700" />
                          <AlertDescription className="text-green-800">
                            <strong>Analysis Status:</strong> Completed successfully. Review results below.
                          </AlertDescription>
                        </Alert>

                        <div className="border border-gray-300 p-4 bg-white">
                          <h3 className="font-bold text-gray-900 mb-3 border-b border-gray-300 pb-2">PRIMARY DIAGNOSIS</h3>
                          <div className="grid grid-cols-2 gap-4 mb-3">
                            <div>
                              <p className="text-gray-700"><strong>Condition:</strong></p>
                              <p className="font-medium">{analysisResult.condition ?? 'N/A'}</p>
                            </div>
                            <div>
                              <p className="text-gray-700"><strong>ICD-10 Code:</strong></p>
                              <p className="font-medium">{analysisResult.icd10 ?? '—'}</p>
                            </div>
                            <div>
                              <p className="text-gray-700"><strong>Confidence Level:</strong></p>
                              <p className="font-medium">{typeof analysisResult.confidence !== 'undefined' ? `${analysisResult.confidence}%` : '—'}</p>
                            </div>
                            <div>
                              <p className="text-gray-700"><strong>Severity:</strong></p>
                              <p className="font-medium">{analysisResult.severity ?? '—'}</p>
                            </div>
                          </div>
                          <div>
                            <p className="text-gray-700 mb-2"><strong>Clinical Description:</strong></p>
                            <p className="text-gray-800">{analysisResult.description ?? ''}</p>
                          </div>
                        </div>

                        <div className="border border-gray-300 p-4 bg-white">
                          <h3 className="font-bold text-gray-900 mb-2">ANATOMICAL LOCATIONS</h3>
                          <ul className="list-disc list-inside text-gray-800 space-y-1">
                            {(analysisResult.areas || []).map((area, index) => (
                              <li key={index}>{area}</li>
                            ))}
                          </ul>
                        </div>

                        <div className="border border-gray-300 p-4 bg-white">
                          <h3 className="font-bold text-gray-900 mb-2">TREATMENT RECOMMENDATIONS</h3>
                          <ol className="list-decimal list-inside text-gray-800 space-y-2">
                            {(analysisResult.recommendations || []).map((rec, index) => (
                              <li key={index} className="text-sm">{rec}</li>
                            ))}
                          </ol>
                        </div>

                        <div className="border border-gray-300 p-4 bg-white">
                          <h3 className="font-bold text-gray-900 mb-2">DIFFERENTIAL DIAGNOSIS</h3>
                          <ul className="list-disc list-inside text-gray-800 space-y-1">
                            {(analysisResult.differentialDx || []).map((dx, index) => (
                              <li key={index}>{dx}</li>
                            ))}
                          </ul>
                        </div>

                        <Alert className="border-yellow-400 bg-yellow-50">
                          <AlertTriangle className="h-4 w-4 text-yellow-700" />
                          <AlertDescription className="text-yellow-800 text-xs">
                            <strong>DISCLAIMER:</strong> This analysis tool is intended to provide general health information and support for individuals who have not consulted a healthcare professional. It is not a substitute for medical advice, diagnosis, or treatment. If you have a pressing or serious health concern, you should consult a qualified healthcare provider for proper evaluation and care.
                          </AlertDescription>
                        </Alert>
                      </div>
                    </CardContent>
                  </Card>
                </div>
              )}
            </div>
          </div>
        </main>
      </div>
    </div>
  );
};

export default ImageUpload;
