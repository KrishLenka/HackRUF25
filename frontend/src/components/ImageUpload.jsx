import React, { useState, useRef, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Progress } from './ui/progress';
import { Alert, AlertDescription } from './ui/alert';
import { 
  Upload, 
  X, 
  Activity, 
  FileText,
  AlertTriangle,
  CheckCircle,
  Home as HomeIcon,
  Search,
  MessageCircle,
  Library
} from 'lucide-react';
import { Link } from 'react-router-dom';

const Header = () => {
  return (
    <header className="bg-white border-b-2 border-blue-800">
      {/* Top Government Bar */}
      <div className="bg-blue-800 text-white text-xs py-1">
          <div className="w-full px-4 flex justify-between items-center">
          <span>Made by Krish Lenka, Aarav Loomba, Ayaan Faisal, and Wayne Zhen</span>
          <span>Dermatology Analysis Tool</span>
        </div>
      </div>
      
  {/* Main Header */}
  <div className="w-full px-4">
        <div className="flex items-center justify-between py-4">
          <Link to="/home" className="flex items-center">
            <div className="w-12 h-12 bg-blue-800 flex items-center justify-center mr-4">
              <Activity className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-bold text-blue-800 leading-tight">
                Skindex
              </h1>
              <p className="text-sm text-gray-600 leading-tight">
                Your trusted AI dermatology consultant
              </p>
            </div>
          </Link>
        </div>
        
        {/* Navigation */}
        <nav className="border-t border-gray-300 py-2">
          <div className="flex space-x-8 text-sm">
            <Link 
              to="/home" 
              className="text-blue-800 hover:text-blue-900 font-medium py-2 border-b-2 border-transparent hover:border-blue-800"
            >
              <HomeIcon className="w-4 h-4 inline mr-2" />
              Home
            </Link>
            <Link 
              to="/" 
              className="text-blue-800 hover:text-blue-900 font-medium py-2 border-b-2 border-blue-800"
            >
              <Upload className="w-4 h-4 inline mr-2" />
              Image Analysis
            </Link>
            <Link 
              to="/assistant" 
              className="text-blue-800 hover:text-blue-900 font-medium py-2 border-b-2 border-transparent hover:border-blue-800"
            >
              <MessageCircle className="w-4 h-4 inline mr-2" />
              Clinical Assistant
            </Link>
            <Link 
              to="/library" 
              className="text-blue-800 hover:text-blue-900 font-medium py-2 border-b-2 border-transparent hover:border-blue-800"
            >
              <Library className="w-4 h-4 inline mr-2" />
              Component Library
            </Link>
          </div>
        </nav>
      </div>
    </header>
  );
};

/*
const Sidebar = () => {
  return (
    <aside className="w-64 bg-gray-100 border-r border-gray-300">
      <div className="p-4">
        <h3 className="font-bold text-gray-800 text-sm mb-4 uppercase tracking-wide">
          Analysis Tools
        </h3>
        <ul className="space-y-2 text-sm">
          <li>
            <a href="#" className="text-blue-800 hover:underline block py-1 font-medium">
              Image Upload & Analysis
                  <div className="w-full px-4">
          </li>
          <li>
            <a href="#" className="text-blue-800 hover:underline block py-1">
              Batch Processing
            </a>
          </li>
          <li>
            <a href="#" className="text-blue-800 hover:underline block py-1">
              Historical Results
            </a>
          </li>
          <li>
            <a href="#" className="text-blue-800 hover:underline block py-1">
              Export Reports
            </a>
          </li>
        </ul>
        
        <h3 className="font-bold text-gray-800 text-sm mb-4 mt-6 uppercase tracking-wide">
          Guidelines
        </h3>
        <ul className="space-y-2 text-sm">
          <li>
            <a href="#" className="text-blue-800 hover:underline block py-1">
              Image Quality Standards
            </a>
          </li>
          <li>
            <a href="#" className="text-blue-800 hover:underline block py-1">
              Diagnostic Criteria
            </a>
          </li>
          <li>
            <a href="#" className="text-blue-800 hover:underline block py-1">
              Privacy Compliance
            </a>
          </li>
        </ul>
      </div>
    </aside>
  );
};
*/

const ImageUpload = () => {
  const [uploadedImage, setUploadedImage] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisProgress, setAnalysisProgress] = useState(0);
  const [analysisResult, setAnalysisResult] = useState(null);
  const fileInputRef = useRef(null);
  const analysisRef = useRef(null);
  const [previewExpanded, setPreviewExpanded] = useState(true);

  const handleImageUpload = (event) => {
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
    }
  };

  const handleAnalyze = () => {
    if (!uploadedImage) return;
    
    setIsAnalyzing(true);
    setAnalysisProgress(0);
    
    // Simulate analysis progress
    const interval = setInterval(() => {
      setAnalysisProgress(prev => {
        if (prev >= 100) {
          clearInterval(interval);
          setIsAnalyzing(false);
          // Mock analysis result
          setAnalysisResult({
            condition: "Atopic Dermatitis (Eczema)",
            icd10: "L20.9",
            confidence: 87,
            severity: "Moderate",
            description: "Chronic inflammatory skin condition characterized by pruritic, erythematous lesions with scaling and lichenification.",
            recommendations: [
              "Topical corticosteroids (moderate potency) - Apply twice daily to affected areas",
              "Emollient therapy - Liberal use of fragrance-free moisturizers",
              "Trigger avoidance - Identify and avoid known allergens and irritants",
              "Follow-up consultation recommended within 2-3 weeks"
            ],
            areas: ["Antecubital fossa", "Dorsal forearm"],
            differentialDx: [
              "Contact dermatitis",
              "Seborrheic dermatitis", 
              "Psoriasis"
            ],
            additionalTests: [
              "Patch testing if contact allergy suspected",
              "KOH prep to rule out fungal infection"
            ]
          });
          return 100;
        }
        return prev + Math.random() * 15;
      });
    }, 300);
  };

  // When analysisResult becomes available, scroll to the analysis section
  useEffect(() => {
    if (analysisResult && analysisRef.current) {
      // small timeout to allow layout to settle
      setTimeout(() => {
        try {
          analysisRef.current.scrollIntoView({ behavior: 'smooth', block: 'start' });
        } catch (e) {
          // fallback: simple window scroll
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
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  return (
    <div className="min-h-screen bg-white">
      <Header />
      
      {/* Mini preview anchored to top-right when results exist (hidden on small screens) */}
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
          {/* Breadcrumb removed */}
          
          <div className="w-full">      
            <h1 className="text-2xl font-bold text-gray-900 mb-4">
              Skindex Skin Consultation
            </h1>
            
            <div className="bg-blue-50 border border-blue-200 p-4 mb-6">
              <p className="text-blue-800 text-sm">
                <strong>Instructions:</strong> Upload images of skin for AI-powered analysis. 
                Ensure images are well-lit, in focus, and show the entire area of concern. 
                System supports JPEG, PNG, and TIFF formats up to 10MB.
              </p>
            </div>

            <div className="flex flex-col gap-6">
              {/* Upload Section (main focus) */}
              <Card className="border-gray-300">
                <CardHeader className="bg-gray-50 border-b border-gray-300">
                  <CardTitle className="text-base text-gray-900 flex items-center">
                    <Upload className="w-5 h-5 mr-2 text-blue-800" />
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
                        className="bg-blue-800 hover:bg-blue-900 mb-3 px-6 py-3"
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
                    // Show compact image preview and analyze button when image is uploaded
                    <div className="space-y-4 flex flex-col items-center">
                      <div className="w-64 h-64 border border-gray-300 overflow-hidden">
                        <img src={uploadedImage.preview} alt="preview" className="w-full h-full object-cover" />
                      </div>
                      <div className="text-sm text-gray-700">
                        <div className="font-medium">{uploadedImage.name}</div>
                        <div>Size: {formatFileSize(uploadedImage.size)}</div>
                      </div>
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
              {/* Results area: hidden until analysis completes */}
              {analysisResult && (
                <div ref={analysisRef}>
                  <Card className="border-gray-300">
                    <CardHeader className="bg-gray-50 border-b border-gray-300">
                      <CardTitle className="text-base text-gray-900 flex items-center">
                        <FileText className="w-5 h-5 mr-2 text-blue-800" />
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