import React, { useState, useRef, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Textarea } from './ui/textarea';
import { 
  Send, 
  Activity, 
  User, 
  Clock,
  FileText,
  Home as HomeIcon,
  Search,
  Upload,
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
              className="text-blue-800 hover:text-blue-900 font-medium py-2 border-b-2 border-transparent hover:border-blue-800"
            >
              <Upload className="w-4 h-4 inline mr-2" />
              Image Analysis
            </Link>
            <Link 
              to="/assistant" 
              className="text-blue-800 hover:text-blue-900 font-medium py-2 border-b-2 border-blue-800"
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
          Clinical Resources
        </h3>
        <ul className="space-y-2 text-sm">
          <li>
            <a href="#" className="text-blue-800 hover:underline block py-1 font-medium">
              Clinical Decision Support
            </a>
          </li>
          <li>
            <a href="#" className="text-blue-800 hover:underline block py-1">
              Drug Interaction Checker
            </a>
          </li>
          <li>
            <a href="#" className="text-blue-800 hover:underline block py-1">
              Treatment Protocols
            </a>
          </li>
          <li>
            <a href="#" className="text-blue-800 hover:underline block py-1">
              ICD-10 Code Lookup
            </a>
          </li>
        </ul>
        
        <h3 className="font-bold text-gray-800 text-sm mb-4 mt-6 uppercase tracking-wide">
          Medical References
        </h3>
        <ul className="space-y-2 text-sm">
          <li>
            <a href="#" className="text-blue-800 hover:underline block py-1">
              Clinical Practice Guidelines
            </a>
          </li>
          <li>
            <a href="#" className="text-blue-800 hover:underline block py-1">
              Diagnostic Criteria Database
            </a>
          </li>
          <li>
            <a href="#" className="text-blue-800 hover:underline block py-1">
              Evidence-Based Medicine
            </a>
          </li>
          <li>
            <a href="#" className="text-blue-800 hover:underline block py-1">
              Recent Publications
            </a>
          </li>
        </ul>
      </div>
    </aside>
  );
};
*/

const AIAssistant = () => {
  const [messages, setMessages] = useState([
    {
      id: 1,
      type: 'assistant',
      content: 'I am an AI assistant trained on dermatological guidelines, diagnostic criteria, and evidence-based treatment information. I can provide insights on skin conditions, possible causes, treatment options, and guidance on when it may be appropriate to visit a healthcare professional. \n\nHow may I assist you with your skin concern today?',
      timestamp: new Date().toLocaleTimeString(),
      sessionId: 'CDS-001'
    }
  ]);
  const [inputMessage, setInputMessage] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const clinicalResponses = [
    {
      keywords: ['acne', 'comedones', 'papules', 'pustules'],
      response: 'ACNE VULGARIS - ICD-10: L70.0\n\nCLINICAL PRESENTATION: Inflammatory and non-inflammatory lesions including open/closed comedones, papules, pustules, and potentially nodules/cysts primarily affecting pilosebaceous units.\n\nDIAGNOSTIC CRITERIA: Clinical diagnosis based on characteristic lesion morphology and distribution. Severity grading: mild (comedonal), moderate (inflammatory papulopustular), severe (nodulocystic).\n\nTREATMENT PROTOCOL:\n- Mild: Topical retinoids + benzoyl peroxide or topical antibiotics\n- Moderate: Above + oral antibiotics (doxycycline 100mg BID)\n- Severe: Consider oral isotretinoin consultation\n\nMONITORING: Reassess at 8-12 weeks. Document response, side effects, adherence.\n\nWould you like specific information about treatment protocols, contraindications, or patient counseling points?'
    },
    {
      keywords: ['eczema', 'dermatitis', 'atopic', 'contact'],
      response: 'DERMATITIS CLASSIFICATION - Multiple ICD-10 codes depending on type\n\nDIFFERENTIAL DIAGNOSIS:\n- Atopic Dermatitis (L20): Chronic, pruritic, personal/family history of atopy\n- Contact Dermatitis (L23/L24): Acute/chronic, well-demarcated, exposure history\n- Seborrheic Dermatitis (L21): Erythematous scaling, sebaceous areas\n- Stasis Dermatitis (L87.2): Lower extremities, chronic venous insufficiency\n\nDIAGNOSTIC APPROACH:\n1. Detailed history (triggers, timeline, family history)\n2. Physical examination (distribution pattern, morphology)\n3. Consider patch testing for suspected contact allergens\n4. KOH prep if fungal infection suspected\n\nMANAGEMENT PRINCIPLES:\n- Trigger identification and avoidance\n- Skin barrier restoration (emollients)\n- Anti-inflammatory therapy (topical corticosteroids/calcineurin inhibitors)\n- Patient education on chronic disease management\n\nRequire specific guidance on a particular dermatitis subtype?'
    },
    {
      keywords: ['melanoma', 'mole', 'pigmented lesion', 'abcde'],
      response: 'MELANOCYTIC LESION EVALUATION - High Priority Assessment\n\nABCDE CRITERIA for Melanoma:\n- A: Asymmetry (one half unlike the other)\n- B: Border irregularity (scalloped, poorly defined)\n- C: Color variation (multiple colors within lesion)\n- D: Diameter >6mm (size of pencil eraser)\n- E: Evolution (changes in size, shape, color, symptoms)\n\nUGLY DUCKLING SIGN: Lesion that looks different from patient\'s other moles\n\nHIGH-RISK FEATURES:\n- New pigmented lesion in adult >30 years\n- Changing lesion (patient-reported evolution)\n- Symptomatic lesion (bleeding, itching, pain)\n- Irregular borders, multiple colors, diameter >6mm\n\nIMMEDIATE ACTION REQUIRED:\nAny suspicious lesion warrants urgent dermatology referral for dermoscopy and potential biopsy. DO NOT delay evaluation.\n\nDERMATOSCOPY: If available, assess for specific features (asymmetric pigment pattern, blue-white veil, irregular vessels).\n\nDOCUMENTATION: Photograph lesions, measure dimensions, document patient concerns.\n\nTimeframe for referral and patient counseling guidance needed?'
    }
  ];

  const commonQueries = [
    "Differential diagnosis for erythematous scaly patches",
    "First-line treatment options for moderate acne",
    "When to refer suspicious pigmented lesions",
    "Topical corticosteroid potency classification"
  ];

  const handleSendMessage = () => {
    if (!inputMessage.trim()) return;

    const newUserMessage = {
      id: Date.now(),
      type: 'user',
      content: inputMessage,
      timestamp: new Date().toLocaleTimeString(),
      sessionId: 'USER-001'
    };

    setMessages(prev => [...prev, newUserMessage]);
    setInputMessage('');
    setIsProcessing(true);

    // Simulate AI processing
    setTimeout(() => {
      const lowerInput = inputMessage.toLowerCase();
      let responseContent = "Thank you for your clinical inquiry. Based on current evidence-based guidelines and clinical protocols, I recommend consulting the most recent clinical practice guidelines for this condition. For specific patient cases, please ensure all clinical findings, patient history, and examination results are considered in your diagnostic and treatment decisions.\n\nCould you provide more specific clinical details about the presentation, patient demographics, or particular aspect of management you're seeking guidance on?";

      // Check for clinical responses
      for (let response of clinicalResponses) {
        if (response.keywords.some(keyword => lowerInput.includes(keyword))) {
          responseContent = response.response;
          break;
        }
      }

      const assistantMessage = {
        id: Date.now() + 1,
        type: 'assistant',
        content: responseContent,
        timestamp: new Date().toLocaleTimeString(),
        sessionId: 'CDS-001'
      };

      setMessages(prev => [...prev, assistantMessage]);
      setIsProcessing(false);
    }, 2000);
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const handleQueryClick = (query) => {
    setInputMessage(query);
  };

  return (
    <div className="min-h-screen bg-white">
      <Header />
      
      <div>
        <main className="w-full p-6">
          {/* Breadcrumb removed */}
          
          <div className="w-full">     
            <h1 className="text-2xl font-bold text-gray-900 mb-4">
              Clinical Decision Support System
            </h1>
            
            <div className="bg-yellow-50 border border-yellow-400 p-4 mb-6">
              <p className="text-yellow-800 text-sm">
                <strong>DISCLAIMER:</strong> This chatbot is intended to provide general health information and support for individuals who have not yet consulted a healthcare professional. It is not a substitute for medical advice, diagnosis, or treatment. If you have a pressing or serious health concern, you should consult a qualified healthcare provider for proper evaluation and care.
              </p>
            </div>

            <Card className="h-[600px] flex flex-col border-gray-300">
              <CardHeader className="bg-gray-50 border-b border-gray-300">
                <CardTitle className="text-base text-gray-900 flex items-center justify-between">
                  <div className="flex items-center">
                    <Activity className="w-5 h-5 text-blue-800 mr-2" />
                    Clinical Decision Support Interface
                  </div>
                  <div className="text-xs text-gray-600 bg-green-100 px-2 py-1 rounded border">
                    Session Active | Version 0.0.1
                  </div>
                </CardTitle>
              </CardHeader>

              {/* Messages Area */}
              <CardContent className="flex-1 overflow-y-auto p-4 space-y-4 bg-white">
                {messages.map((message) => (
                  <div
                    key={message.id}
                    className={`flex gap-3 ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}
                  >
                    {message.type === 'assistant' && (
                      <div className="w-10 h-10 bg-blue-800 flex items-center justify-center flex-shrink-0 text-white font-bold text-xs">
                        CDS
                      </div>
                    )}
                    
                    <div
                      className={`w-full border p-4 text-sm ${
                        message.type === 'user'
                          ? 'bg-blue-50 border-blue-200'
                          : 'bg-gray-50 border-gray-300'
                      }`}
                    >
                      <pre className="whitespace-pre-wrap font-sans leading-relaxed">{message.content}</pre>
                      <div className="flex items-center gap-2 mt-3 pt-2 border-t border-gray-200 text-xs text-gray-600">
                        <Clock className="w-3 h-3" />
                        <span>{message.timestamp}</span>
                        <span>•</span>
                        <span>Session: {message.sessionId}</span>
                      </div>
                    </div>

                    {message.type === 'user' && (
                      <div className="w-10 h-10 bg-gray-600 flex items-center justify-center flex-shrink-0">
                        <User className="w-5 h-5 text-white" />
                      </div>
                    )}
                  </div>
                ))}

                {isProcessing && (
                  <div className="flex gap-3 justify-start">
                    <div className="w-10 h-10 bg-blue-800 flex items-center justify-center flex-shrink-0 text-white font-bold text-xs">
                      CDS
                    </div>
                    <div className="bg-gray-50 border border-gray-300 p-4 text-sm">
                      <div className="flex items-center gap-2 text-gray-600">
                        <div className="w-2 h-2 bg-gray-500 rounded-full animate-pulse"></div>
                        <div className="w-2 h-2 bg-gray-500 rounded-full animate-pulse" style={{animationDelay: '0.2s'}}></div>
                        <div className="w-2 h-2 bg-gray-500 rounded-full animate-pulse" style={{animationDelay: '0.4s'}}></div>
                        <span className="ml-2">Processing clinical inquiry...</span>
                      </div>
                    </div>
                  </div>
                )}
                
                <div ref={messagesEndRef} />
              </CardContent>

              {/* Quick Queries */}
              <div className="border-t border-gray-300 p-4 bg-gray-50">
                <div className="flex items-center gap-2 mb-3">
                  <FileText className="w-4 h-4 text-blue-800" />
                  <span className="text-sm font-medium text-gray-800">Common Clinical Queries:</span>
                </div>
                <div className="grid grid-cols-2 gap-2 mb-4">
                  {commonQueries.map((query, index) => (
                    <Button
                      key={index}
                      variant="outline"
                      size="sm"
                      onClick={() => handleQueryClick(query)}
                      className="text-xs border-gray-400 hover:border-blue-600 hover:text-blue-800 justify-start text-left h-auto py-2 px-3"
                    >
                      {query}
                    </Button>
                  ))}
                </div>
                
                {/* Input Area */}
                <div className="flex gap-2">
                  <div className="flex-1">
                    <Textarea
                      value={inputMessage}
                      onChange={(e) => setInputMessage(e.target.value)}
                      onKeyPress={handleKeyPress}
                      placeholder="Enter your clinical question or patient presentation details..."
                      className="min-h-[80px] resize-none border-gray-400 text-sm"
                      disabled={isProcessing}
                    />
                  </div>
                  <Button
                    onClick={handleSendMessage}
                    disabled={!inputMessage.trim() || isProcessing}
                    className="bg-blue-800 hover:bg-blue-900 self-end px-6"
                  >
                    <Send className="w-4 h-4 mr-2" />
                    Submit
                  </Button>
                </div>
                
                {/* Confidentiality notice removed as requested */}
              </div>
            </Card>
          </div>
        </main>
      </div>
    </div>
  );
};

export default AIAssistant;