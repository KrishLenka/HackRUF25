import React from "react";
import "./App.css";
import { BrowserRouter, Routes, Route, Link } from "react-router-dom";
import { Card, CardContent, CardHeader, CardTitle } from "./components/ui/card";
import ComponentLibrary from "./components/ComponentLibrary";
import ImageUpload from "./components/ImageUpload";
import AIAssistant from "./components/AIAssistant";
import { Activity, Upload, MessageCircle, Library, Home as HomeIcon } from "lucide-react";

// Simple, single Header component (no breadcrumb, no search box)
const Header = () => (
  <header className="bg-white border-b-2 border-blue-800">
    <div className="bg-blue-800 text-white text-xs py-1">
      <div className="w-full px-4 flex justify-between items-center">
        <span>Made by Krish Lenka, Aarav Loomba, Ayaan Faisal, and Wayne Zhen</span>
        <span>Dermatology Analysis Tool</span>
      </div>
    </div>

    <div className="w-full px-4">
      <div className="flex items-center justify-between py-4">
        <Link to="/home" className="flex items-center">
          <div className="w-12 h-12 bg-blue-800 flex items-center justify-center mr-4">
            <Activity className="w-6 h-6 text-white" />
          </div>
          <div>
            <h1 className="text-xl font-bold text-blue-800 leading-tight">Skindex</h1>
            <p className="text-sm text-gray-600 leading-tight">Your trusted AI dermatology consultant</p>
          </div>
        </Link>
      </div>

      <nav className="border-t border-gray-300 py-2">
        <div className="flex space-x-8 text-sm">
          <Link to="/home" className="text-blue-800 hover:text-blue-900 font-medium py-2">
            <HomeIcon className="w-4 h-4 inline mr-2" /> Home
          </Link>
          <Link to="/" className="text-blue-800 hover:text-blue-900 font-medium py-2">
            <Upload className="w-4 h-4 inline mr-2" /> Image Analysis
          </Link>
          <Link to="/assistant" className="text-blue-800 hover:text-blue-900 font-medium py-2">
            <MessageCircle className="w-4 h-4 inline mr-2" /> Clinical Assistant
          </Link>
          <Link to="/library" className="text-blue-800 hover:text-blue-900 font-medium py-2">
            <Library className="w-4 h-4 inline mr-2" /> Component Library
          </Link>
        </div>
      </nav>
    </div>
  </header>
);

const Home = () => (
  <div className="min-h-screen bg-white">
    <Header />
    <main className="w-full p-6">
      <div className="w-full">
        <h1 className="text-2xl font-bold text-gray-900 mb-4">Skindex Platform</h1>

        <div className="bg-blue-50 border border-blue-200 p-4 mb-6">
          <h2 className="font-bold text-blue-900 mb-2">Advanced AI-Powered Dermatological Analysis System</h2>
          <p className="text-blue-800 text-sm">This platform provides patients and healthcare providers with cutting-edge artificial intelligence tools for dermatological condition analysis and consultation.</p>
        </div>

        <h2 className="text-lg font-bold text-gray-900 mb-4 border-b border-gray-300 pb-2">Available Services</h2>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
          <Card className="border-gray-300">
            <CardHeader className="bg-gray-50 border-b border-gray-300">
              <CardTitle className="text-base text-gray-900 flex items-center">
                <Upload className="w-5 h-5 mr-2 text-blue-800" /> Dermatological Image Analysis
              </CardTitle>
            </CardHeader>
            <CardContent className="p-4">
              <p className="text-sm text-gray-700 mb-3">Upload dermatological images for AI-powered analysis and condition identification. System provides detailed diagnostic information and clinical recommendations.</p>
              <Link to="/">
                <button className="bg-blue-800 hover:bg-blue-900 text-xs text-white px-3 py-2 rounded">Access Image Analysis</button>
              </Link>
            </CardContent>
          </Card>

          <Card className="border-gray-300">
            <CardHeader className="bg-gray-50 border-b border-gray-300">
              <CardTitle className="text-base text-gray-900 flex items-center">
                <MessageCircle className="w-5 h-5 mr-2 text-blue-800" /> Clinical Decision Support
              </CardTitle>
            </CardHeader>
            <CardContent className="p-4">
              <p className="text-sm text-gray-700 mb-3">Interactive clinical assistant providing evidence-based information on dermatological conditions, treatments, and diagnostic criteria.</p>
              <Link to="/assistant">
                <button className="bg-blue-800 hover:bg-blue-900 text-xs text-white px-3 py-2 rounded">Access Clinical Assistant</button>
              </Link>
            </CardContent>
          </Card>
        </div>
      </div>
    </main>

    <footer className="bg-gray-100 border-t-2 border-gray-300 py-6">
      <div className="w-full px-4">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 text-xs">
          <div>
            <h4 className="font-bold text-gray-900 mb-2">Contact Information</h4>
            <p className="text-gray-700">Wayne Zhen: wayne1zhen@gmail.com<br/>Aarav Loomba:<br/>Krish Lenka:<br/>Ayaan Faisal:</p>
          </div>
          <div>
            <h4 className="font-bold text-gray-900 mb-2">Policies</h4>
            <ul className="text-gray-700 space-y-1"><li><a href="#" className="hover:underline">Privacy Policy</a></li></ul>
          </div>
          <div>
            <h4 className="font-bold text-gray-900 mb-2">Related Sites</h4>
            <ul className="text-gray-700 space-y-1"><li><a href="#" className="hover:underline">NIH.gov</a></li></ul>
          </div>
          <div>
            <h4 className="font-bold text-gray-900 mb-2">Last Updated</h4>
            <p className="text-gray-700">Page last updated: September 2024<br/>Content reviewed: September 2024<br/>System version: 2.1.4</p>
          </div>
        </div>
        <div className="border-t border-gray-300 mt-4 pt-4 text-center">
          <p className="text-gray-600 text-xs">U.S. Department of Health and Human Services | National Institutes of Health | Dermatology Analysis Tool</p>
        </div>
      </div>
    </footer>
  </div>
);

function App() {
  return (
    <div className="App">
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<ImageUpload />} />
          <Route path="/home" element={<Home />} />
          <Route path="/assistant" element={<AIAssistant />} />
          <Route path="/library" element={<ComponentLibrary />} />
        </Routes>
      </BrowserRouter>
    </div>
  );
}

export default App;