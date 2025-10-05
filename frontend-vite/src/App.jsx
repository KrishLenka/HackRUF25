// frontend-vite/src/App.jsx
import React from "react";
import { BrowserRouter, Routes, Route, Link } from "react-router-dom";
import ImageUpload from "@/components/ImageUpload"; // uses alias '@' → ./src
import { Activity, Upload, MessageCircle } from "lucide-react";

const Header = () => (
  <header className="bg-white border-b-2 border-blue-800">
    <div className="w-full px-4">
      <div className="flex items-center justify-between py-4">
        <div className="flex items-center">
          <div className="w-12 h-12 bg-blue-800 flex items-center justify-center mr-4">
            <Activity className="w-6 h-6 text-white" />
          </div>
          <div>
            <h1 className="text-xl font-bold text-blue-800 leading-tight">Skindex</h1>
            <p className="text-sm text-gray-600 leading-tight">Your trusted AI dermatology consultant</p>
          </div>
        </div>
      </div>
      <nav className="border-t border-gray-300 py-2">
        <div className="flex space-x-8 text-sm">
          <Link to="/" className="text-blue-800 hover:text-blue-900 font-medium py-2 border-b-2 border-blue-800">
            <Upload className="w-4 h-4 inline mr-2" /> Image Analysis
          </Link>
          <Link to="/assistant" className="text-blue-800 hover:text-blue-900 font-medium py-2 border-b-2 border-transparent hover:border-blue-800">
            <MessageCircle className="w-4 h-4 inline mr-2" /> Clinical Assistant
          </Link>
        </div>
      </nav>
    </div>
  </header>
);

export default function App() {
  return (
    <BrowserRouter>
      <Header />
      <Routes>
        <Route path="/" element={<ImageUpload />} />
        {/* add more routes later */}
      </Routes>
    </BrowserRouter>
  );
}
