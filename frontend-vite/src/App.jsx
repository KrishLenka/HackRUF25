// src/App.jsx
import React from "react";
import { BrowserRouter, Routes, Route, Link } from "react-router-dom";
import { Activity, Upload, Home as HomeIcon } from "lucide-react";

import Home from "@/components/Home";          // ← make sure this file exists
import ImageUpload from "@/components/ImageUpload"; // ← and this too

const Header = () => (
  <header className="relative">
    {/* soft skin gradient background */}
    <div className="absolute inset-0 bg-gradient-to-b from-rose-50 via-orange-50 to-amber-100" />
    {/* subtle top glow */}
    <div className="pointer-events-none absolute -top-10 left-1/2 -translate-x-1/2 h-40 w-[36rem] rounded-full bg-white/50 blur-2xl opacity-60" />
    
    <div className="relative border-b border-rose-200/70 backdrop-blur-sm">
      <div className="w-full px-4">
        <div className="flex items-center justify-between py-4">
          <div className="flex items-center">
            <div className="w-12 h-12 bg-rose-600 rounded-lg flex items-center justify-center mr-4 shadow-sm">
              <Activity className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-bold text-rose-900 leading-tight">Skindex</h1>
              <p className="text-sm text-rose-800/80 leading-tight">Your trusted AI dermatology consultant</p>
            </div>
          </div>
        </div>

        <nav className="border-t border-rose-200/70 py-2">
          <div className="flex space-x-6 text-sm">
            <Link to="/" className="text-rose-900/90 hover:text-rose-900 font-medium py-2">
              <Upload className="w-4 h-4 inline mr-2" /> Home
            </Link>
            <Link to="/analyze" className="text-rose-900/90 hover:text-rose-900 font-medium py-2">
              <Upload className="w-4 h-4 inline mr-2" /> Image Analysis
            </Link>
          </div>
        </nav>
      </div>
    </div>
  </header>
);


export default function App() {
  return (
    <BrowserRouter>
      <Header />
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/analyze" element={<ImageUpload />} />
      </Routes>
    </BrowserRouter>
  );
}
